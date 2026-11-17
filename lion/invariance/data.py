"""All-word cached block-12 vectors and genre-held-out rotation plans."""
import json
import math

import torch
from datasets import load_from_disk
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer

from ..cache import cache_path
from ..data import signature
from ..diagnostics.common import absolute, dataset_config, write_csv
from ..diagnostics.data import token_path, metadata
from ..labels import LABELS
from ..partition.data import ud_raw


def paths(cfg):
    key = signature(OmegaConf.create({"encoder": cfg.encoder, "dtype": cfg.cache.dtype,
                                      "amp": cfg.runtime.amp, "tf32": cfg.runtime.tf32,
                                      "protocol": "all-word-invariance-v1"}))
    return absolute(cfg.invariance.results), absolute(cfg.invariance.cache) / key


def prepare(cfg, output):
    results, cache = paths(cfg)
    cache.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    manifest, sizes = {}, []
    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder.name, use_fast=True, local_files_only=True)
    for source in ("ud_upos", "ud_discourse"):
        target = cache / (source + "_vectors.pt")
        if target.exists(): continue
        local = dataset_config(cfg, source)
        folder = absolute(cache_path(local))
        tokens = load_from_disk(token_path(local))
        vectors = {}
        for split in tokens:
            chunks, counts, sentences = [], [], []
            for path in sorted((folder / split).glob("*.pt")):
                saved = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
                chunks.append(saved["features"][:, 12].clone())
                counts.extend(saved["counts"].tolist()); sentences.extend(saved["sentence_ids"].tolist())
            if sentences != list(range(len(tokens[split]))) or counts != [len(r) for r in tokens[split]["words"]]:
                raise ValueError(f"Full-cache sentence alignment mismatch: {source}/{split}")
            vectors[split] = torch.cat(chunks)
        torch.save(vectors, target)
    gum = None
    for name in cfg.invariance.datasets:
        target = cache / (name + ".pt")
        if not target.exists():
            source = "ud_discourse" if name in {"ud_discourse", "gum_upos"} else name
            local = dataset_config(cfg, source)
            tokens = load_from_disk(token_path(local))
            raw = metadata(source)
            genres = sorted({r["genre"] for rows in raw.values() for r in rows})
            labels = LABELS["ud_upos"] if name == "gum_upos" else LABELS[name]
            if name == "gum_upos" and gum is None: gum = ud_raw("en_gum")
            ys, zs, words, documents, sentence_z, sentence_split, offsets = [], [], [], [], [], [], [0]
            for native, (split, dataset) in enumerate(tokens.items()):
                for row in dataset:
                    y = row["labels"]
                    if name == "gum_upos":
                        original = gum[split][row["sentence_id"]]
                        encoded = tokenizer(original["tokens"], is_split_into_words=True, truncation=True,
                                            max_length=cfg.encoder.max_length, add_special_tokens=cfg.encoder.special_tokens)
                        if encoded["input_ids"] != row["ids"]: raise ValueError("GUM UPOS alignment mismatch")
                        ids = encoded.word_ids()
                        kept = [w for i,w in enumerate(ids) if w is not None and (i==0 or w!=ids[i-1])]
                        y = [labels.index(original["upos"][w]) for w in kept]
                    info = raw[split][row["sentence_id"]]
                    z = genres.index(info["genre"])
                    ys += y; zs += [z]*len(y); words += row["words"]
                    documents += [info["document"]]*len(y)
                    sentence_z.append(z); sentence_split.append({"train":0,"validation":1,"test":2}[split])
                    offsets.append(offsets[-1]+len(y))
                sizes.append({"dataset":name,"split":split,"sentences":len(dataset),"words":sum(len(x) for x in dataset['words'])})
            vocab = {w:i for i,w in enumerate(sorted(set(words)))}
            torch.save({"y":torch.tensor(ys),"z":torch.tensor(zs),"type":torch.tensor([vocab[w] for w in words]),
                        "vocab":list(vocab),"document":documents,"offsets":torch.tensor(offsets),
                        "sentence_z":torch.tensor(sentence_z),"sentence_split":torch.tensor(sentence_split),
                        "labels":labels,"genres":genres,
                        "vectors":str(cache/(("ud_discourse" if source=="ud_discourse" else "ud_upos")+'_vectors.pt'))},target)
        manifest[name] = str(target)
    (results/'banks.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if sizes: write_csv(results/'samples.csv',sizes)
    return {"datasets":len(manifest)}


def banks(cfg):
    results,_=paths(cfg)
    for name,path in json.loads((results/'banks.json').read_text()).items():
        bank=torch.load(path,map_location='cpu',weights_only=False)
        source=torch.load(bank['vectors'],map_location='cpu',mmap=True,weights_only=False)
        bank['features']=torch.cat(list(source.values()))
        yield name,bank


def random_types(frequency, target, generator):
    """Random types with exact training-token mass; reserve singleton types to close the count."""
    total=int(frequency.sum())
    if target==0:return torch.zeros(len(frequency),dtype=torch.bool)
    if target==total:return torch.ones(len(frequency),dtype=torch.bool)
    assignment=torch.rand(len(frequency),generator=generator)<target/total
    assignment[frequency>0]=False
    ones=(frequency==1).nonzero().flatten()
    other=(frequency>1).nonzero().flatten()
    remaining=target
    for i in other[torch.randperm(len(other),generator=generator)].tolist():
        if remaining<=len(ones):break
        if int(frequency[i])<=remaining:
            assignment[i]=True;remaining-=int(frequency[i])
    if remaining>len(ones):raise ValueError('Not enough singleton types to close random-split token share')
    assignment[ones[torch.randperm(len(ones),generator=generator)[:remaining]]]=True
    return assignment


def heads(cfg):
    result=[{'rule':'R0','lambda':0.,'threshold':-1.,'draw':-1}]
    for rule in ('R1','R2'):
        result += [dict(rule=rule,**{'lambda':float(lam)},threshold=-1.,draw=-1) for lam in cfg.invariance.strengths]
    for threshold in cfg.invariance.thresholds:
        for rule in ('R3','R4','R5'):
            for draw in (range(cfg.invariance.random_draws) if rule=='R5' else [-1]):
                result += [dict(rule=rule,**{'lambda':float(lam)},threshold=float(threshold),draw=draw) for lam in cfg.invariance.strengths]
    return result


def rotations(name,bank,cfg):
    tests=cfg.invariance.gum_test_genres if name in {'gum_upos','ud_discourse'} else bank['genres']
    for genre in tests:
        test=bank['genres'].index(genre); validation=(test+1)%len(bank['genres'])
        train=[i for i in range(len(bank['genres'])) if i not in (test,validation)]
        yield genre,train,validation,test


def plan(bank, train_genres, cfg, specifications=None):
    sentence_train=torch.isin(bank['sentence_z'],torch.tensor(train_genres)) & (bank['sentence_split']==0)
    token_train=torch.repeat_interleave(sentence_train,bank['offsets'].diff()) & (bank['y']>=0)
    classes=len(bank['labels']); types=len(bank['vocab'])
    counts=torch.bincount(bank['type'][token_train]*classes+bank['y'][token_train],minlength=types*classes).reshape(types,classes)
    frequency=counts.sum(-1); probability=counts.float()/frequency[:,None].clamp_min(1)
    entropy=-torch.special.xlogy(probability,probability).sum(-1)
    groups={};assignments={};generator=torch.Generator().manual_seed(cfg.diagnostics.sample_seed)
    for threshold in cfg.invariance.thresholds:
        group=torch.where(frequency<cfg.invariance.minimum_frequency,2,torch.where(entropy<=threshold,0,1))
        groups[float(threshold)]=group
        target=int(frequency[group==0].sum())
        for draw in range(cfg.invariance.random_draws):
            assignments[(float(threshold),draw)]=random_types(frequency,target,generator)
    specifications=heads(cfg) if specifications is None else specifications
    anti=torch.stack([torch.ones(types,dtype=torch.bool) if h['rule']=='R1' else
                      torch.zeros(types,dtype=torch.bool) if h['rule'] in {'R0','R2'} else
                      groups[h['threshold']]==0 if h['rule']=='R3' else
                      groups[h['threshold']]!=0 if h['rule']=='R4' else assignments[(h['threshold'],h['draw'])]
                     for h in specifications],dim=1)
    joint=torch.bincount(bank['z'][token_train]*classes+bank['y'][token_train],minlength=len(bank['genres'])*classes).reshape(-1,classes)[train_genres].double()
    p=joint/joint.sum(-1,keepdim=True)
    binary=torch.stack([p,1-p],dim=-1)
    mean=binary.mean(0)
    shift=(torch.special.xlogy(binary,binary)-torch.special.xlogy(binary,mean)).sum(-1).mean(0).float()
    train_scores=shift[bank['y'][token_train]]
    cuts=torch.quantile(train_scores,torch.tensor([1/3,2/3]))
    return {'train_sentences':sentence_train.nonzero().flatten(),'training_tokens':token_train,
            'frequency':frequency,'entropy':entropy,'groups':groups,'assignments':anti,'heads':specifications,
            'label_shift':shift,'shift_cuts':cuts}


class Sentences(Dataset):
    def __init__(self,bank,ids):self.bank,self.ids=bank,ids
    def __len__(self):return len(self.ids)
    def __getitem__(self,index):
        sentence=int(self.ids[index]);a,b=self.bank['offsets'][sentence:sentence+2].tolist()
        valid=self.bank['y'][a:b]>=0
        return {'x':self.bank['features'][a:b][valid].float(),'y':self.bank['y'][a:b][valid],
                'z':self.bank['z'][a:b][valid],'type':self.bank['type'][a:b][valid],
                'index':torch.arange(a,b)[valid]}


def collate(rows):return {key:torch.cat([row[key] for row in rows]) for key in rows[0]}


class GenreBatches(Sampler):
    def __init__(self,dataset,genres,batch_size,seed):
        self.groups=[(dataset.bank['sentence_z'][dataset.ids]==g).nonzero().flatten() for g in genres]
        self.per=max(1,batch_size//len(genres));self.seed=seed;self.epoch=0
        self.steps=max(math.ceil(len(g)/self.per) for g in self.groups)
    def __len__(self):return self.steps
    def __iter__(self):
        generator=torch.Generator().manual_seed(self.seed+self.epoch);self.epoch+=1
        columns=[]
        for group in self.groups:
            needed=self.steps*self.per;chunks=[]
            while needed>0:
                order=group[torch.randperm(len(group),generator=generator)][:needed]
                chunks.append(order);needed-=len(order)
            columns.append(torch.cat(chunks).reshape(self.steps,self.per))
        for step in range(self.steps):yield torch.cat([c[step] for c in columns]).tolist()


def loader(bank,ids,cfg,train_genres=None,seed=0):
    dataset=Sentences(bank,ids)
    options={'collate_fn':collate,'num_workers':cfg.runtime.workers,'pin_memory':cfg.runtime.device.startswith('cuda')}
    if train_genres is not None:return DataLoader(dataset,batch_sampler=GenreBatches(dataset,train_genres,cfg.train.batch_size,seed),**options)
    return DataLoader(dataset,batch_size=cfg.runtime.batch_size,shuffle=False,**options)
