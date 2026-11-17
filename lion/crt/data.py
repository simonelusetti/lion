"""All eligible training focals and one uniformly selected CRT focal per sentence."""
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from ..datasets import UD_REPO
from ..labels import LABELS
from ..data import signature
from ..diagnostics.common import absolute,save_json,write_csv
from ..encoder import build_masked_encoder
from ..runtime import autocast,parallel

TASKS=['synthetic','upos','deprel']
CLASSES=[2,len(LABELS['ud_upos']),len(LABELS['ud_deprel'])]


def paths(cfg):
    key=signature(OmegaConf.create({'protocol':'crt-v1','encoder':cfg.encoder,'radius':cfg.crt.radius,'row_seed':cfg.crt.row_seed,'noise_seed':cfg.crt.noise_seed,'amp':cfg.runtime.amp}))
    return absolute(cfg.crt.results),absolute(cfg.crt.cache)/key


def window(ids,focal,cfg):
    lo=max(0,focal-cfg.crt.radius);hi=min(len(ids),focal+cfg.crt.radius+1)
    return [101,*ids[lo:hi],102],focal-lo+1


def prepare(cfg,output):
    results,cache=paths(cfg);cache.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(cfg.encoder.name,use_fast=True,local_files_only=True);vocab=tokenizer.get_vocab()
    def vocabulary(words):
        return {vocab[w] for word in words for w in (word,word.capitalize()) if w in vocab}
    determiners=vocabulary('the a an this that these those my your his her its our their'.split())
    prepositions=vocabulary('of in on at to for with from by'.split())
    noise=torch.Generator().manual_seed(cfg.crt.noise_seed);row_rng=torch.Generator().manual_seed(cfg.crt.row_seed)
    windows={};rows=[];counts=[];counter=0
    for split,source in [('train','train'),('validation','dev'),('test','test')]:
        file=hf_hub_download(UD_REPO,f'parquet/en_ewt/{source}.parquet',repo_type='dataset',local_files_only=True)
        raw=pq.read_table(file,columns=['tokens','upos','deprel','sent_id']).to_pylist()
        encoded=tokenizer([r['tokens'] for r in raw],is_split_into_words=True,add_special_tokens=False,truncation=False)
        part=[];skipped=0
        for sid,record in enumerate(raw):
            ids=encoded['input_ids'][sid];word_ids=encoded.word_ids(sid);word_count={w:word_ids.count(w) for w in set(word_ids) if w is not None}
            eligible=[(i,w) for i,w in enumerate(word_ids) if w is not None and word_count[w]==1]
            flipped=torch.rand(len(eligible),generator=noise)<.1;labels=[]
            for j,(position,word) in enumerate(eligible):
                oracle=(position>=2 and ids[position-2] in determiners) or (position+1<len(ids) and ids[position+1] in prepositions)
                labels.append([int(bool(oracle)^bool(flipped[j])),LABELS['ud_upos'].index(record['upos'][word]),LABELS['ud_deprel'].index(record['deprel'][word])])
                if split!='test':
                    value,focal=window(ids,position,cfg);part.append({'ids':value,'focal':focal,'y':labels[-1],'sentence':sid})
            if not eligible:skipped+=1;continue
            if len(ids)+2>512:raise ValueError(f'Full sentence exceeds MLM limit: {split}/{sid}')
            if split!='train':
                selected=int(torch.randint(len(eligible),(),generator=row_rng));position,word=eligible[selected]
                rows.append({'row':counter,'split':split,'sentence':sid,'source_id':record['sent_id'],'document':record['sent_id'].rsplit('-',1)[0],
                             'ids':ids,'focal':position,'word':word,'y':labels[selected]});counter+=1
        if split!='test':windows[split]=part
        counts.append({'split':split,'sentences':len(raw),'eligible_focals':sum(sum(v==w for v in encoded.word_ids(i))==1 for i in range(len(raw)) for w in set(encoded.word_ids(i)) if w is not None),
                       'no_eligible_sentence':skipped,'crt_rows':sum(r['split']==split for r in rows)})
    torch.save(windows,cache/'windows.pt');torch.save(rows,cache/'rows.pt')
    save_json(cache/'vocabulary.json',{'determiners':sorted(determiners),'prepositions':sorted(prepositions),'special_ids':tokenizer.all_special_ids,'cls':tokenizer.cls_token_id,'sep':tokenizer.sep_token_id,'mask':tokenizer.mask_token_id})
    write_csv(output/'samples.csv',counts);return {'crt_rows':len(rows),'train_focals':len(windows['train'])}


class Queries(Dataset):
    def __init__(self,items):self.items=items
    def __len__(self):return len(self.items)
    def __getitem__(self,index):return {**self.items[index],'index':index}


def collate(items):
    lengths=torch.tensor([len(i['ids']) for i in items]);ids=pad_sequence([torch.tensor(i['ids']) for i in items],batch_first=True)
    result={'ids':ids,'attention_mask':torch.arange(ids.shape[1])[None]<lengths[:,None],
            'positions':torch.tensor([i['focal'] for i in items]),'index':torch.tensor([i['index'] for i in items])}
    if 'y' in items[0]:result['y']=torch.tensor([i['y'] for i in items]);result['sentence']=torch.tensor([i['sentence'] for i in items])
    return result


def model(cfg):
    dtype=torch.float16 if cfg.runtime.device.startswith('cuda') and cfg.runtime.amp=='fp16' else torch.float32
    return parallel(build_masked_encoder(cfg.encoder).to(device=cfg.runtime.device,dtype=dtype).eval(),cfg)


@torch.no_grad()
def features(encoder,items,cfg):
    result=torch.empty(len(items),768,dtype=torch.float16)
    for batch in DataLoader(Queries(items),batch_size=cfg.runtime.batch_size,collate_fn=collate,num_workers=cfg.runtime.workers):
        with autocast(cfg):value=encoder(batch['ids'].to(cfg.runtime.device),batch['attention_mask'].to(cfg.runtime.device),batch['positions'].to(cfg.runtime.device))['features']
        result[batch['index']]=value.cpu().half()
    return result


def extract(cfg,output):
    _,cache=paths(cfg);windows=torch.load(cache/'windows.pt',weights_only=False);encoder=model(cfg)
    for split,items in windows.items():
        target=cache/(split+'.pt')
        if cfg.crt.resume and target.exists():continue
        values=features(encoder,items,cfg)
        torch.save({'features':values,'y':torch.tensor([i['y'] for i in items]),'sentence':torch.tensor([i['sentence'] for i in items])},target)
        print(f'Cached {split}: {len(items)} full-train eligible focal windows',flush=True)
    return {'splits':len(windows)}
