"""Seed-paired arms with identical optimizer steps and labelled-token batch budgets."""
import time
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from tagger.ensemble import IndependentReadouts,clip_heads
from ..runtime import autocast,parallel
from ..diagnostics.common import save_json,write_csv
from ..partition.train import predict,metric_vectors
from ..certification import data
from ..certification.classify import span_counts,evaluate
from ..labels import LABELS
from ..encoder import load_masked_encoder
from .data import paths,ARMS


def schedule(sentence,cfg,seed):
    order=sentence.argsort(stable=True);_,counts=sentence[order].unique_consecutive(return_counts=True)
    offsets=torch.cat([torch.zeros(1,dtype=torch.long),counts.cumsum(0)]);words=[order[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
    generator=torch.Generator().manual_seed(seed);stream=[];sizes=[];epochs=[];permutations=[]
    for epoch in range(cfg.train.epochs):
        permutation=torch.randperm(len(words),generator=generator) if cfg.train.shuffle else torch.arange(len(words));permutations.append(permutation)
        stream.append(torch.cat([words[int(i)] for i in permutation]))
        lengths=counts[permutation]
        sizes.extend(int(part.sum()) for part in lengths.split(cfg.train.batch_size));epochs.extend([epoch+1]*len(lengths.split(cfg.train.batch_size)))
    stream=torch.cat(stream);batches=[];baseline=original=variant=carry=0
    for size,epoch in zip(sizes,epochs):
        n_original=(size+carry)//2;carry=(size+carry)%2;n_variant=size-n_original
        batches.append({'baseline':stream[baseline:baseline+size],'original':stream[original:original+n_original],'variant':stream[variant:variant+n_variant],'epoch':epoch})
        baseline+=size;original+=n_original;variant+=n_variant
    return batches,{'sentence_permutations':torch.stack(permutations),'original_stream':stream,'labelled_tokens_per_step':torch.tensor(sizes),
                    'baseline_original_presentations':baseline,'augmented_original_presentations':original,'augmented_variant_presentations':variant,
                    'optimizer_steps':len(batches),'sentence_batch_size':cfg.train.batch_size}


class Steps(Dataset):
    def __init__(self,features,batches):self.features,self.batches=features,batches
    def __len__(self):return len(self.batches)
    def __getitem__(self,index):
        batch=self.batches[index];f=self.features;original=batch['original'];variant=batch['variant']
        x=torch.stack([f['raw'][batch['baseline']],torch.cat([f['raw'][original],f['selector'][variant]]),torch.cat([f['raw'][original],f['random'][variant]])],dim=1).float()
        augmented=torch.cat([f['y'][original],f['y'][variant]])
        return {'x':x,'y':torch.stack([f['y'][batch['baseline']],augmented,augmented],dim=1),'epoch':batch['epoch'],'step':index}


def collate(rows):return rows[0]


@torch.no_grad()
def evaluation_data(cfg,output):
    target=output/'evaluation.pt'
    if cfg.augmentation.resume and target.exists():return torch.load(target,mmap=True,weights_only=False)
    bank=data.load(cfg);sets=dict(bank['sets']);full=torch.load(data.paths(cfg)/'scorer'/'full_features.pt',mmap=True,weights_only=False)
    start=max(r['sentence'] for rows in sets.values() for r in rows)+1
    transfer,domains,coverage=data.ner_transfer_sets(cfg,'ner',start)
    if {r['target'] for r in coverage if r['status']=='available'}!={'wnut','wikiann'}:raise FileNotFoundError('Both raw WNUT and WikiANN test sets are required.')
    sets.update(transfer);bank['domains'].update(domains);encoder=load_masked_encoder(cfg)
    full.update(data.full_features(encoder,{name:rows for name,rows in sets.items() if name!='train' and name not in full},cfg))
    evaluated={name:{'data':data.readout_data(rows,full[name],'ner'),'target':bank['domains'][name]} for name,rows in sets.items() if name!='train'}
    result={'sets':evaluated,'labels':bank['labels']['ner']};torch.save(result,target);write_csv(output/'target_coverage.csv',coverage)
    return result


def experiment(cfg,output):
    source=paths(cfg);f=torch.load(source/'generate'/'features.pt',mmap=True,weights_only=False);evaluations=evaluation_data(cfg,output);labels=evaluations['labels'];validation=evaluations['sets']['validation']['data'];records=[];timings=[]
    for seed in cfg.augmentation.seeds:
        folder=output/f'seed{seed}';folder.mkdir(exist_ok=True)
        if cfg.augmentation.resume and (folder/'performance.pt').exists():
            saved=torch.load(folder/'performance.pt',weights_only=False);records.extend(saved['records']);timings.append(saved['timing']);continue
        started=time.monotonic();torch.manual_seed(seed);core=IndependentReadouts(cfg.tagger,768,len(labels),3).to(cfg.runtime.device);model=parallel(core,cfg)
        optimizer=torch.optim.AdamW(core.parameters(),lr=cfg.train.lr,weight_decay=cfg.train.weight_decay)
        batches,plan=schedule(f['sentence'],cfg,seed);torch.save(plan,folder/'schedule.pt')
        steps=DataLoader(Steps(f,batches),batch_size=1,collate_fn=collate,num_workers=cfg.runtime.workers,pin_memory=cfg.runtime.device.startswith('cuda'))
        best=torch.full((3,),-torch.inf);best_epoch=torch.zeros(3,dtype=torch.long);state={k:v.detach().cpu().clone() for k,v in core.state_dict().items()};history=[]
        fit_start=time.monotonic()
        for number,batch in enumerate(steps):
            model.train();optimizer.zero_grad(set_to_none=True)
            with autocast(cfg):logits=model(batch['x'].to(cfg.runtime.device)).float()
            ce=F.cross_entropy(logits.permute(1,2,0),batch['y'].to(cfg.runtime.device).T,reduction='none').mean(-1)
            ce.sum().backward()
            if cfg.train.clip is not None:clip_heads(core,cfg.train.clip)
            optimizer.step()
            epoch=batch['epoch']
            if number+1<len(batches) and batches[number+1]['epoch']==epoch:continue
            scores=metric_vectors(predict(model,validation,cfg),validation['y'],len(labels));criterion=scores['macro_f1'];improved=criterion>best
            for k,v in core.state_dict().items():state[k][improved]=v.detach().cpu()[improved]
            best[improved]=criterion[improved];best_epoch[improved]=epoch
            for h,arm in enumerate(ARMS):history.append({'epoch':epoch,'arm':arm,'validation_macro_f1':float(criterion[h]),'validation_accuracy':float(scores['accuracy'][h])})
            print(f'seed{seed} epoch {epoch}: '+', '.join(f'{arm} dev F1 {float(criterion[h]):.4f}' for h,arm in enumerate(ARMS)),flush=True)
        fit_seconds=time.monotonic()-fit_start;core.load_state_dict(state)
        torch.save({'state':state,'best_epoch':best_epoch,'validation_macro_f1':best,'labels':labels,'seed':seed,'heads':3},folder/'model.pt');write_csv(folder/'history.csv',history)
        seed_records=[]
        for split,item in evaluations['sets'].items():
            if split=='validation':continue
            target=item['target'];part=item['data'];logits=predict(model,part,cfg)
            spaces=['native','shared_ner'] if split=='id' else ['shared_ner']
            for space in spaces:
                metrics=evaluate(logits,part,'conll2003',target,labels,space)
                correct,proposed,expected=span_counts(logits.argmax(-1),part['y'],part['sentence'],labels,LABELS[target] if space=='shared_ner' else labels,shared=space=='shared_ner')
                for h,arm in enumerate(ARMS):seed_records.append({'seed':seed,'arm':arm,'target':target,'split':split,'label_space':space,'span_f1':float(metrics['span_f1'][h]),
                                                                 'precision':float(correct[h]/proposed[h].clamp_min(1)),'recall':float(correct[h]/expected.clamp_min(1)),
                                                                 'correct_spans':int(correct[h]),'predicted_spans':int(proposed[h]),'gold_spans':int(expected),'best_epoch':int(best_epoch[h])})
            torch.save({'logits':logits,'y':part['y'],'sentence':part['sentence'],'source_labels':labels,'target':target},folder/(split+'.pt'))
        timing={'seed':seed,'fit_seconds_all_three_arms':fit_seconds,'training_and_evaluation_seconds':time.monotonic()-started,'optimizer_steps_per_arm':len(batches),
                'labelled_token_presentations_per_arm':plan['baseline_original_presentations'],'variant_token_share_augmented':plan['augmented_variant_presentations']/plan['baseline_original_presentations']}
        torch.save({'records':seed_records,'timing':timing},folder/'performance.pt');records.extend(seed_records);timings.append(timing)
        write_csv(output/'per_seed.csv',records);write_csv(output/'timing.csv',timings)
    write_csv(output/'per_seed.csv',records);write_csv(output/'timing.csv',timings)
    save_json(output/'timing.json',{'fit_seconds_all_seeds':sum(r['fit_seconds_all_three_arms'] for r in timings),'training_evaluation_seconds_all_seeds':sum(r['training_and_evaluation_seconds'] for r in timings)})
    return {'seed_arm_evaluations':len(records)}
