"""The matched baseline recipe, using only original T1∪T2 words."""
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from tagger.ensemble import IndependentReadouts,clip_heads
from ..augmentation.train import schedule
from ..partition.train import predict,metric_vectors,load_model
from ..certification import data
from ..certification.classify import span_counts,evaluate
from ..encoder import load_masked_encoder
from ..runtime import autocast,parallel
from ..diagnostics.common import write_csv,absolute


class Steps(Dataset):
    def __init__(self,part,batches):self.part,self.batches=part,batches
    def __len__(self):return len(self.batches)
    def __getitem__(self,index):
        b=self.batches[index];ids=b['baseline']
        return {'x':self.part['features'][ids].float(),'y':self.part['y'][ids],'epoch':b['epoch']}


def collate(rows):return rows[0]


def experiment(cfg,output):
    task=data.task_names(cfg)[0];bank=data.load(cfg);labels=bank['labels'][task];full=torch.load(data.paths(cfg)/'scorer'/'full_features.pt',mmap=True,weights_only=False)
    rows=bank['sets']['train'];include=torch.tensor([r['fold'] in ('g1','g2') for r in rows]);length=torch.tensor([len(r['focals']) for r in rows])
    training=data.readout_data([r for r in rows if r['fold'] in ('g1','g2')],full['train'][include.repeat_interleave(length)],task)
    validation=data.readout_data(bank['sets']['validation'],full['validation'],task)
    if not (output/'raw_test.pt').exists():
        encoder=load_masked_encoder(cfg);features=data.full_features(encoder,{'id':bank['sets']['id']},cfg)['id'];del encoder
        torch.save(data.readout_data(bank['sets']['id'],features,task),output/'raw_test.pt')
    test=torch.load(output/'raw_test.pt',mmap=True,weights_only=False);records=[]
    for seed in cfg.augmentation.seeds:
        folder=output/f'seed{seed}';folder.mkdir(exist_ok=True)
        if not (folder/'model.pt').exists():
            torch.manual_seed(seed);core=IndependentReadouts(cfg.tagger,768,len(labels),1).to(cfg.runtime.device);model=parallel(core,cfg)
            optimizer=torch.optim.AdamW(core.parameters(),lr=cfg.train.lr,weight_decay=cfg.train.weight_decay)
            batches,plan=schedule(training['sentence'],cfg,seed);torch.save(plan,folder/'schedule.pt')
            steps=DataLoader(Steps(training,batches),batch_size=1,collate_fn=collate,num_workers=cfg.runtime.workers,pin_memory=cfg.runtime.device.startswith('cuda'))
            best=-float('inf');history=[]
            for number,batch in enumerate(steps):
                model.train();optimizer.zero_grad(set_to_none=True)
                with autocast(cfg):logits=model(batch['x'].to(cfg.runtime.device)).float()
                loss=F.cross_entropy(logits[:,0],batch['y'].to(cfg.runtime.device));loss.backward();clip_heads(core,cfg.train.clip);optimizer.step()
                epoch=batch['epoch']
                if number+1<len(batches) and batches[number+1]['epoch']==epoch:continue
                scores=metric_vectors(predict(model,validation,cfg),validation['y'],len(labels));criterion=float(scores['macro_f1'][0])
                if criterion>best:best=criterion;best_epoch=epoch;state={k:v.detach().cpu().clone() for k,v in core.state_dict().items()}
                history.append({'epoch':epoch,'validation_macro_f1':criterion,'validation_accuracy':float(scores['accuracy'][0])})
                print(f'Baseline seed{seed} epoch{epoch}: dev token macro-F1 {criterion:.4f}',flush=True)
            core.load_state_dict(state)
            torch.save({'state':state,'labels':len(labels),'dimension':768,'heads':1,'linear':False,'seed':seed,'epoch':best_epoch,'validation_score':best},folder/'model.pt')
            write_csv(folder/'history.csv',history)
        model=load_model(folder,cfg);logits=predict(model,test,cfg);torch.save({'logits':logits},folder/'test.pt')
        records.append(score(logits,test,labels,'baseline',seed,task,cfg.certification.source))
    scorer=load_model(data.paths(cfg)/'scorer'/task,cfg);scorer_record=score(predict(scorer,test,cfg),test,labels,'scorer',-1,task,cfg.certification.source)
    if cfg.replication.get('conll_comparison',True):
        old=torch.load(absolute(cfg.replication.conll_run)/'train'/'evaluation.pt',mmap=True,weights_only=False)
        oldmodel=load_model(absolute(cfg.replication.conll_source)/'scorer'/'ner',cfg);oldpart=old['sets']['id']['data']
        oldscore=evaluate(predict(oldmodel,oldpart,cfg),oldpart,'conll2003','conll2003',old['labels'],'native')
        scorer_record['conll_span_f1']=float(oldscore['span_f1'][0])
    records.append(scorer_record)
    write_csv(output/'performance.csv',records)
    return {'baselines':len(cfg.augmentation.seeds),'test_words':len(test['y'])}


def score(logits,part,labels,model,seed,task='ner',source='fewnerd'):
    if task!='ner':
        metrics=evaluate(logits,part,source,source,labels,'native')
        return {'dataset':source+'_'+task,'classifier':model,'seed':seed,**{k:float(v[0]) for k,v in metrics.items()}}
    metrics=evaluate(logits,part,'fewnerd','fewnerd',labels,'native');correct,proposed,expected=span_counts(logits.argmax(-1),part['y'],part['sentence'],labels,labels)
    return {'dataset':'fewnerd','classifier':model,'seed':seed,**{k:float(v[0]) for k,v in metrics.items()},'precision':float(correct[0]/proposed[0].clamp_min(1)),'recall':float(correct[0]/expected.clamp_min(1)),'gold_spans':int(expected)}
