"""Frozen scorer, document ANOVA and global subset selection."""
import json
import torch
from scipy.stats import f as f_distribution
from tagger.ensemble import IndependentReadouts
from ..runtime import autocast
from ..diagnostics.common import save_json,write_csv
from ..invariance.data import paths
from ..partition.clusters import kmeans,assign
from .data import load,cache,rotation,SOURCES,SLOTS
from .train import genres_for


def invariance(probability,y,environment,document,weights=None):
    # [tokens,conditions,labels] -> equal-weight document residual means.
    names=sorted(set(document));lookup={d:i for i,d in enumerate(names)};inverse=torch.tensor([lookup[d] for d in document]);n=len(names)
    residual=torch.nn.functional.one_hot(y,probability.shape[-1])[:,None].float()-probability
    means=torch.zeros(n,*probability.shape[1:]).index_add_(0,inverse,residual)/torch.bincount(inverse)[:,None,None]
    env=torch.empty(n,dtype=torch.long);env[inverse]=environment
    domains=env.unique();g=len(domains);single=weights is None
    if single:weights=torch.ones(1,n)
    if g<2 or n<=g:
        value=torch.full((len(weights),probability.shape[1]),torch.nan)
        return {'statistic':value[0] if single else value,'p_value':value[0] if single else value,'documents':n,'environments':g}
    group_weights=weights[:,None,:]*(env[None,None,:]==domains[None,:,None])
    counts=group_weights.sum(-1);total=counts.sum(-1)
    pooled=torch.einsum('bgd,dhc->bghc',group_weights,means)/counts[:,:,None,None].clamp_min(1)
    grand=torch.einsum('bd,dhc->bhc',weights,means)/total[:,None,None]
    between=((pooled-grand[:,None]).square()*counts[:,:,None,None]).sum(1)/(g-1)
    within=(torch.einsum('bd,dhc->bhc',weights,means.square())-(pooled.square()*counts[:,:,None,None]).sum(1)).clamp_min(0)/(total-g)[:,None,None]
    statistic=torch.where(within>0,between/within,torch.where(between>1e-12,torch.inf,0.))
    p=torch.tensor(f_distribution.sf(statistic.double().numpy(),g-1,n-g),dtype=torch.float32)
    statistic=statistic.max(-1).values;p=(p.min(-1).values*probability.shape[-1]).clamp_max(1)
    return {'statistic':statistic[0] if single else statistic,'p_value':p[0] if single else p,'documents':n,'environments':g}


@torch.no_grad()
def score(core,features,y,cfg):
    size,conditions,dimension=features.shape;flat=features.reshape(-1,dimension);values=[]
    for start in range(0,len(flat),2048):
        with autocast(cfg):logits=core(flat[start:start+2048].to(cfg.runtime.device)[:,None])[:,0]
        values.append(logits.float().cpu())
    logits=torch.cat(values).reshape(size,conditions,-1)
    loss=-logits.log_softmax(-1).gather(-1,y[:,None,None].expand(-1,conditions,1)).squeeze(-1).mean(0)
    return logits.softmax(-1),loss


def cluster_environments(task,bank,genres,cfg):
    from ..diagnostics.common import absolute
    manifest=json.loads(absolute('results/invariance/banks.json').read_text());full=torch.load(manifest[task],weights_only=False)
    vectors=torch.cat(list(torch.load(full['vectors'],mmap=True,weights_only=False).values()))
    names=sorted(set(full['document']));lookup={d:i for i,d in enumerate(names)};doc=torch.tensor([lookup[d] for d in full['document']])
    means=torch.zeros(len(names),768).index_add_(0,doc,vectors.float())/torch.bincount(doc)[:,None]
    native=torch.repeat_interleave(full['sentence_split'],full['offsets'].diff());train=(native==0)&torch.isin(full['z'],torch.tensor(genres));train_docs=doc[train].unique()
    centers,_=kmeans(means[train_docs],len(genres),cfg.pilot.scorer_seed,cfg)
    environment=assign(means,centers,cfg)
    return torch.tensor([environment[lookup[d]] for d in bank['document']]),centers


def experiment(cfg,output):
    results,_=paths(cfg);selected={};curves=[]
    for task in cfg.invariance.datasets:
        bank=load(cfg,task);features=torch.load(cache(cfg)/SOURCES[task]/'validation.pt',mmap=True,weights_only=False)
        mapping=torch.full((len(bank['z']),),-1,dtype=torch.long);mapping[features['index']]=torch.arange(len(features['index']))
        for genre in genres_for(task,bank,cfg):
            genres,ids=rotation(bank,genre);validation=ids['id_validation'];folder=results/'scorer'/task/genre
            saved=torch.load(folder/'model.pt',weights_only=False);core=IndependentReadouts(cfg.tagger,768,len(bank['labels']),1).to(cfg.runtime.device);core.load_state_dict(saved['state']);core.eval()
            probability,loss=score(core,features['features'][mapping[validation]],bank['y'][validation],cfg)
            docs=[bank['document'][i] for i in validation.tolist()];cluster,centers=cluster_environments(task,bank,genres,cfg)
            test_genre=invariance(probability,bank['y'][validation],bank['z'][validation],docs)
            test_free=invariance(probability,bank['y'][validation],cluster[validation],docs)
            popcount=torch.tensor([i.bit_count() for i in range(256)])
            sufficient=(loss<=1.01*loss.min()).nonzero().flatten().tolist();suff=min(sufficient,key=lambda i:(int(popcount[i]),float(loss[i]),i))
            record={'subsets':{'sel_suff':suff},'fallback':{},'intersection':{},'accepted_count':{}}
            for rule,test in [('sel_inv_genre',test_genre),('sel_inv_free',test_free)]:
                accepted=(torch.isfinite(test['p_value'])&(test['p_value']>=cfg.pilot.alpha)).nonzero().flatten().tolist()
                record['subsets'][rule]=min(accepted,key=lambda i:(float(loss[i]),int(popcount[i]),i)) if accepted else 0
                record['fallback'][rule]=not bool(accepted);record['accepted_count'][rule]=len(accepted)
                intersection=255
                for subset in accepted:intersection&=subset
                record['intersection'][rule]=intersection if accepted else None
            record['loss']=loss.tolist();selected.setdefault(task,{})[genre]=record
            target=output/task/genre;target.mkdir(parents=True,exist_ok=True);torch.save({'centers':centers,'environment':cluster},target/'clusters.pt')
            for subset in range(256):curves.append({'task':task,'rotation':genre,'subset':subset,'slots':'|'.join(SLOTS[j] for j in range(8) if (subset>>j)&1),'loss':float(loss[subset]),
                                                   'genre_statistic':float(test_genre['statistic'][subset]),'genre_p':float(test_genre['p_value'][subset]),'free_statistic':float(test_free['statistic'][subset]),'free_p':float(test_free['p_value'][subset])})
            save_json(output/'selections.json',selected);write_csv(output/'subsets.csv',curves)
    return {'rotations':sum(map(len,selected.values()))}
