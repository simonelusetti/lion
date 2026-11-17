"""Token-capped fixed-mode MMD; sampling indices are shared across readouts."""
import torch
from .loss import genre_sum


def samples(batch, genres, cfg, generator):
    def sample(ids, minimum):
        parts=[]
        for genre in genres:
            cell=ids[batch['z'][ids]==genre]
            if len(cell)>=minimum:
                parts.append(cell[torch.randperm(len(cell),generator=generator)[:cfg.invariance.token_cap]])
        return torch.cat(parts) if parts else torch.empty(0,dtype=torch.long)
    all_ids=torch.arange(len(batch['y']))
    marginal=sample(all_ids,1)
    conditional=[]
    for label in batch['y'].unique():
        ids=(batch['y']==label).nonzero().flatten()
        selected=sample(ids,cfg.invariance.conditional_minimum)
        if len(selected):conditional.append((selected,len(ids)/len(all_ids)))
    return marginal,conditional


def discrepancy(p,z,genres,minimum,kernel,bandwidth):
    if kernel=='linear':
        membership=(z[:,None]==genres).T.float()
        counts=membership.sum(-1)
        means=torch.einsum('gn,ncl->cgl',membership,p)/counts[None,:,None].clamp_min(1)
        distance=(means[:,:,None]-means[:,None,:]).square().sum(-1)
        valid=counts>=minimum
        pairs=(valid[:,None]&valid[None,:]).triu(1)
        return (distance*pairs).sum((-1,-2))
    distance=torch.cdist(p.transpose(0,1),p.transpose(0,1)).square()
    gram=torch.exp(-distance/(2*bandwidth[:,None,None]))
    membership=torch.ones(p.shape[1],len(p),dtype=torch.bool,device=p.device)
    return genre_sum(gram,z,membership,genres,minimum)[0]


def penalty(probability,z,sampled,rules,genres,cfg):
    marginal,conditional=sampled
    marginal=marginal.to(probability.device)
    if cfg.invariance.kernel=='rbf':
        with torch.no_grad():
            p=probability[marginal].transpose(0,1)
            distances=torch.cdist(p,p).square().flatten(1)
            bandwidth=torch.nanmedian(distances.masked_fill(distances<=0,torch.nan),dim=-1).values.nan_to_num(nan=1.).clamp_min(1e-8)
    else:bandwidth=probability.new_ones(probability.shape[1])
    result=probability.new_zeros(probability.shape[1])
    for rule in ('R1','R2'):
        heads=[i for i,r in enumerate(rules) if r==rule]
        if not heads:continue
        cells=[(marginal,1.)] if rule=='R2' else conditional
        minimum=1 if rule=='R2' else cfg.invariance.conditional_minimum
        value=probability.new_zeros(len(heads))
        for ids,share in cells:
            ids=ids.to(probability.device)
            value=value+share*discrepancy(probability[ids][:,heads],z[ids],genres,minimum,cfg.invariance.kernel,bandwidth[heads])
        result[heads]=value
    return result
