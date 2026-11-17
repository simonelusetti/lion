"""Within-label centered log-probability cross-covariance penalties."""
import torch


def center(log_probability,y):
    classes=log_probability.shape[-1]
    total=log_probability.new_zeros(classes,*log_probability.shape[1:]).index_add_(0,y,log_probability)
    count=torch.bincount(y,minlength=classes)
    return log_probability-total[y]/count[y,None,None].clamp_min(1)


def covariance_norm(r,context):
    # Empirical cross covariance, divisor n, with ordinary batch centering.
    if len(r)<2:return r.sum((0,2))*0
    covariance=torch.einsum('ncl,nd->cld',r-r.mean(0),context-context.mean(0))/len(r)
    return covariance.square().sum((-1,-2))


def penalty(log_probability,y,z,context,random_context,covered,rules,genres):
    residual=center(log_probability,y)
    result=log_probability.new_zeros(log_probability.shape[1])
    for rule in ('Rz','Rg','Rrand'):
        heads=[i for i,r in enumerate(rules) if r==rule]
        if not heads:continue
        if rule=='Rg':
            nuisance=(z[:,None]==genres).float();value=covariance_norm(residual[:,heads],nuisance)
        else:
            nuisance=context if rule=='Rz' else random_context
            value=covariance_norm(residual[covered][:,heads],nuisance[covered])
        result[heads]=value
    return result
