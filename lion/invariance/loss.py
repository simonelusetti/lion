"""Differentiable biased RBF MMD², with a detached batch median bandwidth."""
import torch


def genre_sum(kernel, domain, membership, genres, minimum):
    # kernel [heads,words,words]; membership [heads,words].
    groups=(domain[None,:,None]==genres[None,None,:]).permute(0,2,1) & membership[:,None]
    count=groups.sum(-1)
    mass=groups.float()/count[:,:,None].clamp_min(1)
    means=torch.bmm(torch.bmm(mass,kernel),mass.transpose(1,2))
    diagonal=means.diagonal(dim1=-2,dim2=-1)
    discrepancy=(diagonal[:,:,None]+diagonal[:,None,:]-2*means).clamp_min(0)
    available=count>=minimum
    pairs=available[:,:,None]&available[:,None,:]&torch.ones(len(genres),len(genres),device=kernel.device,dtype=torch.bool).triu(1)
    return (discrepancy*pairs).sum((-1,-2)),pairs.sum((-1,-2))


def penalty(probability,y,z,anti,genres,minimum,label_indices):
    # probability [words,heads,labels], anti [words,heads].
    p=probability.transpose(0,1)
    distance=torch.cdist(p,p).square()
    positive=distance.detach().flatten(1)
    bandwidth=torch.nanmedian(positive.masked_fill(positive<=0,torch.nan),dim=-1).values.nan_to_num(nan=1.).clamp_min(1e-8)
    kernel=torch.exp(-distance/(2*bandwidth[:,None,None]))
    anti=anti.T
    marginal,pairs=genre_sum(kernel,z,~anti,genres,1)
    marginal=marginal*(~anti).float().mean(-1)
    conditional=torch.zeros_like(marginal)
    conditional_pairs=torch.zeros_like(pairs)
    active_mass=torch.zeros_like(marginal)
    for ids in label_indices:
        mode=anti[:,ids]
        # Restrict the kernel to the label to avoid multiplying large sparse masks.
        value,n=genre_sum(kernel[:,ids][:,:,ids],z[ids],mode,genres,minimum)
        share=mode.float().sum(-1)/len(y)
        conditional=conditional+value*share
        conditional_pairs+=n
        active_mass+=share*(n>0)
    return conditional+marginal,{'conditional':conditional.detach(),'marginal':marginal.detach(),
                                'conditional_pairs':conditional_pairs.float(),'marginal_pairs':pairs.float(),
                                'conditional_supported_mass':active_mass.detach()}
