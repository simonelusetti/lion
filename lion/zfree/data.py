"""Other-sentence document context, training-only PCA and label residualization."""
import torch
from ..invariance.data import banks,rotations,paths
from ..diagnostics.common import save_json,write_csv


def document_features(bank):
    lengths=bank['offsets'].diff();sentence=torch.repeat_interleave(torch.arange(len(lengths)),lengths)
    # Native split is part of the key: no preprocessing mixes held-out partitions.
    documents={key:i for i,key in enumerate(sorted(set(zip(bank['document'],bank['sentence_split'][sentence].tolist()))))}
    document=torch.tensor([documents[key] for key in zip(bank['document'],bank['sentence_split'][sentence].tolist())])
    sentence_document=document[bank['offsets'][:-1]]
    x=bank['features'].float();classes=len(bank['labels'])
    sentence_sum=torch.zeros(len(lengths),x.shape[-1]).index_add_(0,sentence,x)
    document_sum=torch.zeros(len(documents),x.shape[-1]).index_add_(0,document,x)
    counts=torch.bincount(document,minlength=len(documents))
    other_count=counts[sentence_document]-lengths
    context=(document_sum[sentence_document]-sentence_sum)/other_count[:,None].clamp_min(1)
    valid=bank['y']>=0
    cells=sentence[valid]*classes+bank['y'][valid]
    sentence_hist=torch.bincount(cells,minlength=len(lengths)*classes).reshape(-1,classes).float()
    document_hist=torch.zeros(len(documents),classes).index_add_(0,sentence_document,sentence_hist)
    other_hist=document_hist[sentence_document]-sentence_hist
    own=sentence_hist/sentence_hist.sum(-1,keepdim=True).clamp_min(1)
    other=other_hist/other_hist.sum(-1,keepdim=True).clamp_min(1)
    return {'sentence':sentence,'document':document,'context':context,'own_hist':own,'other_hist':other,
            'covered_sentence':other_count>0,'lengths':lengths}


def residualize(bank,base,training,cfg,seed):
    sentence=base['sentence'];covered=base['covered_sentence'][sentence]&(bank['y']>=0)
    train=training&covered
    train_sentences=torch.bincount(sentence[train],minlength=len(base['lengths'])).float()
    weight=train_sentences/train_sentences.sum();x=base['context'];mean=(x*weight[:,None]).sum(0)
    centered=x-mean
    covariance=centered.T@(centered*weight[:,None])
    eigenvalues,basis=torch.linalg.eigh(covariance);basis=basis[:,-cfg.zfree.pca_dimensions:]
    projected=centered@basis
    scale=(projected.square()*weight[:,None]).sum(0).sqrt().clamp_min(1e-6)
    projected=projected/scale
    y=bank['y'];onehot=torch.nn.functional.one_hot(y.clamp_min(0),len(bank['labels'])).float()
    summary=torch.cat([torch.ones(len(y),1),onehot,base['own_hist'][sentence],base['other_hist'][sentence]],dim=1)
    design=summary[train].double();target=projected[sentence[train]].double()
    gram=design.T@design/len(design);regularizer=torch.eye(gram.shape[0],dtype=gram.dtype)*cfg.zfree.ridge;regularizer[0,0]=0
    coefficients=torch.linalg.solve(gram+regularizer,design.T@target/len(design))
    residual=projected[sentence]-(summary.double()@coefficients).float();residual[~covered]=0
    # Fixed document derangement plus random donor-token matching; never use genres.
    generator=torch.Generator().manual_seed(seed)
    docs=base['document'][train].unique();order=docs[torch.randperm(len(docs),generator=generator)]
    donor=torch.full((int(base['document'].max())+1,),-1,dtype=torch.long);donor[order]=order.roll(1)
    source=torch.full((len(y),),-1,dtype=torch.long)
    ids=train.nonzero().flatten();doc_ids=base['document'][ids]
    ordered=ids[doc_ids.argsort(stable=True)]
    counts=torch.bincount(doc_ids,minlength=len(donor));starts=counts.cumsum(0)-counts
    donor_ids=donor[doc_ids]
    positions=starts[donor_ids]+(torch.rand(len(ids),generator=generator)*counts[donor_ids]).long()
    source[ids]=ordered[positions]
    return {'residual':residual,'covered':covered,'random_source':source,'document':base['document'],'sentence':sentence,
            'pca_mean':mean,'pca_basis':basis,'pca_scale':scale,'ridge_coefficients':coefficients,
            'pca_variance_fraction':float(eigenvalues[-cfg.zfree.pca_dimensions:].sum()/eigenvalues.sum())}


def experiment(cfg,output):
    rows=[]
    for name,bank in banks(cfg):
        base=document_features(bank)
        for index,(genre,genres,validation,test) in enumerate(rotations(name,bank,cfg)):
            sentence_train=torch.isin(bank['sentence_z'],torch.tensor(genres))&(bank['sentence_split']==0)
            training=sentence_train[base['sentence']]&(bank['y']>=0)
            folder=output/name/genre;folder.mkdir(parents=True,exist_ok=True)
            target=folder/'context.pt'
            if not (cfg.invariance.resume and target.exists()):
                context=residualize(bank,base,training,cfg,cfg.zfree.permutation_seed+index)
                torch.save(context,target)
            else:context=torch.load(target,weights_only=False)
            masks={'train':training,'id_validation':torch.isin(bank['sentence_z'][base['sentence']],torch.tensor(genres))&(bank['sentence_split'][base['sentence']]==1),
                   'id_test':torch.isin(bank['sentence_z'][base['sentence']],torch.tensor(genres))&(bank['sentence_split'][base['sentence']]==2)}
            for split,mask in masks.items():
                mask=mask&(bank['y']>=0)
                rows.append({'dataset':name,'test_genre':genre,'split':split,'tokens':int(mask.sum()),'covered':int((mask&context['covered']).sum()),
                             'coverage':float(context['covered'][mask].float().mean()),'pca_variance_fraction':context['pca_variance_fraction']})
        write_csv(output/'coverage.csv',rows)
    return {'rotations':len(rows)//3}
