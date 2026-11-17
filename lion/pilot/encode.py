"""Re-encode actual MASK replacements; deduplicate identical masked sentences."""
import torch
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from ..encoder import build_encoder
from ..runtime import autocast,parallel
from ..invariance.data import paths
from .data import cache,masks,FIXED


class Requests(Dataset):
    def __init__(self,rows,kind,menus,previous,mask_id):self.rows,self.kind,self.menus,self.previous,self.mask_id=rows,kind,menus,previous,mask_id
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];sample=torch.tensor(row['sample_ids']);focal=torch.tensor(row['positions'])[row['focals']]
        if self.previous:
            ids=torch.tensor([row['previous_ids']+row['ids']]);inverse=torch.zeros(len(sample),dtype=torch.long);focal=focal+len(row['previous_ids']);conditions=1
        else:
            menu=self.menus[sample] if self.menus is not None and self.menus.ndim==2 else self.menus
            keep,_=masks(row,self.kind,menu);conditions=keep.shape[1]
            visible=keep.flatten(0,1)[:,torch.tensor(row['word_ids']).clamp_min(0)]
            values=torch.tensor(row['ids'])[None].expand(len(visible),-1).clone();values[~visible]=self.mask_id
            ids,inverse=torch.unique(values,dim=0,return_inverse=True);focal=focal[:,None].expand(-1,conditions).flatten()
        target=(sample[:,None]*conditions+torch.arange(conditions)).flatten()
        return {'ids':ids,'inverse':inverse,'focal':focal,'target':target}


def collate(rows):
    inputs=[];inverse=[];focal=[];target=[];offset=0
    for row in rows:
        inputs.extend(row['ids']);inverse.append(row['inverse']+offset);offset+=len(row['ids']);focal.append(row['focal']);target.append(row['target'])
    lengths=torch.tensor([len(r) for r in inputs]);ids=pad_sequence(inputs,batch_first=True)
    return {'ids':ids,'attention':torch.arange(ids.shape[1])[None]<lengths[:,None],
            'inverse':torch.cat(inverse),'focal':torch.cat(focal),'target':torch.cat(target)}


@torch.no_grad()
def encode(bank,kind,model,cfg,menus=None,rows=None,previous=False):
    rows=bank['rows'] if rows is None else rows
    rows=sorted(rows,key=lambda r:len(r['ids'])+(len(r['previous_ids']) if previous else 0))
    count=1 if previous else (menus.shape[-1] if menus is not None else len(kind))
    tokenizer=AutoTokenizer.from_pretrained(cfg.encoder.name,use_fast=True,local_files_only=True)
    result=torch.zeros(len(bank['z'])*count,768,dtype=torch.float16);requests=Requests(rows,kind,menus,previous,tokenizer.mask_token_id)
    loader=DataLoader(requests,batch_size=cfg.pilot.encoding_batch_sentences,collate_fn=collate,num_workers=cfg.runtime.workers)
    encoded=0
    for batch in loader:
        length=batch['ids'].shape[1];chunk=min(cfg.pilot.encoding_chunk,max(1,cfg.runtime.attention_budget//length**2))
        for start in range(0,len(batch['ids']),chunk):
            stop=start+chunk;selected=((batch['inverse']>=start)&(batch['inverse']<stop)).nonzero().flatten()
            with autocast(cfg):h=model(batch['ids'][start:stop].to(cfg.runtime.device),batch['attention'][start:stop].to(cfg.runtime.device),all_layers=False)['features'][:,:,0]
            values=h[(batch['inverse'][selected]-start).to(h.device),batch['focal'][selected].to(h.device)]
            result[batch['target'][selected]]=values.cpu().half();encoded+=len(h)
    print(f'Encoded {encoded} unique inputs for {len(rows)} sentences, {count} conditions',flush=True)
    return result.reshape(len(bank['z']),count,768)


def base(cfg,output):
    model=parallel(build_encoder(cfg.encoder).to(cfg.runtime.device).eval(),cfg)
    for source in ('ewt','gum'):
        bank=torch.load(cache(cfg)/(source+'.pt'),weights_only=False);folder=cache(cfg)/source;folder.mkdir(exist_ok=True)
        if not (folder/'fixed.pt').exists():
            features=encode(bank,FIXED[:-1],model,cfg)
            previous=encode(bank,['full_prev'],model,cfg,previous=True)
            torch.save(torch.cat([features,previous],dim=1),folder/'fixed.pt')
        if not (folder/'random.pt').exists():
            generator=torch.Generator().manual_seed(cfg.pilot.scorer_seed);ids=torch.randint(256,(len(bank['z']),cfg.pilot.random_masks),generator=generator)
            torch.save(ids,folder/'random_ids.pt');torch.save(encode(bank,'menu',model,cfg,menus=ids),folder/'random.pt')
        if not (folder/'validation.pt').exists():
            rows=[r for r in bank['rows'] if bank['native'][r['sample_ids'][0]]==1]
            # Store compactly; evaluation maps global sample indices to these rows.
            full=encode(bank,'menu',model,cfg,menus=torch.arange(256),rows=rows)
            ids=(bank['native']==1).nonzero().flatten();torch.save({'index':ids,'features':full[ids]},folder/'validation.pt')
    return {'sources':2}


def selected(cfg,output):
    import json
    results=paths(cfg)[0]
    chosen=json.loads((results/'select'/'selections.json').read_text())
    validity=torch.randperm(256,generator=torch.Generator().manual_seed(cfg.pilot.scorer_seed+20))[:cfg.pilot.validity_subsets].sort().values.tolist()
    model=parallel(build_encoder(cfg.encoder).to(cfg.runtime.device).eval(),cfg)
    for source in ('ewt','gum'):
        from .data import SOURCES
        ids=set(validity)
        for task,rotations in chosen.items():
            if SOURCES[task]==source:
                for selection in rotations.values():ids.update(selection['subsets'].values())
        bank=torch.load(cache(cfg)/(source+'.pt'),weights_only=False);folder=cache(cfg)/source
        validation=torch.load(folder/'validation.pt',mmap=True,weights_only=False)
        rows=[r for r in bank['rows'] if bank['native'][r['sample_ids'][0]]!=1]
        for subset in sorted(ids):
            target=folder/f'menu_{subset:03d}.pt'
            if target.exists():continue
            values=encode(bank,'menu',model,cfg,menus=torch.tensor([subset]),rows=rows)[:,0]
            values[validation['index']]=validation['features'][:,subset];torch.save(values,target)
    return {'validity_subsets':validity}
