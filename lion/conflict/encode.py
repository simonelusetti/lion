"""Batched single-slot queries: remove the focal pieces and insert exactly one token."""
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,DataLoader
from ..encoder import build_masked_encoder
from ..runtime import autocast,parallel


class Slots(Dataset):
    def __init__(self,bank,split,index=None,mode='mask',word=None):
        self.rows=bank['sets'][split];self.slots=bank['slots'][split];self.index=torch.arange(len(self.slots['y'])) if index is None else index
        self.mode,self.word,self.mask_id=mode,word,bank['mask_id']
        length=torch.tensor([len(r['ids']) for r in self.rows])[self.slots['owner'][self.index]]
        self.length=length if mode=='original' else length-self.slots['end'][self.index]+self.slots['start'][self.index]+1
    def __len__(self):return len(self.index)
    def __getitem__(self,i):
        index=int(self.index[i]);owner=int(self.slots['owner'][index]);ids=self.rows[owner]['ids'];a,b=int(self.slots['start'][index]),int(self.slots['end'][index])
        if self.mode!='original':
            value=self.mask_id if self.word is None else int(self.word[i]);ids=torch.cat((ids[:a],torch.tensor([value]),ids[b:]));b=a+1
        return {'index':i,'slot':index,'ids':ids,'start':a,'end':b}


def collate(rows):
    ids=pad_sequence([r['ids'] for r in rows],batch_first=True);length=torch.tensor([len(r['ids']) for r in rows]);p=torch.arange(ids.shape[1])[None]
    start=torch.tensor([r['start'] for r in rows])[:,None];end=torch.tensor([r['end'] for r in rows])[:,None]
    return {'ids':ids,'attention':p<length[:,None],'weight':((p>=start)&(p<end)).float()/(end-start),'index':torch.tensor([r['index'] for r in rows]),'slot':torch.tensor([r['slot'] for r in rows])}


def loader(dataset,cfg):
    # Length buckets bound padded token work and keep all token/MLM operations vectorized.
    order=dataset.length.argsort(stable=True).tolist();batches=[];batch=[]
    for i in order:
        if batch and (len(batch)>=cfg.conflict.max_sequences or (len(batch)+1)*int(dataset.length[i])>cfg.conflict.token_budget):batches.append(batch);batch=[]
        batch.append(i)
    if batch:batches.append(batch)
    return DataLoader(dataset,batch_sampler=batches,collate_fn=collate,num_workers=cfg.runtime.workers,pin_memory=cfg.runtime.device.startswith('cuda'))


class Encoder(nn.Module):
    def __init__(self,src,valid):
        super().__init__();self.model=build_masked_encoder(src.encoder).model;self.register_buffer('valid',valid)
    def forward(self,ids,attention,weight,mlm=False):
        hidden=self.model.bert(input_ids=ids,attention_mask=attention,output_hidden_states=True,return_dict=True).hidden_states
        with torch.autocast(device_type=ids.device.type,enabled=False):
            features=torch.stack([torch.einsum('bt,btd->bd',weight.float(),hidden[l].float()) for l in (8,12)],1)
        result={'features':features}
        if mlm:result['probability']=self.model.cls(features[:,1]).double().softmax(-1)[:,self.valid]
        return result


def model(src,bank):return parallel(Encoder(src,bank['lexicon']['ids']).to(src.runtime.device).eval().requires_grad_(False),src,frozen=True)


@torch.no_grad()
def forward(model,batch,src,mlm=False):
    with autocast(src):return model(batch['ids'].to(src.runtime.device),batch['attention'].to(src.runtime.device),batch['weight'].to(src.runtime.device),mlm=mlm)
