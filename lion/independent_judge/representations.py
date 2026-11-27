"""Word-aligned frozen AutoModel representations, separate from readout heads."""
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel,AutoTokenizer
from ..diagnostics.common import absolute
from ..runtime import autocast


def load_encoder(cfg):
    options={'cache_dir':str(absolute(cfg.joint.judge_cache)),'local_files_only':True}
    tokenizer=AutoTokenizer.from_pretrained(cfg.joint.judge_name,use_fast=True,**options)
    model=AutoModel.from_pretrained(cfg.joint.judge_name,**options).to(cfg.runtime.device).eval().requires_grad_(False)
    return model,tokenizer


def collate(rows):
    width=max(len(r['ids']) for r in rows);ids=torch.zeros(len(rows),width,dtype=torch.long);attention=torch.zeros_like(ids,dtype=torch.bool);dest=torch.full_like(ids,-1)
    for i,r in enumerate(rows):
        n=len(r['ids']);ids[i,:n]=r['ids'];attention[i,:n]=True;dest[i,:n]=r['dest']
    return ids,attention,dest


@torch.no_grad()
def encode(cfg,model,tokenizer,records):
    offsets=[0]
    for r in records:offsets.append(offsets[-1]+len(r['focals']))
    chunks=[]
    for start in range(0,len(records),512):
        block=records[start:start+512]
        encoded=tokenizer([r['words'] for r in block],is_split_into_words=True,add_special_tokens=True,truncation=True,max_length=cfg.encoder.max_length,return_overflowing_tokens=True)
        for i,owner in enumerate(encoded['overflow_to_sample_mapping']):
            mapping={int(f):offsets[start+owner]+j for j,f in enumerate(block[owner]['focals'])}
            chunks.append({'ids':torch.tensor(encoded['input_ids'][i]),'dest':torch.tensor([mapping.get(w,-1) for w in encoded.word_ids(i)])})
    total=offsets[-1];accum=torch.zeros(total,model.config.hidden_size);counts=torch.zeros(total)
    batches=DataLoader(chunks,batch_size=cfg.runtime.batch_size,collate_fn=collate,num_workers=cfg.runtime.workers)
    for number,(ids,attention,dest) in enumerate(batches):
        ids,attention=ids.to(cfg.runtime.device),attention.to(cfg.runtime.device);active=dest>=0
        with autocast(cfg):hidden=model(input_ids=ids,attention_mask=attention).last_hidden_state
        target=dest[active];accum.index_add_(0,target,hidden.cpu()[active].float());counts.index_add_(0,target,torch.ones(len(target)))
        if (number+1)%500==0:print(f'Judge encoding {number+1}/{len(batches)} batches',flush=True)
    if (counts==0).any():raise ValueError('Judge tokenizer omitted a requested focal word')
    accum/=counts[:,None]
    return accum.half(),{'sentences':len(records),'requested_words':total,'encoder_chunks':len(chunks),'overflow_chunks':len(chunks)-len(records),'pooling':'Mean final-layer WordPieces; overflow chunks have no stride or piece duplication.'}
