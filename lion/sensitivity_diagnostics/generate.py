"""Record token draws while running the unchanged step-1 sampling procedure."""
from collections import defaultdict
from functools import lru_cache
import torch
from transformers import AutoTokenizer
from ..sensitivity.data import generate,source_config


class Recorder:
    def __init__(self,tokenizer):
        self.tokens=tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    @lru_cache(maxsize=100000)
    def word(self,ids):return ''.join(self.tokens[i].removeprefix('##') for i in ids)
    def start(self,dataset,fold):self.parts=defaultdict(list)
    def capture(self,batch,keep,fills,kind):
        ids=batch['ids'].cpu();attention=batch['attention'].cpu();words=batch['word_ids'].cpu();counts=batch['counts'].cpu();focal=batch['focal'].cpu();keep=keep.cpu();fills=fills.cpu()
        if kind=='selector':
            for key,value in [('original_ids',ids[attention].int()),('word_ids',words[attention].int()),('piece_lengths',attention.sum(-1)),('word_lengths',counts)]:self.parts[key].append(value)
        self.parts[kind+'_token_ids'].append(fills[attention[:,None].expand_as(fills)].int())
        active=torch.arange(keep.shape[1])[None]<counts[:,None]
        self.parts[kind+'_drop_words'].append((~keep)[active])
        change=((fills!=ids[:,None])&(words[:,None]>=0)).long()
        changed=torch.zeros(len(ids),fills.shape[1],keep.shape[1],dtype=torch.long).scatter_add_(2,words[:,None].clamp_min(0).expand_as(fills),change)>0
        # Alternate WordPiece decompositions can spell exactly the same case-sensitive word.
        for row,draw,word in changed.nonzero().tolist():
            positions=words[row]==word
            if self.word(tuple(ids[row,positions].tolist()))==self.word(tuple(fills[row,draw,positions].tolist())):changed[row,draw,word]=False
        distance=(torch.arange(keep.shape[1])[None,None]-focal[:,None,None]).abs().expand_as(changed)
        nearest=distance.masked_fill(~changed,100000).min(-1).values;nearest[~changed.any(-1)]=-1
        self.parts[kind+'_changed_count'].append(changed.sum(-1));self.parts[kind+'_nearest_changed'].append(nearest)
    def finish(self):
        result={k:torch.cat(v) for k,v in self.parts.items()}
        for name,lengths in [('piece_offsets',result['piece_lengths']),('word_offsets',result['word_lengths'])]:result[name]=torch.cat([torch.zeros(1,dtype=torch.long),lengths.cumsum(0)])
        return result


def experiment(cfg,output):
    source=source_config(cfg);tokenizer=AutoTokenizer.from_pretrained(source.encoder.name,use_fast=True,local_files_only=True)
    return generate(cfg,output,recorder=Recorder(tokenizer))
