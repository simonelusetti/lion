"""Pinned supervised data, coarse IO→BIO tags, source roles and runtime projection."""
import hashlib,json
import torch
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from ..certification import data as source_data
from ..diagnostics.common import absolute,save_json,write_csv
from ..sensitivity_diagnostics.metadata import spans


def cost(rows):
    # Every focal incurs sentence-length attention/MLP work during selector learning/fills.
    return sum(len(r['focals'])*(len(r['ids'])+len(r['ids'])**2/768) for r in rows)


def inputs(cfg,source):
    bank=source_data.load(source)
    raw=torch.load(absolute(cfg.sensitivity.results)/'train'/'raw_test.pt',mmap=True,weights_only=False)
    yield 'fewnerd',source_data.attach_features(bank['sets']['id'],raw['features'][:,0].float()),raw,bank


def estimate(cfg,sets):
    old=absolute(cfg.replication.conll_run);reference=torch.load(absolute(cfg.replication.conll_source)/'prepare'/'sentences.pt',weights_only=False)['sets']
    ratio=cost([r for r in sets['train'] if r['fold'] in ('g1','g2')])/cost([r for r in reference['train'] if r['fold'] in ('g1','g2')])
    part1={name:json.loads((old/name/'runtime.json').read_text())['seconds']*ratio for name in ('scorer','selector','calibrate','bounds')}
    part1['baseline_and_features']=60*ratio
    part2=720*cost(sets['id'])/cost(reference['id'])+60
    return {'part1_seconds':sum(part1.values()),'part2_seconds':part2,'total_seconds':sum(part1.values())+part2,'part1_stages':part1,'training_cost_ratio':ratio,
            'method':'CoNLL measured stage times × sum_words(piece_length + piece_length²/768) ratio; Part 2 CoNLL generation 720 s + 60 s inference/analysis. Same GPU throughput assumed; allow roughly ±50%.'}


def prepare(cfg,output):
    root=source_data.paths(cfg);root.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(cfg.encoder.name,use_fast=True,local_files_only=True)
    generator=torch.Generator().manual_seed(cfg.replication.seed);sets={};files=[];counter=0;labels=None;discarded={}
    for split,name in [('train','train'),('validation','validation'),('test','id')]:
        filename=f'supervised/{split}-00000-of-00001.parquet'
        path=hf_hub_download('DFKI-SLT/few-nerd',filename,repo_type='dataset',revision=cfg.replication.revision,local_files_only=True)
        table=pq.read_table(path,columns=['tokens','ner_tags']);raw=table.to_pylist()
        coarse=json.loads(table.schema.metadata[b'huggingface'])['info']['features']['ner_tags']['feature']['names']
        labels=['O']+[f'{prefix}-{kind}' for kind in coarse[1:] for prefix in ('B','I')]
        files.append({'split':split,'file':filename,'sha256':hashlib.sha256(open(path,'rb').read()).hexdigest(),'native_sentences':len(raw)})
        count=cfg.replication.training_sentences if split=='train' else round(cfg.replication.training_sentences*3250/14041) if split=='validation' else len(raw)
        order=torch.randperm(len(raw),generator=generator)[:count].tolist();rows=[];words=0;dropped=0
        # Only tokenize the prefix needed for the word-budgeted test sample.
        for offset in range(0,len(order),128):
            indices=order[offset:offset+128];batch=[raw[i]['tokens'] for i in indices]
            encoded=tokenizer(batch,is_split_into_words=True,add_special_tokens=True,truncation=True,max_length=cfg.encoder.max_length)
            for j,index in enumerate(indices):
                ids=encoded['input_ids'][j];word_ids=encoded.word_ids(j);kept=list(dict.fromkeys(w for w in word_ids if w is not None));lookup={w:i for i,w in enumerate(kept)}
                previous=0;bio=[]
                for tag in raw[index]['ner_tags']:
                    bio.append(0 if tag==0 else 1+2*(tag-1)+int(tag==previous));previous=tag
                gold=torch.tensor([bio[w] for w in kept]);key=f'fewnerd/{split}/{index}'
                rows.append({'ids':torch.tensor(ids),'word_ids':torch.tensor([lookup.get(w,-1) for w in word_ids]),'focals':torch.arange(len(kept)),
                             'gold':{'ner':gold},'words':[raw[index]['tokens'][w] for w in kept],'source_id':key,'document':key,'sentence':counter})
                counter+=1;words+=len(kept);dropped+=len(raw[index]['tokens'])-len(kept)
                if split=='test' and words>=cfg.replication.test_words:break
            if split=='test' and words>=cfg.replication.test_words:break
        sets[name]=rows;discarded[name]=dropped
    docs=sorted(r['document'] for r in sets['train']);order=torch.randperm(len(docs),generator=torch.Generator().manual_seed(cfg.seed)).tolist();docs=[docs[i] for i in order]
    a=round(len(docs)*cfg.certification.scorer_fraction);v=round(a*cfg.certification.scorer_validation_fraction);middle=a+(len(docs)-a)//2
    assignment={d:role for role,group in zip(('scorer_validation','scorer_train','g1','g2'),(docs[:v],docs[v:a],docs[a:middle],docs[middle:])) for d in group}
    for row in sets['train']:row['fold']=assignment[row['document']]
    initial=estimate(cfg,sets);target=cfg.replication.test_words;halved=initial['total_seconds']>cfg.replication.runtime_budget_seconds
    if halved:
        target=(target+1)//2;end=0;words=0
        while words<target:words+=len(sets['id'][end]['focals']);end+=1
        sets['id']=sets['id'][:end]
    final=estimate(cfg,sets);save_json(root.parent/'estimate.json',{'initial':initial,'used':final,'test_halved':halved,'test_target_words':target})
    bank={'sets':sets,'labels':{'ner':labels},'special':{'mask':tokenizer.mask_token_id,'excluded':tokenizer.all_special_ids},'domains':{k:'fewnerd' for k in sets},'group_unit':'sentence'}
    torch.save(bank,output/'sentences.pt');OmegaConf.save(cfg,root/'config.yaml')
    counts=[{'split':k,'sentences':len(rows),'words':sum(len(r['focals']) for r in rows)} for k,rows in sets.items()]
    for role in ('scorer_train','scorer_validation','g1','g2'):
        rows=[r for r in sets['train'] if r['fold']==role];counts.append({'split':role,'sentences':len(rows),'words':sum(len(r['focals']) for r in rows)})
    write_csv(output/'counts.csv',counts)
    write_csv(output/'sample.csv',[{'split':k,'source_id':r['source_id'],'role':r.get('fold','evaluation'),'sentence':r['sentence'],'words':len(r['focals'])} for k,rows in sets.items() for r in rows])
    training={}
    for r in sets['train']:
        if r['fold'] not in ('g1','g2'):continue
        for a,b,kind in spans([labels[int(y)] for y in r['gold']['ner']]):training.setdefault(tuple(r['words'][a:b]),set()).add(kind)
    seen=same=total=0
    for r in sets['id']:
        for a,b,kind in spans([labels[int(y)] for y in r['gold']['ner']]):
            types=training.get(tuple(r['words'][a:b]),set());total+=1;seen+=bool(types);same+=kind in types
    save_json(output/'provenance.json',{'dataset':'DFKI-SLT/few-nerd','config':'supervised','revision':cfg.replication.revision,'files':files,'sampling_seed':cfg.replication.seed,
              'source_sentences':len(sets['train']),'classifier_sentences':sum(r['fold'] in ('g1','g2') for r in sets['train']),'validation_sampling':'round(10531 × 3250 / 14041); calibration 70% of opposite selector half, held-out 30%, unchanged',
              'tagging':'Coarse IO converted to BIO: B on non-O type change/after O, I on continuation. Contiguous same-coarse-type entities merge because IO lacks boundaries; fine tags are not used.',
              'discarded_words_due_to_512_piece_truncation_before_test_halving':discarded,'test_target_words':target,'test_words':counts[2]['words'],
              'test_gold_spans':total,'seen_test_gold_spans':seen,'seen_same_type_test_gold_spans':same,'seen_test_span_share':seen/total,'seen_same_type_test_span_share':same/total})
    print(json.dumps({'counts':counts,'estimate':final,'test_halved':halved},indent=2),flush=True)
    return {'source_sentences':len(sets['train']),'test_words':counts[2]['words']}
