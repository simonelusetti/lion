"""Pinned EWT dependencies with document-disjoint scorer/selector roles."""
import hashlib,json
import torch
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from ..certification import data
from ..datasets import UD_REPO
from ..diagnostics.common import absolute,write_csv,save_json
from ..fewnerd_replication.data import estimate


def prepare(cfg,output):
    data.prepare(cfg,output);bank=data.load(cfg);lookup={};files=[]
    for split in ('train','dev','test'):
        name=f'parquet/en_ewt/{split}.parquet';path=hf_hub_download(UD_REPO,name,revision=cfg.replication.revision,repo_type='dataset',local_files_only=True)
        rows=pq.read_table(path,columns=['tokens','sent_id']).to_pylist()
        lookup.update({r['sent_id']:r['tokens'] for r in rows})
        files.append({'file':name,'sha256':hashlib.sha256(open(path,'rb').read()).hexdigest(),'native_sentences':len(rows)})
    for rows in bank['sets'].values():
        for row in rows:row['words']=lookup[row['source_id']][:len(row['focals'])]
    projected=estimate(cfg,bank['sets']);initial=projected;halved=projected['total_seconds']>cfg.replication.runtime_budget_seconds
    if halved:
        target=(sum(len(r['focals']) for r in bank['sets']['id'])+1)//2
        order=torch.randperm(len(bank['sets']['id']),generator=torch.Generator().manual_seed(cfg.seed)).tolist();selected=[];words=0
        for i in order:
            selected.append(bank['sets']['id'][i]);words+=len(selected[-1]['focals'])
            if words>=target:break
        bank['sets']['id']=selected;projected=estimate(cfg,bank['sets'])
    torch.save(bank,output/'sentences.pt');OmegaConf.save(cfg,data.paths(cfg)/'config.yaml')
    records=[]
    for name,rows in [(k,v) for k,v in bank['sets'].items()]+[(role,[r for r in bank['sets']['train'] if r['fold']==role]) for role in ('scorer_train','scorer_validation','g1','g2')]:
        records.append({'split':name,'sentences':len(rows),'documents':len({r['document'] for r in rows}),'words':sum(len(r['focals']) for r in rows),'labelled_words':sum(int((r['gold']['deprel']>=0).sum()) for r in rows)})
    write_csv(output/'counts.csv',records)
    write_csv(output/'sample.csv',[{'split':split,'sentence':r['sentence'],'source_id':r['source_id'],'document':r['document'],'role':r.get('fold','evaluation')} for split,rows in bank['sets'].items() for r in rows])
    save_json(data.paths(cfg).parent/'estimate.json',{'initial':initial,'used':projected,'test_halved':halved})
    save_json(output/'provenance.json',{'dataset':UD_REPO,'treebank':'en_ewt','revision':cfg.replication.revision,'files':files,'sampling_seed':cfg.seed,'labels':bank['labels']['deprel'],
        'task':'Predict each word’s base UD dependency relation from sentence text. No head positions are supplied or predicted; this is relation tagging, not dependency parsing.',
        'label_mapping':'Strip colon subtypes, matching the existing certification deprel task. Include punctuation and root. No BIO/span metric.',
        'roles':'Random source cap of 10531 sentences, followed by document-grouped A/T1/T2 split with unchanged 25% scorer fraction, 20% within-A scorer validation. All three baseline seeds use T1 union T2.',
        'calibration':'Single all-token group; existing 70/30 sentence split of the opposite selector half; one random focal per sentence.',
        'test':'Native EWT test, no duplication to reach NER word counts; halved only if projected total exceeds six hours.',
        'seen':'Exact case-sensitive focal word seen in T1/T2, with or without the same base dependency label. Entity span analysis is inapplicable.'})
    print(json.dumps({'counts':records,'estimate':projected,'test_halved':halved},indent=2),flush=True)
    return {'source_sentences':len(bank['sets']['train']),'test_words':records[2]['words']}


def inputs(cfg,source):
    bank=data.load(source);raw=torch.load(absolute(cfg.sensitivity.results)/'train'/'raw_test.pt',mmap=True,weights_only=False)
    yield 'en_ewt_deprel',data.attach_features(bank['sets']['id'],raw['features'][:,0].float()),raw,bank


def metadata(cfg,output):
    bank=data.load(cfg);known={}
    for row in bank['sets']['train']:
        if row['fold'] not in ('g1','g2'):continue
        for word,label in zip(row['words'],row['gold']['deprel'].tolist()):known.setdefault(word,set()).add(label)
    forms=[];ambiguity=[];seen=[];gold=[]
    for row in bank['sets']['id']:
        for word,label in zip(row['words'],row['gold']['deprel'].tolist()):
            labels=known.get(word,set());forms.append(word);gold.append(label);ambiguity.append(0 if not labels else 1 if len(labels)==1 else 2);seen.append(0 if not labels else 2 if label in labels else 1)
    raw=torch.load(absolute(cfg.sensitivity.results)/'train'/'raw_test.pt',mmap=True,weights_only=False)
    torch.save({'forms':forms,'truth':torch.tensor(gold),'seen':torch.tensor(seen),'ambiguity':torch.tensor(ambiguity),'sentence':raw['sentence'],'labels':bank['labels']['deprel']},output/'en_ewt_deprel.pt')
    return {'test_words':len(forms),'word_forms_in_classifier_training':len(known)}
