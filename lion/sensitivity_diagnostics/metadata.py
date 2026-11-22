"""Case-sensitive source word types and training gold-entity span membership."""
from functools import lru_cache
import torch
from ..diagnostics.common import dataset_config
from ..data import prepare_data
from ..labels import LABELS
from ..metrics import shared_label
from ..sensitivity.data import source_root,source_config,inputs


def canonical(tag):
    if tag=='O':return tag
    core=shared_label(tag)
    return tag[:2]+core if core!='O' else tag


def spans(tags):
    result=[];start=None;previous='O'
    for i,tag in enumerate([*tags,'O']):
        kind=tag[2:] if tag.startswith(('B-','I-')) else 'O'
        if start is not None and (kind!=previous or tag.startswith('B-')):
            result.append((start,i,previous));start=None
        if kind!='O' and start is None:start=i
        previous=kind
    return result


def experiment(cfg,output):
    source=source_config(cfg);bank=torch.load(source_root(cfg)/'prepare'/'sentences.pt',weights_only=False);source_labels=bank['labels']['ner']
    @lru_cache(maxsize=None)
    def token_data(dataset):
        local=dataset_config(source,dataset);local.encoder.special_tokens=True
        return prepare_data(local)[0]
    def words(row):
        if 'words' in row:return row['words']
        dataset,split,index=row['source_id'].split('/')
        return token_data(dataset)[split][int(index)]['words']
    word_types={};entity_types={}
    for row in bank['sets']['train']:
        if row['fold'] not in ('g1','g2'):continue
        forms=words(row);tags=[source_labels[int(y)] for y in row['gold']['ner']]
        for form,tag in zip(forms,tags):word_types.setdefault(form,set()).add(tag[2:] if tag!='O' else 'O')
        for a,b,kind in spans(tags):entity_types.setdefault(tuple(forms[a:b]),set()).add(kind)
    for dataset,rows,part,_ in inputs(cfg,source):
        forms=[];ambiguity=[];seen=[];truth=[]
        for row in rows:
            labels=source_labels if cfg.sensitivity.get('adapter') else LABELS[dataset]
            text=words(row);tags=[labels[int(y)] if cfg.sensitivity.get('adapter') else canonical(labels[int(y)]) for y in row['gold']['ner']];span_for={}
            for a,b,kind in spans(tags):
                key=tuple(text[a:b]);category=0 if key not in entity_types else (2 if kind in entity_types[key] else 1)
                for i in range(a,b):span_for[i]=category
            for focal in row['focals'].tolist():
                form=text[focal];n=len(word_types.get(form,()));forms.append(form);ambiguity.append(0 if not n else 1 if n==1 else 2)
                seen.append(span_for.get(focal,3));truth.append(source_labels.index(tags[focal]) if tags[focal] in source_labels else -1)
        torch.save({'forms':forms,'ambiguity':torch.tensor(ambiguity),'seen':torch.tensor(seen),'truth':torch.tensor(truth),'sentence':part['sentence'],
                    'error_space':'native BIO with canonical PER/LOC/ORG names; absent source tags cannot be predicted'},output/(dataset+'.pt'))
    return {'training_sentences':sum(r['fold'] in ('g1','g2') for r in bank['sets']['train'])}
