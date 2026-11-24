"""Only the dev-selected cell and shared baseline are evaluated on raw test sets."""
import json
import torch
from tagger.ensemble import IndependentReadouts
from ..partition.train import predict
from ..certification.classify import evaluate,span_counts
from ..diagnostics.common import absolute,write_csv,save_json
from ..labels import LABELS
from .generate import root


@torch.no_grad()
def experiment(cfg,output):
    selection=json.loads((root(cfg)/'select'/'selection.json').read_text());indices=torch.tensor([0,selection['head_index']]);evaluations=torch.load(absolute(cfg.grid_pilot.evaluation_cache),mmap=True,weights_only=False);labels=evaluations['labels'];records=[]
    for seed in cfg.augmentation.seeds:
        saved=torch.load(root(cfg)/'train'/f'seed{seed}'/'model.pt',weights_only=False);core=IndependentReadouts(cfg.tagger,768,len(labels),2).to(cfg.runtime.device)
        core.load_state_dict({k:v[indices] for k,v in saved['state'].items()});core.eval()
        for split,item in evaluations['sets'].items():
            if split=='validation':continue
            part=item['data'];target=item['target'];logits=predict(core,part,cfg)
            for space in (['native','shared_ner'] if split=='id' else ['shared_ner']):
                metrics=evaluate(logits,part,cfg.certification.source,target,labels,space);c,p,e=span_counts(logits.argmax(-1),part['y'],part['sentence'],labels,LABELS[target] if space=='shared_ner' else labels,shared=space=='shared_ner')
                for h,arm in enumerate(['baseline',selection['selected']]):records.append({'seed':seed,'name':arm,'target':target,'space':space,'span_f1':float(metrics['span_f1'][h]),'accuracy':float(metrics['accuracy'][h]),'precision':float(c[h]/p[h].clamp_min(1)),'recall':float(c[h]/e.clamp_min(1))})
            torch.save({'logits':logits,'gold':part['y'],'sentence':part['sentence'],'heads':['baseline',selection['selected']]},output/f'seed{seed}_{split}.pt')
    summary=[]
    for target,space in sorted({(r['target'],r['space']) for r in records}):
        a=torch.tensor([r['span_f1'] for r in records if (r['target'],r['space'],r['name'])==(target,space,'baseline')]);b=torch.tensor([r['span_f1'] for r in records if (r['target'],r['space'],r['name'])==(target,space,selection['selected'])]);d=b-a
        summary.append({'target':target,'space':space,'selected':selection['selected'],'baseline_mean':float(a.mean()),'baseline_sd':float(a.std()),'selected_mean':float(b.mean()),'selected_sd':float(b.std()),'paired_difference':float(d.mean()),'paired_difference_sd':float(d.std())})
    write_csv(output/'test_per_seed.csv',records);write_csv(output/'test_summary.csv',summary);save_json(output/'evaluated_heads.json',{'selection':selection,'evaluated':['baseline',selection['selected']],'unselected_configurations_evaluated_on_test':False})
    return {'selected':selection['selected'],'test_evaluations':len(records)}
