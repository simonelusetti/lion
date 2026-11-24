"""Choose one cell on mean raw dev span F1, before accessing any test feature cache."""
from ..diagnostics.common import read_csv,write_csv,save_json
import torch
from .generate import root,cells


def experiment(cfg,output):
    rows=read_csv(root(cfg)/'train'/'dev_per_seed.csv');summary=[]
    baseline=torch.tensor([float(r['span_f1']) for r in rows if r['name']=='baseline'])
    for name in ['baseline']+[c['name'] for c in cells(cfg)]:
        score=torch.tensor([float(r['span_f1']) for r in rows if r['name']==name]);delta=score-baseline
        summary.append({'name':name,'mean_dev_span_f1':float(score.mean()),'seed_sd':float(score.std()),'difference_from_baseline':float(delta.mean()),'difference_seed_sd':float(delta.std())})
    # Stable grid-order tie breaking; the baseline is a reference, not an additional grid cell.
    winner=max(summary[1:],key=lambda r:r['mean_dev_span_f1']);index=[c['name'] for c in cells(cfg)].index(winner['name'])+1
    write_csv(output/'dev_summary.csv',summary);save_json(output/'selection.json',{'criterion':f'mean native {cfg.certification.source} dev span F1 across seeds 0/1/2, after existing dev token-macro-F1 checkpoint selection','selected':winner['name'],'head_index':index,'dev_span_f1':winner['mean_dev_span_f1'],'baseline_dev_span_f1':summary[0]['mean_dev_span_f1'],'tie_break':'first cell in fraction-then-threshold ascending grid order','test_accessed':False})
    print('DEV SELECTED:',winner,flush=True);return {'selected':winner['name']}
