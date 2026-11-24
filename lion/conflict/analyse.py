"""Three questions only: held-out coverage, raw MLM supply and paired classifier room."""
import json
import math
import torch
from ..diagnostics.common import write_csv
from . import data


def moments(x):
    n=x.numel();return {'rows':n,'mean':float(x.mean()) if n else None,'sd':float(x.std()) if n>1 else None,'row_se':float(x.std()/math.sqrt(n)) if n>1 else None}


def distribution(x,prefix):
    m=moments(x);q=torch.quantile(x,torch.tensor([0.,.25,.5,.75,1.],dtype=x.dtype),interpolation='nearest').tolist() if len(x) else [None]*5
    if len(x) and not torch.isfinite(x).all():m['sd']=m['row_se']=None
    return {prefix+'_'+k:v for k,v in m.items() if k!='rows'}|{prefix+'_'+k:v for k,v in zip(('min','q25','median','q75','max'),q)}


def seeds(x):
    # x: [seed,word], common word support across all seeds; draws already averaged within word.
    common=torch.isfinite(x).all(0);x=x[:,common];n=x.shape[1];point=x.mean(1) if n else x.new_full((x.shape[0],),torch.nan)
    return {'rows':n,'small_bin':n<50,'mean':float(point.mean()) if n else None,'seed_sd':float(point.std()) if n and len(point)>1 else None,'row_se':float(x.mean(0).std()/n**.5) if n>1 else None},point


def groups(bank,slots):
    yield 'all',torch.ones(len(slots['y']),dtype=torch.bool)
    if len(bank['groups'])>1:
        for i,name in enumerate(bank['groups']):yield name,slots['group']==i


def experiment(cfg,output):
    predictor=[];coverage=[];supply=[];room=[];paired=[];per_seed=[];natural=[];audit_pool=[]
    for name in cfg.coverage.datasets:
        bank=data.load(cfg,name);test=bank['slots']['id'];ctx={s:torch.load(data.root(cfg)/'context'/name/f'{s}.pt',weights_only=False) for s in ('validation','id')}
        for split,values in ctx.items():
            slots=bank['slots'][split];p=values['probability'];correct=(p.argmax(-1)==slots['y']).double()
            for group,mask in groups(bank,slots):
                predictor.append({'dataset':name,'split':split,'group':group,'rows':int(mask.sum()),'small_bin':int(mask.sum())<50,'accuracy':moments(correct[mask])['mean'],'accuracy_row_se':moments(correct[mask])['row_se'],**distribution(values['ce'][mask],'ce')})
            for tau in cfg.conflict.thresholds:
                selected=p.max(-1).values>=tau
                for group,mask in groups(bank,slots):
                    for subset in (('all','conflict_prone') if split=='id' else ('all',)):
                        population=mask&(slots['conflict_prone'] if subset=='conflict_prone' else True);chosen=selected&population;n=int(chosen.sum());ce=values['ce'][chosen];m=moments(ce);den=int(population.sum())
                        coverage.append({'dataset':name,'split':split,'tau':tau,'group':group,'subset':subset,'population':den,'selected':n,'small_bin':n<50,'coverage':n/den if den else None,'coverage_row_se':math.sqrt((n/den)*(1-n/den)/den) if den else None,
                            'selected_error':moments(1-correct[chosen])['mean'],'selected_error_row_se':moments(1-correct[chosen])['row_se'],'ce_mean':m['mean'],'ce_sd':m['sd'],'ce_row_se':m['row_se'],'ce_upper_95':m['mean']+1.645*m['row_se'] if n>1 else None})
        sup=torch.load(data.root(cfg)/'supply'/f'{name}.pt',weights_only=False);measured=torch.load(data.root(cfg)/'room'/f'{name}.pt',weights_only=False)
        native_heads=measured['heads'];pred=measured['prediction'].long();original=measured['original'].long();n,k=sup['words'].shape
        id_to_majority=torch.full((len(bank['lexicon']['tokens']),),-1);id_to_majority[bank['lexicon']['ids']]=bank['lexicon']['majority'];word_label=id_to_majority[sup['words'].clamp_min(0)]
        candidate=sup['words']>=0;gold=test['y'];column=torch.arange(k)[None];draws=cfg.conflict.natural_draws
        for tau in cfg.conflict.thresholds:
            selected=ctx['id']['probability'].max(-1).values>=tau
            for group,mask in groups(bank,test):
                chosen=mask&selected;row={'dataset':name,'tau':tau,'group':group,'rows':int(chosen.sum()),'small_bin':int(chosen.sum())<50}
                for key in ('allowed_mass','conflict_mass','nonconflict_mass','max_conflict_probability','priced_bound'):row.update(distribution(sup[key][chosen],key))
                for key,b in [('plausible_conflict',.01),('plausible_conflict',.05),('priced_le',.1),('priced_le',.2)]:
                    x=(sup['max_conflict_probability'][chosen]>=b).double() if key=='plausible_conflict' else (sup['priced_bound'][chosen]<=b).double();m=moments(x);row[f'{key}_{b}']=m['mean'];row[f'{key}_{b}_row_se']=m['row_se']
                row['no_conflict_word_share']=moments((sup['max_conflict_probability'][chosen]==0).double())['mean']
                for key in ('conflict_mass','nonconflict_mass'):row.update(distribution((sup[key]/sup['allowed_mass'])[chosen],'conditional_'+key))
                supply.append(row)
                if int(chosen.sum()):
                    for kind,label in [(-1,'unclassified'),(0,'conflict'),(1,'nonconflict')]:natural.append({'dataset':name,'tau':tau,'group':group,'kind':label,'contexts':int(chosen.sum()),'draws':int(chosen.sum())*draws,'draw_share':float((sup['kind'][chosen,:draws]==kind).double().mean())})
                for scheme in ('natural','top'):
                    scheme_mask=(column<draws) if scheme=='natural' else (column>=draws)
                    masks=[candidate&scheme_mask&(sup['kind']==kind)&chosen[:,None] for kind in (0,1)]
                    for layer in cfg.conflict.layers:
                        h=[i for i,m in enumerate(native_heads) if m['layer']==layer];prediction=pred[:,:,h].permute(2,0,1);orig=(original[:,h].T==gold[None]).double();values=[]
                        for kind,kind_mask in enumerate(masks):
                            count=kind_mask.sum(-1);support=count>0
                            follow_context=((prediction==gold[None,:,None])*kind_mask[None]).sum(-1).double()/count.clamp_min(1)[None];follow_context[:,~support]=torch.nan;values.append(follow_context)
                            different=kind_mask&(word_label!=gold[:,None]);diff_count=different.sum(-1)
                            follow_word=((prediction==word_label[None])*different[None]).sum(-1).double()/diff_count.clamp_min(1)[None];follow_word[:,diff_count==0]=torch.nan
                            for metric,x in [('context_label',follow_context),('word_label_when_different',follow_word),('original_accuracy',orig.masked_fill(~support[None],torch.nan))]:
                                m,point=seeds(x);r={'dataset':name,'tau':tau,'group':group,'scheme':scheme,'layer':layer,'kind':('conflict','nonconflict')[kind],'metric':metric,'insertions':int(kind_mask.sum()),**m};room.append(r)
                                for seed,value in zip(cfg.conflict.seeds,point.tolist()):per_seed.append({**r,'seed':seed,'value':value})
                        common=torch.isfinite(values[0]).all(0)&torch.isfinite(values[1]).all(0);gap=(values[1]-values[0]).masked_fill(~common[None],torch.nan);m,point=seeds(gap)
                        paired.append({'dataset':name,'tau':tau,'group':group,'scheme':scheme,'layer':layer,'comparison':'nonconflict_minus_conflict_context_following',**m,'original_accuracy':float(orig[:,common].mean()) if common.any() else None})
                        for seed,value in zip(cfg.conflict.seeds,point.tolist()):per_seed.append({'dataset':name,'tau':tau,'group':group,'scheme':scheme,'layer':layer,'kind':'paired_gap','metric':'context_label','seed':seed,'value':value,'rows':m['rows']})
        lowest=cfg.conflict.natural_draws
        for slot in ((sup['words'][:,lowest]>=0)&(sup['kind'][:,lowest]==0)).nonzero().flatten().tolist():audit_pool.append((name,slot))
        print(f'{name}: coverage, supply and room summarised',flush=True)
    for filename,rows in [('predictor',predictor),('coverage',coverage),('supply',supply),('room',room),('paired_gap',paired),('per_seed',per_seed),('natural_draws',natural)]:write_csv(output/f'{filename}.csv',rows)
    # One most-plausible conflict per context, uniformly sample 100 contexts across both datasets.
    order=torch.randperm(len(audit_pool),generator=torch.Generator().manual_seed(cfg.conflict.sample_seed+1))[:cfg.conflict.audit_rows].tolist();audit=[];cached={}
    for i in order:
        name,slot=audit_pool[i]
        if name not in cached:cached[name]=(data.load(cfg,name),torch.load(data.root(cfg)/'supply'/f'{name}.pt',weights_only=False),torch.load(data.root(cfg)/'room'/f'{name}.pt',weights_only=False))
        bank,sup,measured=cached[name];s=bank['slots']['id'];r=bank['sets']['id'][int(s['owner'][slot])];form=bank['lexicon']['tokens'][int(sup['words'][slot,cfg.conflict.natural_draws])];focal=int(s['focal'][slot]);lex=bank['lexicon'];index=int((lex['ids']==sup['words'][slot,cfg.conflict.natural_draws]).nonzero()[0]);counts=lex['counts'][index]
        original_forms=[]
        for f in range(len(r['focals'])):
            pieces=r['ids'][r['word_ids']==f].tolist();original_forms.append(''.join(lex['tokens'][p].removeprefix('##') for p in pieces))
        inserted=original_forms.copy();inserted[focal]=form
        entry={'dataset':name,'sentence_id':r['source_id'],'sentence':' '.join(original_forms),'inserted_sentence':' '.join(inserted),'slot':focal,'original_word':original_forms[focal],'inserted_word':form,'y_c':bank['labels'][int(s['y'][slot])],
               'training_counts':json.dumps({label:int(value) for label,value in zip(bank['labels'],counts)}),'training_distribution':json.dumps({label:float(value/counts.sum()) for label,value in zip(bank['labels'],counts)}),
               'majority_label':bank['labels'][int(counts.argmax())],'mlm_probability':float(sup['word_probability'][slot,cfg.conflict.natural_draws]),'context_gold_probability':float(sup['context_gold_probability'][slot]),'tau':min(cfg.conflict.thresholds)}
        for h,meta in enumerate(measured['heads']):entry[f"prediction_layer{meta['layer']}_seed{meta['seed']}"]=bank['labels'][int(measured['prediction'][slot,cfg.conflict.natural_draws,h])]
        audit.append(entry)
    write_csv(output/'audit.csv',audit)
    return {'audit_insertions':len(audit),'room_rows':len(room)}
