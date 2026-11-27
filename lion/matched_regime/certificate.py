"""Joint-error transfer diagnostics, with explicit unsupported groups and units."""
import math
import torch
from scipy.stats import beta
from ..diagnostics.common import absolute,write_csv,save_json
from ..guided_refinement.population import cp
from ..guided_refinement.binwise import minimize
from .data import root,load


def natural(cfg):
    run=absolute(cfg.matched.judge_run);a=load(run/'prepare'/'calibration_A.pt');b=load(run/'judge_outputs'/'predictions.pt')['calibration_probability'];lookup=load(root(cfg)/'prepare'/'majorities.pt');words=load(root(cfg)/'prepare'/'calibration_words.pt');K=len(lookup['labels']);majority=torch.tensor([lookup['majority'].get(w,K) for w in words]);confidence,pred=a['probability'].max(-1);high=confidence>=cfg.joint.tau;departure=majority!=pred;agree=pred==b.argmax(-1);wrong=pred!=a['gold'];result={}
    for name,mask,group,G in [('refined_departure',high&departure,majority*K+pred,(K+1)*K),('label_only_departure',high&departure,pred,K),('label_only_original',high,pred,K)]:
        n=torch.bincount(group[mask],minlength=G);agreed=torch.bincount(group[mask&agree],minlength=G);errors=torch.bincount(group[mask&agree&wrong],minlength=G)
        upper=torch.tensor([float(beta.ppf(1-cfg.matched.alpha_cal/G,int(e)+1,int(d-e))) if d and e<d else 1. for d,e in zip(agreed,errors)],dtype=torch.double)
        result[name]={'groups':G,'natural':n,'agreed':agreed,'wrong':errors,'upper':upper,'natural_agreement':float((mask&agree).sum()/mask.sum()),'natural_tokens':int(mask.sum()),'naive_A_accuracy':float((mask&~wrong).sum()/mask.sum())}
    return result,lookup


def evaluate(cfg,name,pool,indices,b_prediction,reference,lookup,output):
    K=len(lookup['labels']);assigned=pool['y'][indices];a=pool['a_prediction'][indices];majority=torch.tensor([lookup['majority'].get(pool['word'][i],K) for i in indices.tolist()]);agreement=b_prediction==a
    # Mention labels are by design, so only A=B=assigned can certify those labels.
    agreed=agreement&(a==assigned);source_ids=[pool['source_id'][i] for i in indices.tolist()];gold=pool['source_gold'][indices];o=lookup['labels'].index('O')
    composition={'entity_to_O':int(((gold!=o)&(assigned==o)).sum()),'O_to_entity':int(((gold==o)&(assigned!=o)).sum()),'entity_to_different_entity':int(((gold!=o)&(assigned!=o)&(gold!=assigned)).sum())};composition['other']=len(indices)-sum(composition.values())
    summaries=[];groups=[]
    for scheme,ref in reference.items():
        G=ref['groups'];group=majority*K+a if scheme=='refined_departure' else a;mass=torch.bincount(group[agreed],minlength=G);m=int(agreed.sum());intervals=[cp(int(c),m,cfg.matched.alpha_mass/(2*G)) for c in mass];lower=torch.tensor([v[0] for v in intervals],dtype=torch.double);upper=torch.tensor([v[1] for v in intervals],dtype=torch.double)
        minimum,q,_=minimize(lower,upper,-ref['upper']) if m else (0.,None,None);bound=max(0.,1+minimum) if m and q is not None else 0.;unknown=((mass>0)&(ref['agreed']==0)).any();plugin=None if unknown or not m else 1-float((mass.double()/m)@(ref['wrong'].double()/ref['agreed'].clamp_min(1)))
        summaries.append({'pool':name,'scheme':scheme,'pool_size':len(indices),'agreed_assigned':m,'agreement_AB':float(agreement.double().mean()) if len(indices) else None,'agreement_AB_and_assigned':m/len(indices) if len(indices) else None,
            'natural_reference_n':ref['natural_tokens'],'natural_agreement':ref['natural_agreement'],'naive_A_accuracy':ref['naive_A_accuracy'],'certificate_delta0':bound,'plugin_correctness':plugin,
            'delta95':max(0.,bound-.95),'delta90':max(0.,bound-.90),'margin95':bound-.95,'passes95_numerically':m>0 and q is not None and bound>=.95,'distinct_source_sentences':len(set(source_ids)),'source_sentence_repeats':len(source_ids)-len(set(source_ids)),
            'unsupported_agreed_mass':float(mass[ref['agreed']==0].sum()/m) if m else None,'alpha_cal':cfg.matched.alpha_cal,'alpha_mass':cfg.matched.alpha_mass,
            **{f'{key}_share':v/len(indices) if len(indices) else None for key,v in composition.items()},
            'coverage_status':'Requested CP calculation; natural units one focal/sentence. Generated pools share source sentences and five-draw anchors; token-count mass intervals do not establish unconditional generator-population coverage.','population':'A=B=assigned subset of this pool; for MENTION this can be smaller than A=B, since assigned labels come from mention type, not A.'})
        for g in range(G):
            previous=g//K if scheme=='refined_departure' else None;pred=g%K
            groups.append({'pool':name,'scheme':scheme,'majority':'unseen' if previous==K else lookup['labels'][previous] if previous is not None else 'any','A_label':lookup['labels'][pred],
                'natural_reference_n':int(ref['natural'][g]),'natural_agreed':int(ref['agreed'][g]),'natural_agreed_wrong':int(ref['wrong'][g]),'wrong_upper':float(ref['upper'][g]),'generated_agreed':int(mass[g]),'q_lower':float(lower[g]),'q_upper':float(upper[g]),'worst_mass':float(q[g]) if q is not None else None})
    return summaries,groups


def save(cfg,output,summaries,groups):
    write_csv(output/'certificates.csv',summaries);write_csv(output/'groups.csv',groups)
    save_json(output/'method.json',{'majority_pool':'2400 original student-training sentences only; exact case-sensitive word types; deterministic label-order tie break. Unseen is a separate majority group.',
        'assumption':'Within each displayed group, the A=B=assigned generated subset joint-error rate is at most the corresponding natural A=B error rate plus delta. This is an unverified joint-error-transfer assumption; distinct encoder families do not imply independent errors.',
        'references':'refined_departure: A confidence>=.7 and A label != majority; unseen is treated as a departure. label_only_departure uses the same rows with coarser groups. label_only_original reproduces the prior high-confidence-only reference. Gold errors retained. Calibration predictions/focals unchanged.',
        'support':'Generated returns to the majority label are not discarded; their refined-reference diagonal groups have no natural agreements and w=1. Any other empty natural-agreement group also gets w=1.',
        'bounds':'For each pool/scheme: .025/G per natural error upper bound, .025/(2G) per mass endpoint. Bounded-simplex max, checked using independent LP. Confidence is pointwise per pool/scheme, not a selected maximum or a simultaneous claim across all schemes/pools.',
        'units':'Natural sample: one fixed random focal per sentence. Generated outputs share source sentences and fixed five-attempt anchors; distinct-source counts and repeats are shown. CP token-count mass bounds are reported as nominal diagnostics when independence cannot be established. They are not advertised as sentence-independent 95% certificates.',
        'slack':'lower(delta)=max(0,lower(0)-delta); delta* is max(0,lower(0)-threshold), with zero when already failing.',
        'MENTION':'Labelled by donor entity type. Certificate is only for A=B=assigned; it does not silently certify mention labels when A=B disagrees with them. ALL certificates likewise concern the agreed subset, not every ALL training token.'})
