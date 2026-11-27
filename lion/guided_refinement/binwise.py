"""Read-only reanalysis of cached observations; no inference or model fitting."""
import sys,time
import torch
from scipy.optimize import linprog
from scipy.stats import beta
from forge import run,start_run
from omegaconf import OmegaConf
from ..diagnostics.common import ROOT,absolute,read_csv,write_csv,save_json
from .population import load,cp,correlation


def minimize(lower,upper,correctness):
    # Start at lower masses; fill remaining mass in increasing correctness order.
    feasible=bool((lower<=upper).all() and lower.sum()<=1+1e-12 and upper.sum()>=1-1e-12)
    if not feasible:return 0.,None,None
    q=lower.clone();remaining=1-float(q.sum())
    for index in correctness.argsort(stable=True).tolist():
        extra=min(remaining,float(upper[index]-q[index]));q[index]+=extra;remaining-=extra
    value=float(q@correctness)
    independent=linprog(correctness.numpy(),A_eq=torch.ones(1,len(q)).numpy(),b_eq=[1.],bounds=list(zip(lower.tolist(),upper.tolist())),method='highs')
    if not independent.success or abs(independent.fun-value)>1e-9 or abs(float(q.sum())-1)>1e-9:
        raise ArithmeticError('Bounded-simplex solution disagrees with independent linear program')
    if value<float(correctness.min())-1e-10 or value>float(correctness.max())+1e-10:
        raise ArithmeticError('Optimized bound outside bin-correctness range')
    return value,q,float(independent.fun)


def main(cfg):
    OmegaConf.resolve(cfg);cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg);started=time.monotonic()
    source=absolute(cfg.binwise.source);output=absolute(cfg.binwise.results);output.mkdir(parents=True,exist_ok=True);OmegaConf.save(cfg,output/'config.yaml')
    natural=load(source/'diagnostic'/'natural_reference.pt');edges=load(source/'design'/'bins.pt')['edges'];J=len(edges)+1;arms=list(cfg.mechanism_pilot.arms);A=len(arms)
    bins=torch.bucketize(natural['phi'].double(),edges);correct=natural['prediction']==natural['gold']
    n=torch.bincount(bins,minlength=J);k=torch.bincount(bins[correct],minlength=J)
    if int(n.sum())!=13061 or int(k.sum())!=12708:raise ValueError('Natural sample differs from completed experiment')
    L=torch.tensor([float(beta.ppf(.025/J,int(a),int(b-a)+1)) if a>0 and b>0 else 0. for a,b in zip(k,n)],dtype=torch.double)
    theta=k.double()/n.clamp_min(1);old=read_csv(source/'diagnostic'/'quality.csv');old_by_arm={r['arm']:r for r in old};old_bins=read_csv(source/'diagnostic'/'bin_counts.csv')
    natural_rows=[]
    for j in range(J):
        original=next(r for r in old_bins if r['population']=='natural' and int(r['bin'])==j)
        if int(original['count'])!=int(n[j]):raise ValueError('Frozen natural-bin counts changed')
        natural_rows.append({'bin':j,'score_lower':'-inf' if j==0 else float(edges[j-1]),'score_upper':'inf' if j==J-1 else float(edges[j]),'range_convention':'(lower, upper]',
            'n':int(n[j]),'correct':int(k[j]),'empirical_correctness':float(theta[j]) if n[j] else None,'correctness_lower':float(L[j]),'one_sided_tail':.025/J})
    write_csv(output/'natural_bins.csv',natural_rows)
    result=[];mass_rows=[];checks=[];tail=.025/(2*A*J)
    for arm in arms:
        cached=load(source/'diagnostic'/f'{arm}_scores.pt');counts=torch.bincount(torch.bucketize(cached['phi'].double(),edges),minlength=J);m=int(counts.sum())
        accepted=load(source/'generate'/arm/'accepted_features_draw0.pt')
        if m!=int(old_by_arm[arm]['accepted']) or not torch.equal(cached['proposal_indices'],accepted['anchor_id']):raise ValueError(f'{arm}: accepted sample mismatch')
        for j in range(J):
            if int(counts[j])!=int(next(r['count'] for r in old_bins if r['population']==arm and int(r['bin'])==j)):raise ValueError('Accepted-bin frequencies changed')
        empirical=counts.double()/m if m else torch.zeros(J,dtype=torch.double)
        intervals=[cp(int(c),m,tail) for c in counts];lower=torch.tensor([r[0] for r in intervals],dtype=torch.double);upper=torch.tensor([r[1] for r in intervals],dtype=torch.double)
        bound,q,lp=minimize(lower,upper,L) if m else (0.,None,None);feasible=q is not None
        unestimated=bool(((counts>0)&(n==0)).any());plugin=float(empirical@theta) if m and not unestimated else None
        fixed=float(empirical@L) if m else 0.
        previous=old_by_arm[arm]
        result.append({'arm':arm,'accepted':m,**{f'bin_{j}_proportion':float(empirical[j]) for j in range(J)},'plugin_quality':plugin,'binwise_lower':bound,'feasible':feasible,
            'strict_check':'PASS' if m and feasible and bound>=.95 else 'FAIL','margin':bound-.95,'plugin_unestimated':unestimated,
            'fixed_empirical_mass_lower':fixed,'mass_uncertainty_loss':fixed-bound,'calibration_uncertainty_loss':plugin-fixed if plugin is not None else None,
            'previous_worst_ratio_lower':float(previous['Pi_lower']),'previous_worst_ratio_plugin':float(previous['Pi_plugin']),'old_G':float(previous['old_G']),'acceptance':float(previous['acceptance']),
            'change_vs_previous_lower':bound-float(previous['Pi_lower'])})
        for j in range(J):mass_rows.append({'arm':arm,'bin':j,'accepted_count':int(counts[j]),'proportion':float(empirical[j]),'lower':float(lower[j]),'upper':float(upper[j]),'minimizing_mass':float(q[j]) if feasible else None,'tail':tail})
        checks.append({'arm':arm,'feasible':feasible,'greedy_bound':bound,'independent_linprog':lp,'difference':bound-lp if feasible else None,'min_L':float(L.min()),'max_L':float(L.max())})
    write_csv(output/'quality.csv',result);write_csv(output/'accepted_bins.csv',mass_rows)
    # Requested mathematical limiting cases, evaluated directly with both solvers.
    lo=torch.zeros(J,dtype=torch.double);hi=torch.ones(J,dtype=torch.double);known=torch.arange(1,J+1,dtype=torch.double);known/=known.sum()
    cases=[]
    for name,a,b,c,expected in [('constant_correctness',lo,hi,torch.full((J,),.8,dtype=torch.double),.8),('known_masses',known,known,L,float(known@L)),
        ('single_bin_concentration',torch.eye(J,dtype=torch.double)[0],torch.eye(J,dtype=torch.double)[0],L,float(L[0])),
        ('gold_empty_bin',torch.eye(J,dtype=torch.double)[0],torch.eye(J,dtype=torch.double)[0],torch.cat((torch.zeros(1,dtype=torch.double),L[1:])),0.)]:
        value,_,_=minimize(a,b,c)
        if abs(value-expected)>1e-10:raise ArithmeticError(name)
        cases.append({'case':name,'value':value,'expected':expected})
    save_json(output/'mathematical_checks.json',{'derivation':'Shared eta and Qr(input|Bj)=P0(input|Bj) imply E_Qr[T|Bj]=theta_j. Total expectation over accepted-output bins gives pi_r=sum_j q_rj theta_j. On simultaneous events theta>=L and q in C, pi>=min_C q.L. Bonferroni: J calibration tails .025/J plus 2*A*J mass tails .025/(2*A*J), total .05.',
        'arms':checks,'limiting_cases':cases,'no_combined_selection':'Old and new bounds remain separate; their maximum is not reported as a jointly 95%-valid certificate.'})
    # Audit disjointness/provenance without rescoring, changing populations, or refitting.
    design=load(source/'prepare'/'natural_design.pt');reference=load(source/'prepare'/'natural_reference.pt');design_ids={r['source_id'] for r in design};ref_ids=[r['source_id'] for r in reference]
    proposal=load(source/'prepare'/'proposal_indices.pt');audit={'natural_n':int(n.sum()),'natural_correct':int(k.sum()),'natural_correctness':float(k.sum()/n.sum()),
        'design_reference_source_overlap':len(design_ids&set(ref_ids)),'duplicate_reference_source_ids':len(ref_ids)-len(set(ref_ids)),
        'proposal_attempts':len(proposal),'proposal_design':'torch.randint with replacement from fixed 4000-anchor cohort, one independent trajectory per index, shared across arms; accepted conditional outputs supply q, not all proposals.',
        'calibration_provenance':'Reserved native-training calibration fold, excluded from scorer fitting and native-dev checkpoint selection. Earlier labelled calibration sample sentences excluded. Gold correctness uses saved natural gold labels against scorer predictions.',
        'independence_caveat':'Natural inputs were deduplicated and shuffled into disjoint design/reference partitions without replacement. Deduplication couples observations; one example per sentence and disjoint IDs alone do not establish IID Bernoulli trials for an external task population. The original implementation assumed independent natural sentences, but did not justify an exact binomial sampling model after this selection. These computations retain that conditional/nominal interpretation; unconditional exact task-population 95% coverage is not established. A finite-population justification would require specifying the target and sampling design rather than silently changing P0.',
        'within_bin_assumption':'Full input-distribution equality within bins and shared eta remain unverified modeling assumptions. A nominal PASS does not verify either these assumptions or the independence premise.',
        'population_or_bins_changed':False,'new_generation':False,'new_training':False,'new_inference':False}
    save_json(output/'audit.json',audit)
    scores=read_csv(source/'test'/'summary.csv');joined=[];correlations=[]
    for r in result:
        for s in scores:
            if s['head']==r['arm']:joined.append({**r,**s})
    write_csv(output/'arm_results.csv',joined)
    for group,names in [('all_11',arms),('refinement_5',list(cfg.guided.arms))]:
        for target,space in sorted({(r['target'],r['space']) for r in joined}):
            selected=[r for r in joined if r['arm'] in names and (r['target'],r['space'])==(target,space)];y=torch.tensor([float(r['paired_delta_vs_baseline']) for r in selected],dtype=torch.double)
            for metric in ('binwise_lower','plugin_quality','previous_worst_ratio_lower','previous_worst_ratio_plugin','old_G','acceptance'):
                x=torch.tensor([float(r[metric]) if r[metric] is not None else float('nan') for r in selected],dtype=torch.double)
                for kind in ('Pearson','Spearman'):
                    value,reason=correlation(x,y,kind=='Spearman');correlations.append({'group':group,'target':target,'space':space,'metric':metric,'correlation':kind,'value':value,'undefined_reason':reason,'arms':len(selected)})
    write_csv(output/'correlations.csv',correlations)
    lines=['# Direct binwise correctness reanalysis','',
        'No new generation, inference or classifier training. Frozen score bins, natural reference, accepted outputs and test F1 are unchanged. Under shared annotation mechanism and full within-bin input-distribution equality, total expectation gives pi_r = sum_j q_rj theta_j. Calibration tails are .025/J; each mass endpoint receives .025/(2*11*J). Greedy bounded-simplex minimization was independently verified with scipy.optimize.linprog for every arm, along with all requested limiting cases.','',
        '**Coverage caveat:** '+audit['independence_caveat']+' All PASS/FAIL entries below are nominal and conditional on the sampling and within-bin assumptions.','',
        '| Bin | Score range | Natural n | Correct k | Correctness | Lower L_j |','|---|---|---:|---:|---:|---:|']
    for r in natural_rows:lines.append(f"| {r['bin']} | ({r['score_lower']}, {r['score_upper']}] | {r['n']} | {r['correct']} | {100*r['empirical_correctness']:.3f}% | {100*r['correctness_lower']:.3f}% |")
    lines+=['','| Arm | Accepted | Plug-in | New lower | Margin pp | Conditional check | Previous lower | Old G |','|---|---:|---:|---:|---:|---|---:|---:|']
    for r in result:lines.append(f"| {r['arm']} | {r['accepted']} | {100*r['plugin_quality']:.3f}% | {100*r['binwise_lower']:.3f}% | {100*r['margin']:+.3f} | {r['strict_check']} | {100*r['previous_worst_ratio_lower']:.5f}% | {100*r['old_G']:.3f}% |")
    lines+=['','Accepted bin proportions and optimizing masses: accepted_bins.csv. arm_results.csv includes unchanged per-dataset F1 means, seed SDs and paired improvements. correlations.csv includes both scopes and all old/new metrics. audit.json records provenance; mathematical_checks.json records actual LP and limiting-case checks.','',
        '| Arm | Empirical quality | Calibration uncertainty loss pp | Mass uncertainty loss pp |','|---|---:|---:|---:|']
    for r in result:lines.append(f"| {r['arm']} | {100*r['plugin_quality']:.3f}% | {100*r['calibration_uncertainty_loss']:.3f} | {100*r['mass_uncertainty_loss']:.3f} |")
    lines+=['','All five bin-specific correctness lower bounds exceed 95%; consequently every feasible mixture passes the numerical threshold. Empirical quality differs little across arms (about 97.17–97.40%). The mass-uncertainty loss is about 0.024–0.034 percentage points for the five refinement arms, versus up to 0.412 points for the 28-example window-20 arm. The larger differences in the rigorous bounds therefore substantially reflect accepted-sample uncertainty, as well as bin composition. No old reverse-ratio penalty applies when generated mass avoids a natural bin. Neither old/new bounds nor their maximum are used to select or retrain models.','',
        '| Arm | Few-NERD native delta pp | CoNLL delta pp | WNUT delta pp | WikiANN delta pp |','|---|---:|---:|---:|---:|']
    datasets=[('fewnerd','native'),('conll2003','shared_ner'),('wnut','shared_ner'),('wikiann','shared_ner')]
    for arm in arms:
        values=[100*float(next(r['paired_delta_vs_baseline'] for r in joined if r['arm']==arm and (r['target'],r['space'])==dataset)) for dataset in datasets]
        lines.append('| '+arm+' | '+' | '.join(f'{v:+.3f}' for v in values)+' |')
    for scope in ('all_11','refinement_5'):
        lines+=['',f'Correlations across {scope}; cells are Pearson / Spearman.','',
            '| Dataset | New lower | New plug-in | Previous worst-ratio lower | Old G |','|---|---:|---:|---:|---:|']
        for dataset in datasets:
            cells=[]
            for metric in ('binwise_lower','plugin_quality','previous_worst_ratio_lower','old_G'):
                pair=[r for r in correlations if r['group']==scope and (r['target'],r['space'])==dataset and r['metric']==metric]
                cells.append(' / '.join(f"{r['value']:+.3f}" if r['value'] is not None else 'undefined: '+r['undefined_reason'] for r in pair))
            lines.append('| '+dataset[0]+' | '+' | '.join(cells)+' |')
    lines+=['','Correlations are descriptive. Across all arms, the new lower bound has high Pearson association on WNUT (+0.952), while native Few-NERD is weak (+0.163). Within refinement alone the native correlation reverses strongly (-0.955). The new bound therefore does not uniformly rank downstream usefulness better than the older metrics; its WNUT association also reflects accepted-sample-size uncertainty. Certificates remain separate and conditional.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n');save_json(output/'runtime.json',{'seconds':time.monotonic()-started});tracking.finish({'arms':A,'reference_n':int(n.sum())})
    print(output/'RESULTS.md',flush=True);print('Natural bins:',natural_rows,flush=True)
    for r in result:print(r['arm'],'plugin',r['plugin_quality'],'lower',r['binwise_lower'],r['strict_check'],'mass uncertainty loss',r['mass_uncertainty_loss'],flush=True)
    return 0


if __name__=='__main__':run('lion.guided_refinement',sys.argv[1:],main_module='binwise',config_dir=str(ROOT/'conf'),config_name='population_binwise')
