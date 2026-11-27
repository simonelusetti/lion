"""Within-frozen-bin score ECDF distances from cached scores only."""
import math,sys,time
import torch
from forge import run,start_run
from omegaconf import OmegaConf
from ..diagnostics.common import ROOT,absolute,save_json,write_csv
from .population import load


def ecdf(values):
    knots,counts=values.unique(sorted=True,return_counts=True)
    return {'knots':knots,'cdf':counts.cumsum(0).double()/len(values)}


def at(curve,knots):
    index=torch.searchsorted(curve['knots'],knots,right=True)
    return torch.cat((torch.zeros(1,dtype=torch.double),curve['cdf']))[index]


def summaries(population,bin_id,values):
    if not len(values):return {'population':population,'bin':bin_id,'n':0,'min':None,'median':None,'p90':None,'p95':None,'p99':None,'max':None}
    q=torch.quantile(values,torch.tensor([.5,.9,.95,.99],dtype=torch.double))
    return {'population':population,'bin':bin_id,'n':len(values),'min':float(values.min()),'median':float(q[0]),'p90':float(q[1]),'p95':float(q[2]),'p99':float(q[3]),'max':float(values.max())}


def main(cfg):
    OmegaConf.resolve(cfg);cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg);started=time.monotonic()
    source=absolute(cfg.within_bin.source);output=absolute(cfg.within_bin.results);output.mkdir(parents=True,exist_ok=True);OmegaConf.save(cfg,output/'config.yaml')
    natural=load(source/'diagnostic'/'natural_reference.pt')['phi'].double();edges=load(source/'design'/'bins.pt')['edges'];arms=list(cfg.mechanism_pilot.arms);J=len(edges)+1;A=len(arms)
    delta=cfg.within_bin.alpha/(J*(A+1))
    def epsilon(n):return math.sqrt(math.log(2/delta)/(2*n)) if n else None
    natural_bins=torch.bucketize(natural,edges);reference=[natural[natural_bins==j] for j in range(J)]
    curves={'natural':{j:ecdf(v) for j,v in enumerate(reference) if len(v)}}
    quantiles=[summaries('natural',j,v) for j,v in enumerate(reference)];pairs=[];arm_results=[]
    for arm in arms:
        values=load(source/'diagnostic'/f'{arm}_scores.pt')['phi'].double();bins=torch.bucketize(values,edges);curves[arm]={};refuted=[];untestable=[]
        for j in range(J):
            x=reference[j];y=values[bins==j];n,m=len(x),len(y);quantiles.append(summaries(arm,j,y));e0,er=epsilon(n),epsilon(m)
            if n and m:
                a=curves['natural'][j];b=ecdf(y);curves[arm][j]=b
                # Both empirical CDFs are right-continuous. Evaluate after every
                # tied jump at the union of their unique observed score values.
                knots=torch.cat((a['knots'],b['knots'])).unique(sorted=True);difference=(at(a,knots)-at(b,knots)).abs();d=float(difference.max());witness=float(knots[difference.argmax()]);lower=max(0.,d-e0-er);reject=lower>0
                if reject:refuted.append(j)
            else:d=lower=witness=None;reject=False;untestable.append(j)
            pairs.append({'arm':arm,'bin':j,'score_lower':'-inf' if j==0 else float(edges[j-1]),'score_upper':'inf' if j==J-1 else float(edges[j]),'natural_n':n,'generated_n':m,'KS_distance':d,'natural_epsilon':e0,'generated_epsilon':er,'distance_lower':lower,'refuted':reject,'status':'untestable' if not n or not m else 'refuted' if reject else 'inconclusive','max_distance_score':witness})
        arm_results.append({'arm':arm,'accepted':len(values),'refuted_bins':','.join(map(str,refuted)),'refuted_bin_count':len(refuted),'untestable_bins':','.join(map(str,untestable)),'any_refuted':bool(refuted)})
    torch.save(curves,output/'empirical_cdfs.pt');write_csv(output/'pairs.csv',pairs);write_csv(output/'score_quantiles.csv',quantiles);write_csv(output/'arms.csv',arm_results)
    proof={'working_assumptions':'Adopt the previously unverified IID conditions as requested: conditional within-bin score samples are IID from the frozen natural and accepted populations. Bins and policies fixed independently of estimation. No provenance re-audit.',
        'alpha':cfg.within_bin.alpha,'CDF_count':J*(A+1),'delta_each':delta,'bins':J,'arms':A,
        'inequality':'For any CDF, including discrete distributions with ties, P(sup|Fhat-F|>epsilon)<=2 exp(-2*n*epsilon^2).',
        'simultaneous_argument':'Apply DKW conditional on each positive bin sample size. Its failure bound delta holds for every n, hence after averaging over random counts. Union bound over 5 natural and 55 generated CDFs gives total failure <=60*delta=.05; cross-CDF independence is unnecessary. Empty samples are untestable. On the simultaneous event, reverse triangle inequality gives ||F0-Fr||inf >= ||Fhat0-Fhatr||inf-epsilon(n)-epsilon(m). A positive clipped lower bound refutes score equality, and thus full within-bin input equality; probability of any false refutation is <=.05 under the working assumptions.',
        'ties':'Unique union of observed values, right-inclusive searchsorted, exact empirical jumps; no jitter or continuous-null KS p-values.',
        'quantiles':'torch.quantile, linear interpolation; p90,p95,p99. Exact observed maxima. Empty bins have missing quantiles.',
        'interpretation':'Only the necessary within-bin score-distribution consequence is tested. Rejection does not establish failure of correctness transfer or label accuracy; non-rejection does not validate full input equality or imply augmentation cannot improve F1.',
        'source':'Massart (1990), The Tight Constant in the Dvoretzky-Kiefer-Wolfowitz Inequality, https://doi.org/10.1214/aop/1176990746',
        'no_new_generation_scoring_training':True}
    save_json(output/'method.json',proof)
    lines=['# Within-bin NLL distribution comparison','',
        'Saved natural and accepted-output scores only; five existing frozen bins. Adopt IID sampling as a working assumption. No generation, rescoring, fitting, new annotations, or bin changes. Bin IDs below are 1–5; CSV bin IDs are 0–4.','',
        f'DKW–Massart: P(||Fhat−F||∞ > ε) ≤ 2 exp(−2nε²). There are {J*(A+1)} CDFs, so δ={delta:.10f}. Each natural CDF is counted once. Conditional on positive sample counts, apply DKW to each CDF, then union bound and reverse triangle inequality: D_true ≥ max(0,D_emp−ε_n−ε_m). This yields simultaneous 95% lower bounds and at most 5% family-wide false refutation under the working sampling assumptions. Independence between arms or confidence events is not required. Exact tied score values are retained. [Massart (1990)](https://doi.org/10.1214/aop/1176990746).','',
        '| Arm | Accepted | Refuted bins | Untestable bins |','|---|---:|---|---|']
    for r in arm_results:lines.append(f"| {r['arm']} | {r['accepted']} | {', '.join(str(int(j)+1) for j in r['refuted_bins'].split(',') if j) or 'none'} | {', '.join(str(int(j)+1) for j in r['untestable_bins'].split(',') if j) or 'none'} |")
    lines+=['','| Arm | Bin | Natural n | Generated n | KS distance | Simultaneous lower bound | Outcome |','|---|---:|---:|---:|---:|---:|---|']
    for r in pairs:
        distance=f"{r['KS_distance']:.5f}" if r['KS_distance'] is not None else '—';lower=f"{r['distance_lower']:.5f}" if r['distance_lower'] is not None else '—'
        lines.append(f"| {r['arm']} | {r['bin']+1} | {r['natural_n']} | {r['generated_n']} | {distance} | {lower} | {r['status']} |")
    lines+=['','Highest-NLL bin: φ > '+f'{float(edges[-1]):.6f}'+'. Quantiles for every bin are in score_quantiles.csv.','',
        '| Population | n | Median | p90 | p95 | p99 | Maximum |','|---|---:|---:|---:|---:|---:|---:|']
    for r in quantiles:
        if r['bin']==J-1:lines.append('| '+r['population']+' | '+str(r['n'])+' | '+' | '.join(f'{r[k]:.5f}' if r[k] is not None else '—' for k in ('median','p90','p95','p99','max'))+' |')
    lines+=['','Positive lower bounds establish score mismatch hidden inside frozen bins under the working assumptions. Empty and low-count generated bins can be inconclusive because their DKW radii are large. Refutation is of full within-bin input equality through a necessary score consequence; it does not directly test correctness transfer, pseudo-label accuracy, or downstream usefulness. Non-refutation does not validate the assumption.','',
        'Machine-readable results: pairs.csv, arms.csv, score_quantiles.csv. Full exact empirical CDFs: empirical_cdfs.pt. Allocation and simultaneous argument: method.json.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n');save_json(output/'runtime.json',{'seconds':time.monotonic()-started});tracking.finish({'pairs':len(pairs),'refuted':sum(r['refuted'] for r in pairs),'arms_refuted':sum(r['any_refuted'] for r in arm_results)})
    print(output/'RESULTS.md',flush=True)
    for r in arm_results:print(r,flush=True)
    return 0


if __name__=='__main__':run('lion.guided_refinement',sys.argv[1:],main_module='within_bin',config_dir=str(ROOT/'conf'),config_name='population_within_bin')
