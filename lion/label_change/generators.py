"""Six held-out edit generators, fixed scorer/focals; no training or report."""
import json,math,sys,time
import torch
from transformers import AutoTokenizer
from forge import run,start_run
from omegaconf import OmegaConf
from ..certification import data as source,selector
from ..partition.train import load_model
from ..encoder import load_masked_encoder
from ..diagnostics.common import ROOT,absolute,read_csv,write_csv,save_json
from ..runtime import configure
from .data import sampled_rows


@torch.no_grad()
def experiment(cfg,output):
    bank=source.load(cfg);held=[r for r in selector.raw_rows(cfg,bank) if r['fold']=='calibration'];rows=sampled_rows(held,cfg.label_change.calibration_draws,cfg.label_change.seed+3)
    scorer=load_model(source.paths(cfg)/'scorer'/'ner',cfg).requires_grad_(False);encoder=load_masked_encoder(cfg)
    tokenizer=AutoTokenizer.from_pretrained(cfg.encoder.name,local_files_only=True);tokens=tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))));labels=bank['labels']['ner']
    baseline=json.loads((absolute(cfg.label_change.results)/'certify'/'certification.json').read_text());e=baseline['e_hat'];summaries=[];examples=[]
    def word(ids):return ''.join(tokens[i].removeprefix('##') for i in ids)
    for wi,window in enumerate(cfg.generators.windows):
        for fi,fraction in enumerate(cfg.generators.fractions):
            arm=f'{window}_{round(fraction*100)}';clock=time.monotonic();records=[]
            # Same rank draws within each window make the selected word sets nested across fractions.
            rank_rng=torch.Generator(device=cfg.runtime.device).manual_seed(cfg.generators.seed+wi*100)
            fill_rng=torch.Generator(device=cfg.runtime.device).manual_seed(cfg.generators.seed+wi*100+fi+10)
            for batch in selector.focal_batches(source.loader(rows,cfg),'ner',cfg):
                ids,attention,word_ids,focal,gold=(batch[k] for k in ('ids','attention','word_ids','focal','gold'))
                position=torch.arange(int(batch['counts'].max()),device=ids.device)[None];distance=(position-focal[:,None]).abs();eligible=(position<batch['counts'][:,None])&(distance>0)
                if window=='nearby':eligible&=distance<=cfg.generators.radius
                count=torch.floor(eligible.sum(-1)*fraction+.5).long()
                noise=torch.rand(eligible.shape,device=ids.device,generator=rank_rng).masked_fill(~eligible,torch.inf);rank=noise.argsort(-1).argsort(-1)
                dropped=eligible&(rank<count[:,None]);drop=source.word_to_pieces(dropped,word_ids,False)
                fills=selector.independent_fills(encoder,ids,attention,drop,1,bank['special'],cfg,fill_rng)
                vectors=selector.draw_features(encoder,ids,attention,word_ids,focal,fills,cfg)[:,0]
                confidence,prediction=selector.scorer_log_probs(scorer,vectors,cfg).exp().max(-1)
                original=batch['hidden'][batch['owner'],focal];orig_confidence,orig_prediction=selector.scorer_log_probs(scorer,original,cfg).exp().max(-1)
                ids,filled,word_ids,focal,gold,dropped,count,confidence,prediction,orig_confidence,orig_prediction=[v.detach().cpu() for v in (ids,fills[:,0],word_ids,focal,gold,dropped,count,confidence,prediction,orig_confidence,orig_prediction)]
                counts=batch['counts'].cpu().tolist()
                for j,nwords in enumerate(counts):
                    original_words=[word(ids[j,word_ids[j]==w].tolist()) for w in range(nwords)];changed_words=[word(filled[j,word_ids[j]==w].tolist()) for w in range(nwords)]
                    changed=[w for w in range(nwords) if original_words[w]!=changed_words[w]];g=int(gold[j]);p=int(prediction[j]);accepted=bool(changed) and p!=g and float(confidence[j])>=cfg.label_change.tau
                    records.append({'generator':arm,'row':len(records),'sentence_id':int(batch['sentence'][j]),'focal':int(focal[j]),'focal_word':original_words[int(focal[j])],'gold':labels[g],'proposed':labels[p],
                        'confidence':float(confidence[j]),'original_prediction':labels[int(orig_prediction[j])],'original_confidence':float(orig_confidence[j]),'original_correct':int(orig_prediction[j])==g,
                        'selected_words':int(count[j]),'changed_words':len(changed),'selected_positions':json.dumps(dropped[j].nonzero().flatten().tolist()),'changed_positions':json.dumps(changed),
                        'context_changed':bool(changed),'accepted':accepted,'semantic_label_change':labels[g].removeprefix('B-').removeprefix('I-')!=labels[p].removeprefix('B-').removeprefix('I-'),
                        'original_sentence':' '.join(original_words),'modified_sentence':' '.join(changed_words)})
            accepted=[r for r in records if r['accepted']];n=len(records);k=len(accepted);a=k/n
            summary={'generator':arm,'window':window,'fraction':fraction,'rows':n,'accepted':k,'a':a,'a_row_se':math.sqrt(a*(1-a)/n),'fixed_e':e,'a_over_e':a/e if e else None,
                'semantic_accepted':sum(r['semantic_label_change'] for r in accepted),'accepted_from_originally_correct':sum(r['original_correct'] for r in accepted),'changed_candidates':sum(r['context_changed'] for r in records),
                'mean_selected_words':sum(r['selected_words'] for r in records)/n,'mean_changed_words':sum(r['changed_words'] for r in records)/n,'seconds':time.monotonic()-clock}
            summaries.append(summary);write_csv(output/f'{arm}_proposals.csv',records)
            order=torch.randperm(k,generator=torch.Generator().manual_seed(cfg.generators.seed+wi*10+fi))[:cfg.generators.sample_per_generator].tolist();examples.extend(accepted[i] for i in order)
            write_csv(output/'summary.csv',summaries);write_csv(output/'accepted_examples.csv',examples);print(json.dumps(summary),flush=True)
    reference_result=None
    if cfg.generators.get('include_reference',True):
        reference=read_csv(absolute(cfg.label_change.results)/'generate'/'acceptance_sample.csv');n=len(reference);k=sum(r['accepted']=='True' for r in reference)
        reference_result={'rows':n,'accepted':k,'a':k/n,'fixed_e':e,'a_over_e':k/n/e if e else None}
    save_json(output/'comparison.json',{'generators':summaries,'original_single_neighbour':reference_result,
        'protocol':'Exact original acceptance-calibration focal draw stream. Uniform subsets, nearest-integer half-up counts; counts can round to zero. Nearby is ±5 word positions, up to ten context words. Same random ranks make subsets nested within each window. Independent fill RNG per arm. One existing simultaneous MLM pass, preserved focal/piece counts, no retries. Acceptance requires actual decoded context change and native BIO prediction != original gold at confidence >=.9. Fixed e is the original held-out joint high-confidence error, not conditional error. Examples are uniformly sampled accepted proposals for later manual coherence inspection. No unconditional certification or modified-label correctness is asserted.'})
    return {'generators':len(summaries),'saved_examples':len(examples)}


def main(cfg):
    OmegaConf.resolve(cfg);configure(cfg);cfg.forge.store=str(absolute(cfg.forge.store));tracking=start_run(cfg);output=absolute(cfg.generators.results);output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    result=experiment(cfg,output);save_json(output/'runtime.json',{'seconds':time.monotonic()-started});tracking.finish(result);return 0


if __name__=='__main__':run('lion.label_change',sys.argv[1:],main_module='generators',config_dir=str(ROOT/'conf'),config_name='label_change_generators')
