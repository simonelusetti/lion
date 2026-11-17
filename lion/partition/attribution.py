"""X2 sparse-occlusion training, word/band removals and partial-effect regressions."""
from collections import defaultdict
import json
import math

import torch

from ..diagnostics.common import absolute, write_csv
from ..encoder import build_encoder
from transformers import AutoTokenizer
from .data import encode, load_banks, load_features, model_data, paths, pooled
from .probes import conditional
from .stats import bootstrap, interval, weighted_scalar
from .train import fit, load_model, predict


def frequency(cfg, name):
    original = json.loads(absolute(cfg.partition.sources).read_text())
    sources = ("fewnerd", "conll2003") if name == "ner_shared" else ("ud_discourse" if name == "gum_upos" else name,)
    words, numbers = [], []
    for source in sources:
        bank = torch.load(original[source], map_location="cpu", weights_only=False)
        stats = bank["stats"]
        words += list(stats["vocab"])
        numbers.append(stats["counts"].sum(-1))
    vocab = {word: i for i, word in enumerate(sorted(set(words)))}
    counts = torch.zeros(len(vocab), dtype=torch.long).index_add_(0, torch.tensor([vocab[w] for w in words]), torch.cat(numbers))
    return vocab, counts


def removals(cfg, bank, name):
    generator = torch.Generator().manual_seed(cfg.diagnostics.sample_seed + 20000)
    candidates = [(row, f, i) for row in bank["rows"]["test"] for i, f in enumerate(row["focals"])]
    chosen = torch.randperm(len(candidates), generator=generator)[:cfg.partition.attribution_focals].tolist()
    vocab, counts = frequency(cfg, name)
    rows, records = [], []
    for focal_number, original_index in enumerate(chosen):
        row, focal, tree_row = candidates[original_index]
        width = len(row["words"])
        linear = (torch.arange(width) - focal).abs()
        tree = row["tree"][tree_row] if row["tree"] is not None else None
        keep = torch.ones(width, dtype=torch.bool)
        for j in range(width):
            if j == focal: continue
            mask = keep.clone(); mask[j] = False
            rows.append({**row, "focals": [focal], "masks": mask[None, None]})
            tag = row["upos"][j]
            word_class = "content" if tag in cfg.partition.content_upos else "function" if tag in cfg.partition.function_upos else "other" if tag is not None else "unavailable"
            records.append({"focal_number": focal_number, "original_index": original_index, "kind": "word", "band": "single",
                            "word": row["words"][j], "linear": int(linear[j]),
                            "tree": float(tree[j]) if tree is not None else float("nan"),
                            "upos": tag, "word_class": word_class,
                            "frequency": int(counts[vocab[row["words"][j]]]) if row["words"][j] in vocab else 0,
                            "same_form": row["words"][j] == row["words"][focal], "removed": 1})
        bands = [("linear", "1", linear == 1), ("linear", "2", linear == 2),
                 ("linear", "3-4", (linear >= 3) & (linear <= 4)),
                 ("linear", "5-8", (linear >= 5) & (linear <= 8)), ("linear", "9+", linear >= 9)]
        if tree is not None:
            bands += [("tree", "1", tree == 1), ("tree", "2", tree == 2), ("tree", "3+", (tree >= 3) & torch.isfinite(tree))]
        for family, label, remove in bands:
            number = int(remove.sum())
            if not number: continue
            eligible = (linear > 0).nonzero().flatten()
            random = eligible[torch.randperm(len(eligible), generator=generator)[:number]]
            control = torch.zeros(width, dtype=torch.bool); control[random] = True
            for kind, drop in ((family, remove), ("random_" + family, control)):
                rows.append({**row, "focals": [focal], "masks": (~drop)[None, None]})
                records.append({"focal_number": focal_number, "original_index": original_index, "kind": kind,
                                "band": label, "removed": number})
    return rows, records, torch.tensor(chosen)


def vector_prediction(model, features, cfg):
    return predict(model, {"features": features[:, None], "y": torch.zeros(len(features), dtype=torch.long),
                           "sentence": torch.arange(len(features))}, cfg)[:, 0].log_softmax(-1)


def regression(records, responses, cfg, name, response):
    columns = ["intercept", "log_linear", "log_frequency", "same_form"]
    has_tree = name != "ner_shared"
    if has_tree: columns += ["log_tree", "function_vs_content", "other_vs_content"]
    matrix = []
    for row in records:
        value = [1., math.log1p(row["linear"]), math.log1p(row["frequency"]), float(row["same_form"])]
        if has_tree:
            value += [math.log1p(row["tree"]), float(row["word_class"] == "function"), float(row["word_class"] == "other")]
        matrix.append(value)
    x = torch.tensor(matrix, dtype=torch.float64)
    valid = torch.isfinite(x).all(-1) & torch.isfinite(responses).all(0)
    ids = valid.nonzero().flatten()
    if len(ids) <= len(columns): return [], []
    x, y = x[ids], responses[:, ids].double()
    for col in [1, 2, *([4] if has_tree else [])]:
        x[:, col] = (x[:, col] - x[:, col].mean()) / x[:, col].std().clamp_min(1e-12)
    data = {"document": [records[i]["document"] for i in ids.tolist()],
            "z": torch.tensor([records[i]["z"] for i in ids.tolist()])}
    inverse, weights = bootstrap(data, cfg)
    xx = torch.zeros(weights.shape[1], x.shape[1], x.shape[1], dtype=torch.float64).index_add_(0, inverse, x[:, :, None] * x[:, None, :])
    xy = torch.stack([torch.zeros(weights.shape[1], x.shape[1], dtype=torch.float64).index_add_(0, inverse, x * s[:, None]) for s in y])
    yy = torch.stack([torch.zeros(weights.shape[1], dtype=torch.float64).index_add_(0, inverse, s.square()) for s in y])
    xtx = torch.einsum("bd,dpq->bpq", weights.double(), xx)
    xty = torch.einsum("bd,sdp->sbp", weights.double(), xy)
    yty = torch.einsum("bd,sd->sb", weights.double(), yy)
    beta = torch.einsum("bpq,sbq->sbp", torch.linalg.pinv(xtx), xty)
    coefficients = [{"dataset": name, "response": response, "predictor": col, "pairs": len(ids),
                     **interval(beta[:, :, i])} for i, col in enumerate(columns)]
    sse = yty - (beta * xty).sum(-1)
    effects, blocks = [], {}
    for group, dropped in (("distance", [1, 4] if has_tree else [1]), ("word_class", [5, 6] if has_tree else [])):
        if not dropped: continue
        keep = [i for i in range(len(columns)) if i not in dropped]
        reduced_xx, reduced_xy = xtx[:, keep][:, :, keep], xty[:, :, keep]
        b = torch.einsum("bpq,sbq->sbp", torch.linalg.pinv(reduced_xx), reduced_xy)
        reduced_sse = yty - (b * reduced_xy).sum(-1)
        partial = (reduced_sse - sse) / reduced_sse.clamp_min(1e-12)
        blocks[group] = partial
        effects.append({"dataset": name, "response": response, "block": group, **interval(partial)})
    if has_tree:
        effects.append({"dataset": name, "response": response, "block": "word_class_minus_distance",
                        **interval(blocks["word_class"] - blocks["distance"])})
    return coefficients, effects


def experiment(cfg, output):
    _, cache = paths(cfg)
    encoder = None
    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder.name, use_fast=True, local_files_only=True)
    summaries, associations, coefficients, partials, band_rows, coverage, word_rows = [], [], [], [], [], [], []
    for name, bank in load_banks(cfg):
        path = cache / bank["feature_source"] / "attribution.pt"
        if path.exists():
            perturbations = torch.load(path, map_location="cpu", weights_only=False)
        else:
            rows, records, chosen = removals(cfg, bank, name)
            if encoder is None: encoder = build_encoder(cfg.encoder).to(cfg.runtime.device)
            perturbations = {"features": encode(rows, encoder, tokenizer, cfg)[:, 0], "records": records, "chosen": chosen}
            torch.save(perturbations, path)
        chosen = perturbations["chosen"]
        records = [dict(row) for row in perturbations["records"]]
        part = bank["data"]["test"]
        for row in records:
            i = row["original_index"]
            row.update(document=part["document"][i], z=int(part["z"][i]))
        original_ids = torch.tensor([r["original_index"] for r in records])
        selected_index = torch.tensor([r["focal_number"] for r in records])
        baseline = load_features(cfg, bank)["test"]["features"][:, bank["conditions"].index("full")][chosen].float()
        sparse = load_features(cfg, bank, "occlusion")
        offset = sum(len(bank["data"][s]["y"]) for s in ("train", "validation"))
        seed_r, seed_n = [], []
        for seed in cfg.diagnostics.seeds:
            model, _ = fit(model_data(bank, sparse), len(bank["labels"]), cfg, seed, output / name / f"seed{seed}" / "Y", mixture=True)
            before = vector_prediction(model, baseline, cfg)
            after = vector_prediction(model, perturbations["features"].float(), cfg)
            target = part["y"][original_ids]
            r = before[selected_index, target] - after[torch.arange(len(target)), target]
            _, models = conditional(pooled(sparse), bank, cfg, seed, output / name / f"seed{seed}" / "Z", mixture=True)
            n = torch.full_like(r, torch.nan)
            for (label, fold), info in models.items():
                mask = (part["y"][original_ids] == label) & (bank["fold"][offset + original_ids] == fold)
                mask &= bank["eligible"][label, part["z"][original_ids]]
                ids = mask.nonzero().flatten()
                if not len(ids): continue
                probe = load_model(info["path"], cfg)
                old = vector_prediction(probe, baseline, cfg)
                changed = vector_prediction(probe, perturbations["features"][ids].float(), cfg)
                mapping = torch.full((len(bank["z_names"]),), -1, dtype=torch.long)
                mapping[info["domains"]] = torch.arange(len(info["domains"]))
                z = mapping[part["z"][original_ids[ids]]]
                n[ids] = old[selected_index[ids], z] - changed[torch.arange(len(ids)), z]
            seed_r.append(r); seed_n.append(n)
        r, n = torch.stack(seed_r), torch.stack(seed_n)
        folder = output / name; folder.mkdir(parents=True, exist_ok=True)
        torch.save({"r": r, "n": n, "records": records, "chosen": chosen}, folder / "attribution.pt")
        single_ids = torch.tensor([i for i, row in enumerate(records) if row["kind"] == "word"])
        single = [records[i] for i in single_ids.tolist()]
        rs, ns = r[:, single_ids], n[:, single_ids]
        coverage.append({"dataset": name, "sampled_test_focals": len(chosen), "word_pairs": len(single),
                         "word_pairs_with_nuisance_scores": int(torch.isfinite(ns).all(0).sum()),
                         "missing_n_reason": "label/domain does not meet common document-CV probe support"})
        for i, record in enumerate(single):
            word_rows.append({"dataset": name, **record,
                              "r_mean": float(rs[:, i].mean()), "r_seed_sd": float(rs[:, i].std()),
                              "n_mean": float(ns[:, i].mean()), "n_seed_sd": float(ns[:, i].std())})
        c, p = regression(single, rs, cfg, name, "r"); coefficients += c; partials += p
        c, p = regression(single, ns, cfg, name, "n"); coefficients += c; partials += p
        valid = torch.isfinite(ns).all(0)
        if valid.any():
            paired = {"document": [single[i]["document"] for i in valid.nonzero().flatten().tolist()],
                      "z": torch.tensor([single[i]["z"] for i in valid.nonzero().flatten().tolist()])}
            inverse, weights = bootstrap(paired, cfg)
            correlations = []
            for a, b in zip(rs[:, valid], ns[:, valid]):
                moment = weighted_scalar(torch.stack([a, b, a*b, a*a, b*b], dim=-1), inverse, weights)
                covariance = moment[:, 2] - moment[:, 0]*moment[:, 1]
                scale = ((moment[:, 3]-moment[:, 0].square())*(moment[:, 4]-moment[:, 1].square())).clamp_min(1e-20).sqrt()
                correlations.append(covariance/scale)
            associations.append({"dataset": name, "metric": "pairwise_pearson", "pairs": int(valid.sum()), **interval(torch.stack(correlations))})
        focal_values, focal_meta, focal_null = [], [], []
        for focal in range(len(chosen)):
            ids = [i for i, row in enumerate(single) if row["focal_number"] == focal]
            if not ids or not torch.isfinite(ns[:, ids]).all(): continue
            count, top = len(ids), math.ceil(.2*len(ids))
            values = []
            for a,b in zip(rs[:,ids],ns[:,ids]):
                aa, bb = a.argsort(descending=True,stable=True)[:top],b.argsort(descending=True,stable=True)[:top]
                intersection = int(torch.isin(aa,bb).sum())
                values.append(intersection/(2*top-intersection))
            focal_values.append(values); focal_meta.append(single[ids[0]])
            intersections = torch.arange(max(0, 2*top-count),top+1,dtype=torch.float64)
            def logchoose(n,k):
                return torch.lgamma(torch.as_tensor(n+1.,dtype=torch.float64))-torch.lgamma(k+1)-torch.lgamma(n-k+1)
            probability=(logchoose(top,intersections)+logchoose(count-top,top-intersections)-logchoose(count,torch.tensor(float(top)))).exp()
            focal_null.append(float((probability*intersections/(2*top-intersections)).sum()))
        if focal_values:
            meta={"document":[r["document"] for r in focal_meta],"z":torch.tensor([r["z"] for r in focal_meta])}
            inv,w=bootstrap(meta,cfg)
            stat=torch.stack([weighted_scalar(s[:,None],inv,w)[:,0] for s in torch.tensor(focal_values).T])
            associations.append({"dataset":name,"metric":"top20_jaccard","focals":len(focal_values),**interval(stat)})
            null=weighted_scalar(torch.tensor(focal_null)[:,None],inv,w)[:,0]
            associations.append({"dataset":name,"metric":"top20_excess_over_random","focals":len(focal_values),**interval(stat-null[None])})
        groups = defaultdict(list)
        for i,row in enumerate(records):
            if row["kind"] != "word": groups[(row["kind"],row["band"])].append(i)
        for (kind,band), ids in groups.items():
            meta={"document":[records[i]["document"] for i in ids],"z":torch.tensor([records[i]["z"] for i in ids])}
            inv,w=bootstrap(meta,cfg)
            result={"dataset":name,"kind":kind,"band":band,"focals":len(ids),"removed_mean":sum(records[i]["removed"] for i in ids)/len(ids)}
            for response,values in (("r",r[:,ids]),("n",n[:,ids])):
                valid=torch.isfinite(values).all(0)
                if valid.any():
                    stat=torch.stack([weighted_scalar(s[valid,None],inv[valid],w)[:,0] for s in values])
                    result.update({response+"_"+key:value for key,value in interval(stat).items()})
            band_rows.append(result)
        for feature in ("linear","tree","word_class","frequency","same_form"):
            keys = sorted({str(row.get(feature)) for row in single})
            if feature=="frequency": keys=["0","1-5","6-20","21+"]
            if feature=="tree" and name=="ner_shared":continue
            for key in keys:
                ids=[i for i,row in enumerate(single) if (str(row.get(feature))==key if feature!="frequency" else
                     (row[feature]==0 if key=="0" else 1<=row[feature]<=5 if key=="1-5" else 6<=row[feature]<=20 if key=="6-20" else row[feature]>=21))]
                if not ids: continue
                meta={"document":[single[i]["document"] for i in ids],"z":torch.tensor([single[i]["z"] for i in ids])}
                inv,w=bootstrap(meta,cfg)
                row={"dataset":name,"predictor":feature,"value":key,"pairs":len(ids)}
                for response,values in (("r",rs[:,ids]),("n",ns[:,ids])):
                    valid=torch.isfinite(values).all(0)
                    if valid.any():
                        stat=torch.stack([weighted_scalar(s[valid,None],inv[valid],w)[:,0] for s in values])
                        row.update({response+"_"+k:v for k,v in interval(stat).items()})
                summaries.append(row)
    write_csv(output/"strata.csv",summaries)
    write_csv(output/"association.csv",associations)
    write_csv(output/"coefficients.csv",coefficients)
    write_csv(output/"partial_r2.csv",partials)
    write_csv(output/"bands.csv",band_rows)
    write_csv(output/"coverage.csv",coverage)
    write_csv(output/"word_scores.csv",word_rows)
    return {"datasets":len({r['dataset'] for r in associations})}
