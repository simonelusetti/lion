"""Full native training banks, case-sensitive word forms and nested sentence subsets."""
import copy
import math
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from ..certification import data as cert
from ..diagnostics.common import absolute, write_csv, save_json


def source(cfg, name):
    src = OmegaConf.load(absolute(cfg.coverage.datasets[name].source_config))
    src.runtime = OmegaConf.merge(src.runtime, cfg.runtime)
    return src


def old_bank(cfg, name):
    """The bank of the layer run: T1∪T2 roles, native dev and test."""
    return torch.load(absolute(cfg.coverage.datasets[name].source_config).parent / 'prepare' / 'sentences.pt', weights_only=False)


def full_bank(cfg, name):
    path = cfg.coverage.datasets[name].full_bank
    return old_bank(cfg, name) if path is None else torch.load(absolute(path) / 'sentences.pt', weights_only=False)


def task(bank):
    return next(iter(bank['labels']))


def forms(rows, tokens):
    """Case-sensitive word strings from the WordPiece IDs of each word slot: one list per sentence."""
    result = []
    for row in rows:
        ids, words = row['ids'].tolist(), row['word_ids'].tolist()
        pieces = [[] for _ in range(len(row['focals']))]
        for i, w in zip(ids, words):
            if w >= 0:
                pieces[w].append(tokens[i])
        result.append([''.join(p.removeprefix('##') for p in word) for word in pieces])
    return result


def flat(rows, key):
    return torch.cat([row['gold'][key] for row in rows])


def subsets(n, fractions, seed):
    """Nested: each fraction takes the first ceil(f·n) sentences of one fixed permutation."""
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return {f: order[:math.ceil(f * n)].sort().values for f in fractions}


def prepare(cfg, output):
    records = []
    for name, spec in cfg.coverage.datasets.items():
        src = source(cfg, name)
        if spec.full_bank is not None and not (absolute(spec.full_bank) / 'sentences.pt').exists():
            local = copy.deepcopy(src)
            local.certification.train_sentence_cap = None
            absolute(spec.full_bank).mkdir(parents=True, exist_ok=True)
            cert.prepare(local, absolute(spec.full_bank))
        tokenizer = AutoTokenizer.from_pretrained(src.encoder.name, use_fast=True, local_files_only=True)
        tokens = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
        old, full = old_bank(cfg, name), full_bank(cfg, name)
        key = task(full)
        t12 = [r for r in old['sets']['train'] if r['fold'] in ('g1', 'g2')]
        # Native dev/test must be the rows the layer-run heads were evaluated on.
        for split in ('validation', 'id'):
            assert [r['source_id'] for r in old['sets'][split]] == [r['source_id'] for r in full['sets'][split]]
        result = {'labels': full['labels'][key],
                  'full_forms': forms(full['sets']['train'], tokens), 'full_gold': [r['gold'][key] for r in full['sets']['train']],
                  't12_forms': [w for s in forms(t12, tokens) for w in s], 't12_gold': flat(t12, key),
                  'test_forms': [w for s in forms(full['sets']['id'], tokens) for w in s], 'test_gold': flat(full['sets']['id'], key),
                  'subsets': subsets(len(full['sets']['train']), list(cfg.coverage.fractions), cfg.coverage.subset_seed)}
        torch.save(result, output / f'{name}.pt')
        for f, index in result['subsets'].items():
            records.append({'dataset': name, 'fraction': f, 'sentences': len(index),
                            'words': sum(len(result['full_forms'][i]) for i in index.tolist())})
        records.append({'dataset': name, 'fraction': 'T1∪T2 (part A)', 'sentences': len(t12), 'words': len(result['t12_forms'])})
        print(name, records[-6:], flush=True)
    write_csv(output / 'subsets.csv', records)
    return {'datasets': len(cfg.coverage.datasets)}
