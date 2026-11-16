"""Eight sequence labelling datasets, keeping native splits."""

from datasets import Dataset, DatasetDict, load_dataset


def build_conll2003() -> DatasetDict:
    """CoNLL-2003's original train, validation, and test splits."""
    from huggingface_hub import hf_hub_download
    files = {split: hf_hub_download(
        "eriktks/conll2003", f"conll2003/{split}/0000.parquet",
        repo_type="dataset", revision="refs/convert/parquet")
        for split in ("train", "validation", "test")}
    return (load_dataset("parquet", data_files=files)
            .rename_column("ner_tags", "labels")
            .remove_columns(["id", "pos_tags", "chunk_tags"]))


def build_wikiann() -> DatasetDict:
    return (load_dataset("unimelb-nlp/wikiann", "en").rename_column("ner_tags", "labels")
            .remove_columns(["spans", "langs"]))


FEWNERD_REPO = 'DFKI-SLT/few-nerd'


def _build_fewnerd(column):
    dataset = load_dataset(FEWNERD_REPO, "supervised")
    dataset = dataset.remove_columns([name for name in dataset["train"].column_names
                                      if name not in {"tokens", column}])
    return dataset.rename_column(column, "labels")


def build_fewnerd() -> DatasetDict:
    """Few-NERD coarse labels: O plus eight entity types."""
    return _build_fewnerd('ner_tags')

WNUT_REPO = 'leondz/wnut_17'


def build_wnut() -> DatasetDict:
    from huggingface_hub import hf_hub_download
    files = {split: hf_hub_download(
        WNUT_REPO, f"wnut_17/{split}/0000.parquet", repo_type="dataset",
        revision="refs/convert/parquet") for split in ("train", "validation", "test")}
    dataset = load_dataset("parquet", data_files=files).rename_column('ner_tags', 'labels')
    dataset = dataset.remove_columns([name for name in dataset['train'].column_names if name not in ('tokens', 'labels')])
    return dataset


def build_conll2000() -> DatasetDict:
    """CoNLL-2000 chunk labels from NLTK's original train/test splits."""
    import nltk
    from nltk.corpus import conll2000
    try:
        conll2000.fileids()
    except LookupError:
        nltk.download('conll2000')

    def split(file_id: str) -> Dataset:
        return Dataset.from_list([
            {"tokens": [word for word, _, _ in sentence],
             "labels": [tag for _, _, tag in sentence]}
            for sentence in conll2000.iob_sents(file_id)])
    return DatasetDict({'train': split('train.txt'), 'test': split('test.txt')})

UD_REPO = 'universal-dependencies/universal_dependencies'


def _ud_split(treebank: str) -> DatasetDict:
    from huggingface_hub import hf_hub_download
    files = {split: hf_hub_download(UD_REPO, f"parquet/{treebank}/{split}.parquet", repo_type="dataset")
             for split in ("train", "dev", "test")}
    return load_dataset('parquet', data_files=files)


def _ud_with_labels(column: str) -> DatasetDict:
    raw = _ud_split('en_ewt')
    mapped = raw.map(lambda row: {"tokens": row["tokens"], "labels": row[column]},
                     remove_columns=raw["train"].column_names, desc=f"Extracting UD {column}")
    return DatasetDict({"validation" if name == "dev" else name: split for name, split in mapped.items()})


def build_ud_upos() -> DatasetDict:
    return _ud_with_labels('upos')


def build_ud_deprel() -> DatasetDict:
    return _ud_with_labels('deprel')


def _gum_discourse_labels(example: dict) -> list[str]:
    """Forward-fill an EDU's relation from its first token across that EDU."""
    labels = []
    current = None
    for misc in example['misc']:
        if misc and 'Discourse=' in misc:
            current = misc.split('Discourse=')[1].split('|')[0].split(':')[0]
        labels.append(current)
    return labels


def build_ud_discourse() -> DatasetDict:
    raw = _ud_split('en_gum')
    mapped = raw.map(lambda row: {"tokens": row["tokens"], "labels": _gum_discourse_labels(row)},
                     remove_columns=raw["train"].column_names, desc="Extracting GUM discourse relations")
    return DatasetDict({"validation" if name == "dev" else name: split for name, split in mapped.items()})

BUILDERS = {
    "conll2000": build_conll2000,
    "ud_upos": build_ud_upos,
    "ud_deprel": build_ud_deprel,
    "ud_discourse": build_ud_discourse,
    "conll2003": build_conll2003,
    "wikiann": build_wikiann,
    "wnut": build_wnut,
    "fewnerd": build_fewnerd,
}
