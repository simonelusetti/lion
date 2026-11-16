"""Word classification, span scores where defined, and shared NER transfer."""
import torch
from sklearn.metrics import classification_report, f1_score
from seqeval.metrics import f1_score as span_f1


def bio(tags):
    previous = "O"
    output = []
    for tag in tags:
        output.append("O" if tag == "O" else ("I-" if tag == previous else "B-") + tag)
        previous = tag
    return output


def shared_label(name):
    bare = name[2:] if name.startswith(("B-", "I-")) else name
    return {"person": "PER", "organization": "ORG", "location": "LOC", "building": "LOC",
            "corporation": "ORG", "group": "ORG", "PER": "PER", "ORG": "ORG", "LOC": "LOC"}.get(bare, "O")


def scores(logits, targets, counts, labels, dataset):
    prediction = logits.argmax(-1)
    report = classification_report(targets.tolist(), prediction.tolist(),
                                   labels=list(range(len(labels))), target_names=labels,
                                   output_dict=True, zero_division=0)
    result = {"accuracy": float((prediction == targets).float().mean()),
              "macro_f1": report["macro avg"]["f1-score"],
              "cross_entropy": float(torch.nn.functional.cross_entropy(logits.float(), targets)),
              "per_class": report}
    if "O" in labels:
        outside = labels.index("O")
        result["entity_f1"] = float(f1_score((targets != outside).tolist(),
                                             (prediction != outside).tolist(), zero_division=0))
    is_bio = any(label.startswith(("B-", "I-")) for label in labels)
    if is_bio or dataset == "fewnerd":
        gold = [[labels[index] for index in row] for row in targets.split(counts)]
        pred = [[labels[index] for index in row] for row in prediction.split(counts)]
        if not is_bio:
            gold, pred = list(map(bio, gold)), list(map(bio, pred))
        result["span_f1"] = float(span_f1(gold, pred, zero_division=0))
    return result


def transfer_scores(logits, targets, counts, source_labels, target_labels):
    labels = ["O", "PER", "ORG", "LOC"]
    source = torch.tensor([labels.index(shared_label(tag)) for tag in source_labels])
    target = torch.tensor([labels.index(shared_label(tag)) for tag in target_labels])
    pred, gold = source[logits.argmax(-1)], target[targets]
    def spans(sequence, names):
        tags = [names[int(index)] for index in sequence]
        if any(tag.startswith(("B-", "I-")) for tag in names):
            return [tag[:2] + shared_label(tag) if shared_label(tag) != "O" else "O" for tag in tags]
        return bio([shared_label(tag) for tag in tags])
    gold_spans = [spans(row, target_labels) for row in targets.split(counts)]
    pred_spans = [spans(row, source_labels) for row in logits.argmax(-1).split(counts)]
    return {"accuracy": float((pred == gold).float().mean()),
            "macro_f1": float(f1_score(gold, pred, labels=range(4), average="macro", zero_division=0)),
            "entity_macro_f1": float(f1_score(gold, pred, labels=[1, 2, 3], average="macro", zero_division=0)),
            "span_f1": float(span_f1(gold_spans, pred_spans, zero_division=0))}
