import json
from tqdm import tqdm

# ========= CONFIG =========
PRED_PATH = "/u4/m2ponoma/context/data/v3/predicted_reddit_binary_llama3b.jsonl"#"/u4/m2ponoma/context/data/v3/predicted_reddit.jsonl"#"./data/v1/predicted_finetuned_llama_8b.jsonl"#"./data/new/predicted_pretrained_1.7B.jsonl"#"./data/new/predicted_finetuned.jsonl"

# ========= HELPERS =========
def char_f1(pred, gold):
    pred_chars = list(pred)
    gold_chars = list(gold)

    inter = 0
    used = [False] * len(gold_chars)

    for c in pred_chars:
        for i, gc in enumerate(gold_chars):
            if not used[i] and c == gc:
                inter += 1
                used[i] = True
                break

    if inter == 0:
        return 0

    precision = inter / len(pred_chars)
    recall = inter / len(gold_chars)
    return 2 * precision * recall / (precision + recall)

import string
def normalize(s):
    return s.lower().translate(str.maketrans('', '', string.punctuation))

def token_f1(pred, gold):
    pred = normalize(pred)
    gold = normalize(gold)

    if len(pred.split()) == len(gold.split()) == 1:
        return char_f1(pred, gold)

    pred_tokens = set(pred.split())
    gold_tokens = set(gold.split())

    inter = len(pred_tokens & gold_tokens)
    if inter == 0: return 0

    precision = inter / len(pred_tokens)
    recall = inter / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def flatten_gold(piis_dict):
    """Convert dict of dicts -> list of tuples (span, type, relevance)."""
    items = []
    for span, vals in piis_dict.items():
        items.append((span.strip().lower(), vals["type"].lower(), vals["relevance"].lower()))
    return items

def flatten_pred(pred_json):
    """Flatten model output JSON into same (span, type, relevance) tuples."""
    items = []
    try:
        for p in pred_json.keys():
            span = str(p).strip().lower()
            typ = str(pred_json[p].get("type", "")).strip().lower()
            rel = str(pred_json[p].get("relevance", "")).strip().lower()
            if span:
                items.append((span, typ, rel))
    except:
        return items
    return items


def safe_div(a, b):
    return a / b if b > 0 else 0

def f1(p, r):
    return safe_div(2*p*r, p+r) if (p+r) else 0

def compute_scores(pred_tuples, gold_tuples):
    """Compute span, type, relevance precision/recall/F1."""
    # Correct matches
    golden_spans = [g[0] for g in gold_tuples]
    predicted_spans = [pred[0] for pred in pred_tuples]
    correct_spans = 0
    for pr in predicted_spans:
        if pr in golden_spans:
            correct_spans += 1

    correct_spans = sum(pred[0] in [g[0] for g in gold_tuples] for pred in pred_tuples)

    correct_types = sum(pred[1] == gold[1] for pred in pred_tuples for gold in gold_tuples if pred[0] == gold[0])
    correct_rels  = sum(pred[2] == gold[2] for pred in pred_tuples for gold in gold_tuples if pred[0] == gold[0])
    n_pred, n_gold = len(pred_tuples), len(gold_tuples)

    span_prec = safe_div(correct_spans, n_pred)
    span_rec  = safe_div(correct_spans, n_gold)
    type_prec = safe_div(correct_types, n_pred)
    type_rec  = safe_div(correct_types, n_gold)
    rel_prec  = safe_div(correct_rels, n_pred)
    rel_rec   = safe_div(correct_rels, n_gold)

    return {
        "span_precision": span_prec,
        "span_recall": span_rec,
        "span_f1": f1(span_prec, span_rec),
        "type_precision": type_prec,
        "type_recall": type_rec,
        "type_f1": f1(type_prec, type_rec),
        "relevance_precision": rel_prec,
        "relevance_recall": rel_rec,
        "relevance_f1": f1(rel_prec, rel_rec),
    }

def compute_scores1(pred_tuples, gold_tuples, span_threshold=0.2):
    """
    pred_tuples = [(span, type, rel), ...]
    gold_tuples = [(span, type, rel), ...]
    """
    bad_ones = []
    matched = []  # (best_f1, pred_tuple, gold_tuple)

    for pred in pred_tuples:
        pred_span, pred_type, pred_rel = pred
        best_score = 0.0
        best_gold = None

        # find best gold match
        for gold in gold_tuples:
            gold_span, gold_type, gold_rel = gold
            f1 = token_f1(pred_span, gold_span)
            if f1 > best_score:
                best_score = f1
                best_gold = gold

        if best_gold is not None and best_score > span_threshold:
            matched.append((best_score, pred, best_gold))

    # ---- SPAN METRICS ----
    tp_span = len(matched)                         # matched predictions
    fp_span = len(pred_tuples) - tp_span           # predicted but no match
    fn_span = len(gold_tuples) - tp_span           # gold not predicted

    precision_span = tp_span / (tp_span + fp_span) if (tp_span + fp_span) > 0 else 0.0
    recall_span = tp_span / (tp_span + fn_span) if (tp_span + fn_span) > 0 else 0.0
    f1_span = (2 * precision_span * recall_span) / (precision_span + recall_span) if (precision_span + recall_span) > 0 else 0.0

    # ---- TYPE METRICS (only over matched spans) ----
    tp_type = sum(1 for f1, pred, gold in matched if pred[1] == gold[1])
    fp_type = len(matched) - tp_type
    fn_type = sum(1 for gold in gold_tuples if gold not in [m[2] for m in matched or []])  # type missing because gold not found

    try:
        accuracy_type = tp_type / len(matched)
    except:
        accuracy_type = 0

    precision_type = tp_type / (tp_type + fp_type) if (tp_type + fp_type) > 0 else 0.0
    recall_type = tp_type / (tp_type + fn_type) if (tp_type + fn_type) > 0 else 0.0
    f1_type = 2 * precision_type * recall_type / (precision_type + recall_type) if (precision_type + recall_type) > 0 else 0.0

    # ---- RELEVANCE METRICS (only over matched spans) ----
    tp_rel = sum(1 for f1, pred, gold in matched if pred[2] == gold[2])
    fp_rel = len(matched) - tp_rel
    fn_rel = fn_type  # same logic: missing gold PII implies missed relevance too

    precision_rel = tp_rel / (tp_rel + fp_rel) if (tp_rel + fp_rel) > 0 else 0.0
    recall_rel = tp_rel / (tp_rel + fn_rel) if (tp_rel + fn_rel) > 0 else 0.0
    f1_rel = 2 * precision_rel * recall_rel / (precision_rel + recall_rel) if (precision_rel + recall_rel) > 0 else 0.0

    matched_low = [(f1, pred, gold) for f1, pred, gold in matched if gold[2].lower() == "low"]
    matched_high = [(f1, pred, gold) for f1, pred, gold in matched if gold[2].lower() == "high"]

    gold_low = [g for g in gold_tuples if g[2].lower() == "low"]
    gold_high = [g for g in gold_tuples if g[2].lower() == "high"]

    # LOW metrics
    tp_low = sum(1 for f1, pred, gold in matched_low if pred[2].lower() == "low")
    fp_low = len(matched_low) - tp_low
    fn_low = len(gold_low) - tp_low

    precision_low = tp_low / (tp_low + fp_low) if (tp_low + fp_low) > 0 else 0
    recall_low = tp_low / (tp_low + fn_low) if (tp_low + fn_low) > 0 else 0
    f1_low = 2 * precision_low * recall_low / (precision_low + recall_low) if (precision_low + recall_low) > 0 else 0

    # HIGH metrics
    tp_high = sum(1 for f1, pred, gold in matched_high if pred[2].lower() == "high")
    fp_high = len(matched_high) - tp_high
    fn_high = len(gold_high) - tp_high

    precision_high = tp_high / (tp_high + fp_high) if (tp_high + fp_high) > 0 else 0
    recall_high = tp_high / (tp_high + fn_high) if (tp_high + fn_high) > 0 else 0
    f1_high = 2 * precision_high * recall_high / (precision_high + recall_high) if (precision_high + recall_high) > 0 else 0
    
    return {
        "span_precision": precision_span,
        "span_recall": recall_span,
        "span_f1": f1_span,

        "type_accuracy": accuracy_type,
        "type_precision": precision_type,
        "type_recall": recall_type,
        "type_f1": f1_type,

        "relevance_precision": precision_rel,
        "relevance_recall": recall_rel,
        "relevance_f1": f1_rel,
        "rel_low_precision": precision_low,
        "rel_low_recall": recall_low,
        "rel_low_f1": f1_low,

        "rel_high_precision": precision_high,
        "rel_high_recall": recall_high,
        "rel_high_f1": f1_high,

        "num_pred": len(pred_tuples),
        "num_gold": len(gold_tuples),
        "matched": len(matched),
        "avg_soft_span_f1": sum([m[0] for m in matched]) / len(matched) if matched else 0.0,
    }

# ========= LOAD PREDICTIONS =========

def main():
    results = []
    bad_ones = []
    with open(PRED_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Evaluating"):
            sample = json.loads(line)
            gold_tuples = flatten_gold(sample.get("groundtruth", {}))
            pred_tuples = flatten_pred(sample.get("parsed", {}))
           
            res = compute_scores1(pred_tuples, gold_tuples)
            results.append(res)
            bad_ones.append((res["rel_high_precision"], sample.get("context"), sample.get("question")))
            #break
    bad_ones.sort(key=lambda x: x[0])
    print(bad_ones[:10])       
    # ========= AGGREGATE =========
    def avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return sum(vals)/len(vals) if vals else 0

    final = {
        #"coverage": avg("coverage"),
        "span_precision": avg("span_precision"),
        "span_recall": avg("span_recall"),
        "span_f1": avg("span_f1"),
        "type_precision": avg("type_precision"),
        "type_recall": avg("type_recall"),
        "type_f1": avg("type_f1"),
        "relevance_precision": avg("relevance_precision"),
        "relevance_recall": avg("relevance_recall"),
        "relevance_f1": avg("relevance_f1"),
        # "rel_low_precision": avg("rel_low_precision"),
        # "rel_low_recall": avg("rel_low_recall"),
        # "rel_low_f1": avg("rel_low_f1"),

        # "rel_high_precision": avg("rel_high_precision"),
        # "rel_high_recall": avg("rel_high_recall"),
        # "rel_high_f1": avg("rel_high_f1"),
    }

    print("\n==== FINAL METRICS (Fine-tuned model) ====")
    for k,v in final.items():
        print(f"{k:25s}: {v:.4f}")

if __name__ == '__main__':
    #print(token_f1("21 years", "21 years old"))
    main()