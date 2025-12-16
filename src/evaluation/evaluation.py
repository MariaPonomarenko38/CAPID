import json
from tqdm import tqdm
import argparse

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
    items = []
    for span, vals in piis_dict.items():
        items.append((span.strip().lower(), vals["type"].lower(), vals["relevance"].lower()))
    return items

def flatten_pred(pred_json):
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

def compute_scores(pred_tuples, gold_tuples, span_threshold=0.2):
 
    matched = []  
    coverage = tp_span = fp_span = fn_span = 0
    tp_type = tp_rel = tp_rel_low = tp_rel_high = 0
    gold_covered_low = gold_covered_high = 0
    gold_covered = set()

    for pred in pred_tuples:
        pred_span, pred_type, pred_rel = pred
        best_score = 0.0
        best_gold = None
        best_gold_span = None

        for gold in gold_tuples:
            gold_span, gold_type, gold_rel = gold
            f1 = token_f1(pred_span, gold_span)
            if f1 > best_score:
                best_score = f1
                best_gold = gold
                best_gold_span = gold_span

        if best_gold is not None and best_score > span_threshold:
            if best_gold not in gold_covered:
                gold_covered.add(best_gold)
                matched.append((best_score, pred, best_gold))
                coverage += token_f1(pred_span, best_gold_span)
                if best_gold[2] == '0':
                    gold_covered_low += 1

                if best_gold[2] == '1':
                    gold_covered_high += 1

                if pred_type == best_gold[1]:
                    tp_type += 1
                if pred_rel == best_gold[2]:
                    tp_rel += 1
                    if best_gold[2]== '0':
                        tp_rel_low += 1
                    if best_gold[2] == '1':
                        tp_rel_high += 1
    try:
        coverage = coverage / len(matched)
    except:
        coverage = 0
    tp_span = len(gold_covered)                         
    fp_span = len(pred_tuples) - tp_span           
    fn_span = len(gold_tuples) - tp_span           

    precision_span = tp_span / (tp_span + fp_span) if (tp_span + fp_span) > 0 else 0.0
    recall_span = tp_span / (tp_span + fn_span) if (tp_span + fn_span) > 0 else 0.0
    f1_span = (2 * precision_span * recall_span) / (precision_span + recall_span) if (precision_span + recall_span) > 0 else 0.0

    accuracy_type = tp_type / tp_span if tp_span > 0 else 0.0
    accuracy_rel = tp_rel / tp_span if tp_span > 0 else 0.0

    accuracy_rel_low = (
        tp_rel_low / gold_covered_low if gold_covered_low > 0 else 0.0
    )
    accuracy_rel_high = (
        tp_rel_high / gold_covered_high if gold_covered_high > 0 else 0.0
    )
      
    return {
        "coverage": coverage,
        "span_precision": precision_span,
        "span_recall": recall_span,
        "span_f1": f1_span,

        "type_accuracy": accuracy_type,
        "rel_accuracy": accuracy_rel,
        "rel_low_accuracy": accuracy_rel_low,
        "rel_high_accuracy": accuracy_rel_high,
        "matched": len(matched),
        "avg_soft_span_f1": sum([m[0] for m in matched]) / len(matched) if matched else 0.0,
    }


def main(args):
    results = []
    with open(args.pred_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Evaluating"):
            sample = json.loads(line)
            gold_tuples = flatten_gold(sample.get("groundtruth", {}))
            pred_tuples = flatten_pred(sample.get("parsed", {}))
           
            res = compute_scores(pred_tuples, gold_tuples)
            results.append(res)
           
    def avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return sum(vals)/len(vals) if vals else 0

    final = {
        "coverage": avg("coverage"),
        "span_precision": avg("span_precision"),
        "span_recall": avg("span_recall"),
        "span_f1": avg("span_f1"),
        "type_accuracy": avg("type_accuracy"),
        "rel_accuracy": avg("rel_accuracy"),
        "rel_low_accuracy": avg("rel_low_accuracy"),
        "rel_high_accuracy": avg("rel_high_accuracy"),
    }

    print("\n==== FINAL METRICS (Fine-tuned model) ====")
    for k,v in final.items():
        print(f"{k:25s}: {v:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True, )

    args = parser.parse_args()
    main(args)