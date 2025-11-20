import json
from contextual_privacy_llm import PrivacyAnalyzer, run_single_query
from evaluation import compute_scores1
from tqdm import tqdm

from contextual_privacy_llm import PrivacyAnalyzer, run_single_query

# analyzer = PrivacyAnalyzer(
#     model="llama3.1:8b-instruct-fp16",
#     prompt_template="llama",
#     experiment="dynamic"
# )

# result = run_single_query(
#     query_text="My child has autism and I’m in Paris. What support exists for moms like me?",
#     query_id="001",
#     model="llama3.1:8b-instruct-fp16",
#     prompt_template="llama",
#     experiment="dynamic"
# )

# print(result['reformulated_text'])
# # → "What autism support exists for parents in Paris?"

# print(result)
model = "llama3.2:3b-instruct-fp16"
analyzer = PrivacyAnalyzer(
    model=model,
    prompt_template="llama",
    experiment="dynamic"
)

results = []

def convert_prediction(result):
    pred_tuples = []

    # related_context → high relevance
    for span in result.get("related_context", []):
        pred_tuples.append((span, None, "high"))

    # not_related_context → low relevance
    for span in result.get("not_related_context", []):
        pred_tuples.append((span, None, "low"))

    return pred_tuples

def extract_gold(entry):
    return [(span, meta["type"], meta["relevance"]) for span, meta in entry["piis"].items()]


results = []

with open("./data/test.jsonl", "r") as f:
    for line in tqdm(f, desc="Evaluating dataset"):
        entry = json.loads(line)

        result = run_single_query(
            query_text=entry["context"] + " " + entry["question"],
            query_id=entry["id"],
            model=model,
            prompt_template="llama",
            experiment="dynamic"
        )

        pred = convert_prediction(result)
        gold = extract_gold(entry)

        scores = compute_scores1(pred, gold)
        scores["id"] = entry["id"]
        results.append(scores)

# Aggregate results
def avg(key):
    vals = [r[key] for r in results if r[key] is not None]
    return sum(vals)/len(vals) if vals else 0

final = {
    "span_precision": avg("span_precision"),
    "span_recall": avg("span_recall"),
    "span_f1": avg("span_f1"),
    # "type_precision": avg("type_precision"),
    # "type_recall": avg("type_recall"),
    # "type_f1": avg("type_f1"),
    "relevance_precision": avg("relevance_precision"),
    "relevance_recall": avg("relevance_recall"),
    "relevance_f1": avg("relevance_f1"),
}

print("\n==== FINAL METRICS (Fine-tuned model) ====")
for k,v in final.items():
    print(f"{k:25s}: {v:.4f}")