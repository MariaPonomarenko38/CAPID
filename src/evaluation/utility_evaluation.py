import json
from tqdm import tqdm
from collections import Counter
from capid.llm import LLM
from capid.utils import call_openai, call_anthropic
from datetime import datetime
from tqdm import tqdm
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import argparse

MAX_WORKERS = 8     
BATCH_SIZE = 10      

def generate_answer(llm, context, question):
    prompt = f"""Answer the question given into account the context.\nContext: {context}.\nQuestion: {question}.
    Utilize the provided information as much as possible.
    The answer should be consice, around 3 sentences. Output just the answer."""
    output = llm.ask(prompt)
    return output

def generate_answer_ibm(llm, query):
    prompt = f"""Answer the following query.
    Query: {query}
    The answer should be consice, around 3 sentences. Output just the answer."""
    output = llm.ask(prompt)
    return output

def evaluate_pair_v2(llm, context, question, ans_file1, ans_file2):
    '''
    Ans_a - file 1, ans_b - file 2
    '''
    prompt = f"""You are an expert evaluator. 
    You are provided a context, question, and two answers below. 
    Your task is to determine which answer most directly answers the question based on the context provided. 
    If two answers are equal then output equal.
    Make sure to choose the most accurate answer based on the context available.

    Context: {context}
    Question: {question}
    
    Answer A: {ans_file1}
    Answer B: {ans_file2}
      
    First, think step-by-step and discuss your reasoning. After, response with either ``<b>A</b>'' or ``<b>B</b>'' or ``<b>Equal</b>'' corresponding to your choice"""
    
    return llm.ask(prompt, model="gpt-5-chat-latest")

def mask_relevance_piis(context, piis, level="1"):
    masked = context
    if level in ["0", "1"]:
        for pii in piis.keys():
            if piis[pii].get("relevance") == level:
                placeholder = f"[MASKED]"
                masked = masked.replace(pii, placeholder)
    if level == "all":
        for pii in piis.keys():
            placeholder = f"[MASKED]"
            masked = masked.replace(pii, placeholder)
    return masked

def run_experiment(llm, args, evaluator_model=None):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_path = os.path.join(args.results_dir, f"utility_evaluation_{timestamp}.jsonl")

    with open(args.file_1_path, "r") as f:
        data1 = [json.loads(line) for line in f]
    
    with open(args.file_2_path, "r") as f:
        data2 = [json.loads(line) for line in f]

    def make_key(rec):
        ctx = rec.get("context", "").strip()
        q = rec.get("question", "").strip()
        return ctx + " || " + q

    samples_map = {make_key(rec): rec for rec in data2}

    pairs = []
    missing = 0

    for sample in data1:
        key = make_key(sample)
        if key not in samples_map:
            missing += 1
            continue
        sample_ibm = samples_map[key]
        pairs.append((sample, sample_ibm))

    print(f"Matched pairs: {len(pairs)}")
    print(f"Missing pairs (skipped): {missing}")

    results = []

    def process_sample(sample_1, sample_2):
        time.sleep(10)

        context = sample["context"]
        question = sample["question"]
        piis_file1 = sample_1.get("parsed", {})
        piis_file2 = sample_2.get("parsed", {})

        masked1 = mask_relevance_piis(context, piis_file1, level=args.mode1)
        masked2 = mask_relevance_piis(context, piis_file2, level=args.mode2)

        file1_context = masked1
        file2_context = masked2

        ans_file1 = generate_answer(llm, file1_context, question)
        ans_file2 = generate_answer(llm, file2_context, question)

        decision = evaluate_pair_v2(
            evaluator_model or llm,
            context,
            question,
            ans_file1=ans_file1, 
            ans_file2=ans_file2
        )

        alt_d = ""
        if "<b>A</b>" in decision:
            alt_d = "<b>A</b>"
        elif "<b>B</b>" in decision:
            alt_d = "<b>B</b>"
        elif "<b>Equal</b>" in decision:
            alt_d = "<b>Equal</b>"

        return {
            "question": question,
            "context": context,
            "file1_context": file1_context,
            "file2_context": file2_context,
            "ans_file1": ans_file1,
            "ans_file2": ans_file2,
            "decision": alt_d,
            "full_decision": decision
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_sample, s, s1): (s, s1)
            for s, s1 in pairs
        }

        for i, future in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc="Running experiment")
        ):
            try:
                record = future.result()
                results.append(record)

                if len(results) % BATCH_SIZE == 0:
                    with open(results_path, "a", encoding="utf-8") as f:
                        for r in results:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    results.clear()

            except Exception as e:
                s, s1 = futures[future]
                print(f"Error in sample {i} for pair {s} and {s1}: {e}")

    if results:
        with open(results_path, "a") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    all_results = []
    with open(results_path, "r") as f:
        for line in f:
            try:
                all_results.append(json.loads(line))
            except:
                continue

    counts = Counter(r["decision"] for r in all_results)
    total = sum(counts.values())
    revealed_better = counts.get("<b>B</b>", 0) / total
    masked_better = counts.get("<b>A</b>", 0) / total
    equal = counts.get("<b>Equal</b>", 0) / total
    utility_drop = revealed_better - masked_better

    summary = {
        "total": total,
        "revealed_better": revealed_better,
        "masked_better": masked_better,
        "equal": equal,
        "utility_drop": utility_drop,
    }

    with open(results_path, "a") as f:
        f.write(json.dumps({"summary": summary}) + "\n")

    print("\n===== Results =====")
    print(f"Saved results to: {results_path}")
    print(f"Total samples: {total}")
    print(f"Revealed better: {revealed_better*100:.1f}%")
    print(f"Masked better:   {masked_better*100:.1f}%")
    print(f"Equal:           {equal*100:.1f}%")
    print(f"Utility drop (A - B): {utility_drop:.3f}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--file-1-path", type=str, required=True)
    parser.add_argument("--file-2-path", type=str, required=True)
    parser.add_argument("--mode1", type=str, required=True)
    parser.add_argument("--mode2", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)

    args = parser.parse_args()

    llm = LLM(call_openai, max_retries=3, backoff=1.6)
    run_experiment(llm, args, evaluator_model=None)


