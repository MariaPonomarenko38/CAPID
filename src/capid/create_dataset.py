# batch_run.py
from __future__ import annotations

import json
import os
import argparse
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Iterable, List, Optional, Tuple
from tqdm import tqdm
from threading import Semaphore
from capid.llm import LLM
from dotenv import load_dotenv
from capid.pipeline import SampleGenerator
from utils import call_openai  

class Throttler:
 
    def __init__(self, max_concurrent: int, min_interval_s: float = 0.0):
        self.sem = Semaphore(max_concurrent)
        self.min_interval_s = min_interval_s

    def __enter__(self):
        self.sem.acquire()
        if self.min_interval_s > 0.0:
            time.sleep(self.min_interval_s)

    def __exit__(self, exc_type, exc, tb):
        self.sem.release()

def run_pipeline_for_context(
    idx: int,
    p1,
    supporting_pii_category,
    subtopic,
    topic,  
    llm: LLM,
    throttler: Optional[Throttler] = None,
) -> Tuple[int, dict]:
 
    sample_gen = SampleGenerator(llm=llm,
                                pii_category=p1,
                                supporting_pii_category=supporting_pii_category,
                                subtopic=subtopic,
                                topic=topic)
    if throttler:
        with throttler:
            result = sample_gen.generate_sample()
    else:
        result = sample_gen.generate_sample()

    return idx, result

def run_batch(
    pii_category: Iterable[str],
    supporting_pii_category: Iterable[str],
    subtopic: Iterable[str],
    topic: Iterable[str],
    max_workers: int = 8,
    max_concurrent_llm: Optional[int] = None,
    min_interval_s: float = 0.0,
    save_path: str = "inetmediate_results.json",
    save_every: int = 10,
) -> List[dict]:
   

    llm = LLM(call_openai, max_retries=3, backoff=1.6)
    n = len(pii_category)
    results: List[Optional[dict]] = [None] * n

    throttler = (
        Throttler(max_concurrent_llm or max_workers, min_interval_s)
        if (max_concurrent_llm or min_interval_s > 0)
        else None
    )

    if os.path.exists(save_path):
        try:
            with open(save_path, "r") as f:
                saved = json.load(f)
                for idx, entry in enumerate(saved):
                    if entry is not None:
                        results[idx] = entry
            print(f"Loaded {sum(r is not None for r in results)} previous results.")
        except Exception as e:
            print(f"Warning: could not load previous results: {e}")

    completed = sum(r is not None for r in results)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(
                run_pipeline_for_context,
                i,
                p1,
                supporting_pii_category[i],
                subtopic[i],
                topic[i],               
                llm,
                throttler,
            ): i
            for i, p1 in enumerate(pii_category)
            if results[i] is None 
        }

        for count, fut in enumerate(tqdm(as_completed(futs), total=len(futs), desc="Running pipeline"), start=1):
            i = futs[fut]
            try:
                idx, json_obj = fut.result()
                results[idx] = json_obj
            except Exception as e:
                results[i] = {
                    "context": None,
                    "question": None,
                    "piis": {},
                    "error": f"{type(e).__name__}: {e}",
                }

            if count % save_every == 0 or count == len(futs):
                try:
                    with open(save_path, "w") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"Warning: failed to save intermediate results: {e}")

    return [r for r in results if r is not None]

def save_jsonl(records: List[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main(args):

    df = pd.read_csv(args.topics_path)
    pii_category  = list(df["pii_category"])
    supporting_pii_category  = list(df["supporting_pii_category"])
    subtopic  = list(df["subtopic"])
    topic  = list(df["topic"])

    MAX_WORKERS = 10
    MAX_CONCURRENT_LLM = 5   
    MIN_INTERVAL_S = 2   

    batch = run_batch(
        pii_category,
        supporting_pii_category,
        subtopic,
        topic,
        max_workers=MAX_WORKERS,
        max_concurrent_llm=MAX_CONCURRENT_LLM,
        min_interval_s=MIN_INTERVAL_S,
    )

    print(json.dumps(batch[0], indent=2, ensure_ascii=False))

    save_jsonl(batch, args.save_path)

if __name__ == "__main__":
    load_dotenv()  
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics_path", type=str, required=True, )
    parser.add_argument("--save_path", type=str, required=True, )

    args = parser.parse_args()
    main(args)

    