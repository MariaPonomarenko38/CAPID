import json, torch, re
from tqdm import tqdm
from torch.utils.data import DataLoader
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from instructions import INSTRUCTIONS_PRETRAINED, INSTRUCTION, ALPACA_PROMPT
import argparse

BATCH_SIZE = 4         
MAX_NEW_TOKENS = 2048

def collate_fn(batch):
    prompts, metas = zip(*batch)
    return list(prompts), list(metas)

def try_parse_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None

def extract_json_block(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1].strip()
    return ""

def main(args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token 
    FastLanguageModel.for_inference(model)

    with open(args.input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    if args.finetuning:
        prompts = [
            ALPACA_PROMPT.format(INSTRUCTION, s["context"], s["question"], "")
            for s in data
        ]
    else:
        prompts = [
            ALPACA_PROMPT.format(INSTRUCTIONS_PRETRAINED, s["context"], s["question"], "")
            for s in data
        ]

    loader = DataLoader(
        list(zip(prompts, data)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,   
    )

    with open(args.save_path, "w", encoding="utf-8") as fout:
        for batch_prompts, batch_data in tqdm(loader, desc="Generating"):
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
            
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                )

            for out, sample, prompt in zip(outputs, batch_data, batch_prompts):
                decoded = tokenizer.decode(out, skip_special_tokens=True)
                decoded = decoded[len(prompt):]
            
                gen_only = extract_json_block(decoded)
                parsed = try_parse_json(gen_only)
        

                record = {
                    "context": sample["context"],
                    "question": sample["question"],
                    "groundtruth": sample.get("piis", {}),
                    "decoded": decoded,
                    "generated_text": gen_only,
                    "parsed": parsed,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"\n✅ Saved all predictions to {args.save_path}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, )
    parser.add_argument("--input_path", type=str, required=True, )
    parser.add_argument("--save_path",type=str,)

    args = parser.parse_args()
    main(args)