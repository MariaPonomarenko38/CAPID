import json, torch, re
from tqdm import tqdm
from unsloth import FastLanguageModel

# ========= MODEL LOAD =========
model_name = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"#"unsloth/Qwen3-8B-unsloth-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)   # enable optimized inference
model = torch.compile(model)              # optional JIT for extra speed

# ========= PROMPTS =========
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
Text: {}
Question: {}

### Response:
{}"""

instruction_pretrained = """You are given the text and the question.
Find all PIIs (Personally Identifiable Information) in the text and output them separated by commas.
Classify them into one of the following types: nationality, age, occupation, education, location, public organization, health, sexual orientation, finance, family.
Classify their relevance to the question: high, low.

Example:
Text: "John Smith, a 22-year-old student from Canada, works for the University of Toronto."
Question: "What are the educational institutions mentioned in the text?"
Output:
{
  "John Smith": {"type": "family", "relevance": "low"},
  "22-year-old": {"type": "age", "relevance": "low"},
  "student": {"type": "occupation", "relevance": "low"},
  "Canada": {"type": "nationality", "relevance": "low"},
  "University of Toronto": {"type": "education", "relevance": "high"}
}

No explanations or extra text beyond this JSON structure.
"""

# ========= HELPERS =========
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

def try_parse_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None

# ========= LOAD DATA =========
with open("./data/new/test.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# ========= GENERATE =========
output_path = "./data/new/predicted_pretrained_fast_gt.jsonl"
batch_size = 8
max_new_tokens = 256

with open(output_path, "w", encoding="utf-8") as fout:
    for i in tqdm(range(0, len(data), batch_size), desc="Generating", ncols=80):
        batch = data[i:i+batch_size]
        prompts = [
            alpaca_prompt.format(instruction_pretrained, s["context"], s["question"], "")
            for s in batch
        ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                use_cache=True,
            )

        for j, s in enumerate(batch):
            # isolate newly generated tokens only
            new_tokens = outputs[j][inputs["input_ids"].shape[1]:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            decoded = re.sub(r"```json|```", "", decoded).strip()

            json_str = extract_json_block(decoded)
            parsed = try_parse_json(json_str)

            record = {
                "context": s["context"],
                "question": s["question"],
                "groundtruth": s.get("piis", {}),  # ✅ ground truth here
                "generated_text": json_str,
                "parsed": parsed,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"\n✅ All predictions saved to: {output_path}")