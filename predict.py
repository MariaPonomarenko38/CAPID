from unsloth import FastLanguageModel
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
import re

MODEL_NAME = "/u4/m2ponoma/context/models/v1/context-pii-detection-Llama-3.2-3B-v1"
#MODEL_NAME = "ponoma16/context-pii-detection-Llama-3.1-8B"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME, #"unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token 
FastLanguageModel.for_inference(model)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
Text: {}
Question: {}

### Response:
{}"""

instruction = """You are given the text and the question.
Find all PIIs (Personally Identifiable Information) in the text and output them separated by commas.
Classify them into one of the following types: health, location, sexual orientation, occupation, age, belief, relationship, name, education, appearance, code, organization, finance, datetime, demographic.
Classify their relevance to the question: high, low.
Output result in the JSON format.
"""

instruction_pretrained = """You are given the text and the question.
Find all PIIs (Personally Identifiable Information) in the text and output them separated by commas.
Classify them into one of the following types: health, location, sexual orientation, occupation, age, belief, relationship, name, education, appearance, code, organization, finance, datetime, demographic.
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

# ========= JSON PARSER =========
def try_parse_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None

def extract_json_block(text: str) -> str:
    """
    Extracts the first full JSON object {...} from the text.
    Works even if there is extra text before/after.
    Returns the JSON substring, or an empty string if not found.
    """
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

    # If we never closed all braces
    return ""

def extract_valid_pii_objects(text):
    """
    Extracts all valid {"value": ..., "type": ..., "relevance": ...} objects
    from potentially malformed model outputs.
    Returns a list of dicts.
    """
    # Match independent JSON-like dicts
    pattern = r'\{[^{}]*?"(?:value|span)"\s*:\s*".*?"[^{}]*?"type"\s*:\s*".*?"[^{}]*?"relevance"\s*:\s*".*?"[^{}]*?\}'
    matches = re.findall(pattern, text, re.DOTALL)

    objs = []
    for m in matches:
        try:
            # Clean trailing commas or stray tokens
            m_clean = re.sub(r',\s*([\]}])', r'\1', m)
            obj = json.loads(m_clean)
            objs.append(obj)
        except json.JSONDecodeError:
            continue
    return objs

context = """i’m pretty sure i have a severe coffee addiction. i start my day off with an iced coffee and always add extra instant coffee on top of my shot.
there was one day where i was in a rush in the morning and skipped my morning coffee - got a terrible migraine."""
question = """What can i do in order to feel better?"""
input_str = alpaca_prompt.format(
    instruction,
    context,
    question,
    "",
)

# Tokenize and move to GPU
inputs = tokenizer([input_str], return_tensors="pt").to("cuda")

# Generate
with torch.no_grad():
    #outputs = model.generate(**inputs, max_new_tokens=2048)
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        #repetition_penalty=1.1,
    )
    output_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    decoded = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

# Decode full output
#decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Remove prompt prefix
gen_only = extract_json_block(decoded)#decoded[len(input_str):].strip()

# Try parsing JSON
parsed = try_parse_json(gen_only)
print(parsed)
        
   