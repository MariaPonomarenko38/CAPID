import json, torch, re
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForLanguageModeling

# ------------------------
# 1) Load base model (NOT a previous fine-tune)
# ------------------------
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B-unsloth-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ------------------------
# 2) Prompt template
#    IMPORTANT: the label starts RIGHT AFTER "### Response:\n"
#    We'll mask everything before that.
# ------------------------
alpaca_prompt = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{}\n\n"
    "### Input:\nText: {}\nQuestion: {}\n\n"
    "### Response:\n"
)

instruction = (
    "You are given the text and the question.\n"
    "Find all PIIs (Personally Identifiable Information) in the text and output them separated by commas.\n"
    "Classify them into one of the following types: nationality, age, occupation, education, location, "
    "public organization, health, sexual orientation, finance, family.\n"
    "Classify their relevance to the question: high, low.\n"
    "Output result in the JSON format."
)

RESPONSE_TAG = "### Response:\n"   # <- used for masking

# ------------------------
# 3) Load data, clean, split BEFORE formatting
# ------------------------
with open("./data/new/train_permuted.jsonl", "r", encoding="utf-8") as f:
    raw = [json.loads(line) for line in f]

# optional field cleanup (remove 'location' key inside each PII dict)
for ex in raw:
    piis = ex.get("piis", {})
    if isinstance(piis, dict):
        for _, meta in list(piis.items()):
            if isinstance(meta, dict) and "location" in meta:
                del meta["location"]

ds = Dataset.from_list(raw).train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = ds["train"], ds["test"]

# ------------------------
# 4) Map into prompt + completion
#    We DO NOT put the answer inside the prompt; it’s appended as completion.
# ------------------------
def to_text(example):
    prompt = alpaca_prompt.format(instruction, example["context"], example["question"])
    # label is the JSON ONLY
    label = json.dumps(example["piis"], ensure_ascii=False)
    # full text = prompt (context) + label (target) + eos
    return {"text": prompt + label + tokenizer.eos_token}

train_ds = train_ds.map(to_text)
val_ds   = val_ds.map(to_text)

# ------------------------
# 5) Collator that masks the prompt and trains only on the completion
#    We create a label mask by marking tokens BEFORE RESPONSE_TAG as -100.
# ------------------------
def make_completion_only_collator(tokenizer, response_tag):
    # Pre-tokenize the response tag so we can find it in the sequence
    tag_ids = tokenizer.encode(response_tag, add_special_tokens=False)

    def collate(batch):
        texts = [b["text"] for b in batch]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        labels = input_ids.clone()

        # For each sequence, find where RESPONSE_TAG starts; mask everything before it.
        for i in range(input_ids.size(0)):
            seq = input_ids[i].tolist()
            # find first occurrence of tag_ids in seq
            start = -1
            for j in range(0, len(seq) - len(tag_ids) + 1):
                if seq[j : j + len(tag_ids)] == tag_ids:
                    start = j + len(tag_ids)  # label starts AFTER the tag
                    break
            if start == -1:
                # if not found, mask all (fail-safe)
                labels[i] = -100
            else:
                labels[i, :start] = -100  # mask prompt
                # leave the completion (after start) as labels

        return {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

    return collate

data_collator = make_completion_only_collator(tokenizer, RESPONSE_TAG)

# ------------------------
# 6) Trainer with validation; you can add early stopping later
# ------------------------
args = SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    warmup_ratio=0.03,
    learning_rate=1e-4,
    weight_decay=0.05,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    output_dir="outputs",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    dataset_text_field="text",      # we built 'text' ourselves
    data_collator=data_collator,    # <- completion-only loss
    max_seq_length=max_seq_length,
    args=args,
)

trainer.train()

SAVE_DIR = "models/context-pii-detection-qwen-fixed"
trainer.model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"✅ LoRA fine-tuned model saved to {SAVE_DIR}")