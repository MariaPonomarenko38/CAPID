import json
from datasets import Dataset

EOS_TOKEN = "<|eot_id|>"
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
Text: {}
Question: {}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = [
        """You are given the text and the question.
        Find all PIIs (Personally Identifiable Information) in the text and output them separated by commas.
        Classify them into one of the following types: health, location, sexual orientation, occupation, age, belief, relationship, name, education, appearance, code, organization, finance, datetime, demographic.
        Classify their relevance to the question: 1 (high), 0 (low).
        When classifying the relevance, pay to attention to how each PII can be helpful for answering the question.
        When it is highly helpful, its a high (1) relevance.

        Output result in the JSON format.
        """
    ] * len(examples["context"])

    inputs_contexts = examples["context"]
    inputs_questions = examples["question"]
    outputs = examples["piis"]
    texts = []
    for instruction, input_c, input_q, output in zip(instructions, inputs_contexts,inputs_questions, outputs):
        text = alpaca_prompt.format(instruction, input_c, input_q, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }


def prepare_data(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    for sample in data:
        if isinstance(sample.get("piis"), dict):
            sample["piis"] = json.dumps(sample["piis"])
        else:
            sample["piis"] = ""
    dataset = Dataset.from_list(data)
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    return dataset