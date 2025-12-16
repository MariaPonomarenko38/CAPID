INSTRUCTIONS_PRETRAINED = """You are given the text and the question.
Find all PIIs (Personally Identifiable Information) in the text and output them separated by commas.
Classify them into one of the following types: health, location, sexual orientation, occupation, age, belief, relationship, name, education, appearance, code, organization, finance, datetime, demographic.
Classify their relevance to the question: 1 (high), 0 (low).

Example:
Text: "John Smith, a 22-year-old student from Canada, works for the University of Toronto."
Question: "What are the educational institutions mentioned in the text?"
Response:
{
  "John Smith": {"type": "family", "relevance": "0"},
  "22-year-old": {"type": "age", "relevance": "0"},
  "student": {"type": "occupation", "relevance": "0"},
  "Canada": {"type": "nationality", "relevance": "0"},
  "University of Toronto": {"type": "education", "relevance": "1"}
}

No explanations or extra text beyond this JSON structure.
"""

INSTRUCTION = """You are given the text and the question.
                Find all PIIs (Personally Identifiable Information) in the text and output them separated by commas.
                Classify them into one of the following types: health, location, sexual orientation, occupation, age, belief, relationship, name, education, appearance, code, organization, finance, datetime, demographic.
                Classify their relevance to the question: 1 (high), 0 (low).
                When classifying the relevance, pay to attention to how each PII can be helpful for answering the question.
                When it is highly helpful, its a high (1) relevance.

                Output result in the JSON format."""

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
Text: {}
Question: {}

### Response:
{}"""