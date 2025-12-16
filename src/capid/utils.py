from pydantic import BaseModel, Field
import anthropic
import os
from dotenv import load_dotenv
import faker
from openai import OpenAI

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
api_key_openai = os.getenv("OPENAI_API_KEY")
api_key_anthropic = os.getenv("ANTHROPIC_API_KEY")

def call_openai(prompt, model="gpt-4.1-mini", schema=None):
    client = OpenAI(api_key=api_key_openai)

    if model == "gpt-5":
        result = client.responses.create(
            model="gpt-5",
            input=prompt,
            reasoning={ "effort": "low" },
            text={ "verbosity": "low" },
        )
        return result.output_text
    else:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            max_output_tokens=1000,
            top_p=1
        )
        return response.output[0].content[0].text

def call_anthropic(prompt, prefill=""):
    api_key_anthropic = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key_anthropic)
    message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=20000,
    temperature=1,
    messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    return message.content[0].text
