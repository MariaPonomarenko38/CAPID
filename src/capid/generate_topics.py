from capid.utils import call_openai
from capid.llm import LLM
import pandas as pd
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from capid.constants import GEN_TOPICS_PIIS

env = Environment(loader=FileSystemLoader("prompts"))
template = env.get_template("topics_generation.j2")

def generate_topics(save_path):
    llm = LLM(call_openai, max_retries=3, backoff=1.6)
    k = 0
    df = pd.DataFrame(columns=["pii_category", "supporting_pii_category", "subtopic", "topic"])
    for i in tqdm(range(len(GEN_TOPICS_PIIS)), desc="PII i", position=0):
        for j in tqdm(range(i + 1, len(GEN_TOPICS_PIIS)), desc=f"PII j for {GEN_TOPICS_PIIS[i]}", position=1, leave=False):
            k += 1
            prompt = template.blocks["generate_topics"](
                template.new_context({
                    "pii1": GEN_TOPICS_PIIS[i],
                    "pii2": GEN_TOPICS_PIIS[j]
                })
            )
            topics = llm.ask(prompt)
            items = [line.split(". ", 1)[1].strip() for line in topics.splitlines() if line.strip()]

            for topic in tqdm(items, desc="Topics", position=2, leave=False):
                prompt = template.blocks["generate_subtopics"](
                    template.new_context({
                        "pii1": GEN_TOPICS_PIIS[i],
                        "pii2": GEN_TOPICS_PIIS[j],
                        "topic": topic
                    })
                )
                subtopics = llm.ask(prompt)
                sub_items = [line.split(". ", 1)[1].strip() for line in subtopics.splitlines() if line.strip()]

                for sub_i in tqdm(sub_items, desc="Subtopics", position=3, leave=False):
                    df.loc[len(df)] = [GEN_TOPICS_PIIS[i], GEN_TOPICS_PIIS[j], sub_i, topic]
    df.to_csv(save_path, index=False)