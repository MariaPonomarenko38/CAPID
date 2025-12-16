from capid.utils import call_openai
from capid.llm import LLM
from colorama import Fore, Style
import pandas as pd
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
import uuid
from capid.pii_generator_global import PIIGenerator
import random

def colorize(s: str, color_code: str) -> str:
    return f"{getattr(Fore, color_code, '')}{s}{Style.RESET_ALL}"

class SampleGenerator:
    def __init__(self, llm: LLM, pii_category, supporting_pii_category, topic, subtopic):
        self.llm = llm 
        self.pii_generator = PIIGenerator(context=f"{topic}-{subtopic}")
        self.pii_category = pii_category
        self.supporting_pii_category = supporting_pii_category
        self.topic = topic
        self.subtopic = subtopic
        self.situation = "" 
        self.extra_facts = ""
        self.noise = ""
        self.question = ""
        self.piis = {}
        self.env = Environment(loader=FileSystemLoader("../../prompts"))
        self.template = self.env.get_template("pipeline.j2")

    def generate_situation(self):
        category_name1, field_name1, self.pii_category_value = self.pii_generator.generate_random(self.pii_category)
        category_name2, field_name2, self.support_pii_catregory_val = self.pii_generator.generate_random(self.supporting_pii_category)

        self.piis[self.pii_category_value] = {"type": category_name1, "relevance": "high"}
        self.piis[self.support_pii_catregory_val] = {"type": category_name2, "relevance": "high"}

        situation_prompt = self.template.blocks["situation"](
            self.template.new_context({
                "topic": self.topic,
                "subtopic": self.subtopic,
                "pii_type1": field_name1,
                "pii_type2": field_name2,
                "pii_value1": self.pii_category_value,
                "pii_value2": self.support_pii_catregory_val
            })
        )
        situation_prompt = "".join(situation_prompt)
        self.situation = self.llm.ask(situation_prompt, model="gpt-5-chat-latest")
    
    def generate_question(self):
        
        question_prompt = self.template.blocks["question_1"](
            self.template.new_context({
                "pii_value1": self.pii_category_value,
                "pii_value2": self.support_pii_catregory_val,
                "situation": self.situation
            })
        )
        general_question_prompt = "".join(question_prompt)

        general_question = self.llm.ask(general_question_prompt, model="gpt-5-chat-latest")
        general_question = general_question.replace(self.pii_category_value, "")
        general_question = general_question.replace(self.supporting_pii_category, "")

        personalized_question_prompt = self.template.blocks["question_2"](
            self.template.new_context({
                "intermediate_question": general_question
            })
        )
        personalized_question_prompt = "".join(personalized_question_prompt)

        high_piis = [f'{key}:{self.piis[key]["type"]}' for key in self.piis.keys() if self.piis[key]["relevance"] == "high"]
        personalized_question = self.llm.ask(personalized_question_prompt, model="gpt-5-chat-latest")
        
        no_hints_question_prompt = self.template.blocks["question_3"](
            self.template.new_context({
                "question": personalized_question,
                "high_piis": high_piis
            })
        )
        no_hints_question_prompt = "".join(no_hints_question_prompt)

        self.question = self.llm.ask(no_hints_question_prompt, model="gpt-5-chat-latest")

    def generate_extra_facts(self):
        self.pii_generator.context = self.situation
        num_piis = random.randint(1, 3)
        generated = []   

        for _ in range(num_piis):
            category_name, field_name, value = self.pii_generator.generate_random()
            if value not in list(self.piis.keys()):
                generated.append((field_name, value))
                self.piis[value] = {"type": category_name, "relevance": "low"}
     
        self.extra_piis_vals = [val for (_, val) in generated]
        formatted_piis = "\n".join([f"{ptype} - {val}" for (ptype, val) in generated])

        extra_facts_prompt = self.template.blocks["peripheral_context"](
            self.template.new_context({
                "topic": self.topic,
                "subtopic": self.subtopic, 
                "low_relevant_piis": formatted_piis
            })
        )
        extra_facts_prompt = "".join(extra_facts_prompt)

        return self.llm.ask(extra_facts_prompt)
    
    def context_opimization(self):
        context = [self.extra_facts, self.situation,  self.noise]
        random.shuffle(context)
        context_joined = " ".join(context)
        all_piis = ", ".join(list(self.piis.keys()))
        paraphrased_context_prompt = self.template.blocks["context_optimization"](
            self.template.new_context({
                "context": context_joined, 
                "piis": all_piis
            })
        )
        paraphrased_context_prompt = "".join(paraphrased_context_prompt)
        paraphrased_context = self.llm.ask(paraphrased_context_prompt)
        return paraphrased_context
    
    def pii_fix(self, context):
        new_piis = {}
        for p in self.piis.keys():
            if p not in context:
                fix_pii_prompt = self.template.blocks["pii_fix"](
                    self.template.new_context({
                        "pii": p, 
                        "context": context
                    })
                )
                fix_pii_prompt = "".join(fix_pii_prompt)
                span = self.llm.ask(fix_pii_prompt)
                new_piis[span] = self.piis[p]
            else:
                new_piis[p] = self.piis[p]
        return new_piis

    def generate_sample(self):
        
        self.generate_situation()
        self.generate_question()
        self.extra_facts = self.generate_extra_facts()
        self.noise = self.generate_extra_facts()
        updated_context = self.context_opimization()
        self.piis = self.pii_fix(updated_context)
        for p in self.piis.keys():
            if p not in updated_context:
                return {
                    "error":"error"
                }

        return {
            "id": str(uuid.uuid4()),
            "context": updated_context,
            "question": self.question,
            "piis": self.piis,
            "situation": self.situation,
            "additional_facts": self.extra_facts,
            "noise": self.noise
        }

if __name__ == '__main__':

    llm = LLM(call_openai, max_retries=3, backoff=1.6)
    pii_category = "health"
    support_pii_catregory = "occupation"
    subtopic = "Shift Work"
    topic = "Job Fatigue"
    s = SampleGenerator(llm=llm,
                        pii_category=pii_category,
                        supporting_pii_category=support_pii_catregory,
                        topic=topic,
                        subtopic=subtopic)
    result = s.generate_sample()
    print(result["context"])
    # I recently visited Prague Care Hospital to learn more about their innovative approach to 
    # patient rehabilitation and wellness programs. On weekends, I enjoy exploring nature trails 
    # because it helps me clear my mind, even though I manage a medical condition called Chronic 
    # Fatigue. As an air traffic controller with disability status due to chronic pain, I often 
    # struggle to stay focused during overnight shifts, especially around 3 a.m., when my back 
    # throbs and fatigue clouds my concentration.

    print( result["piis"])
    # {'chronic pain': {'type': 'health', 'relevance': 'high'}, 
    #  'air traffic controller': {'type': 'occupation', 'relevance': 'high'}, 
    #  'Prague Care Hospital': {'type': 'organization', 'relevance': 'low'}, 
    #  'Chronic Fatigue': {'type': 'health', 'relevance': 'low'}}

    print( result["question"])
    # How can I maintain my focus and performance during demanding night shifts while managing my challenges?

