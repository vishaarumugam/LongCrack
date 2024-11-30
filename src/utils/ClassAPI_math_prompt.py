import abc
import os
import json
from typing import Union
import uuid
import openai
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import huggingface_hub

class MetaProcessor(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def connect(self):
        try:
            load_dotenv('./resources/.env')
            hugging_token = os.getenv('HUGGING_TOKEN')
            if hugging_token is None:
                raise EnvironmentError("HUGGING_TOKEN not found in environment.")
            huggingface_hub.login(token=hugging_token)
        except Exception as e:
            print("Connected to Hugging Face Hub failed:", e)

    def get_params(self, model: str):
        try:
            #print("Model:",model)
            #print("Calling Meta class")
            #with open('./resources/model_configs.json', 'r') as i_file:
            with open('./resources/model_configs_math_prompt.json', 'r') as i_file:
                #print("i_file:",i_file)
                config = json.load(i_file)
                #print("config:",config)
            return config.get(model, {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading configuration: {e}")
            return {}

    @abc.abstractmethod
    def get_single_completion(self, user_prompt, model):
        pass
    
class Gpt(MetaProcessor):
    def connect(self):
        load_dotenv('./resources/.env')
        self.client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def get_single_completion(self, model:str, user_prompt:str) -> Union[str, None]:
        params = self.get_params(model)
        system_prompt = params['system_prompt']
       # user_prompt0 = params['user_prompt']
       # assistant_prompt = params['assistant_prompt']
        few_shot_examples = params.get('few_shot_examples', [])

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Append few-shot examples
        for example in few_shot_examples:
            messages.append({"role": "user", "content": example['user_prompt']})
            messages.append({"role": "assistant", "content": example['assistant_prompt']})

        messages.append({"role": "user", "content": "provide an response as a mathematics problems for the question or instruction provided.\n" + user_prompt})

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **params['gen_params']
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error fetching completion: {e}")
            return None

def select_generator(model_name: str) -> MetaProcessor:
    if model_name == "gpt-4o-mini":
        return Gpt()
    else:
        raise ValueError("Invalid model name.")