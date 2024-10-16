"""
Created on 10/15/2024

@author: Dan
python ./src/generate.py \
    --model="gpt" \
    --num_questions=1 \
    --step_size=100 \
    --malicious_file="SimpleSafteyTest.jsonl"
"""

import json
import os
from utils.ClassAPI import DataProcessor, select_generator
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using language models.")
    parser.add_argument("--model", type=str, required=True, help="Model name to use for generation.")
    parser.add_argument("--all_question", type=bool, required=False, default=False, help="Use all malicious questions.")
    parser.add_argument("--num_questions", type=int, default=1, help="Number of malicious questions to use ('all' or an integer).")
    parser.add_argument("--malicious_file", type=str, required=True, help="Path to file containing malicious questions.")
    parser.add_argument("--step_size", type=int, default=1, help="Step size for generating prompts.")
    return parser.parse_args()

def main():
    # PARSE ARGS
    args = parse_args()
    if args.all_question:
        args.num_questions = "all"

    # DATA SETUP
    data_processor = DataProcessor()
    data_processor.load_benign_questions("data/benign_questions.jsonl")
    data_processor.load_malicious_questions(f"data/{args.malicious_file}", num_questions=args.num_questions)
    data_processor.generate_list_of_prompts(args.step_size)

    # GENERATE TEXT
    generator = select_generator(args.model)
    generator.connect()

    print("LENGTH OF PROMPT LIST:", len(data_processor.prompt_list))

    # Iterate over prompts and generate completions
    for i, prompt in enumerate(data_processor.prompt_list):
        # Join the questions from the dictionary (UUIDs as keys)

        # Track the position of the malicious question in the benign list
        mal_q_position = i

        # Generate completion using the model
        output = generator.get_single_completion(model=args.model, user_prompt=prompt, malicious_uuid=data_processor.malicious_uuid)

        # Build the JSON output structure
        json_output = {
            "idx": i + 1,
            # "user_prompt": prompt,
            "mqp": mal_q_position * args.step_size,
            "generated_output": output
        }

        # ensure output dir exists
        os.makedirs(f"data/generations/{args.model}", exist_ok=True)

        # Write the output to a JSONL
        with open(f"data/generations/{args.model}/output.jsonl", "a") as o_file:
            o_file.write(json.dumps(json_output) + "\n")

if __name__ == "__main__":
    main()
