import json
from utils.ClassAPI import DataProcessor, select_generator
from utils.logging import gen_logger
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using language models.")
    parser.add_argument("--model", type=str, required=True, help="Model name to use for generation.")
    parser.add_argument("--all_questions", type=str, default='false', choices=['true','false'], help="Use all malicious questions (set to true if present).")
    parser.add_argument("--num_questions", type=int, default=1, help="Number of malicious questions to use ('all' or an integer).")
    parser.add_argument("--malicious_file", type=str, required=True, help="Path to file containing malicious questions.")
    parser.add_argument("--step_size", type=int, default=1, help="Step size for generating prompts.")
    parser.add_argument("--output_path", type=str, help="Output directory for generated text.")
    return parser.parse_args()

def main():
    gen_logger(init=True)
    # PARSE ARGS
    args = parse_args()
    print(args)

if __name__ == "__main__":
    main()