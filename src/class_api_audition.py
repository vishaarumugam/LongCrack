"""
Created on 09/26/2024

@author: Dan Schumacher
How to use:
python ./src/class_api_audition.py
"""

from utils.ClassAPI import Gemma, Gpt, Llama

def main():
    # Instantiate processors (replace `mal_q` with actual prompt/question)
    # gpt_processor = Gpt()
    gemma_processor = Gemma()
    # llama_processor = Llama()

    # Connect to models
    # gpt_processor.connect()
    gemma_processor.connect()
    # llama_processor.connect()

    # Get completions from respective models
    # gpt_completion = gpt_processor.get_single_completion("What is the capital of France?", model="gpt-4")
    gemma_completion = gemma_processor.get_single_completion("Describe the theory of relativity.")
    # llama_completion = llama_processor.get_single_completion("Tell me a story about a pirate.")

    # Output the completions
    # print("GPT Completion:", gpt_completion)
    print("Gemma Completion:", gemma_completion)
    # print("Llama Completion:", llama_completion)

if __name__ == "__main__":
    main()