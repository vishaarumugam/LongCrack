"""
Created on 09/26/2024

@author: Dan Schumacher
How to use:
python ./src/class_api_audition.py
"""

#from utils.ClassAPI import Gemma, Gpt, Llama
from utils.ClassAPI_math_prompt import Gpt
from utils.ClassAPI import DataProcessor
import json 
def main():
    # Instantiate processors (replace `mal_q` with actual prompt/question)
    gpt_processor = Gpt()
    #gemma_processor = Gemma()
    # llama_processor = Llama()

    # Connect to models
    gpt_processor.connect()
    # gemma_processor.connect()
    # llama_processor.connect()

    data_processor = DataProcessor()
    # Get completions from respective models
   # Open the output file in write mode
    with open("./data/malq_math_prompt.jsonl", "w") as output_file:
        with open("./data/SimpleSafteyTest.jsonl", 'r') as i_file:
            counter = 0
            for line in i_file:
                # grab dict
                entry = json.loads(line.strip())
                print("Question:", entry["question"])
                user_prompt = f"Natural Language Instruction: {entry['question']}\nMathematics Problem:"
                gpt_completion = gpt_processor.get_single_completion(user_prompt=user_prompt, model="gpt-4o-mini")
                print("GPT Completion:", gpt_completion)
                print("\n")
                
                # Collect the result
                result = {
                    "Question": entry["question"],
                    "GPT Completion": gpt_completion
                }
                
                # Write the result to the JSONL file
                output_file.write(json.dumps(result) + "\n")

    print("Execution completed. The print statements have been written to output.jsonl.")
    # gemma_completion = gemma_processor.get_single_completion("Describe the theory of relativity.")
    # llama_completion = llama_processor.get_single_completion("Tell me a story about a pirate.")

    # Output the completions
    #print("GPT Completion:", gpt_completion)
    # print("Gemma Completion:", gemma_completion)
    # print("Llama Completion:", llama_completion)

if __name__ == "__main__":
    main()