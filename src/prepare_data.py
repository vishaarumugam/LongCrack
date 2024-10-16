"""
Created on 10/15/2024

@author: Dan
python ./src/prepare_data.py
"""
import pandas as pd
import json
import uuid

def main():
    # SIMPLE_SAFTEY_TEST
    df = pd.read_csv("hf://datasets/Bertievidgen/SimpleSafetyTests/sst_test_cases.csv")
    # rename prompt to question
    df = df.rename(columns={"prompt": "question"})
    df.to_json("./data/SimpleSafteyTest.jsonl", lines=True, orient="records")

    # BENIGN QUESTIONS
    input_file_path = './data/original/list_of_questions.json'  # Generic placeholder
    with open(input_file_path, 'r') as i_file:
        data = json.load(i_file)
    
    # Generate dictionary with UUIDs
    output_list = [
        {"question": q, "uuid": str(uuid.uuid4())} 
        for q in data
    ]

    with open('./data/benign_questions.jsonl', 'w') as o_file:
        for entry in output_list:
            json.dump(entry, o_file)
            o_file.write('\n')

if __name__ == "__main__":
    main()