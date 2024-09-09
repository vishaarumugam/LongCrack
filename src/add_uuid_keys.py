import json
import uuid

input_file_path = './data/list_of_questions.json'  # Generic placeholder
with open(input_file_path, 'r') as i_file:
    data = json.load(i_file)

# Generate dictionary with UUIDs
q_dict = {q: str(uuid.uuid4()) for q in data}

# write to data as jsonl file
with open('./data/regular_questions_dict.jsonl', 'w') as o_file:
    for key, value in q_dict.items():
        json.dump({value:  key}, o_file)
        o_file.write('\n')