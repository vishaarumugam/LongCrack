#!/bin/bash
# -----------------------------------------------------------
# To Run:   ./bin/generation.sh 
#           nohup ./bin/generation.sh &
# -----------------------------------------------------------

# CHANGE THESE
model="llama"
num_questions=1
step_size=250
all_questions="false"
malicious_file="SimpleSafteyTest.jsonl"

# THESE STAY THE SAME
output_path="./data/generations/${malicious_file%%.jsonl*}/${model}/nq${num_question}_ss${step_size}.jsonl"

python ./src/generate.py \
    --model=${model} \
    --num_question=${num_questions} \
    --step_size=${step_size} \
    --all_questions=${all_questions} \
    --malicious_file=${malicious_file} \
    --output_path=${output_path}