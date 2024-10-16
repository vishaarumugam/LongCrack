#!/bin/bash
python ./src/generate.py \
    --model="gpt-4o-mini" \
    --num_questions=1 \
    --step_size=100 \
    --malicious_file="SimpleSafteyTest.jsonl" \
    > ./gpt_output.jsonl