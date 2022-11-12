#!/bin/bash
train_file="./data/train.json"
valid_file="./data/valid.json"
mtpch_dir="ckpt/MultipleChoice/mengzi_mask_0.979"
qa_dir='ckpt/QA/chinese-macbert-base_0.802'
context_file="./data/context.json"
test_file="./data/test.json"
output_file="./predict.csv"
# python train_mtpch.py --train_file $train_file --validation_file $valid_file --model_dir $mtpch_dir --context_path $context_file --pad_to_max_length
python test_mtpch.py --test_path "${2}" --model_dir $mtpch_dir --context_path "${1}" --pad_to_max_length
# python train_qa.py --train_file $train_file --validation_file $valid_file --model_dir $qa_dir --context_path $context_file --pad_to_max_length
python test_qa.py --test_path "${2}" --model_dir $qa_dir --context_path "${1}" --output_path "${3}" --pad_to_max_length
