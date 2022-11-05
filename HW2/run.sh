#!/bin/bash
train_file="./data/train.json"
valid_file="./data/valid.json"
model_dir="./model/mengzi-bert-base"
context_path="./data/context.json"
test_file="./data/test.json"

python train_mtpch.py --train_file $train_file --validation_file $valid_file --model_dir $model_dir --context_path $context_path --pad_to_max_length
# python test_mtpch.py --test_path $test_file  --model_dir $model_dir --context_path $context_path --pad_to_max_length