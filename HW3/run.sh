#!/bin/bash
train_file=data/train_split.jsonl
train_full_file=data/train.jsonl
valid_file=data/valid_split.jsonl
test_file=data/public.jsonl
model_name=google/mt5-small
ckpt_model=ckpt/mt5-small_full_best

# ML
# python3.8 train.py --train_file $train_file --validation_file $valid_file --model_name_or_path $model_name --num_epochs 10 --pad_to_max_length --num_beams 8
# python3.8 train.py --train_file $train_full_file --model_name_or_path $model_name --num_epochs 10 --pad_to_max_length --num_beams 8
python3.8 predict.py --test_file $1 --model_dir $ckpt_model --pad_to_max_length --num_beams 8 --output_file $2
# python3.8 ADL22-HW3/eval.py -r $1 -s $2

# RL
# python3.8 train.py --train_file $train_file --validation_file $valid_file --model_name_or_path $model_name --num_epochs 5 \
# --pad_to_max_length --max_target_length 32 \
# --use_rl --learning_rate 5e-5 \
# --num_beams 2 --top_p 0.2 --temperature 0.7 