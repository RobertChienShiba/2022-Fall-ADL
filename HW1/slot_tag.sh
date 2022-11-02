#! /bin/bash
# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
python3.9 test_multitask_crf.py --test_file "${1}" --task_type slot --ckpt_path ./ckpt/slot/best_multitask.pt --pred_file "${2}" --batch_size 64 --hidden_size 512 --init_weights normal --model_name gru