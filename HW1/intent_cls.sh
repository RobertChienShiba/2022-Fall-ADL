#! /bin/bash
# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
python3.9 test_intent.py --test_file "${1}" --ckpt_path ./ckpt/intent/best_argumentation.pt --pred_file "${2}" --init_weights normal --hidden_size 512 --model_name gru