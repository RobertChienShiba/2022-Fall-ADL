#!/bin/bash
python train_intent.py --model_dir bert-base-uncased
# NOTE Change the model you save when training
# python test_intent.py  --model_dir bert-base-uncased 
python train_slot.py --model_dir bert-base-uncased
# NOTE Change the model you save when training
# python test_slot.py  --model_dir bert-base-uncased