# Reproduce for ADL Homework 3

## Dataset
[download link](https://drive.google.com/file/d/186ejZVADY16RBfVjzcMcz9bal9L3inXC/view?usp=sharing)
```shell
gdown --id 186ejZVADY16RBfVjzcMcz9bal9L3inXC --output data.zip 
unzip data.zip
```
if raw data  decode in the wrong way，Please correct it
```shell
python3.8 data_decode.py
```

## Data Preprocessing for validation(If Don't need to Just Skip)
```shell
python3.8 data_split.py
```
Check the `data` folder it will have `train_split.jsonl` for training `valid_split.jsonl` for validating

## Download Model Checkpoints & Learning Curve Image
```shell
bash download.sh
```


## Install tw_rouge package for chinese summarization metrics
```
git clone https://github.com/moooooser999/ADL22-HW3.git
cd ADL22-HW3
pip install -e tw_rouge
```

## Supervised Training !!!
Validation(Image from download)
```shell
python3.8 train.py --train_file ./data/train_split.jsonl --validation_file ./data/valid_split.jsonl --model_name_or_path google/mt5-small --num_epochs 10 --pad_to_max_length --num_beams 8
```
No Validation(Model Checkpoint from download)
```shell
python3.8 train.py --train_file ./data/train.jsonl --model_name_or_path google/mt5-small --num_epochs 10 --pad_to_max_length --num_beams 8
```

## Reinforcement Training !!!
Validation
```shell
python3.8 train.py --train_file ./data/train_split.jsonl  
--validation_file ./data/valid_split.jsonl 
--model_name_or_path google/mt5-small 
--num_epochs 5 --pad_to_max_length 
--max_target_length 32  --use_rl --learning_rate 5e-5 
--num_beams 2 --top_p 0.2 --temperature 0.7
``` 

## Prediction !!!
```shell
bash run.sh ./data/public.jsonl results.jsonl
```