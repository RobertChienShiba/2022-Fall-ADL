# Reproduce for ADL Homework 2


## Environment
```shell
pip install -r requirements.txt
```

## Download raw data and fine-tuned model
```shell
bash download.sh
```
After Download data and model, You may have **ckpt** folder for fine-tuned model checkpoints, **image** folder for model performance and **data** folder for data to train
Check the data folder It must have **train.json** **valid.json** **test.json** and **context.json**

## Model Training !!!
```shell
# train multiple-choice model
# model: mengzi-bert-base
python3.9 train_mtpch.py --train_file './data/train.json' --validation_file './data/valid.json' --model_dir 'Langboat/mengzi-bert-base' --context_path './data/context.json' --pad_to_max_length

# train qa model
# model: chinese-macbert-base
python3.9 train_qa.py --train_file './data/train.json' --validation_file './data/valid.json' --model_dir 'hfl/chinese-macbert-base' --context_path './data/context.json' --doc_stride 32 --pad_to_max_length
```

