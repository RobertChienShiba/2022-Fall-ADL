# Reproduce for ADL Homework 1 

## Environment
```shell
pip install -r requirements.txt
```

## Preprocessing
```shell
# Use TA Sample Code To preprocess intent detection and slot tagging datasets
bash preprocess.sh
```
:::info
After implement shell script,you will have a ==cache== folder and a pre-trained glove embeddings ==txt== file,inside the ==cache== folder,you will see two folders,each folder for your specific task have three files,==embeddings.pt== record every token embeddings,==label2idx.json== is a dictionary you can covert your class into integer,==vocab.pkl== record every train and eval data's token.
:::

## Basic Model Training
```shell
# train origin intent model
# Performance on Kaggle 0.928
python3.9 train_intent.py --init_weights normal --num_epoch 20 --dropout 0.4 --model_name gru --num_layers 2 --hidden_size 512 --max_len 28 --batch_size 128

# train origin slot model
# Performance on Kaggle 0.789
python3.9 train_slot.py --init_weights normal --num_epoch 20 --dropout 0.4 --model_name gru --num_layer 2  --hidden_size 512 --max_len 35 --batch_size 128
```
:::info
After training slot model you will see a classification report for your validation set produced by python ==seqeval== library
![](https://i.imgur.com/Av0QZQg.png)
:::
## Advance Model Training
```shell
# best intent model train with data argumentation and increase epoch to 40 
# Performance on Kaggle improve from 0.928 to 0.943.
python3.9 train_intent.py --init_weights normal --num_epoch 40 --dropout 0.4 --model_name gru --num_layers 2 --hidden_size 512 --max_len 28 --batch_size 128 --data_argumentation

# best slot model train with multitask learning and increase epoch to 40 
# Performance on Kaggle improve from 0.789 to 0.826.
python3.9 train_multitask_crf.py --init_weights normal --num_epoch 40 --dropout 0.4 --model_name gru --num_layer 2 --hidden_size 512 --batch_size 64 --max_len 35
```
:::warning
In ==train_multitask_crf.py== we do not use ==CRF== layer behind the BiLSTM because it does not improve our performance in my test.
:::
:::info
After training slot model you will see a classification report for your validation set produced by python ==seqeval== library
![](https://i.imgur.com/qLSl8qP.png)
:::