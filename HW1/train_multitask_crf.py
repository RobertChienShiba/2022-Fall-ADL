import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List
import logging
from termcolor import colored
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from torch.nn import functional as F

from dataset import SeqTaggingClsDataset, SeqClsDataset
from model import MultitaskNet
from utils import Vocab, same_seed, argument_tagging, argument_tokens

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
INTENT = "intent"
SLOT = "slot"
TASKS = [INTENT, SLOT]

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def train(model:MultitaskNet,
          optimizer: optim,
          criterion: nn.CrossEntropyLoss,
          train_pbar: Dict[str, tqdm],
          device: torch.device,
          tagging_corpus,
          vocab):

    model.train()

    intent_loss_record = []
    intent_acc_record = []
    slot_loss_record = []
    slot_acc_record = []

    for batch_intent in train_pbar[INTENT]:
        intent_acc = 0
        slot_acc = 0
        optimizer.zero_grad()
        batch_slot = next(iter(train_pbar[SLOT]))
        # data -> (Batch, Seq_len) ; label -> (Batch)
        intent_data, intent_label = batch_intent['data'].to(device), batch_intent['label'].to(device)
        # data, label, mask size -> (Batch, Seq_len)
        slot_data, slot_label, slot_mask = batch_slot['data'].to(device), batch_slot['label'].to(device), batch_slot['mask'].to(device)

        # argument_intent = argument_tokens(intent_data, vocab[INTENT]).to(device)
        # argument_slot = argument_tagging(slot_data, slot_label, dict(tagging_corpus), vocab[SLOT]).to(device)

        # output size -> (Batch, num_class)
        intent_output = model(intent_data, INTENT)
        if model.crf:
            # output size -> (Batch, num_class, Seq_len)
            slot_output = model(slot_data, SLOT)
            slot_output = slot_output.transpose(1, 2)
            slot_loss = -model.crf(slot_output, slot_label, slot_mask, reduction='sum')
            crf_pred = model.crf.decode(slot_output, slot_mask)
            intent_loss = criterion(F.log_softmax(intent_output, dim=1), intent_label)
            # info : mask size -> (Batch, Seq_len)
            slot_acc += sum([crf_pred[idx] == (slot_label[idx].masked_select(text_mask).cpu().detach().numpy().tolist()) \
                for idx, text_mask in enumerate(slot_mask)])
        else:
            # output size -> (Batch, num_class, Seq_len)
            slot_output = model(slot_data, SLOT)
            slot_loss = criterion(slot_output, slot_label)
            intent_loss = criterion(intent_output, intent_label)
            # pred size -> (Batch, Seq_len)
            slot_pred = slot_output.argmax(dim=1)
            # info : mask size -> (Batch, Seq_len)
            slot_acc += sum([slot_pred[idx].masked_select(text_mask).equal(slot_label[idx].masked_select(text_mask)) \
                for idx, text_mask in enumerate(slot_mask)])

        intent_pred = intent_output.argmax(dim=1)

        loss = intent_loss + slot_loss
        loss.backward()
        optimizer.step()
        intent_loss_record.append(intent_loss.item())
        slot_loss_record.append(slot_loss.item())

        # calculate acc
        intent_acc = (intent_pred == intent_label).sum().item() / len(intent_data)
        intent_acc_record.append(intent_acc)
        slot_acc_record.append(slot_acc / len(slot_data))

    mean_intent_loss = np.mean(intent_loss_record)
    mean_intent_acc = np.mean(intent_acc_record)
    mean_slot_loss = np.mean(slot_loss_record)
    mean_slot_acc = np.mean(slot_acc_record)

    return mean_intent_loss, mean_intent_acc, mean_slot_loss, mean_slot_acc

@torch.no_grad()
def validate(model:MultitaskNet,
          criterion:nn.CrossEntropyLoss,
          valid_pbar: Dict[str, tqdm],
          device: torch.device,
          ):

    model.eval()

    intent_loss_record = []
    intent_acc_record = []
    slot_loss_record = []
    slot_acc_record = []


    for batch_intent in valid_pbar[INTENT]:
        intent_acc = 0
        slot_acc = 0
        batch_slot = next(iter(valid_pbar[SLOT]))
        # data, label, mask size -> (Batch, Seq_len)
        intent_data, intent_label = batch_intent['data'].to(device), batch_intent['label'].to(device)
        slot_data, slot_label, slot_mask = batch_slot['data'].to(device), batch_slot['label'].to(device), batch_slot['mask'].to(device)

        # argument_data = argument_tagging(data, label, dict(tagging_corpus), vocab)
        # output size -> (Batch, num_class)
        intent_output = model(intent_data, INTENT)
        if model.crf:
            slot_output = model(slot_data, SLOT)
            slot_output = slot_output.transpose(1, 2)
            slot_loss = -model.crf(slot_output, slot_label, slot_mask, reduction='sum')
            crf_pred = model.crf.decode(slot_output, slot_mask)
            intent_loss = criterion(F.log_softmax(intent_output, dim=1), intent_label)
            # info : mask size -> (Batch, Seq_len)
            slot_acc += sum([crf_pred[idx] == (slot_label[idx].masked_select(text_mask).cpu().detach().numpy().tolist()) \
                for idx, text_mask in enumerate(slot_mask)])
        else:
            # output size -> (Batch, num_class, Seq_len)
            slot_output = model(slot_data, SLOT)
            slot_loss = criterion(slot_output, slot_label)
            intent_loss = criterion(intent_output, intent_label)
            # pred size -> (Batch, Seq_len)
            slot_pred = slot_output.argmax(dim=1)
            # info : mask size -> (Batch, Seq_len)
            slot_acc += sum([slot_pred[idx].masked_select(text_mask).equal(slot_label[idx].masked_select(text_mask)) \
                for idx, text_mask in enumerate(slot_mask)])

        # pred size -> (Batch, Seq_len)
        intent_pred = intent_output.argmax(dim=1)

        intent_loss_record.append(intent_loss.item())
        slot_loss_record.append(slot_loss.item())

        # calculate acc
        intent_acc = (intent_pred == intent_label).sum().item() / len(intent_data)
        intent_acc_record.append(intent_acc)
        slot_acc_record.append(slot_acc / len(slot_data))


    mean_intent_loss = np.mean(intent_loss_record)
    mean_intent_acc = np.mean(intent_acc_record)
    mean_slot_loss = np.mean(slot_loss_record)
    mean_slot_acc = np.mean(slot_acc_record)

    return mean_intent_loss, mean_intent_acc, mean_slot_loss, mean_slot_acc

@torch.no_grad()
def seqeval_eval(model: MultitaskNet,
             valid_pbar: tqdm, 
             device: torch.device):

    model.eval()

    preds: List[List[int]] = []
    ground_truth: List[List[int]] = []

    for batch in valid_pbar:
        # data, label, mask size ->(Batch, Seq_len)
        data, label, mask = batch['data'].to(device), batch['label'].to(device), batch['mask'].to(device)
        if model.crf:
            slot_output = model(data, SLOT)
            slot_output = slot_output.transpose(1, 2)
            crf_pred = model.crf.decode(slot_output, mask)
            preds.extend(crf_pred)

        else:
            # output size -> (Batch, num_class, Seq_len)
            slot_output = model(data, SLOT)
            # pred size -> (Batch, Seq_len)
            pred = slot_output.argmax(dim=1)
            preds.extend([pred[idx].masked_select(text_mask).cpu().detach().numpy().tolist() for idx, text_mask in enumerate(mask)])

        ground_truth.extend([label[idx].masked_select(text_mask).cpu().detach().numpy().tolist() for idx, text_mask in enumerate(mask)])

    return preds, ground_truth
     
def main(args):
    # TODO: implement main function
    same_seed(1234)

    task_datasets = [SeqClsDataset, SeqTaggingClsDataset]
    max_lens = [args.icf_max_len, args.slt_max_len]
    label_paths = ['intent2idx.json', 'tag2idx.json']
    idx2label: Dict[str, Dict[int, str]] = {}
    loaders = {INTENT: dict(), SLOT: dict()}
    tagging_corpus = defaultdict(set)
    task_embeddings = {}
    task_class = {}
    vocab = {}

    for task, label_path, task_dataset, max_len in zip(TASKS, label_paths, task_datasets, max_lens):
        (args.ckpt_dir / task).mkdir(parents=True, exist_ok=True)
        with open(args.cache_dir / task / "vocab.pkl", "rb") as f:
            vocab[task]: Vocab = pickle.load(f)
        label_idx_path = args.cache_dir / task / label_path
        label2idx = json.loads(label_idx_path.read_text())
        idx2label[task] = {v : k for k, v in label2idx.items()}
        for split in SPLITS:
            data_path = args.data_dir / task / f'{split}.json'
            data = json.loads(data_path.read_text()) 
            dataset = task_dataset(data, vocab[task], label2idx, max_len)
            if task == SLOT :
                # Ensure INTENT and SLOT task will end at the same time in the training loop
                args.batch_size = int(np.ceil(len(dataset) / len(loaders[INTENT][split])))
                # For data argumentation and calculate different weights for loss function
                for one in data :
                    for token, tag in zip(one['tokens'], one['tags']) :
                        token = vocab[task].token_to_id(token)
                        tag = label2idx[tag] 
                        tagging_corpus[tag].add(token)
            loaders[task][split] = DataLoader(dataset=dataset, batch_size=args.batch_size, pin_memory=False, \
                shuffle=True, collate_fn=dataset.collate_fn)
        task_embeddings[task] = torch.load(args.cache_dir / task / 'embeddings.pt').to(args.device)
        task_class[task] = len(label2idx)

    model = MultitaskNet(input_size=300, embeddings=task_embeddings, hidden_size=args.hidden_size,
                num_layers=args.num_layers,dropout_rate=args.dropout, pad_id=vocab[INTENT].pad_id,
                bidirectional=True, num_class=task_class, model_name=args.model_name,
                init_method=args.init_weights, device=args.device, crf=args.crf).to(args.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.num_epoch, eta_min=args.lr * 1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, eps=1e-5)
    if args.crf :
        criterion = nn.NLLLoss(reduction='sum')
    else:
        criterion = nn.CrossEntropyLoss(reduction='sum')

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    intent_best_acc, slot_best_acc, early_stop_count = 0, 0, 0
    logging.info(f'Use CRF: {args.crf}')

    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights

        intent_train_pbar = tqdm(loaders[INTENT][TRAIN], position=0, leave=True) 
        slot_train_pbar = tqdm(loaders[SLOT][TRAIN], position=0, leave=True)
        intent_valid_pbar = tqdm(loaders[INTENT][DEV], position=0, leave=True)
        slot_valid_pbar = tqdm(loaders[SLOT][DEV], position=0, leave=True)
    
        # training
        logging.info(f' Epoch [{epoch+1}/{args.num_epoch}]')
        logging.info(f' lr: {optimizer.param_groups[0]["lr"]} ')
        intent_train_loss, intent_train_acc, slot_train_loss, slot_train_acc = train(model, optimizer, criterion,
                {INTENT: intent_train_pbar, SLOT: slot_train_pbar},args.device, tagging_corpus, vocab)
        logging.info(f'intent_train_loss: {intent_train_loss:.4f}, intent_train_acc: {intent_train_acc:.3f}')
        logging.info(f'slot_train_loss: {slot_train_loss:.4f}, slot_train_acc: {slot_train_acc:.3f} ')

        # validating
        logging.info(colored('(Valid)', 'yellow')+ f' Epoch [{epoch+1}/{args.num_epoch}]')
        intent_valid_loss, intent_valid_acc, slot_valid_loss, slot_valid_acc = validate(model, criterion, 
                {INTENT: intent_valid_pbar, SLOT: slot_valid_pbar}, args.device)
        logging.info(f'intent_val_loss: {intent_valid_loss:.4f}, intent_val_acc: {intent_valid_acc:.3f}')
        logging.info(f'slot_val_loss: {slot_valid_loss:.4f}, slot_val_acc: {slot_valid_acc:.3f}')

        def imporove_model(task_val_acc, task_best_acc, save_thresold, task):       
            if task_val_acc > task_best_acc :          
                model_dict = dict(loss=(intent_valid_loss if task == 'intent' else slot_valid_loss),
                                dropout = args.dropout,
                                init_weights=args.init_weights,
                                model_name=args.model_name,
                                crf=args.crf,
                                model_state_dict=model.state_dict(),
                                optimizer_state_dict=scheduler.state_dict()
                                )
                if task_val_acc > save_thresold :
                    # Save your best model
                    torch.save(model_dict, args.ckpt_dir / task /
                        'E{}_{}_{}_B{}_H{}_{:.3f}_model.ckpt'.format(epoch+1, model.__class__.__name__, 
                                        args.crf ,loaders[task][DEV].batch_size, args.hidden_size, task_val_acc)) 
                    torch.save(model_dict, args.ckpt_dir / task / f'best_{args.crf}.pt') 
                    logging.info(colored('Saving {} model with acc {:.3f}...'.format(task, task_val_acc), 'red'))
                return task_val_acc
            else:
                return task_best_acc
        
        intent_best_acc = imporove_model(intent_valid_acc, intent_best_acc, 0.935, INTENT)  
        slot_best_acc = imporove_model(slot_valid_acc, slot_best_acc, 0.82, SLOT)

        if intent_best_acc == intent_valid_acc or slot_best_acc == slot_valid_acc:
            early_stop_count = 0
        else: 
            early_stop_count += 1
            logging.warning(colored(f'model has not improved performance for {early_stop_count} epochs', 'red'))

        if early_stop_count > args.patience:
            logging.warning(colored('\nModel is not imporving, so we halt the training session.', 'red'))
            break

        scheduler.step(intent_valid_loss + slot_valid_loss)
    
    # load weights into model
    checkpoint = torch.load(args.ckpt_dir / SLOT / f'best_multitask.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    predictions, ground_truths = seqeval_eval(model, slot_valid_pbar, args.device)
    joint_acc = 0

    stats = defaultdict(list)

    # cal joint acc
    for prediction, ground_truth in zip(predictions, ground_truths):
        # it will have the same length after mask
        assert len(prediction) == len(ground_truth)
        for idx in range(len(prediction)):
            if ground_truth[idx] == prediction[idx]:
                stats[ground_truth[idx]].append(1)
            else:
                stats[ground_truth[idx]].append(0)
            prediction[idx] = idx2label[SLOT][prediction[idx]]
            ground_truth[idx] = idx2label[SLOT][ground_truth[idx]]

        joint_acc += (prediction == ground_truth)

    print({idx2label[SLOT][k] : np.mean(v) for k, v in dict(stats).items()})
    # print({k : len(v) for k, v in dict(stats).items()})
    print('Joint Accuracy = {:.3f}'.format(joint_acc / len(ground_truths)))
    print('Token Accuracy = {:.3f}'.format(accuracy_score(ground_truths, predictions)))
    print('F1 score = {:.2f}'.format(f1_score(ground_truths, predictions)))
    print(classification_report(ground_truths, predictions, scheme=IOB2, mode='strict'))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )

    # data
    parser.add_argument("--icf_max_len", type=int, default=28)
    parser.add_argument("--slt_max_len", type=int, default=35)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--crf", action='store_true', default=False)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    # init weights
    parser.add_argument("--init_weights", type=str, help="choose the initial weights method from \
    [normal, xavier_normal, kaiming_normal, orthogonal, identity]",
    choices=['normal', 'xavier_normal', 'kaiming_normal', 'orthogonal', 'identity'],
    default='identity')

    # which model
    parser.add_argument("--model_name", type=str, help="choose a model from [rnn, gru, lstm] to finish your task",
                        default='rnn', choices=['rnn', 'gru', 'lstm']) 

    # total of epochs to run
    parser.add_argument("--num_epoch", type=int, default=20)

    # early stop patience
    parser.add_argument("--patience", type=int, default=15, help="when meet early stop it will stop training")

    # choose weight decay rate
    parser.add_argument("--weight_decay", type=float, default=5e-3)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

# python ./train_multitask_crf.py --init_weights normal --num_epoch 40 --dropout 0.4 --model_name gru --num_layer 2 --hidden_size 512 --batch_size 64
# python ./train_multitask_crf.py --init_weights normal --num_epoch 30 --dropout 0.4 --model_name gru --num_layer 2 --hidden_size 512 --batch_size 32 --crf