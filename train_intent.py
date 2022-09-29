from ast import arg
from bisect import bisect_right
import json
from optparse import Option
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import numpy as np
from termcolor import colored
import logging

import torch
from tqdm import trange, tqdm
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler


from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier


TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def train(model:SeqClassifier,
          optimizer:optim,
          criterion:torch.nn.CrossEntropyLoss,
          train_pbar:tqdm,
          device:torch.device,
          writer:SummaryWriter,
          step:int):
    model.train()
    loss_record = []
    acc_record = []
    for batch in train_pbar:
        optimizer.zero_grad()
        data, label = batch['data'].to(device), batch['label'].to(device)
        output= model(data)
        loss = criterion(output, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        loss_record.append(loss.item())

        #  # calculate acc
        pred = output.argmax(dim=1)
        acc = (pred == label).sum().item() / len(data)
        acc_record.append(acc)
        # Display current loss on tqdm progress bar.
        train_pbar.set_postfix({'loss': loss.item(), 'acc': acc})

    mean_train_loss = np.mean(loss_record)
    mean_train_acc = np.mean(acc_record)
    # draw in tensorboard
    writer.add_scalar('Loss/train', mean_train_loss, step)
    writer.add_scalar('acc/train', mean_train_acc, step)

    return step

@torch.no_grad()
def validate(model:SeqClassifier,
          criterion:torch.nn.CrossEntropyLoss,
          valid_pbar:tqdm,
          device:torch.device,
          writer:SummaryWriter,
          step:int
        ):
    model.eval()
    loss_record = []
    acc_record = []

    for batch in valid_pbar:
        data, label = batch['data'].to(device), batch['label'].to(device)
        output= model(data)
        loss = criterion(output, label)
        loss_record.append(loss.item())

        # calculate acc
        pred = output.argmax(dim=1)
        acc = (pred == label).sum().item() / len(data)
        acc_record.append(acc)

        # Display current loss on tqdm progress bar.
        valid_pbar.set_postfix({'loss': loss.item(), 'acc': acc})

    mean_valid_loss = np.mean(loss_record)
    mean_valid_acc = np.mean(acc_record)
    # show in tensorboard
    writer.add_scalar('Loss/valid', mean_valid_loss, step)
    writer.add_scalar('acc/train', mean_valid_acc, step)

    return mean_valid_loss, mean_valid_acc



def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: create DataLoader for train / dev datasets
    dataloaders : Dict[str, DataLoader] = {
        split : DataLoader(dataset=split_datasets,
        batch_size=args.batch_size,
        pin_memory=False,
        shuffle=True,
        collate_fn=split_datasets.collate_fn)
        for split, split_datasets in datasets.items()
    }
    # embedding -> ([6491, 300])
    embeddings = torch.load(args.cache_dir / "embeddings.pt")  
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(input_size=embeddings.shape[-1], embeddings=embeddings, hidden_size=args.hidden_size,
                        num_layers=args.num_layers,dropout_rate=args.dropout, pad_id=vocab.pad_id,
                        bidirectional=True, num_class=len(intent2idx), model_name=args.model_name,
                        init_method=args.init_weights, device=args.device).to(args.device)

    # TODO: init optimizer
    optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.num_epoch, eta_min=args.lr * 1e-1)
    scheduler = GradualWarmupScheduler(optimizer,multiplier=1.5,total_epoch=args.num_epoch*0.2,after_scheduler=scheduler_cosine)
    criterion = torch.nn.CrossEntropyLoss()

    # Writer of tensoboard.
    writer = SummaryWriter() 

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_loss = np.inf
    early_stop_count, step = 0, 0

    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        
        train_pbar = tqdm(dataloaders[TRAIN], position=0, leave=True) 
        valid_pbar = tqdm(dataloaders[DEV], position=0, leave=True)

        # training
        logging.info(f' Epoch [{epoch+1}/{args.num_epoch}]')
        step = train(model, optimizer, criterion, train_pbar, args.device, writer, step)
        
        #validating
        logging.info(colored('(Valid)', 'yellow')+ f' Epoch [{epoch+1}/{args.num_epoch}]')
        valid_loss, valid_acc = validate(model, criterion, valid_pbar, args.device, writer, step)

        if valid_loss < best_loss :          
            best_loss = valid_loss
            model_dict = dict(epochs=epoch+1,
                              loss=valid_loss,
                              acc=valid_acc,
                              batch_size=args.batch_size,
                              init_weights=args.init_weights,
                              model_state_dict=model.state_dict(),
                              optimizer_state_dict=scheduler.state_dict()
                            )
            if round(valid_acc,3) > 0.9 :
                # Save your best model
                torch.save(model_dict, args.ckpt_dir / f'{round(valid_acc,3)}_model.ckpt') 
                logging.info(colored('Saving model with loss {:.3f}...'.format(best_loss), 'red'))
            early_stop_count = 0
        else: 
            early_stop_count += 1
            logging.warning(colored(f'model has not improved performance for {early_stop_count} epochs', 'red'))

        if early_stop_count > args.patience:
            logging.warning(colored('\nModel is not imporving, so we halt the training session.', 'red'))
            return 

        scheduler.step()
    
    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=28)

    # model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )

    # init weights
    parser.add_argument("--init_weights", type=str, help="choose the init weights method from \
    [uniform, normal, xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, orthogonal, identity]",
    default='identity',
    choices=["uniform", "normal", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", \
    "orthogonal", "identity"]
    )

    # which model
    parser.add_argument("--model_name", type=str, help="choose a model from [rnn, gru, lstm] to finish your task",
                        default='lstm', choices=['rnn', 'gru', 'lstm']) 

    # total of epochs to run
    parser.add_argument("--num_epoch", type=int, default=50)

    # early stop patience
    parser.add_argument("--patience", type=int, default=50, help="when meet early stop it will stop training")

    # choose weight decay rate
    parser.add_argument("--weight_decay", type=float, default=5e-3)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
