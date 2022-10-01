from distutils.log import debug
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import logging
from termcolor import colored
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab, same_seed

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def train(model:SeqTagger,
          optimizer: optim,
          criterion: nn.CrossEntropyLoss,
          train_pbar: tqdm,
          device: torch.device,
          writer: SummaryWriter,
          step: int):

    model.train()

    loss_record = []
    acc_record = []

    for batch in train_pbar:
        acc = 0
        optimizer.zero_grad()
        # data, label, mask size ->(Batch, Seq_len)
        data, label, mask = batch['data'].to(device), batch['label'].to(device), batch['mask'].to(device)
        # output size -> (Batch, num_class, Seq_len)
        output= model(data)
        loss = criterion(output, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # step += 1
        loss_record.append(loss.item())

        # calculate acc
        # pred size -> (Batch, Seq_len)
        pred = output.argmax(dim=1)
        acc += sum([pred[idx].masked_select(text_mask).equal(label[idx].masked_select(text_mask)) \
                for idx, text_mask in enumerate(mask)])
        acc_record.append(acc / len(data))
        # Display current loss on tqdm progress bar.
        train_pbar.set_postfix({'loss': loss.item(), 'acc': acc / len(data)})

    mean_train_loss = np.mean(loss_record)
    mean_train_acc = np.mean(acc_record)
    # draw in tensorboard
    # writer.add_scalar('Loss/train', mean_train_loss, step)
    # writer.add_scalar('acc/train', mean_train_acc, step)

    return mean_train_loss, mean_train_acc

@torch.no_grad()
def validate(model:SeqTagger,
          criterion:nn.CrossEntropyLoss,
          valid_pbar: tqdm,
          device: torch.device,
          writer: SummaryWriter,
          step: int):

    model.eval()

    loss_record = []
    acc_record = []

    for batch in valid_pbar:
        acc = 0
        # data, label, mask size ->(Batch, Seq_len)
        data, label, mask = batch['data'].to(device), batch['label'].to(device), batch['mask'].to(device)
        # output size -> (Batch, num_class, Seq_len)
        output= model(data)
        loss = criterion(output, label)
        loss_record.append(loss.item())

        # calculate acc
        pred = output.argmax(dim=1)
        acc += sum([pred[idx].masked_select(text_mask).equal(label[idx].masked_select(text_mask)) \
                for idx, text_mask in enumerate(mask)])
        acc_record.append(acc / len(data))

        # Display current loss on tqdm progress bar.
        valid_pbar.set_postfix({'loss': loss.item(), 'acc': acc / len(data)})

    mean_valid_loss = np.mean(loss_record)
    mean_valid_acc = np.mean(acc_record)
    # show in tensorboard
    # writer.add_scalar('Loss/valid', mean_valid_loss, step)
    # writer.add_scalar('acc/train', mean_valid_acc, step)

    return mean_valid_loss, mean_valid_acc


def main(args):
    # TODO: implement main function
    same_seed(12345)

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }

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

    model = SeqTagger(input_size=embeddings.shape[-1], embeddings=embeddings, hidden_size=args.hidden_size,
                num_layers=args.num_layers,dropout_rate=args.dropout, pad_id=vocab.pad_id,
                bidirectional=True, num_class=len(tag2idx), model_name=args.model_name,
                init_method=args.init_weights, device=args.device).to(args.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.num_epoch, eta_min=args.lr * 1e-2)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter() 

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_acc, step, early_stop_count = 0, 0, 0
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        
        train_pbar = tqdm(dataloaders[TRAIN], position=0, leave=True) 
        valid_pbar = tqdm(dataloaders[DEV], position=0, leave=True)

        # training
        logging.info(f' Epoch [{epoch+1}/{args.num_epoch}]')
        train_loss, train_acc = train(model, optimizer, criterion, train_pbar, args.device, writer, step)
        logging.info(f'train_loss: {train_loss:.4f}, train_acc: {train_acc:.3f}')

        # validating
        logging.info(colored('(Valid)', 'yellow')+ f' Epoch [{epoch+1}/{args.num_epoch}]')
        valid_loss, valid_acc = validate(model, criterion, valid_pbar, args.device, writer, step)
        logging.info(f'val_loss: {valid_loss:.4f}, val_acc: {valid_acc:.3f}')

        if valid_acc > best_acc :          
            best_acc = valid_acc
            model_dict = dict(epochs=epoch+1,
                              loss=valid_loss,
                              acc=valid_acc,
                              batch_size=args.batch_size,
                              hidden_size = args.hidden_size,
                              dropout = args.dropout,
                              init_weights=args.init_weights,
                              num_layers = args.num_layers,
                              model_name = args.model_name,
                              model_state_dict=model.state_dict(),
                              optimizer_state_dict=scheduler.state_dict()
                            )
            if valid_acc > 0.795 :
                # Save your best model
                torch.save(model_dict, args.ckpt_dir / '{:.3f}_model.ckpt'.format(valid_acc)) 
                torch.save(model_dict, args.ckpt_dir / 'best.pt') 
                logging.info(colored('Saving model with acc {:.3f}...'.format(best_acc), 'red'))
            early_stop_count = 0
        else: 
            early_stop_count += 1
            logging.warning(colored(f'model has not improved performance for {early_stop_count} epochs', 'red'))

        if early_stop_count > args.patience:
            logging.warning(colored('\nModel is not imporving, so we halt the training session.', 'red'))
            return 

        scheduler.step()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=35)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    # init weights
    parser.add_argument("--init_weights", type=str, help="choose the initial weights method from \
    [normal, xavier_normal, kaiming_normal, orthogonal, identity]",
    default='identity')

    # which model
    parser.add_argument("--model_name", type=str, help="choose a model from [rnn, gru, lstm] to finish your task",
                        default='gru', choices=['rnn', 'gru', 'lstm']) 

    # total of epochs to run
    parser.add_argument("--num_epoch", type=int, default=20)

    # early stop patience
    parser.add_argument("--patience", type=int, default=15, help="when meet early stop it will stop training")

    # choose weight decay rate
    parser.add_argument("--weight_decay", type=float, default=5e-2)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)

# python ./train_slot.py --init_weights normal --num_epoch 40 --dropout 0.4 --model_name gru --num_layer 2 
# python ./train_slot.py --init_weights normal --num_epoch 40 --dropout 0.4 --model_name gru --num_layer 2 --batch_size 32 --hidden_size 512