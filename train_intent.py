from ast import arg
from bisect import bisect_right
import json
from optparse import Option
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import numpy as np

import torch
from tqdm import trange
import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier


TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


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
                        num_layers=args.num_layers,dropout_rate=args.dropout_rate, pad_id=vocab.pad_id,
                        bidirectional=args.bidirectional, num_class=len(intent2idx), model_name=args.model_name,
                        init_method=args.init_weights, device=args.device)

    # TODO: init optimizer
    optimizer = optim.AdamW(params=model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.num_epoch, eta_min=args.lr * 1e-1)
    criterion = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter() # Writer of tensoboard.

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_loss = np.inf
    step = 0
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.train()
        loss_record = []

        train_pbar = tqdm(dataloaders[TRAIN], position=0, leave=True) 
        valid_pbar = tqdm(dataloaders[DEV], position=0, leave=True)

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
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
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
    parser.add_argument("--init_weights", type=str, help="choose the init weights method from \
    [uniform, normal, constant, xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, orthogonal]",
    default='normal',
    choices=["uniform", "normal", "constant", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "orthogonal"]
    )

    # which model
    parser.add_argument("--model_name", type=str, help="choose a model from [rnn, gru, lstm] to finish your task",
                        default='rnn', choices=['rnn', 'gru', 'lstm']) 

    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
