import json
import math
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

from accelerate import Accelerator
import transformers
from transformers import (
    PreTrainedModel,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_scheduler,
)

from dataset import SeqTaggingClsDataset
from utils import same_seed

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def train(model:PreTrainedModel,
          optimizer: optim,
          dataloader: DataLoader,
          lr_scheduler,
          idx2tag     
    ):

    model.train()
    loss_record = []
    acc_record = []
    train_pbar = tqdm(dataloader, position=0, leave=True) 

    for step, batch in enumerate(train_pbar, 1):
        acc = 0
        batch.pop('ids')
        outputs = model(**batch)
        loss = outputs.loss
        if len(dataloader) % args.gradient_accumulation_steps != 0 \
            and len(dataloader) - step < args.gradient_accumulation_steps:
            loss = loss / (len(dataloader) % args.gradient_accumulation_steps)
        else:
            loss = loss / args.gradient_accumulation_steps
        accelerator.backward(loss)
        # Update model parameters
        if step % args.gradient_accumulation_steps == 0 or step == len(dataloader):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        loss_record.append(loss.item())

        # calculate acc
        # preds size -> (Batch, Seq_len)
        preds = outputs.logits.argmax(dim=-1)
        labels = batch['labels']
        # Remove ignored index (special tokens)
        true_predictions = [
            [int(p) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, labels)
        ]
        true_labels = [
            [int(l) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, labels)
        ]
        acc += sum([torch.tensor(prediction).equal(torch.tensor(label)) for prediction, label in zip(true_predictions, true_labels)])
        acc_record.append(acc / len(true_labels))
        # Display current loss on tqdm progress bar.
        train_pbar.set_postfix({'loss': loss.item(), 'acc': acc / len(true_labels)})

    mean_train_loss = np.mean(loss_record)
    mean_train_acc = np.mean(acc_record)

    return mean_train_loss, mean_train_acc

@torch.no_grad()
def validate(model:PreTrainedModel,
          dataloader: DataLoader,
          idx2tag
        ):

    model.eval()
    loss_record = []
    acc_record = []
    valid_pbar = tqdm(dataloader, position=0, leave=True)

    for step, batch in enumerate(valid_pbar):
        acc = 0
        batch.pop('ids')
        outputs = model(**batch)
        loss = outputs.loss
        loss_record.append(loss.item())

        # calculate acc
        # pred size -> (Batch, Seq_len)
        preds = outputs.logits.argmax(dim=-1)
        labels = batch['labels']
        # Remove ignored index (special tokens)
        true_predictions = [
            [int(p) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, labels)
        ]
        true_labels = [
            [int(l) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(preds, labels)
        ]
        acc += sum([torch.tensor(prediction).equal(torch.tensor(label)) for prediction, label in zip(true_predictions, true_labels)])
        acc_record.append(acc / len(true_labels))

        # Display current loss on tqdm progress bar.
        valid_pbar.set_postfix({'loss': loss.item(), 'acc': acc / len(true_labels)})

    mean_valid_loss = np.mean(loss_record)
    mean_valid_acc = np.mean(acc_record)

    return mean_valid_loss, mean_valid_acc
    
def main(args):
    # TODO: implement main function
    same_seed(12345)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    global accelerator
    accelerator = Accelerator()

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())
    idx2tag: Dict[int, str] = {v : k for k, v in tag2idx.items()}
    print(idx2tag)

    config = AutoConfig.from_pretrained(args.model_dir, id2label=idx2tag)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, do_lower_case=True)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir, config=config)

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}

    datasets: Dict[str, SeqTaggingClsDataset] = {}

    for split, split_data in data.items():
        datasets[split] = SeqTaggingClsDataset(split_data, tokenizer, tag2idx, args.max_len, True if split == TRAIN else False)
    
    dataloaders : Dict[str, DataLoader] = {
        split : DataLoader(dataset=split_datasets,
        batch_size=args.batch_size,
        pin_memory=False,
        shuffle=True,
        collate_fn=split_datasets.collate_fn)
        for split, split_datasets in datasets.items()
    }
 
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    update_steps_per_epoch = math.ceil(len(dataloaders[TRAIN]) / args.gradient_accumulation_steps)
    args.max_update_steps = args.num_epoch * update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.sched_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.max_update_steps * args.warmup_ratio),
        num_training_steps=args.max_update_steps,
    )
    model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloaders[TRAIN],dataloaders[DEV], lr_scheduler
    )

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_acc = 0

    total_train_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logging.info("\n******** Running training ********")
    logging.info(f"Num train examples = {len(datasets[TRAIN])}")
    logging.info(f"Num Epochs = {args.num_epoch}")
    logging.info(f"Instantaneous batch size per device = {args.batch_size}")
    logging.info(f"Total train batch size (w/ parallel, distributed & accumulation) = {total_train_batch_size}")
    logging.info(f"Instantaneous steps per epoch = {len(train_dataloader)}")
    logging.info(f"Update steps per epoch = {update_steps_per_epoch}")
    logging.info(f"Total update steps = {args.max_update_steps}")

    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights

        # training
        logging.info(f' Epoch [{epoch+1}/{args.num_epoch}]')
        train_loss, train_acc = train(model, optimizer, train_dataloader, lr_scheduler, idx2tag)
        logging.info(f'train_loss: {train_loss:.4f}, train_acc: {train_acc:.3f}')

        # validating
        logging.info(colored('(Valid)', 'yellow')+ f' Epoch [{epoch+1}/{args.num_epoch}]')
        valid_loss, valid_acc = validate(model, valid_dataloader, idx2tag)
        logging.info(f'val_loss: {valid_loss:.4f}, val_acc: {valid_acc:.3f}')

        if valid_acc > best_acc :          
            best_acc = valid_acc
            if valid_acc > 0.81 :
                # Save your best model
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                save_name = (args.model_dir).split('/')[-1] + f'_{valid_acc:.3f}'
                unwrapped_model.save_pretrained(args.ckpt_dir / save_name, save_function=accelerator.save)
                logging.info("Saving config and model to {}...".format(args.ckpt_dir))
                tokenizer.save_pretrained(args.ckpt_dir / save_name)
        else: 
            logging.warning(f'model has not improved performance')
    
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
    parser.add_argument("--max_len", type=int, default=128)

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=4)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    # which model
    parser.add_argument("--model_dir", type=str, default="bert-base-uncased")

    # total of epochs to run
    parser.add_argument("--num_epoch", type=int, default=5)

    # choose weight decay rate
    parser.add_argument("--weight_decay", type=float, default=5e-3)

    # lr_scheduler
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--sched_type", type=str, default="linear", choices=["linear", "cosine", "constant"])
    parser.add_argument("--warmup_ratio", type=float, default=0.1)


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)

# python ./train_slot.py --init_weights normal --num_epoch 20 --dropout 0.4 --model_name gru --num_layer 2 --hidden_size 512
# python ./train_slot.py --init_weights normal --num_epoch 40 --dropout 0.4 --model_name gru --num_layer 2 --batch_size 64 --hidden_size 512