import json
import math
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

from accelerate import Accelerator
import transformers
from transformers import (
    PreTrainedModel,
    AutoConfig,
    AutoModelForSequenceClassification,
    BertTokenizer,
    get_scheduler,
)

from dataset import SeqClsDataset
from utils import same_seed


TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def train(model: PreTrainedModel,
          optimizer: optim,
          dataloader: DataLoader,
          lr_scheduler
    ):
    model.train()
    loss_record = []
    acc_record = []

    train_pbar = tqdm(dataloader, position=0, leave=True) 

    for step, batch in enumerate(train_pbar, 1):
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
        preds = outputs.logits.argmax(dim=-1)
        labels = batch['labels']
        acc = (preds == labels).sum().item() / len(batch['input_ids'])
        acc_record.append(acc)
        # Display current loss on tqdm progress bar.
        train_pbar.set_postfix({'loss': loss.item(), 'acc': acc})

        if accelerator.sync_gradients:
            train_pbar.update(1)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)

    mean_train_loss = np.mean(loss_record)
    mean_train_acc = np.mean(acc_record)

    return mean_train_loss, mean_train_acc


@torch.no_grad()
def validate(model:PreTrainedModel,
          dataloader: DataLoader
        ):

    model.eval()
    loss_record = []
    acc_record = []

    valid_pbar = tqdm(dataloader, position=0, leave=True)

    for step, batch in enumerate(valid_pbar):
        batch.pop('ids')
        outputs= model(**batch)
        loss = outputs.loss
        loss_record.append(loss.item())

        # calculate acc
        labels = batch['labels']
        preds = outputs.logits.argmax(dim=-1)
        acc = (preds == labels).sum().item() / len(batch['input_ids'])
        acc_record.append(acc)

        # Display current loss on tqdm progress bar.
        valid_pbar.set_postfix({'loss': loss.item(), 'acc': acc})

    mean_valid_loss = np.mean(loss_record)
    mean_valid_acc = np.mean(acc_record)

    return mean_valid_loss, mean_valid_acc


def main(args):

    same_seed(12345)
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    global accelerator
    accelerator = Accelerator()

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    idx2intent = {v: k for k, v in intent2idx.items()}

    config = AutoConfig.from_pretrained(args.model_dir, id2label=idx2intent)
    tokenizer = BertTokenizer.from_pretrained(args.model_dir, use_fast=True, do_lower_case=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, config=config)

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, tokenizer, intent2idx, args.max_len, True if split == TRAIN else False)
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

    # for name, param in model.named_parameters():
    #     print(name, param)

    # TODO: init optimizer
    optimizer = optim.AdamW(params=model.parameters(), lr=args.lr)

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
    best_acc= 0

    total_train_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logging.info("******** Running training ********")
    logging.info(f"Num train examples = {len(datasets[TRAIN])}")
    logging.info(f"Num Epochs = {args.num_epoch}")
    logging.info(f"Instantaneous batch size per device = {args.batch_size}")
    logging.info(f"Total train batch size (w/ parallel, distributed & accumulation) = {total_train_batch_size}")
    logging.info(f"Instantaneous steps per epoch = {len(dataloaders[TRAIN])}")
    logging.info(f"Update steps per epoch = {update_steps_per_epoch}")
    logging.info(f"Total update steps = {args.max_update_steps}")

    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        
        # training
        logging.info(f' Epoch [{epoch+1}/{args.num_epoch}]')
        train_loss, train_acc = train(model, optimizer, train_dataloader, lr_scheduler)
        logging.info(f'train_loss: {train_loss:.4f}, train_acc: {train_acc:.3f}')
       
        
        #validating
        logging.info(colored('(Valid)', 'yellow')+ f' Epoch [{epoch+1}/{args.num_epoch}]')
        valid_loss, valid_acc = validate(model, valid_dataloader)
        logging.info(f'val_loss: {valid_loss:.4f}, val_acc: {valid_acc:.3f}')

        if valid_acc > best_acc :          
            best_acc = valid_acc       
            if valid_acc > 0.965 :
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
    parser.add_argument("--max_len", type=int, default=32)

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=4)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda:0", default="cuda:0"
    )
    # which model
    parser.add_argument("--model_dir", type=str, default="bert-base-uncased")

    # total of epochs to run
    parser.add_argument("--num_epoch", type=int, default=5)

    # choose weight decay rate
    parser.add_argument("--weight_decay", type=float, default=1e-2)

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

# python ./train_intent.py --init_weights normal --num_epoch 40 --dropout 0.4 --model_name gru --num_layers 2 --hidden_size 512 --data_argumentation
# python ./train_intent.py --init_weights normal --num_epoch 20 --dropout 0.4 --model_name gru --num_layers 2 --hidden_size 512