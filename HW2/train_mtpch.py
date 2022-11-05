#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on multiple choice relying on the accelerate library without using a Trainer.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
from sched import scheduler
from statistics import mode
from typing import Optional, Union

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    BertForMultipleChoice,
    BertTokenizerFast,
    SchedulerType,
    get_scheduler,
)

from HFdataset_format import MultipleChoiceDataset, DataCollatorForMultipleChoice

logger = get_logger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def main(args):
    global accelerator
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    config = AutoConfig.from_pretrained(args.model_dir)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir, use_fast=True)
    model = BertForMultipleChoice.from_pretrained(
        args.model_dir,
        config=config,
    )

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    dataset = MultipleChoiceDataset(args.context_path, args.pad_to_max_length, args.max_length, tokenizer, True,
            train_path=args.train_file, trainHF_path='./data/train_mtpch.json', validation_path=args.validation_file, 
            validationHF_path='./data/valid_mtpch.json')

    # DataLoaders creation:

    train_dataloader = DataLoader(
        dataset.train_dataset.select(range(100)), shuffle=True, batch_size=args.per_device_train_batch_size,
        collate_fn=DataCollatorForMultipleChoice(is_train=True, tokenizer=tokenizer)
    )
    eval_dataloader = DataLoader(
        dataset.eval_dataset.select(range(100)),  batch_size=args.per_device_eval_batch_size,
        collate_fn=DataCollatorForMultipleChoice(is_train=False, tokenizer=tokenizer)
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Metrics
    # metric = evaluate.load("accuracy")

    # Create output directory
    output_dir = args.ckpt_dir / 'MultipleChoice'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset.train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    epoch_pbar = trange(args.num_epochs, desc="Epoch")
    best_val_acc = 0

    for epoch in epoch_pbar:
        # training
        train_loss, train_acc = train(model, optimizer, train_dataloader, lr_scheduler)
        logger.info(f'epoch: {epoch + 1}, lr: {optimizer.param_groups[0]["lr"]}')
        logger.info(f'train_loss: {train_loss:.4f}, train_acc: {train_acc:.3f}')

        # validating
        val_loss, val_acc = validate(model, eval_dataloader)
        logger.info(f'val_loss: {val_loss:.4f}, val_acc: {val_acc:.3f}')

        if val_acc > best_val_acc:
            logger.info(f'Validation accuracy improved from {best_val_acc:.3f}')
            best_val_acc = val_acc
            if best_val_acc > 0.9:
                # save your best model
                accelerator.wait_for_everyone()
                model = accelerator.unwrap_model(model)
                save_name = (args.model_dir).split('/')[-1] + f'_E{epoch+1}_{val_acc:.3f}'
                model.save_pretrained(output_dir / save_name, is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save)
                tokenizer.save_pretrained(output_dir / save_name)
        else:
            logger.info('Validation accuracy did not improve')


@torch.no_grad()
def validate(model, dataloader):
    model.eval()
    val_acc = 0
    val_loss = 0
    valid_pbar = tqdm(dataloader, position=0, leave=True)

    # Evaluation loop - calculate accuracy and save model weights
    for batch in valid_pbar:
        labels = batch['labels']
        logger.info(labels)
        outputs = model(**batch)
        loss = outputs.loss
        val_loss += loss.detach().float()
        
        preds = outputs.logits.argmax(dim=1)
        val_acc += (preds == labels).sum().item()

        if accelerator.sync_gradients:
            valid_pbar.update(1)

    return val_loss / len(dataloader), val_acc / len(dataloader.dataset)

def train(model, optimizer, dataloader, lr_scheduler):
    model.train()
    train_acc = 0
    train_loss = 0
    train_pbar = tqdm(dataloader, position=0, leave=True)

    # Training loop - iterate over train dataloader and update model weights
    for batch in train_pbar:
        with accelerator.accumulate(model):
            labels = batch['labels']
            logger.info(labels)
            outputs = model(**batch)
            loss = outputs.loss
            train_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        preds = outputs.logits.argmax(dim=1)
        train_acc += (preds == labels).sum().item()

        if accelerator.sync_gradients:
            train_pbar.update(1)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)

    return  train_loss / len(dataloader), train_acc / len(dataloader.dataset)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
    parser.add_argument(
        "--train_file", type=str, default="./data/train.json", help="A csv or a json file containing the training data.", required=False
    )
    parser.add_argument(
        "--validation_file", type=str, default="./data/valid.json", help="A csv or a json file containing the validation data.", required=False
    )
    parser.add_argument(
        "--context_path", type=str, default="./data/context.json", help="A csv or a json file containing the context data.", required=False
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. num_epochsSequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        # default=True,
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./model/mengzi-bert-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=5e-3, help="Weight decay to use.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=32,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--ckpt_dir", type=Path, default="./ckpt", help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)