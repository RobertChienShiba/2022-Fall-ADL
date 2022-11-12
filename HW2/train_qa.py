#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning a 🤗 Transformers model for question answering using 🤗 Accelerate.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from collections import defaultdict
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import transformers
import datasets
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForQuestionAnswering,
    BertTokenizerFast,
    SchedulerType,
    get_scheduler,
)

from utils import post_processing_function, create_and_fill_np_array
from datasets import QADataset, DataCollatorForQA

logger = get_logger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
warnings.filterwarnings("ignore")

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

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(args.model_dir)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_dir,
        config=config
    )
    # Preprocessing the datasets.
    dataset = QADataset(
        args.context_path, args.pad_to_max_length, args.max_length, tokenizer, True, args.doc_stride,
        train_path=args.train_file, trainHF_path='./data/train_qa.json', validation_path=args.validation_file, 
        validationHF_path='./data/valid_qa.json'
    )
    # DataLoaders creation:
    data_collator = DataCollatorForQA()

    train_dataloader = DataLoader(
        dataset.train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    eval_dataloader = DataLoader(
        dataset.eval_dataset['preprocessed'], collate_fn=data_collator, batch_size = args.per_device_eval_batch_size
    )


    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.max_train_steps * args.gradient_accumulation_steps * 0.1,
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

    # Create output directory
    output_dir = args.ckpt_dir / 'QA'
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

    # Only show the progress bar once on each machine.
    epoch_pbar = trange(args.num_epochs, desc="Epoch")
    best_em= 0
    curve_logs = defaultdict(list)

    for epoch in epoch_pbar:
        # training
        train_loss, train_acc = train(model, optimizer, train_dataloader, lr_scheduler)
        logger.info(f'epoch: {epoch + 1}, lr: {optimizer.param_groups[0]["lr"]}')
        logger.info(f'train_loss: {train_loss:.4f}, train_acc: {train_acc:.3f}')

        # validating
        logger.info("***** Running Evaluation *****")
        logger.info(f"  Num examples = {len(dataset.eval_dataset['preprocessed'])}")
        logger.info(f"  Batch size = {args.per_device_eval_batch_size}")
        val_loss, metrics = validate(model, eval_dataloader, dataset)
        logger.info(f'val_loss: {val_loss:.4f}, EM: {metrics["em"]:.3f}')
        if metrics['em'] > best_em:
            logger.info(f'Validation accuracy improved from {best_em:.3f}')
            best_em = metrics['em'] 
            if best_em > 0.8:
                # save your best model
                accelerator.wait_for_everyone()
                model = accelerator.unwrap_model(model)
                save_name = (args.model_dir).split('/')[-1] + f'_{best_em:.3f}'
                model.save_pretrained(output_dir / save_name, is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save)
                tokenizer.save_pretrained(output_dir / save_name)
            else:
                logger.info('Validation accuracy did not improve')

        curve_logs['train_acc'].append(train_acc)
        curve_logs['val_acc'].append(metrics['em'])
        curve_logs['train_loss'].append(train_loss)
        curve_logs['val_loss'].append(val_loss)
        save_curve_plot(epoch + 1, curve_logs)

def save_curve_plot(epoch, log):
    fig = plt.figure(figsize=(9, 4))
    model_name = (args.model_dir).split('/')[-1]
    fig.suptitle(f"{model_name} {args.num_epochs}epoch")

    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('Loss')
    ax.plot([*range(1, epoch + 1)], log['train_loss'], label='Training')
    ax.plot([*range(1, epoch + 1)], log['val_loss'], label='Validation')
    ax.set_xlim(1, epoch)
    ax.set_xlabel('Epoch')
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('EM')
    ax.plot([*range(1, epoch + 1)], log['train_acc'], label='Training')
    ax.plot([*range(1, epoch + 1)], log['val_acc'], label='Validation')
    ax.set_xlim(1, epoch)
    ax.set_xlabel('Epoch')
    ax.legend()

    fig.savefig(f'without_pretrained_QA.png')
    plt.close(fig)

def train(model, optimizer, dataloader, lr_scheduler):
    train_acc = 0
    train_loss = 0
    real_segment = 0
    model.train()
    train_pbar = tqdm(dataloader, position=0, leave=True)
    for i, batch in enumerate(train_pbar):
        with accelerator.accumulate(model):
            if not isinstance(batch, dict):
                batch = batch[1]
            cls_index = batch.pop('cls_index')
            batch.pop('overflow_to_sample_mapping')
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            train_loss += loss.detach().item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        start_positions = outputs.start_logits.argmax(dim=1)
        end_positions = outputs.end_logits.argmax(dim=1)
        start_labels = batch['start_positions']
        end_labels = batch['end_positions']
        real_start = torch.argwhere(start_labels != cls_index).flatten()
        real_end = torch.argwhere(end_labels != cls_index).flatten()
        assert torch.eq(real_start, real_end).all()
        real_idx = real_start & real_end
        acc = (start_positions == start_labels) & (end_positions == end_labels)
        acc = acc[real_idx]
        real_segment += len(acc)
        train_acc += acc.sum().item()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            train_pbar.update(1)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)

    assert real_segment != 0
    return train_loss / len(dataloader), train_acc / real_segment 

@torch.no_grad()
def validate(model, dataloader, dataset):
    all_example_ids = []
    all_offset_mapping = []
    for batch_data in dataloader:
        all_example_ids += batch_data['example_id']
        all_offset_mapping += batch_data['offset_mapping']
    example_to_features = defaultdict(list)
    for idx, feature_id in enumerate(all_example_ids):
        example_to_features[feature_id].append(idx)

    all_start_logits = []
    all_end_logits = []

    val_loss = 0
    valid_pbar = tqdm(dataloader, position=0, leave=True)
    model.eval()

    for batch in valid_pbar:
        if not isinstance(batch, dict):
            batch = batch[1]
        batch.pop('example_id')
        batch.pop('cls_index')
        batch.pop('overflow_to_sample_mapping')
        batch.pop('offset_mapping')
        outputs = model(**batch)
        loss = outputs.loss
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
            start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
            end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

        all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
        all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

        val_loss += loss.item()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            valid_pbar.update(1)
    
    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
    assert max_len == args.max_length

    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, dataset.eval_dataset['preprocessed'], max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, dataset.eval_dataset['preprocessed'], max_len)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    metrics_dict = post_processing_function(dataset.eval_dataset['non_preprocessed'], all_offset_mapping, outputs_numpy, 
            example_to_features, args.n_best_size)

    return val_loss / len(dataloader), metrics_dict, 

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")
    parser.add_argument(
        "--train_file", type=str, default="./data/train.json", help="A csv or a json file containing the training data.",required= False
    )
    parser.add_argument(
        "--validation_file", type=str, default="./data/valid.json", help="A csv or a json file containing the validation data.,",required=False
    )
    parser.add_argument(
        "--context_path", type=str, default="./data/context.json", help="A csv or a json file containing the context data.", required=False
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
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
        default=16,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--ckpt_dir", type=Path, default="./ckpt", help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=12345, help="A seed for reproducible training.")
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=32,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate when looking for an answer.",
    )

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
