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
import json
import os
import random
from itertools import chain
from pathlib import Path
from sched import scheduler
from statistics import mode
from typing import Optional, Union

import datasets
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    BertForMultipleChoice,
    BertTokenizerFast,
)

from datasets import MultipleChoiceDataset, DataCollatorForMultipleChoice

logger = get_logger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def main(args):
  global accelerator
  accelerator = Accelerator()

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
  dataset = MultipleChoiceDataset(args.context_path, args.pad_to_max_length, args.max_length, tokenizer, False,
          test_path=args.test_path, testHF_path='./data/test_mtpch.json')

  
  # DataLoaders creation:
  test_dataloader = DataLoader(
      dataset.test_dataset, shuffle=False, batch_size=args.per_device_test_batch_size,
      collate_fn=DataCollatorForMultipleChoice(is_train=False, tokenizer=tokenizer)
  )

  # Use the device given by the `accelerator` object.
  device = accelerator.device
  model.to(device)

  # Prepare everything with our `accelerator`.
  model, test_dataloader= accelerator.prepare(model, test_dataloader)

  logger.info("***** Running testing *****")
  logger.info(f"  Num examples = {len(dataset.test_dataset)}")
  logger.info(f"  Instantaneous batch size per device = {args.per_device_test_batch_size}")

  pred_dict = test(model, test_dataloader)

  with open(args.test_path, 'r', encoding='utf-8') as file:
    test_file = json.load(file)
  with open(args.test_path, 'w', encoding='utf-8') as file:
    for data in test_file:
      if data['id'] in pred_dict:
        data['relevant'] = data["paragraphs"][int(pred_dict[data['id']])]
    json.dump(test_file, file, indent=4, ensure_ascii=False)

@torch.no_grad()
def test(model, dataloader):
    model.eval()
    data = dict()
    test_pbar = tqdm(dataloader, position=0, leave=True)

    # Evaluation loop - calculate accuracy and save model weights
    for batch in test_pbar:
        ids = batch.pop('ids')
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=1).detach().cpu().numpy().tolist()

        for id, pred in zip(ids, preds):
            data[id] = pred

        if accelerator.sync_gradients:
            test_pbar.update(1)

    return data

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
    parser.add_argument(
        "--test_path", type=str, default="./data/test.json", help="A csv or a json file containing the testing data.", required=True
    )
    parser.add_argument(
        "--context_path", type=str, default="./data/context.json", help="A csv or a json file containing the context data.", required=True
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
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_test_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the testing dataloader.",
    )
    parser.add_argument("--ckpt_dir", type=Path, default="./ckpt", help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=12345, help="A seed for reproducible training.")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
