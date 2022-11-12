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
import csv
import os
import random
from pathlib import Path
from collections import defaultdict

import datasets
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForQuestionAnswering,
    BertTokenizerFast,

)

from utils import post_processing_function, create_and_fill_np_array
from datasets import QADataset, DataCollatorForQA

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

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(args.model_dir)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_dir,
        config=config,
    )
    # Preprocessing the datasets.
    dataset = QADataset(
        args.context_path, args.pad_to_max_length, args.max_length, tokenizer, False, args.doc_stride,
        test_path=args.test_path, testHF_path='./data/test_qa.json'
    )
    # DataLoaders creation:
    data_collator = DataCollatorForQA()

    test_dataloader = DataLoader(
        dataset.test_dataset['preprocessed'], collate_fn=data_collator, batch_size=args.per_device_test_batch_size
    )

    # Prepare everything with our `accelerator`.
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    logger.info("***** Running testing *****")
    logger.info(f"  Num examples = {len(dataset.test_dataset['preprocessed'])}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_test_batch_size}")

    predictions = test(model, test_dataloader, dataset)

    pred_df = pd.DataFrame(predictions)
    # print(pred_df)
    pred_df.to_csv(args.output_path, index=False, encoding='utf-8')

@torch.no_grad()
def test(model, dataloader, dataset):
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

    test_pbar = tqdm(dataloader, position=0, leave=True)
    model.eval()

    for batch in test_pbar:
        if not isinstance(batch, dict):
            batch = batch[1]
        batch.pop('example_id')
        batch.pop('offset_mapping')
        batch.pop('overflow_to_sample_mapping')
        outputs = model(**batch)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
            start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
            end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

        all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
        all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            test_pbar.update(1)
    
    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
    assert max_len == args.max_length

    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, dataset.test_dataset['preprocessed'], max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, dataset.test_dataset['preprocessed'], max_len)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    predictions = post_processing_function(dataset.test_dataset['non_preprocessed'], all_offset_mapping, outputs_numpy, 
            example_to_features, args.n_best_size)

    return predictions

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")
    parser.add_argument(
        "--test_path", type=str, default="./ckpt/relevant.json", help="A csv or a json file containing the training data.",required=False
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
        # default=True,
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to the output json file.",
        default="./predict.csv"
    )
    parser.add_argument(
        "--per_device_test_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the testing dataloader.",
    )
    parser.add_argument("--ckpt_dir", type=Path, default="./ckpt", help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=12356, help="A seed for reproducible training.")
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
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
