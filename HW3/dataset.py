from dataclasses import dataclass
from typing import Dict, Optional, Union
from itertools import chain
from pathlib import Path
import json
from collections import defaultdict

import torch
import pandas as pd
import numpy as np 

from datasets import load_dataset
from transformers import PreTrainedTokenizer

class SummarizationDataset(object):
    def __init__(self,
        paddings: bool,
        max_source_length: int,
        max_target_length: int,
        tokenizer: PreTrainedTokenizer,
        prefix,
        is_train = True,
        **kwargs
        ) -> None:
        self.paddings = "max_length" if paddings else False
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.prefix = prefix
        self.valid = kwargs.get('validation_path', None)

        if self.is_train and self.valid:
            self.train_dataset, self.eval_dataset = self.preprocess(dict(train=kwargs['train_path'], 
                validation=kwargs['validation_path']))
        elif self.is_train:
            self.train_dataset = self.preprocess(dict(train=kwargs['train_path']))
        else:
            self.test_dataset = self.preprocess(dict(test=kwargs['test_path']))

    def prepare_features(self, examples):
        text_column_name = "maintext"
        summary_column_name = "title"
        prefix = self.prefix

        inputs = examples[text_column_name]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = self.tokenizer(inputs, max_length=self.max_source_length, padding=self.paddings, truncation=True)

        if self.is_train:
            targets = examples[summary_column_name]
            # Tokenize targets with the `text_target` keyword argument
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(text_target=targets, max_length=self.max_target_length, padding=self.paddings, truncation=True)

                # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
                # padding in the loss.
                labels["input_ids"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

                model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def preprocess(self, data_files: Dict):
        raw_datasets = load_dataset('json', data_files=data_files)
        if self.is_train:
            processed_datasets = raw_datasets.map(
                self.prepare_features,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on dataset",
            )

            train_dataset = processed_datasets["train"]
            if self.valid:
                eval_dataset = processed_datasets["validation"]
                return train_dataset, eval_dataset
            return train_dataset
        else:
            processed_datasets = raw_datasets.map(
                self.prepare_features, batched=True, remove_columns=raw_datasets["test"].column_names
            )

            test_dataset = processed_datasets["test"]
            return test_dataset