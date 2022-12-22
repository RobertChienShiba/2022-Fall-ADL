from dataclasses import dataclass
from typing import Dict, Optional, Union
from itertools import chain
from pathlib import Path
import json
from collections import defaultdict
from abc import ABC, abstractmethod

import torch
import pandas as pd
import numpy as np 

from datasets import load_dataset
from transformers import BertTokenizerFast

from utils import mask_tokens

@dataclass
class DataCollatorForMultipleChoice:
    is_train: bool
    tokenizer: BertTokenizerFast

    def __call__(self, features):
        batch = defaultdict(list)
        for feature in features:
            for k, v in feature.items():
                batch[k].append(v)
        batch = dict(map(lambda x: (x[0], torch.tensor(x[1]))if x[0] != 'ids' else (x[0], x[1]), dict(batch).items()))

        special_tokens_mask = batch.pop('special_tokens_mask')
        if self.is_train:
            batch['input_ids'] = mask_tokens(batch['input_ids'],
                paragraph_indices=(batch['token_type_ids'] &
                                    ~special_tokens_mask).bool(),
                mask_id=self.tokenizer.mask_token_id, mask_prob=0.15)
        return batch

class DataCollatorForQA:
    def __call__(self, features):
        batch = defaultdict(list)
        for feature in features:
            for k, v in feature.items():
                batch[k].append(v)
        batch = dict(map(lambda x: (x[0], torch.tensor(x[1]))if x[0] not in ['example_id', 'offset_mapping'] else (x[0], x[1]), dict(batch).items()))
        return batch

class BaseDataset(ABC):
    def __init__(self,
        context_path: str,
        paddings: bool,
        max_length: int,
        tokenizer: BertTokenizerFast,
        is_train = True,
        **kwargs
        ) -> None:
        self.paddings = "max_length" if paddings else False
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.is_train = is_train
        if self.is_train:
            self.convert_HFdata(context_path, kwargs['train_path'], kwargs['trainHF_path'])
            self.convert_HFdata(context_path, kwargs['validation_path'], kwargs['validationHF_path'])
            self.train_dataset, self.eval_dataset = self.preprocess(dict(train=kwargs['trainHF_path'], 
                    validation=kwargs['validationHF_path']))
        else:
            self.convert_HFdata(context_path, kwargs['test_path'], kwargs['testHF_path'])
            self.test_dataset = self.preprocess(dict(test=kwargs['testHF_path']))

    @abstractmethod
    def convert_HFdata(self):
        pass
    
    @abstractmethod
    def preprocess(self):
        pass

class MultipleChoiceDataset(BaseDataset):
    def __init__(self,
        context_path: str,
        paddings: bool,
        max_length: int,
        tokenizer: BertTokenizerFast,
        is_train: bool,
        **kwargs
        ) -> None:
        super(MultipleChoiceDataset, self).__init__(context_path, 
                    paddings, max_length, tokenizer, is_train, **kwargs)

    def prepare_feature(self, examples):
        ending_names = ["paragraphs_0", "paragraphs_1", "paragraphs_2", "paragraphs_3"]
        context_name = "question"
        label_column_name = "label"

        first_sentences = [[context] * 4 for context in examples[context_name]]
        second_sentences = [[f"{examples[end][i]}" for end in ending_names] for i in range(len(examples[context_name]))]

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = self.tokenizer(
            first_sentences,
            second_sentences,
            max_length=self.max_length,
            padding=self.paddings,
            truncation='only_second',
            return_special_tokens_mask=True
        )
        # Un-flatten
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

        if self.is_train:
          labels = examples[label_column_name]
          tokenized_inputs["labels"] = labels
        else:
          tokenized_inputs["ids"] = examples["id"]
        return tokenized_inputs

    def convert_HFdata(self, context_path: str, ori_json: str, HF_json: str) -> None:
        """
        convert to HuggingFace SWAG dataset format
        """
        with open(context_path, 'r', encoding='utf-8') as file:
            context = json.load(file)

        df = pd.read_json(ori_json, orient='records')

        if 'relevant' in df.columns:
            df[['paragraphs_0', 'paragraphs_1', 'paragraphs_2', 'paragraphs_3', 'label']] = df.apply(
                lambda df, contexts: (contexts[df['paragraphs'][0]], contexts[df['paragraphs'][1]], 
                contexts[df['paragraphs'][2]], contexts[df['paragraphs'][3]], df['paragraphs'].index(df['relevant'])), args=(context,),
                axis=1, result_type='expand')
            df[['id', 'question', 'paragraphs_0', 'paragraphs_1', 'paragraphs_2', 'paragraphs_3', 'label']].to_json(HF_json, 
                orient='records', indent=4, force_ascii=False)
        else:
            df[['paragraphs_0', 'paragraphs_1', 'paragraphs_2', 'paragraphs_3']] = df.apply(
                lambda df, contexts: (contexts[df['paragraphs'][0]], contexts[df['paragraphs'][1]], 
                contexts[df['paragraphs'][2]], contexts[df['paragraphs'][3]]), args=(context,),
                axis=1, result_type='expand')
            df[['id', 'question', 'paragraphs_0', 'paragraphs_1', 'paragraphs_2', 'paragraphs_3']].to_json(HF_json,
                orient='records', indent=4, force_ascii=False)
        

    def preprocess(self, data_files: Dict):
        raw_datasets = load_dataset('json', data_files=data_files)
        if self.is_train:
            raw_datasets = raw_datasets.class_encode_column('label')

            processed_datasets = raw_datasets.map(
                self.prepare_feature, batched=True, remove_columns=raw_datasets["train"].column_names
            )

            train_dataset = processed_datasets["train"]
            eval_dataset = processed_datasets["validation"]
            return train_dataset, eval_dataset
        else:
            processed_datasets = raw_datasets.map(
                self.prepare_feature, batched=True, remove_columns=raw_datasets["test"].column_names
            )

            test_dataset = processed_datasets["test"]
            return test_dataset

class QADataset(BaseDataset):
    def __init__(self,
        context_path: str,
        paddings: bool,
        max_length: int,
        tokenizer: BertTokenizerFast,
        is_train: bool,
        doc_stride: int,
        **kwargs
        ) -> None:
        self.doc_stride = doc_stride
        super(QADataset, self).__init__(context_path, 
                paddings, max_length, tokenizer, is_train, **kwargs)

    def prepare_train_features(self, examples):
        question_column_name = "question" 
        context_column_name = "context" 
        answer_column_name = "answers"

        pad_on_right = self.tokenizer.padding_side == "right" 

        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=self.paddings
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples['cls_index'] = []
        tokenized_examples['overflow_to_sample_mapping'] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)
            tokenized_examples['cls_index'].append(cls_index)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples['overflow_to_sample_mapping'].append(sample_index)
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

        # Validation preprocessing
    def prepare_validation_features(self, examples):
        question_column_name = "question" 
        context_column_name = "context" 

        pad_on_right = self.tokenizer.padding_side == "right" 
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=self.paddings,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []
        tokenized_examples['overflow_to_sample_mapping'] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples['overflow_to_sample_mapping'].append(sample_index)
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def convert_HFdata(self, context_path: str, ori_json: str, HF_json: str) -> None:
        """
        convert to HuggingFace SQUAD dataset format
        """
        with open(context_path, 'r', encoding='utf-8') as file:
            context = json.load(file)

        with open('relevant.json', 'r', encoding='utf-8') as file:
            relevant = json.load(file)

        df = pd.read_json(ori_json, orient='records')

        if 'answer' in df.columns:
            df[['context', 'answers']] = df.apply(
                lambda df, context: (context[df['relevant']], {k: [v] for k, v in df['answer'].items()}), args=(context,),
                axis=1, result_type='expand')
            df[['id', 'question', 'context', 'answers']].to_json(HF_json, 
                orient='records', indent=4, force_ascii=False)
        else:
            df['relevant'] = df['id'].map(relevant)
            df['context'] = df['relevant'].apply(
                lambda s, context: context[int(s)] , args=(context,))
            df[['id', 'question', 'context']].to_json(HF_json, 
                orient='records', indent=4, force_ascii=False)

        # print(df.head())

    def preprocess(self, data_files: Dict):
        raw_datasets = load_dataset('json', data_files=data_files)
        if self.is_train:
            train_dataset = raw_datasets["train"]
            train_dataset = train_dataset.map(
                self.prepare_train_features, batched=True, remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on train dataset"
            )

            eval_example = raw_datasets["validation"]
            train_feature = eval_example.map(
                self.prepare_train_features, batched=True, remove_columns=raw_datasets["validation"].column_names,
                desc="Running tokenizer on validation dataset"
            )
            valid_feature = eval_example.map(
                self.prepare_validation_features, batched=True, remove_columns=raw_datasets["validation"].column_names,
                desc="Running tokenizer on validation dataset"
            )
            for column in valid_feature.column_names:
                if column not in train_feature.column_names:
                    assert len(valid_feature[column]) == torch.tensor(train_feature["input_ids"]).shape[0]
                    train_feature = train_feature.add_column(column, valid_feature[column])
                else:
                    assert torch.tensor(train_feature[column]).shape == torch.tensor(valid_feature[column]).shape
            eval_dataset = train_feature
            return train_dataset, dict(preprocessed=eval_dataset, non_preprocessed=eval_example)
        else:
            test_example = raw_datasets['test']
            test_dataset =test_example.map(
                self.prepare_validation_features, batched=True, remove_columns=raw_datasets["test"].column_names,
                desc="Running tokenizer on prediction dataset"
            )

            return dict(preprocessed=test_dataset, non_preprocessed=test_example)
