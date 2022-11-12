from typing import List, Dict
from collections import defaultdict

from torch.utils.data import Dataset
import torch

from transformers import PreTrainedTokenizer

from utils import mask_tokens

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        label_mapping: Dict[str, int],
        max_len: int,
        is_train: bool
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        batch_tokens = defaultdict(list) 
        # intents category id     
        batch_labels: List[int] = [] 
        # sample['id'] number of text
        batch_ids: List[str] = []           
        for sample in samples:
            tokenized_example = self.tokenizer(
                sample['text'], 
                padding="max_length", 
                max_length=self.max_len, 
                truncation=True,
                return_special_tokens_mask=True,
                return_tensors="pt"
            )
            for key, value in tokenized_example.items():
                batch_tokens[key].append(value)
            if sample.get('intent'):
                batch_labels.append(self.label2idx(sample['intent']))
            batch_ids.append(sample['id'])
        batch_tokens = {key: (torch.stack(value)).squeeze(1) for key, value in batch_tokens.items()}
        special_tokens_mask = batch_tokens.pop('special_tokens_mask')
        # if self.is_train:
        #     batch_tokens['input_ids'] = mask_tokens(batch_tokens['input_ids'],
        #         paragraph_indices=(batch_tokens['token_type_ids'] &
        #                             ~special_tokens_mask).bool(),
        #         mask_id=self.tokenizer.mask_token_id, mask_prob=0.15)
        batch_labels = torch.LongTensor(batch_labels)

        return dict(**batch_tokens, labels=batch_labels, ids=batch_ids)

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    def __init__(self,       
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        label_mapping: Dict[str, int],
        max_len: int,
        is_train: bool
    ):

        super(SeqTaggingClsDataset, self).__init__(data, tokenizer, label_mapping, max_len, is_train)

    def collate_fn(self, samples: List[Dict]):
        # TODO: implement collate_fn
        batch_tokens = defaultdict(list)  
        # intents category id     
        batch_labels: List[List[int]] = [] 
        # sample['id'] number of text
        batch_ids: List[str] = []           
        for sample in samples:
            tokenized_example = self.tokenizer(
                sample['tokens'], 
                padding="max_length", 
                max_length=self.max_len, 
                truncation=True,
                return_tensors="pt",
                is_split_into_words=True
            )
            for key, value in tokenized_example.items():
                batch_tokens[key].append(value)
            word_ids = tokenized_example.word_ids()
            prev_word_id = None
            example_labels = []
            for word_id in word_ids:
                if word_id == None:
                    example_labels.append(-100) # Ignored when computing loss
                elif word_id != prev_word_id:   # Put tag only on the first token of word
                    if sample.get('tags'):
                        example_labels.append(self.label2idx(sample['tags'][word_id]))
                    else:
                        example_labels.append(0)    # Put arbitary label if no tag_co
                else:
                    example_labels.append(-100)
                prev_word_id = word_id
            batch_labels.append(example_labels)
            batch_ids.append(sample['id'])
        batch_tokens = {key: (torch.stack(value)).squeeze(1) for key, value in batch_tokens.items()} 
        batch_labels = torch.LongTensor(batch_labels)

        return dict(**batch_tokens, labels=batch_labels, ids=batch_ids)

