from typing import List, Dict

from torch.utils.data import Dataset
import torch

import utils


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: utils.Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

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
        batch_tokens:List[List[int]] = []  
        # intents category id     
        batch_labels: List[int] = [] 
        # sample['id'] number of text
        batch_ids: List[str] = []           
        for sample in samples:
            batch_tokens.append(sample['text'].split())
            if sample.get('intent'):
                batch_labels.append(self.label2idx(sample['intent']))
            batch_ids.append(sample['id'])
        padded_tokens = self.vocab.encode_batch(batch_tokens,self.max_len)
        batch_data = torch.LongTensor(padded_tokens)
        batch_labels = torch.LongTensor(batch_labels)
        
        return dict(data=batch_data, label=batch_labels, ids=batch_ids)

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    def __init__(self,       
        data: List[Dict],
        vocab: utils.Vocab,
        label_mapping: Dict[str, int],
        max_len: int):

        super(SeqTaggingClsDataset, self).__init__(data, vocab, label_mapping, max_len)

    def collate_fn(self, samples: List[Dict]):
        # TODO: implement collate_fn
        batch_tokens:List[List[str]] = []  
        # intents category id     
        batch_labels: List[List[int]] = [] 
        # sample['id'] number of text
        batch_ids: List[str] = []           
        for sample in samples:
            batch_tokens.append(sample['tokens'])
            if sample.get('tags'):
                batch_labels.append([self.label2idx(tag) for tag in sample['tags']])
            batch_ids.append(sample['id'])
        padded_tokens = self.vocab.encode_batch(batch_tokens,self.max_len)
        padded_labels = utils.pad_to_len(batch_labels, self.max_len, self.vocab.pad_id)
        batch_data = torch.LongTensor(padded_tokens)
        batch_labels = torch.LongTensor(padded_labels)
        available_tokens = batch_data != self.vocab.pad_id

        return dict(data=batch_data, label=batch_labels, ids=batch_ids, mask=available_tokens)

