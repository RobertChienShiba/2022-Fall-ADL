from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
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
        batch_tokens  = []       #  List[List[maxlen_tokens2id]]  maybe has padding insisde
        batch_labels = []       # List[int] store intents category id 
        batch_ids = []           # List[str] store sample['id'] number of text
        for sample in samples:
            batch_tokens.append(sample['text'].split())
            batch_labels.append(self.label2idx(sample['intent']))
            batch_ids.append(sample['id'])
        padded_tokens = self.vocab.encode_batch(batch_tokens,self.max_len)
        batch_data = torch.LongTensor(batch_tokens)
        batch_labels = torch.LongTensor(batch_labels)
        return {
            'data': batch_data,
            'label' : batch_labels,
            'id' : batch_ids
        }

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples):
        # TODO: implement collate_fn
        raise NotImplementedError
