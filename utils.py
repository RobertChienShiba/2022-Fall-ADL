from typing import Iterable, List, Dict
import random
import torch
import numpy as np
import os

import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import wordnet

class Vocab:
    PAD = "[PAD]"
    UNK = "[UNK]"

    def __init__(self, vocab: Iterable[str]) -> None:
        self.token2idx = {
            Vocab.PAD: 0,
            Vocab.UNK: 1,
            **{token: i for i, token in enumerate(vocab, 2)},
        }

    @property
    def pad_id(self) -> int:
        return self.token2idx[Vocab.PAD]

    @property
    def unk_id(self) -> int:
        return self.token2idx[Vocab.UNK]

    @property
    def tokens(self) -> List[str]:
        return list(self.token2idx.keys())

    def token_to_id(self, token: str) -> int:
        return self.token2idx.get(token, self.unk_id)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id(token) for token in tokens]

    def encode_batch(
        self, batch_tokens: List[List[str]], to_len: int = None
    ) -> List[List[int]]:
        batch_ids = [self.encode(tokens) for tokens in batch_tokens]
        to_len = max(len(ids) for ids in batch_ids) if to_len is None else to_len
        padded_ids = pad_to_len(batch_ids, to_len, self.pad_id)
        return padded_ids


def pad_to_len(seqs: List[List[int]], to_len: int, padding: int) -> List[List[int]]:
    paddeds = [seq[:to_len] + [padding] * max(0, to_len - len(seq)) for seq in seqs]
    return paddeds


# Fix same seed for reproducibility.
def same_seed(seed=123): 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_synonyms(word: str):
    synonyms = []

    for syn in wordnet.synsets(word):
        for lm in syn.lemmas():
                synonyms.append(lm.name())

    synonyms = list(set(synonyms))

    if synonyms :
        return random.choice(synonyms)
    else:
        return word


def argument_tokens(input_tokens: List[int], vocab: Vocab):
    argument_token = input_tokens.clone()
    randn_value= torch.randn(input_tokens.shape)
    argument_token[randn_value.abs() > 1.5] = vocab.unk_id
    argument_token[randn_value.abs() > 2.5] = torch.randint(2, len(vocab.tokens), argument_token[randn_value.abs() > 2.5].shape)
    _, seq_len = input_tokens.shape
    swap_time = int(np.ceil(seq_len * 0.05))
    for i in range(len(argument_token)):
        for _ in range(swap_time):
              rand = torch.randint(2, seq_len -1, [1])
              token_id = argument_token[i][rand]
              token = list(vocab.token2idx.keys())[token_id]
              synonym = get_synonyms(token)
              synonym_id = vocab.token_to_id(synonym)
              argument_token[i][rand] = synonym_id
            # rand1, rand2 = torch.randint(0, seq_len - 1, [2])
            # while rand1 == rand2 :
            #     rand1, rand2 = torch.randint(0, seq_len - 1, [2])
            # tmp = argument_token[i][rand1].clone()
            # argument_token[i][rand1] = argument_token[i][rand2]
            # argument_token[i][rand2] = tmp

    return argument_token



def argument_tagging(input_tokens: List[int], labels: List[int], tagging_corpus: Dict, vocab:Vocab):
    argument_tokens = input_tokens.clone()
    argument_labels = labels.clone()
    initial_shape = argument_tokens.shape 
    argument_tokens = argument_tokens.flatten()
    argument_labels = argument_labels.flatten()
    # imbalance_cond = (argument_labels == 8) | (argument_labels == 7) | (argument_labels == 5)
    # argument_idx = torch.argwhere(imbalance_cond).flatten()
    bernoulli_idx = torch.bernoulli(torch.full(argument_tokens.shape, 0.2)).bool().cuda()
    rand_idx = torch.bernoulli(torch.full(argument_tokens.shape, 0.05)).bool().cuda()
    mask_idx = torch.argwhere(bernoulli_idx & ~rand_idx).flatten()
    synonym_idx = torch.argwhere(bernoulli_idx & rand_idx).flatten()
    for idx in mask_idx:
        last_token = argument_tokens[idx]
        new_token = random.choice(list(tagging_corpus[argument_labels[idx].item()]))
        while last_token == new_token :
            new_token = random.choice(list(tagging_corpus[argument_labels[idx].item()]))
        argument_tokens[idx] = new_token
    for idx in synonym_idx:
        token_id = argument_tokens[idx]
        token = list(vocab.token2idx.keys())[token_id.item()]
        synonym = get_synonyms(token)
        synonym_id = vocab.token_to_id(synonym)
        argument_tokens[idx] = synonym_id

    argument_tokens = argument_tokens.reshape(*initial_shape)

    return argument_tokens




