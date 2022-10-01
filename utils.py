from typing import Iterable, List
import random
import torch
import numpy as np
import os

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


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
def same_seed(seed=12345): 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_synonyms(glove_path: str, word: str, glove_model = None):
    word2vec_output_file = 'glove.840B.300d.word2vec.txt'
    if not os.path.exists(word2vec_output_file):
        glove2word2vec(glove_path, word2vec_output_file)
    if glove_model is None:
        glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    try :
        synonym , similar_score = glove_model.most_similar(positive=word)[0]
        if similar_score > 0.7 :
            return synonym, glove_model
        else:
            return word, glove_model
    except:
            return word, glove_model

def argument_tokens(input_tokens: List[int], vocab: Vocab):
    argument_token = input_tokens.clone()
    randn_value= torch.randn(input_tokens.shape)
    argument_token[randn_value.abs() > 1.5] = vocab.unk_id
    argument_token[randn_value.abs() > 2.5] = torch.randint(2, len(vocab.tokens), argument_token[randn_value.abs() > 2.5].shape)
    _, seq_len = input_tokens.shape
    swap_time = int(np.ceil(seq_len * 0.1))
    for i in range(len(argument_token)):
        for _ in range(swap_time):
            rand1, rand2 = torch.randint(0, seq_len - 1, [2])
            while rand1 == rand2 :
                rand1, rand2 = torch.randint(0, seq_len - 1, [2])
            tmp = argument_token[i][rand1].clone()
            argument_token[i][rand1] = argument_token[i][rand2]
            argument_token[i][rand2] = tmp
    # replace_tokens = argument_token[randn_value.abs() > 1.5]
    # glove_model = None
    # for idx, token_id in enumerate(replace_tokens):
    #     token = list(vocab.token2idx.keys())[token_id]
    #     synonyms, glove_model = get_synonyms('./glove.840B.300d.txt', token, glove_model)
    #     synonyms_id = vocab.token_to_id(synonyms)
    #     replace_tokens[idx] = synonyms_id

    # argument_token[randn_value.abs() > 1.5] = replace_tokens

    return argument_token




