from typing import Iterable, List, Dict
import random
import torch
import numpy as np

# Fix same seed for reproducibility.
def same_seed(seed=123): 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mask_tokens(origin_input, paragraph_indices, mask_id, mask_prob=0.15):
    mask_input = origin_input.clone()
    mask_indices = torch.bernoulli(torch.full(origin_input.shape, mask_prob)).bool()

    mask_indices = mask_indices & paragraph_indices
    mask_input[mask_indices] = mask_id
    return mask_input



