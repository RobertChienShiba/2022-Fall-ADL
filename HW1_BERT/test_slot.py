import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List
import csv
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from accelerate import Accelerator
import transformers
from transformers import (
    PreTrainedModel,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_scheduler,
)

from dataset import SeqTaggingClsDataset
from utils import same_seed

@torch.no_grad()
def predict(model:PreTrainedModel,
            dataloader: DataLoader,
            idx2tag
        ):
    model.eval()
    test_ids: List[str] = []
    all_preds: List[List[int]] = []
    test_pbar = tqdm(dataloader, position=0, leave=True) 

    for _, batch in enumerate(test_pbar):
        ids = batch.pop('ids')
        labels = batch.pop('labels')
        outputs = model(**batch)
        # preds size -> (Batch, Seq_len)
        preds = outputs.logits.argmax(dim=-1)
        # Remove ignored index (special tokens)
        true_predictions = [
            ' '.join([idx2tag[int(p)] for (p, l) in zip(prediction, label) if l != -100])
            for prediction, label in zip(preds, labels)
        ]
        test_ids.extend(ids)
        all_preds.extend(true_predictions)

    return all_preds, test_ids

def main(args):
    # TODO: implement main function
    same_seed(12345)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    global accelerator
    accelerator = Accelerator()

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())
    idx2tag = {v: k for k, v in tag2idx.items()}

    config = AutoConfig.from_pretrained(args.model_dir, id2label=idx2tag)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, do_lower_case=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir, config=config)

    data = json.loads(args.test_path.read_text())
    dataset = SeqTaggingClsDataset(data, tokenizer, tag2idx, args.max_len, False)
    # TODO: crecate DataLoader for test dataset
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            pin_memory=False,
                            shuffle=False,
                            collate_fn=dataset.collate_fn
                            )
    # Prepare everything with our accelerator.
    model, test_dataloader = accelerator.prepare(
        model, dataloader
    )
    logging.info("\n******** Running predicting ********")
    logging.info(f"Num test examples = {len(dataset)}")
    # TODO: predict dataset
    true_predictions, test_ids = predict(model, test_dataloader, idx2tag)

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file , 'w', encoding='utf-8', newline='')as file:
        csvWriter = csv.writer(file)
        csvWriter.writerow(['id', 'tags'])
        for test_id, pred in zip(test_ids, true_predictions):
            csvWriter.writerow([test_id, pred])

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_path",
        type=Path,
        help="Path to the test file.",
        default="./data/slot/test.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/slot/",
        required=False
    )
    parser.add_argument("--pred_file", type=Path, default="slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=32)


    # data loader
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    # which model
    parser.add_argument("--model_dir", type=str, default="bert-base-uncased")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

# python ./test_slot.py --test_file ./data/slot/test.json  --ckpt_path ckpt/slot/best_slot.pt  --pred_file ./pred/slot --hidden_size 512 --init_weights normal --model_name gru
# bash ./slot_tag.sh ./data/slot/test.json ./pred/slot 