import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import csv

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
import transformers
from transformers import (
    PreTrainedModel,
    AutoConfig,
    AutoModelForSequenceClassification,
    BertTokenizer,
)

from dataset import SeqClsDataset
from utils import same_seed


@torch.no_grad()
def predict(model: PreTrainedModel,
          dataloader: DataLoader,
        ):
    model.eval()
    test_ids: List[str] = []
    preds: List[int] = []

    test_pbar = tqdm(dataloader, position=0, leave=True) 

    for _, batch in enumerate(test_pbar):
        ids = batch.pop('ids')
        batch.pop('labels')
        outputs = model(**batch)
        # calculate acc
        pred = outputs.logits.argmax(dim=-1)
        preds.extend(pred.cpu().detach().numpy().tolist())
        test_ids.extend(ids)

    return preds, test_ids


def main(args):

    same_seed()
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    global accelerator
    accelerator = Accelerator()

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    idx2intent = {v: k for k, v in intent2idx.items()}

    config = AutoConfig.from_pretrained(args.model_dir, id2label=idx2intent)
    tokenizer = BertTokenizer.from_pretrained(args.model_dir, use_fast=True, do_lower_case=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, config=config)

    data = json.loads(args.test_path.read_text())
    dataset = SeqClsDataset(data, tokenizer, intent2idx, args.max_len, False)
    # TODO: crecate DataLoader for test dataset
    test_dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            pin_memory=False,
                            shuffle=False,
                            collate_fn=dataset.collate_fn
                            )
    # Prepare everything with our accelerator.
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

    # TODO: START TEST !!!
    logging.info("******** Running predicting ********")
    logging.info(f"Num test examples = {len(dataset)}")

    preds, test_ids = predict(model, test_dataloader)
    results = {example_id: idx2intent[pred] for example_id, pred in zip(test_ids, preds)}

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w', encoding='utf-8', newline='')as file:
        csvWriter = csv.writer(file)
        csvWriter.writerow(['id', 'intent'])
        for example_id, intent in results.items():
            csvWriter.writerow([example_id, intent])

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_path",
        type=Path,
        help="Path to the test file.",
        default="./data/intent/test.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/intent/",
        required=False
    )
    parser.add_argument("--pred_file", type=Path, default="intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=512)

    # data loader
    parser.add_argument("--batch_size", type=int, default=16)

    # testing 
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

# bash ./intent_cls.sh ./data/intent/test.json ./pred/intent
