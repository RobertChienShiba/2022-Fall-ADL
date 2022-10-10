import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List
import csv
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab, same_seed

@torch.no_grad()
def predict(model:SeqTagger,
          test_pbar:tqdm,
          device:torch.device
        ):
    model.eval()
    test_ids: List[str] = []
    preds: List[List[int]] = []

    for batch in test_pbar:
        data, mask = batch['data'].to(device), batch['mask'].to(device)

        output = model(data)
        # calculate acc
        # pred size -> (Batch, Seq_len)
        pred = output.argmax(dim=1)

        # calculate acc
        preds.extend([pred[idx].masked_select(text_mask) for idx, text_mask in enumerate(mask)])
        test_ids.extend(batch['ids'])

    return preds, test_ids

def main(args):
    # TODO: implement main function
    same_seed()

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            pin_memory=False,
                            shuffle=False,
                            collate_fn=dataset.collate_fn
                            )
    # embedding -> ([6491, 300])
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(input_size=embeddings.shape[-1], embeddings=embeddings, hidden_size=args.hidden_size,
                        num_layers=args.num_layers,dropout_rate=args.dropout, pad_id=vocab.pad_id,
                        bidirectional=True, num_class=len(tag2idx), model_name=args.model_name,
                        init_method=args.init_weights, device=args.device).to(args.device)


    # load weights into model
    checkpoint = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # TODO: predict dataset
    test_pbar = tqdm(dataloader, position=0, leave=True) 
    preds, test_ids = predict(model, test_pbar, args.device)
    tags = [np.array(list(tag2idx.keys()))[ids.cpu().detach().numpy()] for ids in preds]

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file , 'w', encoding='utf-8', newline='')as file:
        csvWriter = csv.writer(file)
        csvWriter.writerow(['id', 'tags'])
        for test_id, tag in zip(test_ids, tags):
            csvWriter.writerow([test_id, ' '.join(tag)])

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
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
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="./pred/slot")

    # data
    parser.add_argument("--max_len", type=int, default=35)

    # model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )

        # init weights
    parser.add_argument("--init_weights", type=str, help="choose the initial weights method from \
    [normal, xavier_normal, kaiming_normal, orthogonal, identity]",
    choices=['normal', 'xavier_normal', 'kaiming_normal', 'orthogonal', 'identity'],
    default='identity')

    # which model
    parser.add_argument("--model_name", type=str, help="choose a model from [rnn, gru, lstm] to finish your task",
                        default='rnn', choices=['rnn', 'gru', 'lstm']) 

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.pred_file == Path('./pred/slot'):
        args.pred_file.mkdir(parents=True, exist_ok=True)
        args.pred_file = args.pred_file / f"{args.model_name}_B{args.batch_size}_H{args.hidden_size}_pred.csv"
    main(args)

# python ./test_slot.py --test_file ./data/slot/test.json  --ckpt_path ckpt/slot/best.pt  --pred_file ./pred/slot --hidden_size 512 --init_weights normal
# bash ./slot_tag.sh ./data/slot/test.json ./pred/slot 