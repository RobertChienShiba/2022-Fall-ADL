import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import csv

import torch
from torch.utils.data import DataLoader

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab, same_seed


@torch.no_grad()
def predict(model:SeqClassifier,
          test_pbar:tqdm,
          device:torch.device
        ):
    model.eval()
    test_ids: List[str] = []
    preds: List[int] = []

    for batch in test_pbar:
        data = batch['data'].to(device)
        output= model(data)
        # calculate acc
        pred = output.argmax(dim=1)
        preds.extend(pred.cpu().detach().numpy().tolist())
        test_ids.extend(batch['ids'])

    return preds, test_ids


def main(args):

    same_seed()

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            pin_memory=False,
                            shuffle=False,
                            collate_fn=dataset.collate_fn
                            )
    # embedding -> ([6491, 300])
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(input_size=embeddings.shape[-1], embeddings=embeddings, hidden_size=args.hidden_size,
                        num_layers=args.num_layers,dropout_rate=args.dropout, pad_id=vocab.pad_id,
                        bidirectional=True, num_class=len(intent2idx), model_name=args.model_name,
                        init_method=args.init_weights, device=args.device).to(args.device)


    # load weights into model
    checkpoint = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # TODO: predict dataset
    test_pbar = tqdm(dataloader, position=0, leave=True) 
    preds, test_ids = predict(model, test_pbar, args.device)
    intents = [list(intent2idx.keys())[ids] for ids in preds]

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w', encoding='utf-8', newline='')as file:
        csvWriter = csv.writer(file)
        csvWriter.writerow(['id', 'intent'])
        for test_id, intent in zip(test_ids, intents):
            csvWriter.writerow([test_id, intent])

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
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="./pred/intent")

    # data
    parser.add_argument("--max_len", type=int, default=28)

    # model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_layers", type=int, default=2)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # testing 
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
    if args.pred_file == Path('./pred/intent'):
        args.pred_file.mkdir(parents=True, exist_ok=True)
        args.pred_file = args.pred_file / f"{args.model_name}_B{args.batch_size}_H{args.hidden_size}_pred.csv"
    main(args)

# bash ./intent_cls.sh ./data/intent/test.json ./pred/intent
