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

from dataset import SeqTaggingClsDataset,SeqClsDataset
from model import MultitaskNet
from utils import Vocab, same_seed

INTENT = "intent"
SLOT = "slot"
TASKS = [INTENT, SLOT]

@torch.no_grad()
def predict(model: MultitaskNet,
          test_pbar:tqdm,
          device:torch.device,
          task: str
        ):
    model.eval()
    test_ids: List[str] = []
    preds = []

    for batch in test_pbar:
        data = batch['data'].to(device)

        output = model(data, task)

        if task == 'intent':
            # pred size -> (Batch, Seq_len)
            pred = output.argmax(dim=1)
            preds.extend(pred.cpu().detach().numpy().tolist())
        elif task == 'slot':
            mask = batch['mask'].to(device)
            if model.crf:
                output = output.transpose(1, 2)
                pred = model.crf.decode(output, mask)
                preds.extend(pred)
            else:
                pred = output.argmax(dim=1)
                preds.extend([pred[idx].masked_select(text_mask).cpu().detach().numpy().tolist()  \
                        for idx, text_mask in enumerate(mask)])
        else:
            raise NameError(f'YOU CHOOSE THE WRONG {task} !!! please choose one of ["intent", "slot"] task')

        test_ids.extend(batch['ids'])

    return preds, test_ids

def main(args):
    # TODO: implement main function

    same_seed()

    task_datasets = [SeqClsDataset, SeqTaggingClsDataset]
    max_lens = [args.icf_max_len, args.slt_max_len]
    label_paths = ['intent2idx.json', 'tag2idx.json']
    idx2label: Dict[str, Dict[int, str]] = {}
    loaders = {INTENT: DataLoader, SLOT: DataLoader}
    task_embeddings = {}
    task_class = {}
    vocab = {}

    for task, label_path, task_dataset, max_len in zip(TASKS, label_paths, task_datasets, max_lens):
        with open(args.cache_dir / task / "vocab.pkl", "rb") as f:
            vocab[task]: Vocab = pickle.load(f)
        label_idx_path = args.cache_dir / task / label_path
        label2idx = json.loads(label_idx_path.read_text())
        idx2label[task] = {v : k for k, v in label2idx.items()}
        task_embeddings[task] = torch.load(args.cache_dir / task / 'embeddings.pt').to(args.device)
        task_class[task] = len(label2idx)
        data = json.loads(args.test_file.read_text()) 
        dataset = task_dataset(data, vocab[task], label2idx, max_len)
        loaders[task] = DataLoader(dataset=dataset, 
                            batch_size=args.batch_size,     
                            pin_memory=False,
                            shuffle=False,
                            collate_fn=dataset.collate_fn)

    model = MultitaskNet(input_size=300, embeddings=task_embeddings, hidden_size=args.hidden_size,
                        num_layers=args.num_layers,dropout_rate=args.dropout, pad_id=vocab[INTENT].pad_id,
                        bidirectional=True, num_class=task_class, model_name=args.model_name,
                        init_method=args.init_weights, device=args.device, crf=args.crf).to(args.device)


    # load weights into model
    checkpoint = torch.load(args.ckpt_path)
    # logging.info(f'Use CRF: {checkpoint["crf"]}')
    model.load_state_dict(checkpoint['model_state_dict'])

    # TODO: predict dataset
    test_pbar = tqdm(loaders[args.task_type], position=0, leave=True) 
    preds, test_ids = predict(model, test_pbar, args.device, args.task_type)

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w', encoding='utf-8', newline='')as file:
        csvWriter = csv.writer(file)
        if args.task_type == 'slot':
            csvWriter.writerow(['id', 'tags'])
            tags = [[idx2label[args.task_type][id] for id in ids]for ids in preds]
            for test_id, tag in zip(test_ids, tags):
                csvWriter.writerow([test_id, ' '.join(tag)])
        elif args.task_type == 'intent':
            csvWriter.writerow(['id', 'intent'])
            intents = [idx2label[args.task_type][ids] for ids in preds]
            for test_id, intent in zip(test_ids, intents):
                csvWriter.writerow([test_id, intent])
        else:
            raise NameError(f'YOU CHOOSE THE WRONG {task} !!! please choose one of ["intent", "slot"] task')

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        required=True,
        help="Path to the test file."
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )

    parser.add_argument("--pred_file", type=Path, default="./pred")

    # data
    parser.add_argument("--icf_max_len", type=int, default=28)
    parser.add_argument("--slt_max_len", type=int, default=35)
    parser.add_argument("--task_type", type=str, required=True, choices=["intent", "slot"])

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--crf", action='store_true', default=False)

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
    if args.pred_file == Path('./pred'):
        (args.pred_file / args.task_type).mkdir(parents=True, exist_ok=True)
        args.pred_file = args.pred_file / args.task_type / f'{args.crf}_{args.batch_size}_{args.hidden_size}_pred.csv'
    main(args)

# python ./test_multitask_crf.py --test_file ./data/slot/test.json --task_type slot --ckpt_path ./ckpt/slot/best_False.pt --num_layers 2 --batch_size 64 --hidden_size 512 --init_weights normal --model_name gru
# python ./test_multitask_crf.py --test_file ./data/intent/test.json --task_type intent --ckpt_path ./ckpt/intent/best_False.pt --num_layers 2 --batch_size 64 --hidden_size 512 --init_weights normal --model_name gru
# python ./test_multitask_crf.py --test_file ./data/slot/test.json --task_type slot --ckpt_path ./ckpt/slot/best_True.pt --num_layers 2 --batch_size 32 --hidden_size 512 --init_weights normal --model_name gru --crf
# python ./test_multitask_crf.py --test_file ./data/intent/test.json --task_type intent --ckpt_path ./ckpt/intent/best_True.pt --num_layers 2 --batch_size 32 --hidden_size 512 --init_weights normal --model_name gru --crf
# bash ./slot_tag.sh ./data/slot/test.json ./pred