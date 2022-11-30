import os
import sys
import argparse
import logging
import math
import numpy as np
from functools import partial
from tqdm import tqdm
import jsonlines

import datasets
from datasets import load_dataset
from accelerate import Accelerator
import torch
from torch.utils.data.dataloader import DataLoader
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)

from dataset import SummarizationDataset

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--test_file", type=str, default=None, required=True, help="A csv or a json file containing the testing data."
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=256,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=64,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_test_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the testing dataloader.",
    )
    parser.add_argument("--output_file", type=str, default="results.jsonl")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--seed", type=int, default=14, help="A seed for reproducible training.")
    args = parser.parse_args()
    
    return args

def main(args):
    global accelerator 
    accelerator = Accelerator()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    
    logger.setLevel(logging.INFO)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    
    config = AutoConfig.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir, config=config)    
    
    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    padding = "max_length" if args.pad_to_max_length else False
    prefix = args.source_prefix if args.source_prefix is not None else ""
    raw_datasets = load_dataset("json", data_files={"test": args.test_file})
    test_examples = raw_datasets['test']
    dataset = SummarizationDataset(padding, args.max_source_length, args.max_target_length, tokenizer, prefix, False, 
        test_path=args.test_file)
    
    data_collator = default_data_collator
    test_dataloader = DataLoader(dataset.test_dataset, collate_fn=data_collator, 
                        batch_size=args.per_device_test_batch_size)

    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    # Test!
    logger.info("\n******** Running testing ********")
    logger.info(f"Num test examples = {len(dataset.test_dataset)}")
    all_predictions = test(model, test_dataloader, tokenizer)
    
    # save_name = f'B_{args.num_beams}_S_{args.do_sample}_TK_{args.top_k}_TP_{args.top_p}_T_{args.temperature}_' + args.output_file
    # save_name = f'full_{args.output_file}'
    save_name = args.output_file
    with jsonlines.open(save_name, 'w') as writer:
        output_data = []
        for id, prediction in zip(test_examples['id'], all_predictions):
            output_data.append({"title": prediction, "id": id})
        writer.write_all(output_data)

def test(model, dataloader, tokenizer):
    model.eval()
    all_predictions = []
    test_pbar = tqdm(dataloader, position=0, leave=True)
    gen_kwargs = {
        "max_length": args.max_target_length,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
    }
    for step, data in enumerate(test_pbar):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                data["input_ids"],
                attention_mask=data["attention_mask"],
                **gen_kwargs,
            )
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            all_predictions += decoded_preds

    return all_predictions

if __name__ == "__main__":
    args = parse_args()
    main(args)
    