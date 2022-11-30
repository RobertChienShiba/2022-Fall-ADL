import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from collections import defaultdict

import datasets
import nltk
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import transformers
from accelerate import Accelerator, GradScalerKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)

from dataset import SummarizationDataset
from utils import postprocess_text, save_curve_plot, rl_loss

logger = get_logger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--train_file", type=str, default=None, required=True, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
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
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=int, default=0.1
    )
    parser.add_argument("--output_dir", type=Path, default='./ckpt', help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=114, help="A seed for reproducible training.")
    parser.add_argument("--grad_max_norm", type=float, default=1)
    parser.add_argument("--use_rl", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--temperature", type=float, default=1)
    args = parser.parse_args()

    return args


def main(args):
    global accelerator
    # grad_scaler = [GradScalerKwargs(init_scale=64)]
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    # accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, \
    #             mixed_precision='fp16', kwargs_handlers=grad_scaler)
    logger.setLevel(logging.INFO)
    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, do_lower_case=True)

    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")


    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # Temporarily set max_target_length for training.
    padding = "max_length" if args.pad_to_max_length else False
    if args.validation_file:
        dataset = SummarizationDataset(padding, args.max_source_length, args.max_target_length, tokenizer, prefix, True, 
                    train_path=args.train_file, validation_path=args.validation_file)
    else:
        dataset = SummarizationDataset(padding, args.max_source_length, args.max_target_length, tokenizer, prefix, True, 
            train_path=args.train_file)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    train_dataloader = DataLoader(
        dataset.train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    if args.validation_file:
        eval_dataloader = DataLoader(dataset.eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.max_train_steps * args.gradient_accumulation_steps * args.warmup_ratio,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    if args.validation_file:
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Metric
    # metric = evaluate.load("rouge")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset.train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    epoch_pbar = trange(args.num_epochs, desc="Epoch")
    best_rouge_mean = 0
    result_logs = defaultdict(list)
 
    for epoch in epoch_pbar:
        # training
        train_loss = train(model, optimizer, train_dataloader, lr_scheduler, tokenizer)
        logger.info(f'epoch: {epoch + 1}, lr: {optimizer.param_groups[0]["lr"]}')
        if args.use_rl:
            logger.info(f'rl_loss: {train_loss:.4f}')
            result_logs['rl_loss'].append(round(train_loss, 4))
        else:
            logger.info(f'ml_loss: {train_loss:.4f}')
            result_logs['ml_loss'].append(round(train_loss, 4))

        # validating
        if args.validation_file: 
            logger.info("***** Running Evaluation *****")  
            logger.info(f"  Num examples = {len(dataset.eval_dataset)}")
            logger.info(f"  Batch size = {args.per_device_eval_batch_size}")              
            rouge_scores = validate(model, eval_dataloader, tokenizer)
            rouge_1 = rouge_scores["rouge-1"]['f']
            rouge_2 = rouge_scores["rouge-2"]['f']
            rouge_L = rouge_scores["rouge-l"]['f']
            rouge_mean = (rouge_1 + rouge_2 + rouge_L) / 3

            if rouge_mean > best_rouge_mean:
                best_rouge_mean = rouge_mean
                if best_rouge_mean > 0.258 or args.use_rl:
                  accelerator.wait_for_everyone()
                  unwrapped_model = accelerator.unwrap_model(model)
                  save_postfix = '_RL' if args.use_rl else f'_{best_rouge_mean:.4f}'
                  save_name = (args.model_name_or_path).split('/')[-1] + save_postfix
                  unwrapped_model.save_pretrained(args.output_dir / save_name, is_main_process=accelerator.is_main_process,
                          save_function=accelerator.save)
                  tokenizer.save_pretrained(args.output_dir / save_name)
                  logger.info("Saving config and model to {}...".format(args.output_dir / save_name))
            else:
                logger.info('Validation rouge did not improve')

            if args.use_rl:
                logger.info(f'rl_r1: {rouge_1:.4f}, rl_r2: {rouge_2:.4f}, rl_rL: {rouge_L:.4f}')
                result_logs['rl_r1'].append(round(rouge_1, 4))
                result_logs['rl_r2'].append(round(rouge_2, 4))
                result_logs['rl_rL'].append(round(rouge_L, 4))
            else:
                logger.info(f'val_r1: {rouge_1:.4f}, val_r2: {rouge_2:.4f}, val_rL: {rouge_L:.4f}')
                result_logs['val_r1'].append(round(rouge_1, 4))
                result_logs['val_r2'].append(round(rouge_2, 4))
                result_logs['val_rL'].append(round(rouge_L, 4))

        elif train_loss < 2.8:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            save_name = (args.model_name_or_path).split('/')[-1] + f'_full_{round(train_loss, 4)}'
            unwrapped_model.save_pretrained(args.output_dir / save_name, is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save)
            tokenizer.save_pretrained(args.output_dir / save_name)
            logger.info("Saving config and model to {}...".format(args.output_dir / save_name))
        
        else:
            logger.info(f'Model not save at loss: {train_loss}')
          
        if os.path.exists('result_log.json'):
            with open('result_log.json', 'r', encoding='utf-8') as f:
                update_log = json.load(f)
                update_log.update(dict(result_logs))
                with open('result_log.json', 'w') as f:
                    json.dump(update_log, f, ensure_ascii=False, indent=2)
        else:
            with open('result_log.json', 'w') as f:
                json.dump(dict(result_logs), f, ensure_ascii=False, indent=2)

    save_curve_plot('result_log.json', args)

def train(model, optimizer, dataloader, lr_scheduler, tokenizer):
    model.train()
    total_loss = 0
    step = 0
    train_pbar = tqdm(dataloader, position=0, leave=True)
    for _, batch in enumerate(train_pbar):
        with accelerator.accumulate(model):
            # with accelerator.autocast():
            outputs = model(**batch)
            if not args.use_rl:
                loss = outputs.loss
            else:
                # Loss & reward function for RL
                greedy_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    do_sample=True,
                    max_length=args.max_target_length,
                )
                greedy_tokens = accelerator.pad_across_processes(
                    greedy_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                greedy_tokens = accelerator.gather(greedy_tokens)
                greedy_tokens = greedy_tokens.cpu().numpy()
                if isinstance(greedy_tokens, tuple):
                    greedy_tokens = greedy_tokens[0]
                greedy_outputs = tokenizer.batch_decode(greedy_tokens, skip_special_tokens=True)
                greedy_labels = batch['labels'].cpu().numpy()
                # Replace -100 in the labels as we can't decode them.
                greedy_labels = np.where(greedy_labels != -100, greedy_labels, tokenizer.pad_token_id)
                greedy_labels = tokenizer.batch_decode(greedy_labels, skip_special_tokens=True)

                sample_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=args.max_target_length,
                    do_sample=args.do_sample,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature
                )
                sample_tokens = accelerator.pad_across_processes(
                    sample_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                sample_tokens = accelerator.gather(sample_tokens)
                sample_tokens = sample_tokens.cpu().numpy()
                if isinstance(sample_tokens, tuple):
                    sample_tokens = sample_tokens[0]
                sample_outputs = tokenizer.batch_decode(sample_tokens, skip_special_tokens=True)
                sample_labels = batch['labels'].cpu().numpy()
                # Replace -100 in the labels as we can't decode them.
                sample_labels = np.where(sample_labels != -100, sample_labels, tokenizer.pad_token_id)
                sample_labels = tokenizer.batch_decode(sample_labels, skip_special_tokens=True)

                try:
                    greedy_rouges = postprocess_text(greedy_outputs, greedy_labels, avg=False)
                    sample_rouges = postprocess_text(sample_outputs, sample_labels, avg=False)
                except:
                    logger.warning('Model Predict Empty Summary !!!')
                    continue
                greedy_rewards = list(map(lambda x: sum([x["rouge-1"]['f'], x["rouge-2"]['f'], x["rouge-l"]['f']]) / 3, greedy_rouges))
                sample_rewards = list(map(lambda x: sum([x["rouge-1"]['f'], x["rouge-2"]['f'], x["rouge-l"]['f']]) / 3, sample_rouges))

                loss_input = outputs.logits[:, :sample_tokens.shape[1], :].reshape(-1, outputs.logits.shape[-1])
                loss_target = torch.tensor(sample_tokens.reshape(-1)).cuda()
                loss = rl_loss(loss_input, loss_target, greedy_rewards, sample_rewards)

            total_loss += loss.detach().item()
            step += 1
            accelerator.backward(loss)
            
            # Update model parameters
            accelerator.clip_grad_norm_(model.parameters(), max_norm=args.grad_max_norm) 
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        train_pbar.set_postfix({'loss': f'{total_loss/(step):>5f}'})
        train_pbar.update(1)

    return total_loss / len(dataloader)

def validate(model, dataloader, tokenizer):
    model.eval()
    valid_pbar = tqdm(dataloader, position=0, leave=True)
    gen_kwargs = {
        "max_length": args.max_target_length,
        "num_beams": args.num_beams,
    }
    all_predictions = []
    all_labels = []
    for step, batch in enumerate(valid_pbar):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]

            generated_tokens, labels = accelerator.gather((generated_tokens, labels))
            generated_tokens = generated_tokens.cpu().numpy()
            labels = labels.cpu().numpy()

            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            for idx, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
              if pred == '':
                empty_pred = decoded_preds.pop(idx)
                # assert empty_pred == ''
                print(decoded_labels.pop(idx))
            assert len(decoded_preds) == len(decoded_labels)
            assert '' not in decoded_preds
            # valid_pbar.set_postfix({'rouge':f'{postprocess_text(decoded_preds, decoded_labels)}'})
            all_predictions += decoded_preds
            all_labels += decoded_labels

    rouge_scores = postprocess_text(all_predictions, all_labels)

    return rouge_scores

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)