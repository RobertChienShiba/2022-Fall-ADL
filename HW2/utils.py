"""
Post-processing utilities for question answering.
"""
import collections
import json
import logging
import os
from typing import Optional, Tuple, Dict, List

import numpy as np
from tqdm import tqdm
import torch

logger = logging.getLogger(__name__)

def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    features_per_example: Dict[str, List[int]],
    n_best_size: int = 20,
    log_level: Optional[int] = logging.INFO,
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.

    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        features_per_example: a map example to its corresponding features.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        log_level (:obj:`int`, `optional`, defaults to ``logging.INFO``):
            ``logging`` log level (e.g., ``logging.WARNING``)
    """
    if len(predictions) != 2:
        raise ValueError("`predictions` should be a tuple with two elements (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    # example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    # features_per_example = collections.defaultdict(list)
    # for i, feature in enumerate(features):
    #     features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()

    # Logging.
    logger.setLevel(log_level)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example in tqdm(examples):
        # Those are the indices of the features associated to the current example.
        # Type: Dict[str, List[int]]
        feature_indices = features_per_example[example]

        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or len(offset_mapping[start_index]) < 2
                        or offset_mapping[end_index] is None
                        or len(offset_mapping[end_index]) < 2
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index:
                        continue

                    prelim_predictions.append(
                        {   
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start": offset_mapping[start_index][0],
                            "end":  offset_mapping[end_index][1],
                        }
                    )

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            pred["text"] = context[pred["start"] : pred["end"]]

        # RAISE ERROR IF SOME EXAMPLE HAVE A NULL ANSWER
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            logger.error(f"example id : {example['id']} have a null answer")
            raise KeyboardInterrupt ("the null answer is not possible.")

        # Compute the softmax of all scores.
        scores = np.array([pred["score"] for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. the null answer is not possible.
        all_predictions[example["id"]] = predictions[0]

    return all_predictions

# Post-processing:
def post_processing_function(examples, features, predictions, example_to_features, n_best_size, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        features_per_example=example_to_features,
        n_best_size=n_best_size,
    )
    formatted_predictions = [{"id": k, "prediction_text": v['text']} for k, v in predictions.items()]

    if 'answers' in examples.column_names:
        references = [{"id": ex["id"], "answers": ex['answers']} for ex in examples]
        return evaluate(predictions=formatted_predictions, references=references)
    else: 
        return formatted_predictions

# Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    """
    Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

    Args:
        start_or_end_logits(:obj:`tensor`):
            This is the output predictions of the model. We can only enter either start or end logits.
        eval_dataset: Evaluation dataset
        max_len(:obj:`int`):
            The maximum length of the output tensor. ( See the model.eval() part for more details )
    """

    step = 0
    # create a numpy array and fill it with -100.
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
    # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
    for _, output_logit in enumerate(start_or_end_logits):  # populate columns
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
        # And after every iteration we have to change the step

        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat

# remove punctuation
def remove_punctuation(in_str):
    in_str = str(in_str).lower().strip()
    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
               '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
               '「','」','（','）','－','～','『','』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)

def calc_em_score(answers, prediction):
    em = 0
    assert len(answers) == 1 and prediction != ""
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em

def evaluate(predictions, references):
    em = 0
    total_count = 0
    skip_count = 0
    pred = dict([(data['id'], data['prediction_text']) for data in predictions])
    ref = dict([(data['id'], data['answers']['text']) for data in references])
    for query_id, answers in ref.items():
        total_count += 1
        if query_id not in pred:
            skip_count += 1
            continue
        prediction = pred[query_id]
        em += calc_em_score(answers, prediction)
    em_score = 100.0 * em / total_count
    assert skip_count == 0
    return {
        'em': em_score, 
        'total': total_count, 
    }

def mask_tokens(origin_input, paragraph_indices, mask_id, mask_prob=0.15):
    mask_input = origin_input.clone()
    mask_indices = torch.bernoulli(torch.full(origin_input.shape, mask_prob)).bool()

    mask_indices = mask_indices & paragraph_indices
    mask_input[mask_indices] = mask_id
    return mask_input

    
# import pandas as pd
# import json
# import datasets
# from pprint import pprint

# import torch
# #显示所有列
# pd.set_option('display.max_columns', None)
# #显示所有行
# # pd.set_option('display.max_rows', None)
# #设置value的显示长度为100，默认为50
# # pd.set_option('max_colwidth',100)
# def mtpch_HF_format(ori_json: str, HF_json: str) -> None:
#     with open('./data/context.json', 'r', encoding='utf-8') as file:
#         context = json.load(file)

#     df = pd.read_json(ori_json, orient='records')

#     if 'relevant' in df.columns:
#         df[['paragraphs_0', 'paragraphs_1', 'paragraphs_2', 'paragraphs_3', 'label']] = df.apply(
#             lambda df, contexts: (contexts[df['paragraphs'][0]], contexts[df['paragraphs'][1]], 
#             contexts[df['paragraphs'][2]], contexts[df['paragraphs'][3]], df['paragraphs'].index(df['relevant'])), args=(context,),
#             axis=1, result_type='expand')
#         df[['id', 'question', 'paragraphs_0', 'paragraphs_1', 'paragraphs_2', 'paragraphs_3', 'label']].to_json(HF_json, 
#             orient='records', indent=4, force_ascii=False)
#     else:
#         df[['paragraphs_0', 'paragraphs_1', 'paragraphs_2', 'paragraphs_3']] = df.apply(
#             lambda df, contexts: (contexts[df['paragraphs'][0]], contexts[df['paragraphs'][1]], 
#             contexts[df['paragraphs'][2]], contexts[df['paragraphs'][3]]), args=(context,),
#             axis=1, result_type='expand')
#         df[['id', 'question', 'paragraphs_0', 'paragraphs_1', 'paragraphs_2', 'paragraphs_3']].to_json(HF_json,
#             orient='records', indent=4, force_ascii=False)

#     print(df.head())

# def qa_HF_format(ori_json: str, HF_json: str) -> None:
#     with open('./data/context.json', 'r', encoding='utf-8') as file:
#         context = json.load(file)

#     df = pd.read_json(ori_json, orient='records')

#     if 'answer' in df.columns:
#         df[['context', 'answers']] = df.apply(
#             lambda df, context: (context[df['relevant']], {k: [v] for k, v in df['answer'].items()}), args=(context,),
#             axis=1, result_type='expand')
#         df[['id', 'question', 'context', 'answers']].to_json(HF_json, 
#             orient='records', indent=4, force_ascii=False)
#     else:
#         df['context'] = df['relevant'].apply(
#             lambda s, context: context[s], args=(context,))
#         df[['id', 'question', 'context']].to_json(HF_json, 
#             orient='records', indent=4, force_ascii=False)

#     print(df.head())

# def mask_tokens(origin_input, paragraph_indices, mask_id, mask_prob=0.15):
#     mask_input = origin_input.clone()
#     mask_indices = torch.bernoulli(torch.full(origin_input.shape, mask_prob)).bool()

#     mask_indices = mask_indices & paragraph_indices
#     mask_input[mask_indices] = mask_id
#     return mask_input

# if __name__ == '__main__':
#     mtpch_HF_format('./data/train.json', './data/train_mtpch.json')
#     mtpch_HF_format('./data/valid.json', './data/valid_mtpch.json')

#     qa_HF_format('./data/train.json', './data/train_qa.json')
#     qa_HF_format('./data/valid.json', './data/valid_qa.json')

#     mtpch_HFdataset = datasets.load_dataset('json', data_files={'train':'./data/train_mtpch.json', 'validation':'./data/valid_mtpch.json'})
#     mtpch_HFdataset = mtpch_HFdataset.class_encode_column('label')
#     pprint(mtpch_HFdataset['train'].features)
#     pprint(mtpch_HFdataset['train'][0])

#     qa_HFdataset = datasets.load_dataset('json', data_files={'train':'./data/train_qa.json', 'validation':'./data/valid_qa.json'})
#     pprint(qa_HFdataset['train'].features)
#     pprint(qa_HFdataset['train'][0])

