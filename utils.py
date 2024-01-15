import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
import glob
from typing import Optional, Sequence, Union, List, Dict

import openai
import tqdm
from openai import openai_object
import copy

import numpy as np
import torch

SEP_TOKEN="<sep>"
STRING_SEP="<:>"

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

QUERY_PROMPT="## Human:\n{request}\n\n## Assistant:\n{response}"

INFER_TMP_FILE="{data_path}_pred_{data_suffix}_results_rank_{rank}.jsonl"

def numpy_sigmoid(x):
    # r_x = x - x.max()
    return 1. / (1. + np.exp(-x))


def read_json_or_jsonl_data(data_path):
    if data_path[-5:] == ".json":
        with open(data_path, 'r') as f:
            data_list = json.load(f)
    else:
        with open(data_path, 'r') as f:
            lines = f.read().strip().split('\n')
            data_list = [json.loads(l) for l in lines]

    print_rank_0(f">>> totally load {len(data_list)} data from {data_path}")
    return data_list

def merge_json_or_jsonl_data(data_path_pattern):
    file_names = glob.glob(data_path_pattern)
    print_rank_0(f"load {len(file_names)} files from {data_path_pattern}.")
    outputs = []
    for file_name in file_names:
        new_data = read_json_or_jsonl_data(file_name)
        if isinstance(new_data, list):
            outputs.extend(new_data)
        elif isinstance(new_data, dict):
            outputs.append(new_data)
    return outputs

    
def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def set_reward_tokenizer(model, tokenizer):
    
    tokenizer.pad_token_id = 3
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.unk_token_id = 0
    tokenizer.sep_token_id = 4

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    print_rank_0(tokenizer)
    return model, tokenizer


           

def calibration_error(
    y_true,
    y_prob,
    n_bins=5,
    strategy="uniform",
):
    if len(y_true) == 0:
        return 0., 0., 0.

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    binids = np.searchsorted(bins[1:-1], y_prob)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    # prob_true = bin_true[nonzero] / bin_total[nonzero]
    # prob_pred = bin_sums[nonzero] / bin_total[nonzero]
   
    # return prob_true, prob_pred, bin_total[nonzero]
    try:
        expected_error = np.abs(bin_sums - bin_true).sum() / len(y_prob)
        average_error = (np.abs(bin_sums[nonzero] - bin_true[nonzero]) / bin_total[nonzero]).mean()
        max_error = (np.abs(bin_sums[nonzero] - bin_true[nonzero]) / bin_total[nonzero]).max()
    except Exception as e:
        print_rank_0(">>>> WARNING: Encounter error in calibration calculation")
        print_rank_0(e)
        expected_error, average_error, max_error = 0., 0., 0.
        
    return expected_error, average_error, max_error
