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

import torch

SEP_TOKEN="<sep>"
STRING_SEP="<:>"

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

QUERY_PROMPT="## Human:\n{request}\n\n## Assistant:\n{response}"

INFER_TMP_FILE="{data_path}_pred_{data_suffix}_results_rank_{rank}.jsonl"

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


           
