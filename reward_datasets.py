import os
import json
from tqdm import tqdm
import gzip
import random
from copy import deepcopy

from utils import print_rank_0
from pprint import pprint
import numpy as np

import torch
from torch.utils.data import Dataset

from transformers import LlamaTokenizer

from datasets import load_dataset
from utils import read_json_or_jsonl_data
from utils import DEFAULT_PAD_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_UNK_TOKEN
from utils import QUERY_PROMPT, SEP_TOKEN, STRING_SEP


class TextRewardDataset(Dataset):
    def __init__(self, data):
        self.data = data 

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self,):
        return len(self.data)
    

def reward_data_collactor(args, batch, tokenizer):
    input_ids, attention_mask = [], []
    query_ids, text, scores, apo_data_mask = [], [], [], []
    
    max_response_num = max([len(item['scores']) for item in batch])
    if args.debug_mode:
        print_rank_0(">>> response padding number: {}".format(max_response_num))
    
    for item1 in batch:
        item = prepare_data_item(args, item1,
                                 tokenizer=tokenizer,
                                 padding=(not len(batch) == 1),
                                 max_response_num=max_response_num)

        scores.append(item['scores'])
        input_ids.append(item['tokens']['input_ids'])
        attention_mask.append(item['tokens']['attention_mask'])
        text.append(item['text'])

        if item.get("type", "hh") == 'apo':
            apo_data_mask.append(1)
            # coeffs.append(args.apo_loss_coeff / args.apo_sample_num)
        else:
            apo_data_mask.append(0)
            # coeffs.append(args.rm_kl_coeff)
        
        if "query_ids" in item:
            query_ids.append(item['query_ids'])
        
    if len(query_ids) > 0:
        assert len(query_ids) == len(scores), f"not all items have key:query_id, in {batch}"

        
    return {
        "scores": scores,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "query_ids": query_ids,
        "text": text,
        "apo_data_mask": apo_data_mask
        # "coeffs": coeffs
    }


def reward_tokenize(sentences, tokenizer, padding="longest", add_sep_token=False):
    if isinstance(sentences, str):
        sentences = [sentences]

    input_ids = []
    for sent in sentences:
        if add_sep_token:            
            query, response = sent.split(SEP_TOKEN)
            query_ids = tokenizer.encode(query, add_special_tokens=False)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            input_ids.append(
                [tokenizer.bos_token_id] + query_ids + [tokenizer.sep_token_id] + response_ids + [tokenizer.eos_token_id]
            )
        else:
            if SEP_TOKEN in sent:
                query, response = sent.split(SEP_TOKEN)
                query_ids = tokenizer.encode(query, add_special_tokens=False)
                response_ids = tokenizer.encode(response, add_special_tokens=False)
                input_ids.append(
                    [tokenizer.bos_token_id] + query_ids + response_ids + [tokenizer.eos_token_id]
                )
            else:
                input_ids.append(
                    [tokenizer.bos_token_id] + tokenizer.encode(sent, add_special_tokens=False) + [tokenizer.eos_token_id]
                )
            
    if padding == 'longest':
        max_input_length = max([len(inp_ids) for inp_ids in input_ids])
        max_length = min(tokenizer.model_max_length, max_input_length)
    else:
        max_length = tokenizer.model_max_length

    outputs = {"input_ids": [], "attention_mask": []}
    for inp_ids in input_ids:        
        attn_mask = [1] * len(inp_ids)
        if len(inp_ids) >= max_length:
            if tokenizer.truncation_side == 'left':
                inp_ids = inp_ids[-max_length :]
                attn_mask = attn_mask[-max_length :]
            else:
                inp_ids = inp_ids[:max_length]
                attn_mask = attn_mask[:max_length]
        else:
            if tokenizer.padding_side == 'left':
                inp_ids = [tokenizer.pad_token_id] * (max_length - len(inp_ids)) + inp_ids
                attn_mask = [0] * (max_length - len(attn_mask)) + attn_mask
            else:
                inp_ids =  inp_ids + [tokenizer.pad_token_id] * (max_length - len(inp_ids)) 
                attn_mask = attn_mask + [0] * (max_length - len(attn_mask))

        outputs['input_ids'].append(deepcopy(inp_ids))
        outputs['attention_mask'].append(deepcopy(attn_mask))
    return outputs


def prepare_data_item(args, item, tokenizer=None, padding=False, max_response_num=1):
    new_item = deepcopy(item)
    if not len(new_item['scores']) == len(new_item['text']):
        ValueError("invalid data point {}".format(new_item))
        return None


    if "query_ids" in new_item and not len(new_item['scores']) == len(new_item['query_ids']):
        ValueError("invalid data point {}".format(new_item))
        return None

    # score_idx = np.argsort(new_item['scores'])
    max_score = max(new_item['scores']) + 1e-5
    min_score = min(new_item['scores']) - 1e-5
    new_item['scores'] = [(score - min_score) / (max_score -min_score) for score in new_item['scores']]

    if padding:
        new_item['text'] += ["\n\nHuman: ?\n\nAssistant: <sep> Some"] * (max_response_num - len(new_item['text']))
        new_item['scores'] += [-1.] * (max_response_num - len(new_item['scores']))
        if "query_ids" in new_item:
            new_item['query_ids'] += [ "unk" + STRING_SEP + "pad" + STRING_SEP + "unk"] * (max_response_num - len(new_item['query_ids']))


    if tokenizer is not None:
        try:
            new_item['tokens'] = reward_tokenize(
                sentences=new_item['text'],
                tokenizer=tokenizer,
                padding="max_length" if padding else "longest",
                add_sep_token=args.add_sep_token
            )
        except:
            raise ValueError(f"get tokenization error with {new_item}")

    return new_item



def load_rejection_samples(data_path):
    data_list = read_json_or_jsonl_data(data_path)
    outputs = []
    for item in data_list:
        # print_rank_0(item)
        if 'query' in item:
            query = str(item['query'])
        else:
            query = str(item['instruction'])
            
        query_id = str(item['query_id'])
        
        for key in item:
            #if "hh_best" in key or "gpt4" in key:
            if "sample_" in key or "gpt4" in key or 'ans_' in key:
                outputs.append({
                    "text": [ query + SEP_TOKEN + str(item[key])],
                    "query_ids": [ data_path + STRING_SEP + query_id + STRING_SEP + key],
                    "scores": [-1]
                })
    print(f">>> totally get {len(outputs)} rejection samples.")
    print(outputs[0])
    return outputs


def load_text_score_dataset(args, data_path):
    print_rank_0("loading text-scores dataset from: \n   {}".format(data_path))

    if args.data_type == "reject_sample":
        data_list = load_rejection_samples(data_path)
    else:
        data_list = read_json_or_jsonl_data(data_path)
        for item in data_list:
            item['query_ids'] = [os.path.split(data_path)[1]] * len(item['text'])

    
            
    print_rank_0("finished loading with {} data.".format(len(data_list)))
    return data_list
        
    
