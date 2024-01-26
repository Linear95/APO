import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import json
import random

import torch
import torch.distributed as dist
import transformers

from torch.utils.data import Dataset
from transformers import Trainer, AutoConfig
from transformers import EvalPrediction


from model import LlamaRewardModel

from reward_datasets import TextRewardDataset, reward_data_collactor
from reward_datasets import load_text_score_dataset
from arguments import CustomTrainingArguments
from trainer import RewardModelTrainer, compute_metrics

from utils import print_rank_0, set_reward_tokenizer, merge_json_or_jsonl_data
from utils import DEFAULT_PAD_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_UNK_TOKEN
from utils import QUERY_PROMPT, SEP_TOKEN, STRING_SEP, INFER_TMP_FILE



def get_eval_datasets(args):
    data_dict = {}

    for data_path in args.eval_data_path:
        eval_data_list = load_text_score_dataset(args=args, data_path=data_path)
              
        eval_dataset = TextRewardDataset(eval_data_list)
        
        data_name = os.path.split(data_path)[-1]
        data_dict[data_name] = eval_dataset
        print_rank_0(">> finished loading {} data with data size = {}".format(data_name, len(eval_dataset)))

        if args.debug_mode:
            print_rank_0(f">>> check loaded data:")        
            print_rank_0(f">>> {eval_dataset[0]}")
        
    return data_dict

def get_train_dataset(args):    
    all_train_data = []
    for train_data_path in args.train_data_path:
        train_data = load_text_score_dataset(args=args, data_path=train_data_path)
        all_train_data.extend(train_data)

    if args.debug_mode:
        print_rank_0(f">>> check loaded data:")        
        print_rank_0(f">>> {all_train_data[0]}")

    train_set = TextRewardDataset(all_train_data)
    return train_set


def train():
    parser = transformers.HfArgumentParser(CustomTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print_rank_0(args)

    # load data
    #---------------------------------------------------------------------------------
    if args.do_train:
        train_dataset = get_train_dataset(args)
    else:
        train_dataset = None
        
    eval_dataset_dict = get_eval_datasets(args)

    # setup model
    #---------------------------------------------------------------------------------
    print_rank_0(f"Begin loading model from {args.model_name_or_path}")
    if args.model_type == "reward":
        model = LlamaRewardModel.from_pretrained(args.model_name_or_path)
    elif args.model_type == "sft":
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)
        
    print_rank_0(model)
    print_rank_0(f"Finished loading model from {args.model_name_or_path}")

    model.is_parallelizable = True
    model.model_parallel = True

    # setup tokenizer
    #---------------------------------------------------------------------------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,      
        model_max_length=args.max_length,        
        padding_side=args.padding_side,
        truncation_side=args.truncation_side,
        use_fast=False,
    )
    
    if args.model_type == "reward":
        model, tokenizer = set_reward_tokenizer(model=model, tokenizer=tokenizer)

    # build trainer
    #---------------------------------------------------------------------------------

    trainer = RewardModelTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=args,
        compute_metrics=lambda x: compute_metrics(args, x),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_dict,
        data_collator=lambda x: reward_data_collactor(args, x, tokenizer)
    )

    if args.do_train:
        if args.eval_at_start:
            for eval_set_name, eval_dataset in eval_dataset_dict.items():
                eval_result = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval_"+eval_set_name)
                print_rank_0(eval_result)


        with torch.autocast("cuda"): 
            if args.resume_from_checkpoint:
                train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
            else:
                train_result = trainer.train()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        trainer.save_state()
        trainer.save_model(output_dir=args.output_dir)


    final_eval_results ={}
    for eval_set_name, eval_dataset in eval_dataset_dict.items():
        args.current_eval_filename = os.path.split(eval_set_name)[-1]
        eval_result = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval_"+eval_set_name)
            
        print_rank_0(eval_result)
        final_eval_results[eval_set_name] = eval_result

        if args.task_type == "inference":
            torch.distributed.barrier()
            if dist.get_rank() == 0:
                print_rank_0(eval_set_name)
                data_path = eval_dataset[0]['query_ids'][0].split(STRING_SEP)[0]

                result_temp = INFER_TMP_FILE.format(data_path=data_path,
                                                    data_suffix=args.data_suffix,
                                                    rank="*")
                print_rank_0(f"begin merge temp file from {result_temp}")
                outputs = merge_json_or_jsonl_data(result_temp)
                with open(f"{data_path}_pred_{args.data_suffix}_results.json", 'w') as f:
                    json.dump(outputs, f, ensure_ascii=False, indent=2)



    with open(f"{args.output_dir}/final_eval_results.json", 'w') as f:
        json.dump(final_eval_results, f, ensure_ascii=False)



if __name__ == "__main__":
    train()
