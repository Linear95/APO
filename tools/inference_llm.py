import os
import argparse
from copy import deepcopy
import json
import glob
from dataclasses import dataclass
from typing import Dict, Sequence
from tqdm import tqdm


import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset
from arguments import CustomTrainingArguments

from utils import print_rank_0, read_json_or_jsonl_data, SEP_TOKEN
from utils import DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN

from reward_datasets import TextRewardDataset, batch_padding

IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "\n\nHuman: {instruction}\n{input}\n\nAssistant: "
    ),
    "prompt_no_input": (
        "\n\nHuman: {instruction}\n\nAssistant: "
    ),
}

B_INST, E_INST = "[INST]", "[/INST]" # for llama-2-chat

def load_query_data_for_generation(args, data_path):
    all_data = read_json_or_jsonl_data(data_path)
    outputs = []
    for idx, item in enumerate(all_data):
        if args.data_type == "comparison_pair":
            query = item['text'][0].split(SEP_TOKEN)[0]                
            outputs.append({
                "query": query,
                "query_id": item.get("query_id", str(idx))
            })
        else:
            outputs.append({
                'query': item['query'],
                "query_id": item.get("query_id", str(idx))
            })
    return TextRewardDataset(outputs)                


def query_data_collactor(args, batch, tokenizer):
    input_ids, attention_mask, labels = [], [], []
    text = [item['query'] for item in batch]
    query_ids = [item['query_id'] for item in batch]

    for sent in text:
        if args.model_prefix == "llama-2-chat":
            # check details at https://huggingface.co/meta-llama/Llama-2-7b-chat
            sent = sent.replace("\nAssistant", f" {E_INST} ").replace("\nHuman", f" {tokenizer.eos_token} {tokenizer.bos_token} {B_INST} ")           
            sent = sent.strip().strip(tokenizer.eos_token)
            input_query_ids = tokenizer.encode(sent, add_special_tokens=False)
           
        else:
            input_query_ids = tokenizer.encode(sent)
            
        input_ids.append(input_query_ids)

    outputs = batch_padding(input_ids, tokenizer)
    outputs['query_ids'] = query_ids
    outputs['text'] = text
    return outputs


def main():
    parser = transformers.HfArgumentParser(CustomTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # setup model
    #---------------------------------------------------------------------------------
    device = torch.cuda.current_device()
    print_rank_0(f"start loading model from {args.model_name_or_path}")
    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        # torch_dtype=torch.float16,       
    )
    print_rank_0(model)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="left",  # for batch decode
        truncation_side='left',
        model_max_length=args.max_length,        
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = 0        
        # tokenizer.pad_token = DEFAULT_PAD_TOKEN
        # smart_tokenizer_and_embedding_resize(
        #     special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        #     tokenizer=tokenizer,
        #     model=model,
        # )

    eval_dataset = load_query_data_for_generation(args, args.data_path)
    
    sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False)
    dataloader = DataLoader(
        eval_dataset, 
        shuffle=False, 
        collate_fn=lambda x: query_data_collactor(args, x, tokenizer), 
        batch_size=args.per_device_eval_batch_size,
        sampler=sampler,
    )

    if args.task_type == "testing":
        generation_config = GenerationConfig(
            temperature=0.3,
            do_sample=True,
            max_new_tokens=512,
            top_k=5,
            top_p=0.85,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=0,
            repetition_penalty=1.05,
            num_return_sequences=1,
        )
    elif args.task_type == "sampling":
        if args.model_prefix == "llama-2-chat":
            temperature = 0.6
            top_p=0.9
        else:
            temperature = 1.2
            top_p=1.

        generation_config = GenerationConfig(
            temperature=temperature,  # default=0.8
            do_sample=True,
            min_length=1,
            max_new_tokens=256,
            top_p=top_p,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=0,
            num_return_sequences=4,
        )


    model.to(device)
    model.eval()
    
    all_outputs = []
    progress_bar = tqdm(range(len(dataloader)), disable=(dist.get_rank() != 0))
    for step, batch in enumerate(dataloader):
        progress_bar.update(1)
        input_ids = torch.Tensor(batch['input_ids']).long().to(model.device)        
        attention_mask = torch.Tensor(batch['attention_mask']).float().to(model.device)
        query_ids = batch['query_ids']
        text = batch['text']

        batch_size = input_ids.shape[0]
        
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
            )
        output_seq = generation_output.sequences.reshape(batch_size, generation_config.num_return_sequences, -1)
        
        inputs_string = tokenizer.batch_decode(input_ids.reshape(batch_size, -1), skip_special_tokens=True)
        
        for idx in range(len(inputs_string)):
            new_item = {"query_id": query_ids[idx], "query": text[idx]}
            output_responses = tokenizer.batch_decode(output_seq[idx], skip_special_tokens=True)
            for res_idx, output_res in enumerate(output_responses):
                response_sample = output_res.replace(inputs_string[idx], '')
                if args.model_prefix == "llama-2-chat":
                    #sent = sent.replace("\nAssistant", f" {E_INST} ").replace("\nHuman", f" {tokenizer.eos_token} {tokenizer.bos_token} {B_INST} ")
                    response_sample = response_sample.replace(E_INST, "\nAssistant").replace(B_INST, "\nHuman")
                    #response_sample = response_sample.replace(E_INST, "\n\nAssistant:").replace(B_INST, "\n\nHuman:")
                
                new_item[f"sample_{res_idx}"] = response_sample                
            
            all_outputs.append(new_item)
        
        if dist.get_rank() == 0 and (step % 10 == 0):
            print_rank_0(f"finished {step} of {len(dataloader)}")
            print_rank_0(all_outputs[-1])


    output_file_prefix = f"{args.output_dir}/{args.model_prefix}_{args.task_type}_{args.data_suffix}"
    with open(f"{output_file_prefix}_rank{dist.get_rank()}.json", 'w') as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)
    print(f"rank {dist.get_rank()} finishs inference.")

    del model
    torch.cuda.empty_cache() 
    dist.barrier()
    if dist.get_rank() == 0:
        result_paths = glob.glob(f"{output_file_prefix}_rank*.json")
        all_results = []
        for res_path in result_paths:
            new_results = read_json_or_jsonl_data(res_path)
            all_results.extend(new_results)

        print(f"totally loaded {len(all_results)} results")
        with open(f"{output_file_prefix}_results.json", 'w') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"finished inference results merge.")

if __name__ == "__main__":
    main()
