import os
import re
import json
import argparse
import glob
from copy import deepcopy
from pprint import pprint

def get_best_key(item, item_scores, filter_pattern=False):
    max_score = -100000000.
    result = None
    for key, value in item_scores.items():
        if len(item_scores) > 1 and key == "hh_best":
            continue
        if value > max_score:
            if item[key].strip() == "":
                continue
            else:
                result = deepcopy(key)
                max_score = value
    return result

def get_scores_from_list(score_list):
    score_dict = {}
    for item in score_list:
        for key, value in item.items():
            query_id, ans_id = key.split(':')
            if query_id in score_dict:
                if ans_id in score_dict[query_id] and value != score_dict[query_id][ans_id]:
                    print(f">>>>> warning!")
                    print(f">>>>> replacing {query_id}: {ans_id} value {score_dict[query_id][ans_id]} with {value}")

                score_dict[query_id][ans_id] = value
            else:
                score_dict[query_id] = {ans_id: value}
    return score_dict

def get_scores(data_path, rm_scorer):
    file_names = glob.glob(f"{data_path}_*pred_{rm_scorer}*rank*.jsonl")
    score_dict = {}
    for file_name in file_names:
        with open(file_name, 'r') as f:
            lines = f.readlines()
            scores = [json.loads(l.strip()) for l in lines]
        for item in scores:
            for key, value in item.items():
                query_id, ans_id = key.split(':')
                if query_id in score_dict:
                    if ans_id in score_dict[query_id] and value != score_dict[query_id][ans_id]:
                        print(f">>>>> warning!")
                        print(f">>>>> replacing {query_id}: {ans_id} value {score_dict[query_id][ans_id]} with {value}")

                    score_dict[query_id][ans_id] = value
                else:
                    score_dict[query_id] = {ans_id: value}
    return score_dict
            
def rejection_sample(data_path, score_path=None, rm_scorer=None):
    with open(data_path, 'r') as f:
        data_list = json.load(f)

    print(f"totally load {len(data_list)} samples for rejection sampling")

    if score_path is not None:
        with open(score_path, 'r') as f:
            # lines = f.readlines()
            # scores_list = [json.loads(l.strip()) for l in lines]
            score_list = json.load(f)
        data_scores = get_scores_from_list(score_list)
    elif rm_scorer is not None:
        data_scores = get_scores(data_path, rm_scorer)
    else:
        raise ValueError('cannot found score data')
    
    hh_best_counter = 0
    outputs = []
    for item in data_list:#[:10]:
        query_id = str(item['query_id'])
        item_scores = data_scores[query_id]

        
        #best_res_key = max(item_scores, key=item_scores.get)
        best_res_key = get_best_key(item, item_scores, filter_pattern=True)
        if best_res_key is None:
            best_res_key = get_best_key(item, item_scores, filter_pattern=False)
            if best_res_key is None:
                print(item)
                continue
        
        item['target'] = item[best_res_key]
        item['scores'] = item_scores

        if best_res_key == "hh_best":
            hh_best_counter += 1
        outputs.append(deepcopy(item))
    print(f"get {hh_best_counter} data with hh_best selected")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description ='parser for preference data processing.')
    parser.add_argument("--data_path", type=str, default="", help="the path to input data.")
    parser.add_argument("--output_dir", type=str, default="", help="the path to output data.")
    parser.add_argument("--output_file_name", type=str, default="", help="the path to output data.")
    parser.add_argument("--score_path", type=str, default="", help="the rm model name to get score")
    parser.add_argument("--rm_scorer", type=str, default="", help="the rm model name to get score")

    parser.add_argument("--domain", type=str, default="general", help="the domain of the preference data, selected from [general, normal, academy, business, entertainment, literature].")

    parser.add_argument("--convert", action='store_true', help="whether convert responses into the preference text-score format.")
    parser.add_argument("--to_pairs", action='store_true', help="whether convert responses into pair comparisons.")
    
    args = parser.parse_args()
    #outputs = rejection_sample(args.data_path, f"{args.data_path}_{args.rm_scorer}_prediction.json")
    outputs = rejection_sample(args.data_path, args.score_path, args.rm_scorer)

    if len(args.output_file_name) == 0:
        
        _, file_name = os.path.split(args.score_path)
        print(file_name)
        args.output_file_name = f"{args.output_dir}/{file_name}_sft.json"
    
    with open(f"{args.output_file_name}", 'w', encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    
