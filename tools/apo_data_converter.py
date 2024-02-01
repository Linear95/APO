import os
import random
import json
import argparse

from pprint import pprint
from tqdm import tqdm

def preprocess_response(response):    
    while "\nHuman:" in response:
        # remove the additional generation of LLM after the current turn responses.
        response = response.split("\nHuman:")[0]

    return response


def convert_item(item, sampling=False):
    sample_names = ['sample_0', 'sample_1', 'sample_2', 'sample_3']
    if "\nHuman:" in item['golden']:
        print(item)
    gpt_response = preprocess_response(item['golden'])

    if sampling:
        sample_names = [random.choice(sample_names)]

    outputs = []
    for sample_name in sample_names:
        query = item['query']
        query_id = str(item['query_id'])
        res_response = preprocess_response(item[sample_name])
        data_point = {
            "text": [query+'<sep>'+gpt_response, query+'<sep>'+res_response],
            "scores": [1., 0.],
            "type": "apo"
        }
        outputs.append(data_point)

    return outputs


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description ='parser for preference data processing.')
    parser.add_argument("--golden_data_path", type=str, default="", help="the path to golden annotation data.")
    parser.add_argument("--sample_data_path", type=str, default="", help="the path to llm sample data.")
    parser.add_argument("--output_dir", type=str, default="", help="the path to output converted data.")
    parser.add_argument("--apo_data_name", type=str, default="", help="the path to output converted data.")
    parser.add_argument("--sampling", action="store_true", help="whether random select one of the llm sample for each query")
    args = parser.parse_args()

           
    with open(args.sample_data_path, 'r') as f:
        sft_samples = json.load(f)
    print(f'finished loadding {len(sft_samples)} samples')
    
    with open(args.golden_data_path, 'r') as f:
        golden_samples = json.load(f)

    print(f'finished loadding {len(golden_samples)} samples')

    merged_data = {}

    for item in tqdm(sft_samples):
        query_id = str(item['query_id'])
        merged_data[query_id] = item
        
    for item in tqdm(golden_samples):
        query_id = str(item['query_id'])
        merged_data[query_id]['golden'] = item['golden']

    score_dict = None
    outputs = []    
    for query_id, item in merged_data.items():
        new_results = convert_item(item, sampling=args.sampling)
        outputs.extend(new_results)
        # except:
        #     pprint(item1)
        #     error_count += 1

    # print(f"get {error_count} error items")

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        
    if args.sampling:
        output_path = f"{args.output_dir}/{args.apo_data_name}_sampled_text_scores.json"
    else:
        output_path = f"{args.output_dir}/{args.apo_data_name}_text_scores.json"
        
    print(f'finished processing {len(outputs)} data at {output_path}')
    with open(output_path, 'w') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
        
        
