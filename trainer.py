import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import json
import datetime

import numpy as np
import sklearn

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers


from transformers import Trainer, AutoConfig
from transformers import EvalPrediction

from utils import print_rank_0, calibration_error, numpy_sigmoid
from utils import QUERY_PROMPT, SEP_TOKEN, STRING_SEP, INFER_TMP_FILE



def rm_calibration_errors(args, labels, probs, masks, num_bins):
    label_list = labels.reshape(-1).tolist()
    prob_list = probs.reshape(-1).tolist()
    mask_list = masks.reshape(-1).tolist()

    y_true, y_prob = [], []
    for label, prob, mask in zip(label_list, prob_list, mask_list):
        if mask:
            y_true.append(label)
            y_prob.append(prob)

    if args.debug_mode:
        print_rank_0(f">>>>> check calibration inputs mask filtered...")
        print_rank_0(f">>>>>>>> y_true: {y_true[:10]}")
        print_rank_0(f">>>>>>>> y_prob: {y_prob[:10]}")
                   
    return calibration_error(np.array(y_true), np.array(y_prob), n_bins=num_bins)
    

def compute_metrics(args, prediction: EvalPrediction):
    logits = torch.from_numpy(prediction.predictions)
    scores = torch.from_numpy(prediction.label_ids)

    if args.debug_mode:    
        print_rank_0(f">> check eval_prediction inputs...")
        print_rank_0(f">>> logits: {logits[:5]}")
        print_rank_0(f">>> scores: {scores[:5]}")

    logits_diff = logits.unsqueeze(1) - logits.unsqueeze(2)  # [batch_size, num_sample, num_sample]

    score_mask_larger = (scores.unsqueeze(1) > scores.unsqueeze(2)) * 1.
    score_mask_smaller = (scores.unsqueeze(1) < scores.unsqueeze(2)) * 1.
    score_mask = score_mask_larger - score_mask_smaller
    pad_mask = (scores >= 0).unsqueeze(1) * 1. * (scores >= 0).unsqueeze(2)


    # calculate accuracy...
    pred_compare = (logits_diff.detach() * score_mask > 0.) * 1.
    total_mask = (score_mask_larger + score_mask_smaller) * pad_mask
    #correct_compare = (pred_compare == score_mask_larger) * total_mask
    correct_compare = pred_compare * total_mask

    all_acc = correct_compare.sum() / total_mask.sum() if total_mask.sum() > 0 else total_mask.sum()
    average_score = logits.mean().item()

    calibration_errors = {}
    if args.rm_calibration:
        for num_bins in args.calibration_bins:
            expected_error, average_error, max_error = rm_calibration_errors(
                args=args,                                                                            
                labels=score_mask_larger,
                #probs=torch.sigmoid(logits_diff),
                probs=numpy_sigmoid(logits_diff.numpy()),
                masks=total_mask,
                num_bins=num_bins
            )
            # if args.save_calibration and args.task_type == "eval":
            #     time = datetime.datetime.now()
            #     time_stamp = time.strftime("%d-%H:%M:%S")
            #     if dist.get_rank() == 0:
            #         outputs = {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}
            #         with open(f"{args.output_dir}/calibration_result_t{args.current_eval_filename}_bin{num_bins}.json", 'w') as f:
            #             json.dump(outputs, f, ensure_ascii=False, indent=2)

            calibration_errors[f"calibration_ECE_bin{num_bins}"] = expected_error
            calibration_errors[f"calibration_ACE_bin{num_bins}"] = average_error
            calibration_errors[f"calibration_MCE_bin{num_bins}"] = max_error

    if args.debug_mode:
        print_rank_0(f">> check eval_prediction outputs...")
        print_rank_0(f">>> correct_compare: {correct_compare}")
        print_rank_0(f">>> total_mask: {total_mask}")
        print_rank_0(f">>> all_acc: {all_acc}")
        print_rank_0(f">>> calibration error: {calibration_errors}")

    return {"Preference Acc": all_acc.item(), "Avg Score": average_score, **calibration_errors}     


def reward_model_loss(logits, scores, coeffs=None, loss_type="ranking"): # `logits`, `scores` with shape [bs, r], `coeffs` with shape [bs]
    logits_diff = logits.unsqueeze(1) - logits.unsqueeze(2)  # shape [bs, r, r]

    score_mask_larger = (scores.unsqueeze(1) > scores.unsqueeze(2)) * 1.
    score_mask_smaller = (scores.unsqueeze(1) < scores.unsqueeze(2)) * 1.
    score_mask = score_mask_larger - score_mask_smaller
    pad_mask = (scores >= 0).unsqueeze(1) * 1. * (scores >= 0).unsqueeze(2)

    total_mask = (score_mask_larger + score_mask_smaller) * pad_mask

    if loss_type == "diff":
        log_prob = logits_diff * score_mask * pad_mask # shape [bs, r, r]
    else: 
        log_prob = torch.nn.functional.logsigmoid(logits_diff * score_mask * pad_mask) # shape [bs, r, r]
        
    if coeffs is not None:
        log_prob = log_prob * coeffs.unsqueeze(-1).unsqueeze(-1)

    total_loss = - (log_prob * total_mask).sum()
    total_pairs = total_mask.sum()

    return  total_loss / total_pairs  if total_pairs > 0 else total_loss
    #return - log_prob.mean()
       

class RewardModelTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[List[str]] = None):
        device = model.device
        labels = torch.Tensor(inputs['scores']).float().to(device)

        with torch.no_grad():
            loss, logits = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
            # logits = outputs.logits

        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, logits, labels)

                
    def compute_loss(self, model, inputs, return_outputs=False):
        device = model.device
        scores  = torch.Tensor(inputs['scores']).float().to(device)    # shape [batch_size, response_num]
        input_ids = torch.Tensor(inputs['input_ids']).long().to(device)    # shape [batch_size, response_num, seq_length]
        attention_mask = torch.Tensor(inputs['attention_mask']).float().to(device) 
        # coeffs = torch.Tensor(inputs['coeffs']).float().to(device)   
        apo_data_mask = torch.Tensor(inputs['apo_data_mask']).float().to(device)    # shape [batch_size]  value 1 if apo data 

        batch_size, response_num, seq_length = input_ids.shape
        
        if self.args.debug_mode:
            print(f">>> input_ids shape {input_ids.shape}")
    
        outputs = model(
            input_ids=input_ids.view(-1, seq_length),
            attention_mask=attention_mask.view(-1, seq_length),
            padding_side=self.args.padding_side,
            pooling_type=self.args.pooling_type
        )
        
        batch_logits = outputs['rm_logits'].view(batch_size, response_num) # shape [bs, r]

        if self.args.task_type == "apo":
            rm_kl_loss = reward_model_loss(batch_logits, scores, coeffs=(1. - apo_data_mask), loss_type="ranking")
            apo_loss = reward_model_loss(batch_logits, scores, coeffs=apo_data_mask, loss_type=self.args.apo_loss_type)        
            total_loss = self.args.rm_kl_coeff * rm_kl_loss + self.args.apo_loss_coeff / self.args.apo_sample_num * apo_loss
        else:
            total_loss = reward_model_loss(batch_logits, scores, coeffs=None, loss_type="ranking")

        if self.args.debug_mode:
            print_rank_0(f">>> debug")
            print_rank_0(f">>> input_ids shape {input_ids.shape}")
            print_rank_0(f">>> Batch rm logits {batch_logits}")
            print_rank_0(f">>> Query ids {query_ids}")

        if self.args.task_type == "inference":
            new_results = []

            for i_bs in range(batch_size):
                for j_sample in range(response_num):
                    data_path, query_id, ans_id = query_ids[i_bs][j_sample].split(STRING_SEP)
                    new_results.append(
                        json.dumps({f"{query_id}:{ans_id}": batch_logits[i_bs][j_sample].item()}, ensure_ascii=False)
                    )                    

            output_file_path = INFER_TMP_FILE.format(data_path=data_path,
                                                     data_suffix=self.args.data_suffix,
                                                     rank=dist.get_rank())
            with open(output_file_path, 'a') as f:
                f.write("\n".join(new_results)+"\n")
        
        return (total_loss, batch_logits) if return_outputs else total_loss
