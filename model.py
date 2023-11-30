import math
from typing import List, Optional, Tuple, Union
from pprint import pprint

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers import LlamaModel, LlamaForCausalLM, LlamaPreTrainedModel, LlamaTokenizer
from transformers import BertModel, BertPreTrainedModel


class LlamaRewardModel(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)        
        self.reward_head = nn.Linear(config.hidden_size, 1, bias=False)        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def floating_point_ops(self, inputs):
        return 0

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pooling_type: str = "average",
        padding_side: str = "right",
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(dim=-1)).to(hidden_states.device)
            else:
                sequence_lengths = -1

        if attention_mask is None:
            attention_mask = torch.ne(input_ids, self.config.pad_token_id).float()

        # print("hidden_states shape {}".format(hidden_states.shape))
        # print("attention_mask shape {}".format(attention_mask.shape))

        attention_mask_ext = attention_mask.unsqueeze(-1)
        if pooling_type in ["last", "eos"]:
            offset = 1 if pooling_type == "eos" else 2
            if padding_side == "right":
                pooled_hidden_state = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths - offset]
            else:
                pooled_hidden_state = hidden_states[torch.arange(batch_size, device=hidden_states.device), - offset]

        elif pooling_type == "average":
            pooled_hidden_state = (hidden_states * attention_mask_ext).sum(dim=1) / attention_mask_ext.sum(dim=1)
        elif pooling_type == "max":
            pooled_hidden_state = (hidden_states * attention_mask_ext).max(dim=1)[0]
        else:
            raise ValueError("The pooling method {} is not implemented!!".format(pooling_type))

        pooled_logits = self.reward_head(pooled_hidden_state)
        
        #pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        return {
            "lm_logits": lm_logits,
            "rm_logits": pooled_logits,
            "hidden_states": transformer_outputs[0],
            "rm_embeddings": pooled_hidden_state
        }
            

        # return SequenceClassifierOutputWithPast(
        #     loss=loss,
        #     logits=pooled_logits,
        #     past_key_values=transformer_outputs.past_key_values,
        #     hidden_states=transformer_outputs.hidden_states,            
        #     attentions=transformer_outputs.attentions,
        # )


class BertRewardModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = BertModel(config)
        self.reward_head = nn.Linear(config.hidden_size, 1, bias=False)        
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pooling_type: str = "average",
        padding_side: str = "right",
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print("debug inputs", input_ids.shape)
        # print("debug attention", attention_mask.shape)

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(dim=-1)).to(hidden_states.device)
            else:
                sequence_lengths = -1

        if attention_mask is None:
            attention_mask = torch.ne(input_ids, self.config.pad_token_id).float()

        # print("hidden_states shape {}".format(hidden_states.shape))
        # print("attention_mask shape {}".format(attention_mask.shape))

        attention_mask_ext = attention_mask.unsqueeze(-1)
        if pooling_type == "last":
            if padding_side == "right":
                pooled_hidden_state = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths-1]
            else:
                pooled_hidden_state = hidden_states[torch.arange(batch_size, device=hidden_states.device), -1]

        elif pooling_type == "average":
            pooled_hidden_state = (hidden_states * attention_mask_ext).sum(dim=1) / attention_mask_ext.sum(dim=1)
        elif pooling_type == "max":
            pooled_hidden_state = (hidden_states * attention_mask_ext).max(dim=1)[0]
        else:
            raise ValueError("The pooling method {} is not implemented!!".format(pooling_type))

        pooled_logits = self.reward_head(pooled_hidden_state)
        
        #pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(hidden_states.device)
            if self.config.problem_type is None:
                self.config.problem_type = "regression"
                # elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                #     self.config.problem_type = "single_label_classification"
                # else:
                #     self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())

            elif self.config.problem_type == "ranking":
                pass
            
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )





if __name__ == '__main__':
    

    
    with open("/apdcephfs_cq3/share_1567347/pengyucheng/workspace/alpaca_datasets/test_text_examples.txt", 'r') as f:
        lines = f.read().split('\n')
    lines = [line for line in lines if len(line) > 10]
    print("finished loading {} lines".format(len(lines)))

    model = LlamaRewardModel.from_pretrained(
        "/apdcephfs_cq3/share_1567347/pengyucheng/saved_models/llama-7b-hf",         
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print(model)

    tokenizer = LlamaTokenizer.from_pretrained("/apdcephfs_cq3/share_1567347/pengyucheng/saved_models/llama-7b-hf")
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference


    batch_size = 32
    num_batches = len(lines) // 32
    batches = [lines[i*batch_size : (i+1)*batch_size] for i in range(num_batches)]

    inputs = tokenizer(batches[0], padding="max_length", max_length=64)

    with torch.autocast("cuda"):     
        input_ids = torch.Tensor(inputs['input_ids']).long().to("cuda")
        attention_mask = torch.Tensor(inputs['attention_mask']).to(torch.float16).to("cuda")

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_type="max",
            padding_side=tokenizer.padding_side
        )

        print(output.logits)    




