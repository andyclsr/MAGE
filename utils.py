from datasets import load_dataset
from transformers import AutoTokenizer

import os
import gc
import copy
import shutil
from dataclasses import dataclass, field
from typing import Optional
from peft import PeftModel
from accelerate import Accelerator
from transformers import AutoTokenizer, LlamaTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
import torch
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk, disable_caching
import numpy as np
import pandas as pd
from tqdm import tqdm
from inference_tools.modeling_llama import LlamaModelForTokenScore

disable_caching()

def clean_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class Instructions:
    response_split = "\n\nAssistant:"
    input_split = "\n\nHuman:"

    @staticmethod
    def get_input(query):
        before_response = Instructions.response_split.join(query.split(Instructions.response_split)[:-1])
        return before_response.rstrip() + ' ' + Instructions.response_split
        
    @staticmethod
    def get_response(response):
        return response.split(Instructions.response_split)[-1].strip()


class Instructions_summary():
    instruction_summary = "Generate a one-sentence summary of this post."
    response_split = "### Response:"
    input_split = "### Input:"
    instruction_split = "### Instruction:"

    @classmethod
    def prompt_input(self, input):
        # formulate the news
        return f"### Instruction: {Instructions_summary.instruction_summary} ### Input: {input} ### Response: "

    def get_prompt(self, query):
        before_response = self.response_split.join(query.split(self.response_split)[:-1])
        return before_response.rstrip() 

    def get_post(self, query):
        before_response = self.get_prompt(query)
        return before_response.split(self.input_split)[1].strip()

    def get_input(self, query):
        return self.get_prompt(query) + ' ' + self.response_split
        
    def get_response(self, response):
        return response.split(self.response_split)[-1].strip()


def build_dataset(path, tokenizer, rm_tokenizer, split='train', size=None, frac = None):
    # ds = load_dataset(path, split=split)
    ds = load_from_disk(path)
    ds = ds[split]

    if size is not None:
        ds = ds.select(range(size))
    
    if frac is not None:
        selected_indices = np.random.choice(len(ds), int(len(ds) * frac), replace=False)
        ds = ds.select(selected_indices)

    
    def tokenize(sample, reject=False):
        if not reject:
            sample['text'] = sample['chosen'] 
        else:
            sample['text'] = sample['rejected'] 
        split_text = sample['text'].split('\n\nAssistant:')
        sample['prompt'] = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        sample['reward_ids'] = rm_tokenizer.encode(sample['text']) # for data filter
        return sample

    ds_concat = ds.map(tokenize, batched=False, fn_kwargs={"reject": False}, num_proc=30)
    ds_concat = ds_concat.filter(lambda x: len(x["input_ids"]) <= 256 and len(x["input_ids"]) >= 8 and len(x['reward_ids']) <= 256 and len(x['reward_ids']) >= 8)
    ds_concat = ds_concat.remove_columns(['rejected', 'chosen', 'reward_ids', 'text']) # TODO: remember to restore it



    ds_concat.set_format(type="torch")
    return ds_concat


def build_dataset_summary(path, tokenizer, rm_tokenizer, split='train', size=None, frac=None):
    ds = load_from_disk(path)
    # ds = load_dataset(path, 'comparisons')
    ds = ds[split]

    if frac is not None:
        selected_indices = np.random.choice(len(ds), int(len(ds) * frac), replace=False)
        ds = ds.select(selected_indices)

    ds = ds.filter(lambda x: x["info"]['post'] is not None and 100 < len(x["info"]['post']) < 1200, batched=False, num_proc=30)

    if size is not None:
        ds = ds.select(range(size))

    def tokenize(sample):
        info_post = sample["info"]["post"].replace("\n", " ")
        prompt_summary = Instructions_summary.prompt_input(info_post)
        sample["prompt"] = prompt_summary
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False,  num_proc=30) 
    ds = ds.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    remove_columns = ['info', 'summaries', 'choice', 'worker', 'batch', 'split', 'extra']
    ds = ds.remove_columns(remove_columns)
    ds.set_format(type="torch")
    return ds


def build_dataset_eval_frac(path, tokenizer, rm_tokenizers_list, split='test', size=None, frac=None, part_idx=None, part_nums=None):
    # ds = load_dataset(path, split=split)

    ds = load_from_disk(path)
    ds = ds[split]
    
    if frac is not None:
        selected_indices = np.random.choice(len(ds), int(len(ds) * frac), replace=False)
        ds = ds.select(selected_indices)
    if size is not None:
        ds = ds.select(range(size))
    else:
        ds = ds.select(range(0, len(ds), 4))
    rm_tokenizer1, rm_tokenizer2 = rm_tokenizers_list[:2]
    def tokenize(sample):
        sample['text'] = sample['chosen'] 
        split_text = sample['text'].split('\n\nAssistant:')
        sample['prompt'] = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
        sample['response'] = split_text[-1].strip()
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        sample["input_ids_rm1"] = rm_tokenizer1.encode(sample["prompt"])
        sample["input_ids_rm2"] = rm_tokenizer2.encode(sample["prompt"])
        return sample

    ds_chosen = ds.map(tokenize, batched=False, num_proc=20)
    ds_concat = ds_chosen
    ds_concat = ds_concat.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8 \
                                 and len(x["input_ids_rm1"]) <= 512 and len(x["input_ids_rm1"]) >= 8
                                 and len(x["input_ids_rm2"]) <= 512 and len(x["input_ids_rm2"]) >= 8 )
    ds_concat = ds_concat.remove_columns(['chosen', 'rejected','input_ids_rm1', 'input_ids_rm2', 'text', 'prompt', 'response', 'query'])
    ds_concat.set_format(type="torch")
    if part_idx is not None and part_nums is not None:
        part_lens = len(ds_concat) // part_nums
        ds_concat = ds_concat.select(range(part_idx * part_lens, (part_idx + 1) * part_lens))
    return ds_concat





def build_dataset_summary_eval_frac(path, tokenizer, rm_tokenizers, split='test', size=None, frac=None, part_idx=None, part_nums=None):
    if split == 'test':
        split = 'validation'
    ds = load_from_disk(path)
    ds = ds[split]
    # ds = ds.select(range(0,len(ds),100)) # TODO remove this line!!!!! important
    ds = ds.filter(lambda x: x["info"]['post'] is not None and 100 < len(x["info"]['post']) < 1200, batched=False, num_proc=30)

    # need to remove duplicated prompts for evaluation
    def remove_duplicate(duplicated_dataset):
        duplicated_dataset = duplicated_dataset.filter(lambda x: x['info']["id"] is not None)
        initial_list = duplicated_dataset.map(lambda x: {"id": x['info']["id"]})
        _ , unique_indices = np.unique(initial_list["id"], return_index=True, axis=0)
        filtered_dataset = duplicated_dataset.select(unique_indices.tolist())
        return filtered_dataset

    ds = remove_duplicate(ds)
    if size is not None:
        ds = ds.select(range(size))
    print('log: before select min2000, the size of the dataset:', len(ds))
    ds = ds.select(range(0, min(len(ds),2000))) # select 2000 data 
    
    if frac is not None:
        selected_indices = np.random.choice(len(ds), int(len(ds) * frac), replace=False)
        ds = ds.select(selected_indices)

    def tokenize(sample):
        info_post = sample["info"]["post"].replace("\n", " ")
        prompt_summary = Instructions_summary.prompt_input(info_post)
        sample["prompt"] = prompt_summary
        sample["input_ids"] = tokenizer.encode(prompt_summary)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False,  num_proc=30) 
    ds = ds.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    remove_columns = ['info', 'summaries', 'choice', 'worker', 'batch', 'split', 'extra']
    ds = ds.remove_columns(remove_columns)
    ds.set_format(type="torch")
    if part_idx is not None and part_nums is not None:
        part_lens = len(ds_concat) // part_nums
        ds_concat = ds_concat.select(range(part_idx * part_lens, (part_idx + 1) * part_lens))
    return ds







def build_dataset_eval(path, tokenizer, rm_tokenizers_list, split='test', size=None, frac=None):
    # ds = load_dataset(path, split=split)

    ds = load_from_disk(path)
    ds = ds[split]
    
    
    if frac is not None:
        selected_indices = np.random.choice(len(ds), int(len(ds) * frac), replace=False)
        ds = ds.select(selected_indices)
    if size is not None:
        ds = ds.select(range(size))
    else:
        ds = ds.select(range(0, len(ds), 4))
    rm_tokenizer1, rm_tokenizer2 = rm_tokenizers_list[:2]
    def tokenize(sample):
        sample['text'] = sample['chosen'] 
        split_text = sample['text'].split('\n\nAssistant:')
        sample['prompt'] = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
        sample['response'] = split_text[-1].strip()
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        sample["input_ids_rm1"] = rm_tokenizer1.encode(sample["prompt"])
        sample["input_ids_rm2"] = rm_tokenizer2.encode(sample["prompt"])
        return sample

    ds_chosen = ds.map(tokenize, batched=False, num_proc=20)
    ds_concat = ds_chosen
    ds_concat = ds_concat.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8 \
                                 and len(x["input_ids_rm1"]) <= 512 and len(x["input_ids_rm1"]) >= 8
                                 and len(x["input_ids_rm2"]) <= 512 and len(x["input_ids_rm2"]) >= 8 )
    ds_concat = ds_concat.remove_columns(['chosen', 'rejected','input_ids_rm1', 'input_ids_rm2', 'text', 'prompt', 'response', 'query'])
    ds_concat.set_format(type="torch")
    return ds_concat


def build_dataset_summary_eval(path, tokenizer, rm_tokenizers, split='test', size=None, frac=None):
    if split == 'test':
        split = 'validation'
    ds = load_from_disk(path)
    ds = ds[split]
    # ds = ds.select(range(0,len(ds),100)) # TODO remove this line!!!!! important
    ds = ds.filter(lambda x: x["info"]['post'] is not None and 100 < len(x["info"]['post']) < 1200, batched=False, num_proc=30)

    # need to remove duplicated prompts for evaluation
    def remove_duplicate(duplicated_dataset):
        duplicated_dataset = duplicated_dataset.filter(lambda x: x['info']["id"] is not None)
        initial_list = duplicated_dataset.map(lambda x: {"id": x['info']["id"]})
        _ , unique_indices = np.unique(initial_list["id"], return_index=True, axis=0)
        filtered_dataset = duplicated_dataset.select(unique_indices.tolist())
        return filtered_dataset

    ds = remove_duplicate(ds)
    if size is not None:
        ds = ds.select(range(size))
    print('log: before select min2000, the size of the dataset:', len(ds))
    ds = ds.select(range(0, min(len(ds),2000))) # select 2000 data 
    
    if frac is not None:
        selected_indices = np.random.choice(len(ds), int(len(ds) * frac), replace=False)
        ds = ds.select(selected_indices)

    def tokenize(sample):
        info_post = sample["info"]["post"].replace("\n", " ")
        prompt_summary = Instructions_summary.prompt_input(info_post)
        sample["prompt"] = prompt_summary
        sample["input_ids"] = tokenizer.encode(prompt_summary)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False,  num_proc=30) 
    ds = ds.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    remove_columns = ['info', 'summaries', 'choice', 'worker', 'batch', 'split', 'extra']
    ds = ds.remove_columns(remove_columns)
    ds.set_format(type="torch")
    return ds








def check_lora_in_model_path(model, path):
    if os.path.exists(path):
        dirnames = os.listdir(path)
        if 'adapter_config.json' in dirnames:
            return True
        
        state_dict_keys = model.state_dict().keys()
        for key in state_dict_keys:
            if 'lora' in key:
                return True
    return False


def build_dataset_assist_explicit_rm(path, tokenizer, rm_tokenizer, split='train', size=None, frac=None):
    # ds = load_dataset(path, split=split)
    ds = load_from_disk(path)
    ds = ds[split]

    if size is not None:
        ds = ds.select(range(size))
    
    if frac is not None:
        selected_indices = np.random.choice(len(ds), int(len(ds) * frac), replace=False)
        ds = ds.select(selected_indices)

    
    def tokenize(sample, reject=False):
        sample['text'] = sample['chosen'] 
        split_text = sample['text'].split('\n\nAssistant:')
        sample['prompt'] = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        sample['response_0'] = tokenizer.encode(sample['prompt'] + split_text[-1].strip())

        sample['text'] = sample['rejected']
        split_text = sample['text'].split('\n\nAssistant:')
        sample['response_1'] = tokenizer.encode(sample['prompt'] + split_text[-1].strip())
        
        sample["query"] = tokenizer.decode(sample["input_ids"])
        sample['reward_ids'] = rm_tokenizer.encode(sample['text']) # for data filter
        return sample

    ds_concat = ds.map(tokenize, batched=False, fn_kwargs={"reject": False}, num_proc=30)
    ds_concat = ds_concat.filter(lambda x: len(x["input_ids"]) <= 256 and len(x["input_ids"]) >= 8 and len(x['reward_ids']) <= 256 and len(x['reward_ids']) >= 8)
    ds_concat = ds_concat.remove_columns(['rejected', 'chosen', 'reward_ids', 'text'])



    ds_concat.set_format(type="torch")
    return ds_concat


def build_dataset_summary_explicit_rm(path, tokenizer, rm_tokenizer, split='train', size=None, frac=None):
    ds = load_from_disk(path)
    # ds = load_dataset(path, 'comparisons')
    ds = ds[split]

    if frac is not None:
        selected_indices = np.random.choice(len(ds), int(len(ds) * frac), replace=False)
        ds = ds.select(selected_indices)

    ds = ds.filter(lambda x: x["info"]['post'] is not None and 100 < len(x["info"]['post']) < 1200, batched=False, num_proc=30)

    if size is not None:
        ds = ds.select(range(size))

    def tokenize(sample):
        info_post = sample["info"]["post"].replace("\n", " ")
        prompt_summary = Instructions_summary.prompt_input(info_post)
        sample["prompt"] = prompt_summary
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        sample["query"] = tokenizer.decode(sample["input_ids"])
        sample['response_0'] = tokenizer.encode(sample["summaries"][0]['text'])
        sample['response_1'] = tokenizer.encode(sample["summaries"][1]['text'])
        return sample

    ds = ds.map(tokenize, batched=False,  num_proc=30) 
    ds = ds.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    remove_columns = ['info', 'summaries', 'choice', 'worker', 'batch', 'split', 'extra']
    ds = ds.remove_columns(remove_columns)
    ds.set_format(type="torch")
    return ds


def build_dataset_assist_explicit_rm_4train(path, tokenizer, size=None, frac=None):
    ds = load_from_disk(path)

    if frac is not None:
        selected_indices = np.random.choice(len(ds), int(len(ds) * frac), replace=False)
        ds = ds.select(selected_indices)

    if size is not None:
        ds = ds.select(range(size))

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["generated_text"])
        return sample

    ds = ds.map(tokenize, batched=False,  num_proc=30) 
    remove_columns = ["prompt", "generated_text"] # TODO: remembert to restore it
    ds = ds.remove_columns(remove_columns) # TODO: remembert to restore it
    ds.set_format(type="torch")
    return ds

def build_dataset_summary_explicit_rm_4train(path, tokenizer, size=None, frac=None):
    ds = load_from_disk(path)['train']

    if frac is not None:
        selected_indices = np.random.choice(len(ds), int(len(ds) * frac), replace=False)
        ds = ds.select(selected_indices)

    if size is not None:
        ds = ds.select(range(size))

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["generated_text"])
        return sample

    ds = ds.map(tokenize, batched=False,  num_proc=30) 
    remove_columns = ["prompt", "generated_text"]
    ds = ds.remove_columns(remove_columns)
    ds.set_format(type="torch")
    return ds


def load_reward_model(reward_peft_path, gpu_id):
    num_labels = 2 if ('humor' in reward_peft_path or 'faithful' in reward_peft_path) else 1
    reward_model = AutoModelForSequenceClassification.from_pretrained(
                    reward_peft_path,
                    num_labels=num_labels, torch_dtype=torch.bfloat16,
                    device_map=gpu_id,
                    )
    if check_lora_in_model_path(reward_model, reward_peft_path):
        reward_model = PeftModel.from_pretrained(reward_model, reward_peft_path)
    if hasattr(reward_model, 'merge_and_unload'):
        reward_model = reward_model.merge_and_unload() # merge lora weights
    return reward_model.to(gpu_id)


def load_main_tokenizer(tokenier_name):
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>" 
    DEFAULT_UNK_TOKEN = "<unk>" 

    # tokenizer = AutoTokenizer.from_pretrained(tokenier_name, use_fast = False)
    tokenizer = AutoTokenizer.from_pretrained(tokenier_name)#, use_fast = False)
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
    return tokenizer


def get_rewards(reward_model, texts_for_rewards, reward_mean_std=None, sub_position=0):
    rewards = []
    # print('log: reward model forwarding ...') # TODO: here changed
    with torch.no_grad():
        # pbar = tqdm(total=len(texts_for_rewards))
        for inputs in texts_for_rewards:
            if sub_position != 0: # for multiple output
                rewards.append(reward_model(**(inputs.to(reward_model.device))).logits[0][sub_position])
            else:
                rewards.append(reward_model(**(inputs.to(reward_model.device))).logits[0])
            # pbar.update(1)
    
    if reward_mean_std is None:
        rewards = [np.round(r.cpu().detach().item(), 5) for r in rewards]
    else:
        mean_reward, std_reward = reward_mean_std
        rewards = [np.round((r.cpu().detach().item() - mean_reward) / std_reward, 5) for r in rewards]
    return rewards



def save_configs(config, path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'training_config.txt'), 'w+') as f:
        if type(config) == dict:
            lines = [key + ' : ' + config[key] + '\n' for key in config.keys()]
            f.writelines(lines)
        else:
            f.writelines(str(config))


def get_average_state_dict(state_dicts, coefficients):
    i = 0
    with torch.no_grad():
        for state_dict, coefficient in zip(state_dicts, coefficients):
            current_weights = state_dict
            for key in list(current_weights.keys()):
                if i == 0:
                    state_dicts[0][key] = coefficient * current_weights[key]
                else:
                    state_dicts[0][key] += coefficient * current_weights[key]
            i += 1
    return state_dicts[0]

@torch.no_grad()
def merge_weights_with_preference(base_model_names, preference, temp_save_path):
    models = []
    for base_model_name in base_model_names:
        model_tmp = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map='cpu',
        )
        models.append(model_tmp)
    
    state_dicts = [model_tmp.state_dict() for model_tmp in models]
    average_weights = get_average_state_dict(state_dicts, preference)
    model_1 = models[0]
    model_1.load_state_dict(average_weights, strict=False)
    if os.path.exists(temp_save_path):
        shutil.rmtree(temp_save_path, ignore_errors=True)
    model_1.save_pretrained(temp_save_path)

    while len(models):
        del models[0]
    while len(state_dicts):
        del state_dicts[0]
    del average_weights
    gc.collect()
    torch.cuda.empty_cache()


from torch.nn.utils import parameters_to_vector, vector_to_parameters

@torch.no_grad()
def normalize(x, dim=0):
    min_values, _ = torch.min(x, dim=dim, keepdim=True)
    max_values, _ = torch.max(x, dim=dim, keepdim=True)
    y = (x - min_values) / (max_values - min_values)
    return y

@torch.no_grad()
def clamp(x, min_ratio=0, max_ratio=0):
    if len(x.size())==1:
        d = x.size(0)
        sorted_x, _ = torch.sort(x)
        min=sorted_x[int(d * min_ratio)]
        max=sorted_x[int(d * (1-max_ratio)-1)]
    else:
        d = x.size(1)
        sorted_x, _ = torch.sort(x, dim=1)
        min=sorted_x[:, int(d * min_ratio)].unsqueeze(1)
        max=sorted_x[:, int(d * (1-max_ratio)-1)].unsqueeze(1)
    clamped_x= torch.clamp(x, min, max)
    return clamped_x

@torch.no_grad()
def act(x):
    y = torch.tanh(x)  # torch.relu(x)
    return y

@torch.no_grad()
def get_average_param_vector(tvs, preference):
    preference = preference.view(-1, 1)
    weighted_sum = torch.sum(tvs * preference, dim=0)
    return weighted_sum

@torch.no_grad()
def PCB_merge(flat_task_checks, preference, pcb_ratio=0.1):
    all_checks = flat_task_checks.clone()
    n, d = all_checks.shape
    all_checks_abs = clamp(torch.abs(all_checks), min_ratio=0.0001, max_ratio=0.0001)
    clamped_all_checks = torch.sign(all_checks)*all_checks_abs
    self_pcb = normalize(all_checks_abs, 1)**2
    self_pcb_act = torch.exp(n*self_pcb)
    cross_pcb = all_checks * torch.sum(all_checks, dim=0)
    cross_pcb_act = act(cross_pcb)
    task_pcb = self_pcb_act * cross_pcb_act

    scale = normalize(clamp(task_pcb, 1-pcb_ratio, 0), dim=1) # TODO: here dim=1?
    tvs = clamped_all_checks
    merged_tv = get_average_param_vector(tvs * scale, preference) / torch.clamp(torch.sum(scale, dim=0), min=1e-12)
    return merged_tv


from model_merging_methods.merging_methods import MergingMethod

@torch.no_grad()
def merge_weights_with_advanced_merging(base_model_names, preference, args, is_preprocess=False, is_merge=False, to_merge_dict=None):
    models_to_merge = []
    for base_model_name in base_model_names:
        model_tmp = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map='cpu',
        )
        models_to_merge.append(model_tmp)
    
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path3,
        device_map='cpu',
    )
    
    merging_method = MergingMethod(merging_method_name=args.merging_method_name)
    merged_model = pretrained_model
    merged_model = merging_method.get_merged_model(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   preference=preference,
                                                   exclude_param_names_regex=[],
                                                   trainers=[None for _ in range(len(base_model_names))],
                                                   scaling_coefficient=args.scaling_coefficient,
                                                   nums_fisher_examples=None,
                                                   fisher_scaling_coefficients=None,
                                                   normalize_fisher_weight=None,
                                                   minimal_fisher_weight=None,
                                                   nums_regmean_examples=None,
                                                   reduce_non_diagonal_ratio=None,
                                                   param_value_mask_rate=0.8, # TODO: change this
                                                   weight_format=args.weight_format,
                                                   weight_mask_rates=args.weight_mask_rates,
                                                   use_weight_rescale=args.use_weight_rescale,
                                                   mask_strategy=args.mask_strategy,
                                                   mask_apply_method=args.mask_apply_method,
                                                   models_use_deepcopy=False,
                                                   is_preprocess=is_preprocess,
                                                   is_merge = is_merge,
                                                   to_merge_dict = to_merge_dict
                                                   )

    # para_vector_list = [parameters_to_vector(model_tmp.parameters()) for model_tmp in models_to_merge]
    # flat_task_checks = torch.stack(para_vector_list, dim=0)
    
    # merged_tv = PCB_merge(flat_task_checks, preference, pcb_ratio=0.1)
    # vector_to_parameters(merged_tv, models_to_merge[0].parameters())

    while len(models_to_merge)>1:
        del models_to_merge[len(models_to_merge)-1]
    del pretrained_model
    gc.collect()
    torch.cuda.empty_cache()
    return merged_model

@torch.no_grad()
def efficient_merge_weights_with_preference(base_model_names, preference, temp_save_path):
    models = []
    for base_model_name in base_model_names:
        model_tmp = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map='cpu',
        )
        models.append(model_tmp)
    
    state_dicts = [model_tmp.state_dict() for model_tmp in models]
    average_weights = get_average_state_dict(state_dicts, preference)
    models[0].load_state_dict(average_weights, strict=False)

    while len(models)>1:
        del models[len(models)-1]
    while len(state_dicts):
        del state_dicts[0]
    del average_weights
    gc.collect()
    torch.cuda.empty_cache()
    return models[0]


@torch.no_grad()
def efficient_merge_weights_with_preference_acc(base_model_names, preference, temp_save_path):
    models = []
    for base_model_name in base_model_names:
        model_tmp = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map='cpu',
        )
        models.append(model_tmp)
    
    state_dicts = [model_tmp.state_dict() for model_tmp in models]
    average_weights = get_average_state_dict(state_dicts, preference)
    model_tmp = AutoModelForCausalLM.from_pretrained(
        base_model_names[0],
        device_map='auto',
    )
    model_tmp.load_state_dict(average_weights, strict=False)

    while len(models):
        del models[len(models)-1]
    while len(state_dicts):
        del state_dicts[0]
    del average_weights
    gc.collect()
    torch.cuda.empty_cache()
    return model_tmp

@torch.no_grad()
def efficient_merge_weights_with_preference_bf16(base_model_names, preference, temp_save_path):
    models = []
    for base_model_name in base_model_names:
        model_tmp = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map='cpu',
        )
        models.append(model_tmp)
    
    state_dicts = [model_tmp.state_dict() for model_tmp in models]
    average_weights = get_average_state_dict(state_dicts, preference)
    models[0].load_state_dict(average_weights, strict=False)

    while len(models)>1:
        del models[len(models)-1]
    while len(state_dicts):
        del state_dicts[0]
    del average_weights
    gc.collect()
    torch.cuda.empty_cache()
    return models[0]


@torch.no_grad()
def efficient_merge_explicit_token_rm(base_model_names, preference, temp_save_path, device):
    models = []
    for base_model_name in base_model_names:
        model_tmp = LlamaModelForTokenScore.from_pretrained(
            base_model_name,
            device_map='cpu',
        )
        models.append(model_tmp)
    
    state_dicts = [model_tmp.state_dict() for model_tmp in models]
    average_weights = get_average_state_dict(state_dicts, preference)
    
    model_to_return = LlamaModelForTokenScore.from_pretrained(
        base_model_names[0],
        device_map=device,
        load_in_8bit= True
        # torch_dtype=torch.float16,
    )
    model_to_return.load_state_dict(average_weights, strict=False)

    while len(models):
        del models[len(models)-1]
    while len(state_dicts):
        del state_dicts[0]
    del average_weights
    gc.collect()
    torch.cuda.empty_cache()
    return model_to_return


@torch.no_grad()
def efficient_merge_explicit_token_rm_bf16(base_model_names, preference, temp_save_path, device):
    models = []
    for base_model_name in base_model_names:
        model_tmp = LlamaModelForTokenScore.from_pretrained(
            base_model_name,
            device_map='cpu',
        )
        models.append(model_tmp)
    
    state_dicts = [model_tmp.state_dict() for model_tmp in models]
    average_weights = get_average_state_dict(state_dicts, preference)
    
    model_to_return = LlamaModelForTokenScore.from_pretrained(
        base_model_names[0],
        device_map=device,
        load_in_8bit= True
        # torch_dtype=torch.float16,
    )
    model_to_return.load_state_dict(average_weights, strict=False)

    while len(models):
        del models[len(models)-1]
    while len(state_dicts):
        del state_dicts[0]
    del average_weights
    gc.collect()
    torch.cuda.empty_cache()
    return model_to_return


@torch.no_grad()
def efficient_merge_value_model(base_model_names, preference, temp_save_path, device):
    models = []
    for base_model_name in base_model_names:
        model_tmp = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map='cpu',
        )
        models.append(model_tmp)
    
    state_dicts = [model_tmp.state_dict() for model_tmp in models]
    average_weights = get_average_state_dict(state_dicts, preference)
    
    model_to_return = AutoModelForCausalLM.from_pretrained(
        base_model_names[0],
        device_map=device,
        load_in_8bit= True
    )
    model_to_return.load_state_dict(average_weights, strict=False)

    while len(models):
        del models[len(models)-1]
    while len(state_dicts):
        del state_dicts[0]
    del average_weights
    gc.collect()
    torch.cuda.empty_cache()
    return model_to_return


@torch.no_grad()
def efficient_merge_value_model_bf16(base_model_names, preference, temp_save_path, device):
    models = []
    for base_model_name in base_model_names:
        model_tmp = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map='cpu',
        )
        models.append(model_tmp)
    
    state_dicts = [model_tmp.state_dict() for model_tmp in models]
    average_weights = get_average_state_dict(state_dicts, preference)
    
    model_to_return = AutoModelForCausalLM.from_pretrained(
        base_model_names[0],
        device_map=device,
        load_in_8bit= True
    )
    model_to_return.load_state_dict(average_weights, strict=False)

    while len(models):
        del models[len(models)-1]
    while len(state_dicts):
        del state_dicts[0]
    del average_weights
    gc.collect()
    torch.cuda.empty_cache()
    return model_to_return



@torch.no_grad()
def value_head_efficient_merge_value_model(base_model_names, preference, temp_save_path, device):
    # we use value head trl model to return only difference lie in the type of model_to_return
    models = []
    for base_model_name in base_model_names:
        model_tmp = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map='cpu',
        )
        models.append(model_tmp)
    
    state_dicts = [model_tmp.state_dict() for model_tmp in models]
    average_weights = get_average_state_dict(state_dicts, preference)
    
    model_to_return = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model_names[0],
        device_map=device,
        load_in_8bit= True
    )
    model_to_return.load_state_dict(average_weights, strict=False)

    while len(models):
        del models[len(models)-1]
    while len(state_dicts):
        del state_dicts[0]
    del average_weights
    gc.collect()
    torch.cuda.empty_cache()
    return model_to_return


@torch.no_grad()
def efficient_merge_value_model_enhanced(base_model_names, preference, temp_save_path, device):
    pass


def merge_lora_weight(model, path):
    if check_lora_in_model_path(model, path):
        model = PeftModel.from_pretrained(model, path)
        model = model.merge_and_unload()
    return model


def get_clean_data(full_responses, full_prompts, remove_bad=False):
    full_prompts_clean = []
    full_responses_clean = []
    for i, response in enumerate(full_responses):
        full_prompts[i] = full_prompts[i].strip('[PAD] ').strip('[PAD]').strip('<s>').strip('</s>').strip()
        response = response.strip('[PAD] ').strip('[PAD]').strip('<s>').strip('</s>')
        temp_resp = response.replace(full_prompts[i], '').strip().strip('\n\n----').strip('\n\n----- ').strip()
        if '</s>' in temp_resp:
            temp_resp = temp_resp[:temp_resp.rindex('</s>')]
        temp_resp = temp_resp.split('\n\nHuman:')[0].strip()
        temp_resp = temp_resp.split('\nHuman:')[0].strip()
        temp_resp = temp_resp.split('\n\nAssistant:')[0].strip()
        temp_resp = temp_resp.split('\nAssistant:')[0].strip()
        temp_resp = temp_resp.split('\n\n\n')[0].strip()
        clean_resp = full_prompts[i] + ' ' + temp_resp
        if remove_bad and (('.....' in clean_resp) or (clean_resp.count(':)') >= 3)):
            ## pass bad sample
            continue
        full_responses_clean.append(clean_resp)
        full_prompts_clean.append(full_prompts[i])
    return full_prompts_clean, full_responses_clean



def vas_build_dataset(path, tokenizer, rm_tokenizer, split='train', size=None, frac = None):
    # ds = load_dataset(path, split=split)
    ds = load_from_disk(path)
    ds = ds[split]

    if size is not None:
        ds = ds.select(range(size))
    
    if frac is not None:
        selected_indices = np.random.choice(len(ds), int(len(ds) * frac), replace=False)
        ds = ds.select(selected_indices)

    
    def tokenize(sample, reject=False):
        if not reject:
            sample['text'] = sample['chosen'] 
        else:
            sample['text'] = sample['rejected'] 
        split_text = sample['text'].split('\n\nAssistant:')
        sample['prompt'] = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
        sample["input_ids"] = tokenizer.encode(sample["prompt"])# TODO:here original [:-1]
        sample['response'] = tokenizer.encode(split_text[-1].strip())# TODO:here original [1:]
        sample['reward_ids'] = rm_tokenizer.encode(sample['text']) # for data filter
        sample['processed_text'] = tokenizer.decode(tokenizer.encode(sample['text']))
        return sample

    ds_concat = ds.map(tokenize, batched=False, fn_kwargs={"reject": False}, num_proc=30)
    ds_concat = ds_concat.filter(lambda x: len(x["input_ids"]) <= 256 and len(x["input_ids"]) >= 8 and len(x['reward_ids']) <= 256 and len(x['reward_ids']) >= 8)
    ds_concat = ds_concat.remove_columns(['rejected', 'chosen', 'reward_ids', 'text'])


    # we only need input_ids, response and processed_text

    ds_concat.set_format(type="torch")
    return ds_concat


def vas_build_dataset_summary(path, tokenizer, rm_tokenizer, split='train', size=None, frac=None):
    ds = load_from_disk(path)
    # ds = load_dataset(path, 'comparisons')
    ds = ds[split]

    if frac is not None:
        selected_indices = np.random.choice(len(ds), int(len(ds) * frac), replace=False)
        ds = ds.select(selected_indices)

    ds = ds.filter(lambda x: x["info"]['post'] is not None and 100 < len(x["info"]['post']) < 1200, batched=False, num_proc=30)

    if size is not None:
        ds = ds.select(range(size))

    def tokenize(sample):
        info_post = sample["info"]["post"].replace("\n", " ")
        prompt_summary = Instructions_summary.prompt_input(info_post)
        sample["prompt"] = prompt_summary
        sample["input_ids"] = tokenizer.encode(sample["prompt"]) # TODO:here original [:-1]
        sample['response'] = tokenizer.encode(sample["summaries"][0]['text']) # TODO:here original [1:]
        sample['processed_text'] = tokenizer.decode(tokenizer.encode(prompt_summary+sample["summaries"][0]['text']))
        return sample

    # we only need input_ids, response and processed_text

    # print(ds[0])
    ds = ds.map(tokenize, batched=False,  num_proc=30) 
    ds = ds.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    remove_columns = ['info', 'summaries', 'choice', 'worker', 'batch', 'split', 'extra']
    ds = ds.remove_columns(remove_columns)
    ds.set_format(type="torch")
    return ds


def vas_build_evaluate_summary_dataset_half(path, tokenizer, frac=None):
    # ds = load_dataset("csv", data_files=path)['train']
    ds = load_from_disk(path)
    if frac is not None:
        selected_indices = np.random.choice(len(ds), int(len(ds) * frac), replace=False)
        ds = ds.select(selected_indices)
    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["partial_response"]) # TODO:here original [1:]
        return sample

    ds = ds.map(tokenize, batched=False,  num_proc=30)
    removed_columns = ['partial_response', 'prompt']
    ds = ds.remove_columns(removed_columns)
    ds.set_format(type="torch")
    return ds


def vas_build_evaluate_summary_dataset_whole(path, tokenizer, frac=None):
    # ds = load_dataset("csv", data_files=path)['train']
    ds = load_from_disk(path)['validation']
    ds = ds.filter(lambda x: x["info"]['post'] is not None and 100 < len(x["info"]['post']) < 1200, batched=False, num_proc=30)

    # need to remove duplicated prompts for evaluation
    def remove_duplicate(duplicated_dataset):
        duplicated_dataset = duplicated_dataset.filter(lambda x: x['info']["id"] is not None)
        initial_list = duplicated_dataset.map(lambda x: {"id": x['info']["id"]})
        _ , unique_indices = np.unique(initial_list["id"], return_index=True, axis=0)
        filtered_dataset = duplicated_dataset.select(unique_indices.tolist())
        return filtered_dataset

    ds = remove_duplicate(ds)
    print('log: before select min2000, the size of the dataset:', len(ds))
    if frac is not None:
        selected_indices = np.random.choice(len(ds), int(len(ds) * frac), replace=False)
        ds = ds.select(selected_indices)
    def tokenize(sample):
        info_post = sample["info"]["post"].replace("\n", " ")
        prompt_summary = Instructions_summary.prompt_input(info_post)
        sample["prompt"] = prompt_summary
        chosen_id = sample["choice"]
        sample["chosen_input_ids"] = tokenizer.encode(sample["prompt"]+sample["summaries"][chosen_id]['text'])
        sample["rejected_input_ids"] = tokenizer.encode(sample["prompt"]+sample["summaries"][1-chosen_id]['text'])
        return sample

    ds = ds.map(tokenize, batched=False,  num_proc=30)
    remove_columns = ['info', 'summaries', 'worker', 'batch', 'split', 'extra', 'prompt']
    ds = ds.remove_columns(remove_columns)
    ds.set_format(type="torch")
    return ds