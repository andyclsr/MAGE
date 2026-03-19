from datasets import load_dataset
from transformers import AutoTokenizer

import os
import gc
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import HfArgumentParser
from transformers import AutoModelForCausalLM, DataCollatorWithPadding
from torch.utils.data import DataLoader
from trl import set_seed
import tyro
import numpy as np
import pandas as pd

from utils import  build_dataset_eval, build_dataset_summary_eval, load_main_tokenizer, Instructions, Instructions_summary, \
                    get_clean_data

from multi_reward_models import RewardModels
tqdm.pandas()

hhrlhf_dataset_path = 'dataset/hh-rlhf'
summary_dataset_path = 'dataset/summarize_from_feedback'
tokenizer_path = 'models/llama2-7b-hf'


@dataclass
class ScriptArguments:
    reward_names:Optional[str] = field(default='harmless,helpful') 
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "experiment type"})
    run_name: Optional[str] = field(default='vas-summary-sft')
    frac: Optional[float] = field(default=1.0, metadata={"help": "fraction of the dataset to use"})
    decode: Optional[bool] = field(default=False, metadata={"help": "whether to decode"})
    logging_path: Optional[str] = field(default='RiC/ppo/logs_vas')
    model_name: Optional[str]=field(default='models/summary_sft_merged')
    output_dataset_path: Optional[str] = field(default='dataset/vas_summary_sft.csv')
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "mini batch size"})
    repeat_times: Optional[int] = field(default=10, metadata={"help": "repeat times"})
        

parser = HfArgumentParser(ScriptArguments)
script_args = tyro.cli(ScriptArguments)

accelerator = Accelerator()
process_id = Accelerator().local_process_index 
gpu_id = process_id

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
reward_path_tokenizer_dict = {
    'harmless': ['Ray2333/gpt2-large-harmless-reward_model'],
    'helpful': ['Ray2333/gpt2-large-helpful-reward_model'],
    'deberta': ['OpenAssistant/reward-model-deberta-v3-large-v2'],
    'preference2': ['OpenAssistant/reward-model-deberta-v3-large-v2'],
    'summary': ['Tristan/gpt2_reward_summarization'],
    'preference1': ['Tristan/gpt2_reward_summarization'],
    'faithful':['CogComp/bart-faithful-summary-detector'],
    'humor': ['mohameddhiab/humor-no-humor'],
}
reward_model_path_list = []
rm_tokenizer_path_list = []
prefix = 'models/'
for name in reward_names:
    if name not in reward_path_tokenizer_dict.keys():
        raise NotImplementedError
    reward_model_path_list.append(prefix+ reward_path_tokenizer_dict[name][0])
    rm_tokenizer_path_list.append(prefix+ reward_path_tokenizer_dict[name][0])
    
reward_model = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id)
rm_tokenizer = AutoTokenizer.from_pretrained(rm_tokenizer_path_list[0])


set_seed(8888)

tokenier_name = tokenizer_path
exp_type = script_args.exp_type

    
tfrac = None
if script_args.frac < 1.0 :
    tfrac = script_args.frac

tokenizer = load_main_tokenizer(tokenier_name)
if exp_type == 'assistant':
    train_dataset = build_dataset_eval(hhrlhf_dataset_path, tokenizer, reward_model.rm_tokenizers, split='test', frac=tfrac)
    instructions = Instructions()
else:
    train_dataset = build_dataset_summary_eval(summary_dataset_path, tokenizer, reward_model.rm_tokenizers, split='test', frac=tfrac)
    instructions = Instructions_summary()
    
# only remove when the key is in the dataset
if 'prompt' in train_dataset.column_names:
    train_dataset = train_dataset.remove_columns('prompt')
if 'query' in train_dataset.column_names:
    train_dataset = train_dataset.remove_columns('query')

for key in ['key', 'text', 'response']:
    if key in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns(key)
        
train_dataset = train_dataset.shuffle()
print(f"Size of the train set: {len(train_dataset)}")

model_name = script_args.model_name
model = AutoModelForCausalLM.from_pretrained(model_name,
        # device_map=gpu_id,
        # use_cache=True,
        load_in_8bit=True,
    )
model.eval()
model = accelerator.prepare(model)

tokenizer.paddding_side = "left"

## TODO: mention tokenizer pad token, pad side, eos token, add special token

generation_kwargs = {
    "max_new_tokens": 128 if exp_type == 'assistant' else 48,
    'min_length': -1, 
    "top_k": 0.0,
    "top_p": 1.0, 
    "do_sample": True,
    "temperature": 0.7,
    "pad_token_id": tokenizer.pad_token_id,
    # "begin_suppress_tokens": [tokenizer.eos_token_id],
}


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_data_loader = DataLoader(train_dataset, batch_size=script_args.mini_batch_size, drop_last=True, collate_fn=data_collator)
train_data_loader = accelerator.prepare(train_data_loader)

results = []


pbar = tqdm(total=len(train_dataset) // script_args.mini_batch_size // accelerator.num_processes)

with torch.no_grad():
    for i, batch in enumerate(train_data_loader):

        input_ids = batch["input_ids"]
        prompt = tokenizer.decode(input_ids[0])
        response_tensors = accelerator.unwrap_model(model).generate(input_ids, **generation_kwargs)
        start_idx = input_ids.shape[1]
        len_response_tensors = response_tensors.shape[1]-start_idx
        partial_response_tensors = response_tensors[:, :start_idx+len_response_tensors//2]
        partial_response = tokenizer.decode(partial_response_tensors[0], skip_special_tokens=True)
        
        full_responses = []
        full_prompts = []
        
        new_generation_kwargs = {
            "max_new_tokens": 128 if exp_type == 'assistant' else 48,
            'min_length': -1, 
            "top_k": 0.0,
            "top_p": 1.0, 
            "do_sample": True,
            "temperature": 0.7,
            "pad_token_id": tokenizer.pad_token_id,
            # "begin_suppress_tokens": [tokenizer.eos_token_id],
        }
        new_generation_kwargs['max_new_tokens']-=len_response_tensors//2
        for _ in range(script_args.repeat_times):
            response_tensors =  accelerator.unwrap_model(model).generate(partial_response_tensors, **new_generation_kwargs)
            generated_text = tokenizer.decode(response_tensors[0], skip_special_tokens=True)
            full_responses.append(generated_text)
            full_prompts.append(prompt)
            
        full_prompts, full_responses = get_clean_data(full_responses, full_prompts)
        queries_responses = [
            (instructions.get_input(text), instructions.get_response(text))
            for text in full_responses
        ]
        if hasattr(instructions, 'get_post'):
            rewards_list = reward_model.get_reward_model_scores(queries_responses, instructions.get_post)
        else:
            rewards_list = reward_model.get_reward_model_scores(queries_responses)
        
        all_rewards = []
        for i in range(len(reward_names)):
            all_rewards.append(accelerator.gather_for_metrics([np.mean(rewards_list[i])]))
            # if accelerator.is_main_process:
            #     print(f'reward_{reward_names[i]}:', rewards_list[i])
                
        full_prompts = [full_prompts[0]]
        full_responses = [partial_response]
        all_full_prompts = accelerator.gather_for_metrics(full_prompts)
        all_full_responses = accelerator.gather_for_metrics(full_responses)
        
        for idx in range(len(all_full_prompts)):
            item = {
                "prompt": all_full_prompts[idx],
                "partial_response": all_full_responses[idx],
                f"reward_{reward_names[0]}": all_rewards[0][idx],
                f"reward_{reward_names[1]}": all_rewards[1][idx],
            }
            if len(reward_names) > 2:
                for i in range(2, len(reward_names)):
                    item[f"reward_{reward_names[i]}"] = all_rewards[i][idx]
            results.append(item)
            
        if accelerator.is_main_process:
            print('lens: ', len(all_full_prompts))
            print(f'reward_{reward_names[0]}:', all_rewards[0])
            print(f'reward_{reward_names[1]}:', all_rewards[1])
            if len(reward_names) > 2:
                for i in range(2, len(reward_names)):
                    print(f'reward_{reward_names[i]}:', all_rewards[i])
            print('full_prompts:', all_full_prompts[0])
            print('full_responses:', all_full_responses[0])
            print('---'*20)
            
        pbar.update(1)
    

if accelerator.is_main_process:
    output_data = pd.DataFrame(results)

    from datasets import Dataset
    # output_data.to_csv(script_args.output_dataset_path, index=False)
    d = Dataset.from_pandas(output_data)

    d.save_to_disk(script_args.output_dataset_path)