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
from dataclasses import asdict
import numpy as np
import pandas as pd

from utils import  build_dataset_eval, build_dataset_summary_eval, load_main_tokenizer, Instructions, Instructions_summary, \
                    get_clean_data, build_dataset_summary_explicit_rm, build_dataset_assist_explicit_rm

from multi_reward_models import RewardModels
tqdm.pandas()

hhrlhf_dataset_path = '/dataset/hh-rlhf'
summary_dataset_path = '/dataset/summarize_from_feedback'
tokenizer_path = '/models/llama2-7b-hf'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

@dataclass
class ScriptArguments:
    reward_names:Optional[str] = field(default='harmless,helpful') 
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "experiment type"})
    run_name: Optional[str] = field(default='vas-summary-sft')
    frac: Optional[float] = field(default=1.0, metadata={"help": "fraction of the dataset to use"})
    decode: Optional[bool] = field(default=False, metadata={"help": "whether to decode"})
    logging_path: Optional[str] = field(default='/RiC/ppo/logs_vas')
    model_name: Optional[str]=field(default='/models/summary_sft_merged')
    output_dataset_path: Optional[str] = field(default='/dataset/vas_summary_sft.csv')
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "mini batch size"})
    repeat_times: Optional[int] = field(default=10, metadata={"help": "repeat times"})
    is_log: Optional[bool] = field(default=False, metadata={"help": "whether to log"})
        

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
prefix = '/models/'
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

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
import time
import logging

def log_info(*args):
    if accelerator.is_main_process:
        total_str = ''
        for arg in args:
            total_str += str(arg) + ' '
        if script_args.is_log:
            log.info(total_str)
        else:
            print(total_str)

def log_dict(**kwargs):
    if accelerator.is_main_process:
        for key, value in kwargs.items():
            if script_args.is_log:
                log_info(key, ':', value)
            else:
                print(key, ':', value)

# initialize the logger
if gpu_id == 0 and script_args.is_log:
    log_time = time.strftime("%m%d_%H%M%S")
    log_file_name = os.path.join(script_args.logging_path, script_args.run_name, 'log_{}.txt'.format(log_time))
    
    if not os.path.exists(os.path.join(script_args.logging_path, script_args.run_name)):
        os.makedirs(os.path.join(script_args.logging_path, script_args.run_name))

    # 创建 logger
    log = logging.getLogger('vgs')
    log.setLevel(logging.INFO) 
    
    # 配置日志格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 创建控制台 Handler
    console_handler = logging.StreamHandler()  # 输出到标准输出
    console_handler.setLevel(logging.INFO)     # 设置日志级别
    console_handler.setFormatter(formatter)    # 设置日志格式

    # 创建文件 Handler
    file_handler = logging.FileHandler(log_file_name)  # 输出到文件
    file_handler.setLevel(logging.INFO)                  # 设置日志级别
    file_handler.setFormatter(formatter)                 # 设置日志格式

    # 将两个 Handler 添加到 logger
    log.addHandler(console_handler)
    log.addHandler(file_handler)
    
log_dict(**asdict(script_args))
log_info('starting...')
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
    
tfrac = None
if script_args.frac < 1.0 :
    tfrac = script_args.frac

if exp_type == 'assistant':
    train_dataset = build_dataset_assist_explicit_rm(hhrlhf_dataset_path, tokenizer, reward_model.rm_tokenizers[0], split='train', frac=tfrac)
    instructions = Instructions()
else:
    train_dataset = build_dataset_summary_explicit_rm(summary_dataset_path, tokenizer, reward_model.rm_tokenizers, split='train', frac=tfrac)
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
        device_map=gpu_id,
        load_in_8bit=True,
    )
model.eval()


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

        # 对提示文本进行tokenize
        input_ids = batch["input_ids"]
        prompt = tokenizer.decode(input_ids[0])
        # 使用模型生成文本
        full_responses = []
        full_prompts = []
        # if exp_type == 'summary':
        full_prompts.append(prompt)
        full_responses.append(tokenizer.decode(batch['response_0'][0], skip_special_tokens=True))
        
        full_prompts.append(prompt)
        full_responses.append(tokenizer.decode(batch['response_1'][0], skip_special_tokens=True))
            
        for _ in range(script_args.repeat_times):
            response_tensors = model.generate(input_ids, **generation_kwargs)
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
            all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))
        all_full_prompts = accelerator.gather_for_metrics(full_prompts)
        all_full_responses = accelerator.gather_for_metrics(full_responses)
        
        for idx in range(len(all_full_prompts)):
            item = {"prompt": all_full_prompts[idx], "generated_text": all_full_responses[idx]}
            for i in range(len(reward_names)):
                item[f"reward_{reward_names[i]}"] = all_rewards[i][idx]
            results.append(item)
            # results.append({
            #     "prompt": all_full_prompts[idx],
            #     "generated_text": all_full_responses[idx],
            #     f"reward_{reward_names[0]}": all_rewards[0][idx],
            #     f"reward_{reward_names[1]}": all_rewards[1][idx],
            # })
            
        if accelerator.is_main_process:
            log_info('lens: ', len(full_prompts))
            for i in range(len(reward_names)):
                log_info(f'reward_{reward_names[i]}:', rewards_list[i])
            # log_info(f'reward_{reward_names[0]}:', rewards_list[0])
            # log_info(f'reward_{reward_names[1]}:', rewards_list[1])
            log_info('full_prompts:', full_prompts[0])
            for i in range(len(full_prompts)):
                log_info(f'full_responses:{i}', full_responses[i].replace(full_prompts[i], ''))
            log_info('---'*20)
            
        pbar.update(1)
    

if accelerator.is_main_process:
    # 转换为DataFrame
    output_data = pd.DataFrame(results)

    # 保存生成的数据集
    # output_data.to_csv(script_args.output_dataset_path, index=False)
    from datasets import Dataset
    d = Dataset.from_pandas(output_data)

    d.save_to_disk(script_args.output_dataset_path)