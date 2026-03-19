from datasets import load_dataset
from transformers import AutoTokenizer

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import TrainingArguments, AdamW, AutoTokenizer, AutoConfig, DataCollatorWithPadding
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset
from token_rm.modeling_llama import LlamaModelForTokenScore
from torch.nn import functional as F
from transformers import HfArgumentParser
from dataclasses import dataclass, field, asdict
from accelerate import Accelerator
import tyro
from trl import set_seed, DataCollatorForCompletionOnlyLM
from multi_reward_models import RewardModels
from typing import Optional
import os
import gc
from tqdm import tqdm
tqdm.pandas()

from utils import build_dataset_assist_explicit_rm_4train, build_dataset_summary_explicit_rm_4train,  Instructions, Instructions_summary, \
    vas_build_evaluate_summary_dataset_half, vas_build_evaluate_summary_dataset_whole

summary_dataset_path = "RiC/ppo/vas_dataset/summary_dataset_none_gen_frac1"
hhrlhf_dataset_path = "RiC/ppo/vas_dataset/hhrlhf_dataset_none_gen_frac1"

summary_old_path = 'dataset/summarize_from_feedback'

@dataclass
class ScriptArguments:
    reward_names:Optional[str] = field(default='harmless,helpful') 
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "experiment type"})
    run_name: Optional[str] = field(default='rm_summary_faith_llama')
    frac: Optional[float] = field(default=1.0, metadata={"help": "fraction of the dataset to use"})
    logging_path: Optional[str] = field(default='RiC/ppo/logs_vas')
    model_name: Optional[str]=field(default='models/llama2-7b-hf')
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "mini batch size"})
    is_log: Optional[bool] = field(default=True, metadata={"help": "whether to log"})
    learning_rate : Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    summary_valid_dataset_path: Optional[str] = field(default='RiC/ppo/vas_dataset/summary_valid_dataset')
    

parser = HfArgumentParser(ScriptArguments)
script_args = tyro.cli(ScriptArguments)

set_seed(8888)

reward_names = [script_args.reward_names]
exp_type = script_args.exp_type

# 定义超参数
learning_rate = script_args.learning_rate

accelerator = Accelerator()
gpu_id = accelerator.local_process_index


reward_path_tokenizer_dict = {
    'harmless': ['Ray2333/gpt2-large-harmless-reward_model'],
    'helpful': ['Ray2333/gpt2-large-helpful-reward_model'],
    'deberta': ['OpenAssistant/reward-model-deberta-v3-large-v2'],
    'preference2': ['OpenAssistant/reward-model-deberta-v3-large-v2'],
    'summary': ['Tristan/gpt2_reward_summarization'],
    'pref1': ['Tristan/gpt2_reward_summarization'],
    'faithful':['CogComp/bart-faithful-summary-detector'],
    'humor': ['mohameddhiab/humor-no-humor'],
}

prefix = 'models/'
for key in reward_path_tokenizer_dict.keys():
    reward_path_tokenizer_dict[key][0] = prefix + reward_path_tokenizer_dict[key][0]

reward_model_path_list = []
rm_tokenizer_path_list = []
for name in reward_names:
    if name not in reward_path_tokenizer_dict.keys():
        raise NotImplementedError
    reward_model_path_list.append(reward_path_tokenizer_dict[name][0])
    rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name][0])
    
reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id) 

if exp_type == 'assistant':
    instructions = Instructions()
else:
    instructions = Instructions_summary()

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
tokenizer.pad_token = tokenizer.bos_token
tokenizer.padding_side = "left"

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






######################################################################
######################################################################
# XXX: create the valid dataset
valid_dataset = vas_build_evaluate_summary_dataset_whole(summary_old_path, tokenizer, frac=0.1)
valid_data_loader = DataLoader(valid_dataset, batch_size=1, drop_last=True)
valid_data_loader = accelerator.prepare(valid_data_loader)
######################################################################
######################################################################

def evaluate_for_whole_sentence_model():

    acc = []
    with torch.no_grad():
        for batch in valid_data_loader:
            full_responses = [tokenizer.decode(batch['chosen_input_ids'][0], skip_special_tokens=True)]
            queries_responses = [
                (instructions.get_input(text), instructions.get_response(text))
                for text in full_responses
            ]
            if hasattr(instructions, 'get_post'):
                rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
            else:
                rewards_list = reward_models.get_reward_model_scores(queries_responses)
            
            chosen_score = rewards_list[0][0]
            
            ####################################################
            full_responses = [tokenizer.decode(batch['rejected_input_ids'][0], skip_special_tokens=True)]
            queries_responses = [
                (instructions.get_input(text), instructions.get_response(text))
                for text in full_responses
            ]
            if hasattr(instructions, 'get_post'):
                rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
            else:
                rewards_list = reward_models.get_reward_model_scores(queries_responses)
            
            rejected_score = rewards_list[0][0]
            
            acc.append(torch.tensor(float(chosen_score > rejected_score), device=accelerator.device))
        
    acc = accelerator.gather(torch.stack(acc)).float()
    return acc.mean()



######################################################################
######################################################################
# TODO: half-response evaluation

# valid_dataset = vas_build_evaluate_summary_dataset_half(script_args.summary_valid_dataset_path, tokenizer)
# valid_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# valid_data_loader = DataLoader(valid_dataset, batch_size=4, drop_last=True, collate_fn=valid_collator)
# valid_data_loader = accelerator.prepare(valid_data_loader)

# name_base = 'models/token_rm_summary_faith_llama' 
# # model_list = [name_base+"_1.pth", name_base+"_2.pth", name_base+"_3.pth"]
# model_list = [name_base+'.pth', name_base+"_1.pth", name_base+"_2.pth", name_base+"_3.pth"]

# for name in model_list:
#     md = LlamaModelForTokenScore.from_pretrained(
#         script_args.model_name,
#         load_in_8bit=True,
#     )
#     md.load_state_dict(torch.load(name, map_location='cpu'), strict=False)
#     md = accelerator.prepare(md)
#     valid_loss = evaluate_value_model(md, valid_data_loader, reward_name)
#     if accelerator.is_main_process:
#         log_info(f"Model {name} valid loss: {valid_loss}")
#     del md
#     gc.collect()
#     torch.cuda.empty_cache()

# import sys
# sys.exit()
######################################################################
######################################################################

acc = evaluate_for_whole_sentence_model()
log_info(f" valid acc: {acc}")