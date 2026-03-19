import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import TrainingArguments, AdamW, AutoTokenizer
from datasets import load_dataset
from modeling_llama import LlamaModelForTokenScore
from torch.nn import functional as F
from transformers import HfArgumentParser
from dataclasses import dataclass, field, asdict
from accelerate import Accelerator
import tyro
from trl import set_seed, DataCollatorForCompletionOnlyLM
from typing import Optional
import os

from RiC.ppo.utils import build_dataset_assist_explicit_rm_4train, build_dataset_summary_explicit_rm_4train,  Instructions, Instructions_summary

summary_dataset_path = "/RiC/ppo/vas_dataset/summary_dataset_none_gen_frac1"
hhrlhf_dataset_path = "/RiC/ppo/vas_dataset/hhrlhf_dataset_none_gen_frac1"


@dataclass
class ScriptArguments:
    reward_name:Optional[str] = field(default='harmless,helpful') 
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "experiment type"})
    run_name: Optional[str] = field(default='token_rm_summary_faith_llama')
    frac: Optional[float] = field(default=1.0, metadata={"help": "fraction of the dataset to use"})
    logging_path: Optional[str] = field(default='/RiC/ppo/logs_vas')
    model_name: Optional[str]=field(default='/models/Llama-2-7b-hf')
    output_dataset_path: Optional[str] = field(default='/dataset/vas_summary_sft.csv')
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "mini batch size"})
    is_log: Optional[bool] = field(default=False, metadata={"help": "whether to log"})
    output_dir: Optional[str] = field(default='/models/token_rm_summary_faith_llama')
    learning_rate : Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    num_epochs: Optional[int] = field(default=3, metadata={"help": "the number of epochs"})
        

parser = HfArgumentParser(ScriptArguments)
script_args = tyro.cli(ScriptArguments)

set_seed(8888)

reward_name = script_args.reward_name
exp_type = script_args.exp_type

# 定义超参数
learning_rate = script_args.learning_rate
num_epochs = script_args.num_epochs

accelerator = Accelerator()
gpu_id = accelerator.local_process_index

# 加载自定义模型
model = LlamaModelForTokenScore.from_pretrained(script_args.model_name)
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)

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


def compute_loss(
    self,
    input_ids: torch.LongTensor,  # size = (B, L)
    attention_mask: torch.BoolTensor,  # size = (B, L)
    labels: torch.LongTensor,  # size = (B, L)
    ref_rewards: torch.FloatTensor,  # size = (B,)
) -> dict[str, torch.Tensor]:
    
    batch_size = input_ids.size(0)

    output = self.model(
        input_ids=input_ids,  # size = (B, L)
        attention_mask=attention_mask,  # size = (B, L)
    )
    logits = output.logits 

    if self.args.loss_type == "mse":
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_logits = shift_logits.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)  # size = (B, L-1)
        losses = []
        for i in range(batch_size):
            ref_reward_label = ref_rewards[i]
            truncated_rewards = shift_logits[i, labels[i][1:] != -100]
            ref_reward_label = ref_reward_label.to(truncated_rewards.dtype)
            mseloss = F.mse_loss(truncated_rewards, ref_reward_label.repeat(truncated_rewards.size(0)), reduction='mean') 
            losses.append(mseloss)
        loss = torch.stack(losses).mean()
    else:
        raise NotImplementedError
    return loss


tfrac = None
if script_args.frac < 1.0 :
    tfrac = script_args.frac
    
if exp_type == 'assistant':
    train_dataset = build_dataset_assist_explicit_rm_4train(hhrlhf_dataset_path, tokenizer, frac=tfrac)
    instructions = Instructions()
    response_template_ids = tokenizer.encode(Instructions.response_split, add_special_tokens=False)[1:]  
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer, mlm=False)
else:
    train_dataset = build_dataset_summary_explicit_rm_4train(summary_dataset_path, tokenizer, frac=tfrac)
    instructions = Instructions_summary()
    response_template_ids = tokenizer.encode(Instructions_summary.response_split, add_special_tokens=False)[1:]  
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer, mlm=False)

    
train_data_loader = DataLoader(train_dataset, batch_size=script_args.mini_batch_size, drop_last=True, collate_fn=collator)
train_data_loader,model = accelerator.prepare(train_data_loader, model)

# 优化器
optimizer = AdamW(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    model.train()

    for idx, batch in enumerate(train_data_loader):
        input_ids, attention_mask, labels, ref_rewards = batch["input_ids"], batch["attention_mask"], \
            batch["labels"], batch[f"reward_{script_args.reward_name}"]

        loss = compute_loss(
            model=model, 
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            ref_rewards=ref_rewards, 
            loss_type="mse",
        )

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_info(f"Epoch {idx + 1}/{len(train_data_loader)}, Loss: {loss.item():.4f}")

# 保存模型
model.save_pretrained(script_args.output_dir)
