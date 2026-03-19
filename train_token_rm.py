from datasets import load_dataset
from transformers import AutoTokenizer

import torch
from torch import nn
# from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import TrainingArguments,AdamW, AutoTokenizer, \
            AutoConfig, DataCollatorWithPadding, get_cosine_schedule_with_warmup, BitsAndBytesConfig, \
            AutoModelForCausalLM
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset, load_from_disk
from token_rm.modeling_llama import LlamaModelForTokenScore
from torch.nn import functional as F
from transformers import HfArgumentParser
from dataclasses import dataclass, field, asdict
from accelerate import Accelerator
import tyro
from trl import set_seed, DataCollatorForCompletionOnlyLM
from typing import Optional
import os
import gc
from tqdm import tqdm
import shutil
from trl import PPOTrainer, PPOConfig
import time
import logging
from peft import LoraConfig


tqdm.pandas()

from utils import build_dataset_assist_explicit_rm_4train, build_dataset_summary_explicit_rm_4train,  Instructions, Instructions_summary, \
    vas_build_evaluate_summary_dataset_half, vas_build_evaluate_summary_dataset_whole

summary_dataset_path = "RiC/ppo/vas_dataset/summary_dataset_none_gen_frac1"
hhrlhf_dataset_path = "RiC/ppo/vas_dataset/assist_dataset_none_gen_frac1"

summary_old_path = 'dataset/summarize_from_feedback'

summary_valid_dataset_path = 'RiC/ppo/vas_dataset/summary_valid_rm_judge_2000'
assist_valid_dataset_path = 'RiC/ppo/vas_dataset/assist_valid_rm_judge_2000'

half_summary_valid_dataset_path = 'RiC/ppo/vas_dataset/summary_valid_dataset_2000'
half_assist_valid_dataset_path = 'RiC/ppo/vas_dataset/assist_valid_dataset_200'


@dataclass
class ScriptArguments:
    reward_name:Optional[str] = field(default='harmless,helpful') 
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "experiment type"})
    run_name: Optional[str] = field(default='token_rm_summary_faith_llama')
    frac: Optional[float] = field(default=1.0, metadata={"help": "fraction of the dataset to use"})
    logging_path: Optional[str] = field(default='RiC/ppo/logs_vas')
    model_name: Optional[str]=field(default='models/llama2-7b-hf')
    output_dataset_path: Optional[str] = field(default='dataset/vas_summary_sft.csv')
    mini_batch_size: Optional[int] = field(default=4, metadata={"help": "mini batch size"})
    is_log: Optional[bool] = field(default=False, metadata={"help": "whether to log"})
    output_dir: Optional[str] = field(default='RiC/ppo/logs_vas/token_rm_summary_faith_llama/token_rm_summary_faith_llama')
    learning_rate : Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    num_epochs: Optional[int] = field(default=4, metadata={"help": "the number of epochs"})
    gradient_accumulation_steps: Optional[int] = field(default=2, metadata={"help": "gradient accumulation steps"})
    
    copy_head: Optional[bool] = field(default=False, metadata={"help": "whether to copy head"})
    dataset_path: Optional[str] = field(default=None, metadata={"help": "dataset path"})
    use_lora: Optional[bool] = field(default=False, metadata={"help": "whether to use lora"})

    validate: Optional[bool] = field(default=False, metadata={"help": "whether to validate"})

parser = HfArgumentParser(ScriptArguments)
script_args = tyro.cli(ScriptArguments)

set_seed(8888)

reward_name = script_args.reward_name
exp_type = script_args.exp_type

# 定义超参数
learning_rate = script_args.learning_rate
num_epochs = script_args.num_epochs

accelerator = Accelerator(gradient_accumulation_steps = script_args.gradient_accumulation_steps)
gpu_id = accelerator.local_process_index

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
tokenizer.pad_token = tokenizer.bos_token
tokenizer.padding_side = "left"


# 加载自定义模型
####################################################
####################################################
# save_path = 'models/token_rm_summary_faith_llama.pth' 
# TODO: mention here token_rm_summary_faith_llama.pth is llama2-7b-hf model


# script_args.model_name = 'RiC/ppo/logs_vas/token_rm_summary_pref1_llama/token_rm_summary_pref1_llama_b2_1'
# script_args.model_name = 'RiC/ppo/logs_vas/token_rm_summary_faith_llama/token_rm_summary_faith_llama_b2_best'
# script_args.model_name = 'RiC/ppo/logs_vas/token_rm_summary_pref1_llama/token_rm_summary_pref1_llama_b1_no8bit_401'

if script_args.use_lora:
    lora_config = LoraConfig(
        r=64, 
        lora_alpha=128, 
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = LlamaModelForTokenScore.from_pretrained(
        script_args.model_name,
        load_in_8bit=True,
        torch_dtype=torch.bfloat16, # XXX original is float16
        peft_config=lora_config,
        # use_lora= script_args.use_lora,
        # dtype=torch.float16,
        # device_map=gpu_id,
        # quantization_config=quantization_config
    )
else:
    model = LlamaModelForTokenScore.from_pretrained(
        script_args.model_name,
        # load_in_8bit=True,
        torch_dtype=torch.float16, # XXX original is float16
        # dtype=torch.float16,
        # device_map=gpu_id,
        # quantization_config=quantization_config
    )
if script_args.copy_head:
    tmp_model = AutoModelForCausalLM.from_pretrained(script_args.model_name, torch_dtype=torch.float16)
    model.score_head.load_state_dict(tmp_model.lm_head.state_dict())
    del tmp_model

    

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
# optimizer = AdamW(model.parameters(), lr=learning_rate) # XXX original

####################################################
####################################################


#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
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
def initialize():
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
        return log
    return None

log = initialize()
log_dict(**asdict(script_args))
log_info('starting...')
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#


def compute_loss(
    input_ids: torch.LongTensor,  # size = (B, L)
    attention_mask: torch.BoolTensor,  # size = (B, L)
    labels: torch.LongTensor,  # size = (B, L)
    ref_rewards: torch.FloatTensor,  # size = (B,)
) -> dict[str, torch.Tensor]:
    
    batch_size = input_ids.size(0)

    output = model(
        input_ids=input_ids,  # size = (B, L)
        attention_mask=attention_mask,  # size = (B, L)
    )
    logits = output.logits 

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
    
    return loss


######################################################################
######################################################################
# TODO: check whether we need to do it differently with assist task


def evaluate_for_whole_sentence():
    valid_dataset = vas_build_evaluate_summary_dataset_whole(summary_old_path, tokenizer, frac=0.1)
    valid_data_loader = DataLoader(valid_dataset, batch_size=1, drop_last=True)
    valid_data_loader = accelerator.prepare(valid_data_loader)

    name_base = 'models/token_rm_summary_faith_llama' 
    # model_list = [name_base+"_1.pth", name_base+"_2.pth", name_base+"_3.pth"]
    model_list = [name_base+'.pth', name_base+"_1.pth", name_base+"_2.pth", name_base+"_3.pth"]

    log_info('padding_id', tokenizer.pad_token_id)

    for name in model_list:
        md = LlamaModelForTokenScore.from_pretrained(
            script_args.model_name,
            load_in_8bit=True,
        )
        md.load_state_dict(torch.load(name, map_location='cpu'), strict=False)
        md = accelerator.prepare(md)
        acc = []
        for batch in valid_data_loader:
            max_len = max(batch['chosen_input_ids'].size(1), batch['rejected_input_ids'].size(1))
            
            ####################################################
            # pad_for_chosen = max_len-batch['chosen_input_ids'].size(1)
            # pad_for_reject = max_len-batch['rejected_input_ids'].size(1)
            # batch['chosen_input_ids'] = F.pad(batch['chosen_input_ids'], (0, max_len-batch['chosen_input_ids'].size(1)), value=tokenizer.pad_token_id)
            # batch['rejected_input_ids'] = F.pad(batch['rejected_input_ids'], (0, max_len-batch['rejected_input_ids'].size(1)), value=tokenizer.pad_token_id)
            # input_ids = torch.cat([batch['chosen_input_ids'], batch['rejected_input_ids']], dim=0)
            # attention_mask = torch.ones(input_ids.size(), dtype=torch.long)
            # attention_mask[0,:pad_for_chosen] = 0
            # attention_mask[1,:pad_for_reject] = 0
            
            # output = md(input_ids.cuda(), attention_mask=attention_mask.cuda(), return_dict=True)
            # end_scores = output.end_scores.to(torch.float32)
            # acc.append(end_scores[0]>end_scores[1])
            ####################################################
            
            
            ####################################################
            attention_mask_chosen = torch.ones(batch['chosen_input_ids'].size(), dtype=torch.long)
            output_chosen = md(input_ids= batch['chosen_input_ids'], attention_mask = attention_mask_chosen, return_dict=True)
            end_score_chosen = output_chosen.end_scores.to(torch.float32)
            
            attention_mask_rejected = torch.ones(batch['rejected_input_ids'].size(), dtype=torch.long)
            output_rejected = md(input_ids = batch['rejected_input_ids'], attention_mask =attention_mask_rejected, return_dict=True)
            end_score_rejected = output_rejected.end_scores.to(torch.float32)
            
            acc.append(end_score_chosen[0]>end_score_rejected[0])
            ####################################################
            
        
        acc = accelerator.gather(torch.stack(acc)).float()
        if accelerator.is_main_process:
            log_info(f"Model {name} valid acc: {acc.mean()}")
        del md
        gc.collect()
        torch.cuda.empty_cache()

    import sys
    sys.exit()
######################################################################
######################################################################


######################################################################
######################################################################
# TODO: wrong implementation
# valid_dataset = vas_build_evaluate_summary_dataset_whole(summary_old_path, tokenizer, frac=0.1)
# valid_data_loader = DataLoader(valid_dataset, batch_size=1, drop_last=True)
# valid_data_loader = accelerator.prepare(valid_data_loader)

def evaluate_for_whole_sentence_model():

    acc = []
    with torch.no_grad():
        for batch in valid_data_loader:
            ####################################################
            attention_mask_chosen = torch.ones(batch['chosen_input_ids'].size(), dtype=torch.long)
            output_chosen = model(input_ids= batch['chosen_input_ids'], attention_mask = attention_mask_chosen, return_dict=True)
            end_score_chosen = output_chosen.end_scores.to(torch.float32)
            
            attention_mask_rejected = torch.ones(batch['rejected_input_ids'].size(), dtype=torch.long)
            output_rejected = model(input_ids = batch['rejected_input_ids'], attention_mask =attention_mask_rejected, return_dict=True)
            end_score_rejected = output_rejected.end_scores.to(torch.float32)
            
            log_info('end_score_chosen', end_score_chosen, 'end_score_rejected', end_score_rejected)
            acc.append(end_score_chosen[0]>end_score_rejected[0])
            ####################################################
        
    
    acc = accelerator.gather(torch.stack(acc)).float()
    return acc.mean()
######################################################################
######################################################################

if exp_type == 'assistant':
    valid_dataset_name = assist_valid_dataset_path
    half_valid_dataset_name = half_assist_valid_dataset_path
else:
    valid_dataset_name = summary_valid_dataset_path
    half_valid_dataset_name = half_summary_valid_dataset_path

valid_dataset = load_from_disk(valid_dataset_name).shuffle()
valid_dataset = valid_dataset.select(range(len(valid_dataset)//10))
valid_data_loader = DataLoader(valid_dataset, batch_size=1, drop_last=True)
valid_data_loader = accelerator.prepare(valid_data_loader)

# log_info('len: ',len(valid_dataset))

    
def fixed_evaluate_for_whole_sentence_model():
    with torch.no_grad():
        acc = []
        mse_loss_list = []
        pbar = tqdm(total=len(valid_data_loader))
        for batch in valid_data_loader:
            end_score_list = []
            ground_truth_list = []
            for i in range(2):
                input_ids = tokenizer.encode(batch[f'response_{i}'][0], return_tensors='pt').to(accelerator.device)
                attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(accelerator.device)
                output = model(input_ids = input_ids, attention_mask = attention_mask, return_dict=True)
                end_score = output.end_scores.to(torch.float32)
                end_score_list.append(end_score)
                ground_truth_list.append(batch[f'reward_{i}_{reward_name}'])
            
            end_score_list = torch.tensor(end_score_list, dtype=torch.float32).to(accelerator.device)
            ground_truth_list = torch.tensor(ground_truth_list, dtype=torch.float32).to(accelerator.device)
            
            mse_loss = F.mse_loss(end_score_list, ground_truth_list, reduction='mean')
            mse_loss_list.append(mse_loss)
        
            # Calculate pairwise correspondence for accuracy
            for i in range(len(end_score_list)):
                for j in range(i + 1, len(end_score_list)):
                    # Compare the pairwise relationships
                    end_relation = end_score_list[i] > end_score_list[j]
                    ground_truth_relation = ground_truth_list[i] > ground_truth_list[j]
                    
                    # Check if the pairwise relationships match
                    # acc.append(torch.tensor(end_relation == ground_truth_relation, dtype=torch.float32).to(accelerator.device))
                    acc.append((end_relation == ground_truth_relation).float().to(accelerator.device))
            
            pbar.update(1)

        mse_loss = accelerator.gather(torch.stack(mse_loss_list))
        # log_info('mse_loss len:', len(mse_loss))
        mse_loss = mse_loss.mean().item()
        acc = accelerator.gather(torch.stack(acc)).float().mean().item()
    return mse_loss, acc

######################################################################
######################################################################
# TODO: half-response evaluation

half_valid_dataset = vas_build_evaluate_summary_dataset_half(half_valid_dataset_name, tokenizer)
half_valid_data_loader = DataLoader(half_valid_dataset, batch_size=1, drop_last=True)
half_valid_data_loader = accelerator.prepare(half_valid_data_loader)


def fixed_evaluate_for_half_sentence_model():
    with torch.no_grad():
        pbar = tqdm(total=len(half_valid_data_loader))
        end_score_list = []
        ground_truth_list = []
        for batch in half_valid_data_loader:
            input_ids = batch['input_ids']
            attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(accelerator.device)
            output = model(input_ids = input_ids, attention_mask = attention_mask, return_dict=True)
            end_score = output.end_scores.to(torch.float32)
            end_score_list.append(end_score)
            ground_truth_list.append(batch[f'reward_{reward_name}'])
            pbar.update(1)
            
            
        end_score_list = torch.tensor(end_score_list, dtype=torch.float32).to(accelerator.device)
        ground_truth_list = torch.tensor(ground_truth_list, dtype=torch.float32).to(accelerator.device)
        
        end_score_list = accelerator.gather(end_score_list)
        ground_truth_list = accelerator.gather(ground_truth_list)
        
        mse_loss = F.mse_loss(end_score_list, ground_truth_list, reduction='mean').item()

    return mse_loss



if script_args.validate:
    name_base = script_args.output_dir
    model_list = [name_base+"_1", name_base+"_2", name_base+"_3", name_base+'_best']

    for name in model_list:
        model = LlamaModelForTokenScore.from_pretrained(
            name,
            load_in_8bit=True,
        )
        model = accelerator.prepare(model)
        valid_loss = fixed_evaluate_for_half_sentence_model()
        mse_loss, acc = fixed_evaluate_for_whole_sentence_model()
        log_info(f"Half: Model {name} valid half loss: {valid_loss}, valid acc: {acc}, valid mse loss: {mse_loss}")

    import sys
    sys.exit()
######################################################################
######################################################################

# model = accelerator.prepare(model)
# mse_loss, acc = fixed_evaluate_for_whole_sentence_model()
# log_info(f"Model valid acc: {acc}, valid mse loss: {mse_loss}")
# import sys
# sys.exit()





tfrac = None
if script_args.frac < 1.0 :
    tfrac = script_args.frac
    
if exp_type == 'assistant':
    if script_args.dataset_path is not None:
        hhrlhf_dataset_path = script_args.dataset_path
    train_dataset = build_dataset_assist_explicit_rm_4train(hhrlhf_dataset_path, tokenizer, frac=tfrac)
    train_dataset = train_dataset.shuffle()
    instructions = Instructions()
    response_template_ids = tokenizer.encode(Instructions.response_split, add_special_tokens=False)[1:]  
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer, mlm=False)
else:
    train_dataset = build_dataset_summary_explicit_rm_4train(summary_dataset_path, tokenizer, frac=tfrac)
    train_dataset = train_dataset.shuffle()
    instructions = Instructions_summary()
    response_template_ids = tokenizer.encode(Instructions_summary.response_split, add_special_tokens=False)[1:]  
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer, mlm=False)


train_data_loader = DataLoader(train_dataset, batch_size=script_args.mini_batch_size, drop_last=True, collate_fn=collator)
train_data_loader, model, optimizer = accelerator.prepare(train_data_loader, model, optimizer)
scheduler = CosineAnnealingLR(optimizer, T_max=len(train_data_loader), eta_min=0)
# scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_data_loader)//20, num_training_steps=len(train_data_loader))
scheduler = accelerator.prepare(scheduler)

######################################################################
# config = PPOConfig(model_name = 'test')
# ppo_trainer = PPOTrainer(model = model, config=config, tokenizer=tokenizer, dataset = train_dataset, data_collator=collator, optimizer=optimizer)
######################################################################

pbar = tqdm(total=num_epochs*len(train_dataset) // script_args.mini_batch_size // accelerator.num_processes)
best_acc = 0.0
# 训练循环
for epoch in range(num_epochs):
    model.train()

    for idx, batch in enumerate(train_data_loader):
        with accelerator.accumulate(model):
            input_ids, attention_mask, labels, ref_rewards = batch["input_ids"], batch["attention_mask"], \
                batch["labels"], batch[f"reward_{script_args.reward_name}"]

            loss = compute_loss(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels, 
                ref_rewards=ref_rewards, 
            )

            # 反向传播和优化
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            # if (idx+1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            loss_item = accelerator.gather_for_metrics(loss.detach())
            mean_loss = loss_item.mean().item()
            
            if idx % 20 == 0:
                log_info(f"Epoch{epoch}, Batch {idx + 1}/{len(train_data_loader)}, Loss: {mean_loss :.4f}, lr: {scheduler.get_last_lr()[0]}")
            if idx % 100 == 0:
                mse_loss, acc = fixed_evaluate_for_whole_sentence_model()
                half_loss = fixed_evaluate_for_half_sentence_model()
                log_info(f"Epoch {epoch}, Batch {idx + 1}/{len(train_data_loader)}, valid acc: {acc}, valid mse loss: {mse_loss}, valid half loss: {half_loss}")
                if accelerator.is_main_process and idx > 0:
                    if acc > best_acc:
                        best_acc = acc
                        unwrapped_model =  accelerator.unwrap_model(model)
                        # if os.path.exists(script_args.output_dir+f'_no8bit_{idx+1}'):
                        #     shutil.rmtree(script_args.output_dir+f'_no8bit_{idx+1}')
                            
                        if os.path.exists(script_args.output_dir+'_best'):
                            shutil.rmtree(script_args.output_dir+'_best')
                        # accelerator.save_model(model, script_args.output_dir+'_best')
                        unwrapped_model.save_pretrained(script_args.output_dir+'_best')
                        log_info("Best model saved.")
                
            pbar.update(1)
        # gc.collect()
        # torch.cuda.empty_cache()
    
    accelerator.wait_for_everyone()
    # acc = fixed_evaluate_for_whole_sentence_model()
    if accelerator.is_main_process:
        # log_info(f"Epoch {epoch}, valid acc: {acc}")
        # if acc > best_acc:
        # unwrapped_model =  accelerator.unwrap_model(model)
        # state_dict = unwrapped_model.state_dict()
        # cleaned_state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
        # torch.save(cleaned_state_dict, script_args.output_dir+f'_{epoch+1}.pth')
        if os.path.exists(script_args.output_dir+f'_{epoch+1}'):
            shutil.rmtree(script_args.output_dir+f'_{epoch+1}')
        accelerator.unwrap_model(model).save_pretrained(script_args.output_dir+f'_{epoch+1}')

# if accelerator.is_main_process:
#     # save the model of the last epoch
#     unwrapped_model =  accelerator.unwrap_model(model)
#     state_dict = unwrapped_model.state_dict()
#     cleaned_state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
#     torch.save(cleaned_state_dict, script_args.output_dir+'last.pth')