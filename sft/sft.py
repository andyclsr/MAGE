from datasets import load_dataset
from transformers import AutoTokenizer

import os
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, HfArgumentParser, TrainingArguments
from trl import SFTTrainer, set_seed, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
import numpy as np
import pandas as pd
from transformers import BitsAndBytesConfig
from datasets import load_from_disk

from utils import load_main_tokenizer, Instructions_summary, build_dataset_summary, Instructions, build_dataset
tqdm.pandas()

# define paths for two datasets
# hhrlhf_dataset_path = '/dataset/Anthropic/hh-rlhf'
hhrlhf_dataset_path = '/dataset/hh-rlhf'
summary_dataset_path = '/dataset/summarize_from_feedback'


@dataclass
class ScriptArguments:
    log_with: Optional[str] = field(default='none', metadata={"help": "use 'wandb' to log with wandb"})
    save_directory: Optional[str] = field(default='./logs_trl/')
    learning_rate: Optional[float] = field(default=1.4e-4, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=1, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "the number of gradient accumulation steps"})
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "loading model in 8 bit or bfloat16"})
    wandb_name: Optional[str] = field(default='summary_sft_all_bs1_lora64', metadata={"help": "Name for this experiment"})
    exp_type: Optional[str] = field(default='summary', metadata={"help": "exp type, 'summary' or 'assistant' "})
    base_model_name: Optional[str] = field(default=None, metadata={"help": "local path to the base model or the huggingface id"})
    max_steps: Optional[int] = field(default=60000, metadata={"help": "max steps"})
    save_steps: Optional[int] = field(default=30000, metadata={"help": "save steps"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type = script_args.exp_type
base_model_name = '/models/llama2-7b-hf'
if script_args.base_model_name is not None:
    base_model_name = script_args.base_model_name
tokenier_name = base_model_name # we use the same tokenizer for the base model
print('base model: ', base_model_name)
os.makedirs(os.path.join(script_args.save_directory, script_args.wandb_name), exist_ok=True)

training_args = TrainingArguments(
        max_steps=script_args.max_steps,  # TODO: ver2 for llama is 60000
        output_dir=os.path.join(script_args.save_directory, script_args.wandb_name),
        dataloader_drop_last=True,
        eval_steps=70000,
        save_steps=script_args.save_steps,
        logging_steps=10,
        save_strategy='steps',
        per_device_train_batch_size=script_args.batch_size,
        per_device_eval_batch_size=script_args.batch_size,
        learning_rate=script_args.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=0,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=False,
        weight_decay=0.01,
        run_name=script_args.wandb_name,
        report_to=script_args.log_with,
        ddp_find_unused_parameters=False,
    )

process_id = Accelerator().local_process_index 
gpu_id = process_id 
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))


# set seed before initializing value head for deterministic eval
set_seed(8888)
current_device = Accelerator().local_process_index
print(current_device)

lora_config = LoraConfig(
    r=64, 
    lora_alpha=128, 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer = load_main_tokenizer(tokenier_name)
if 'gpt2' not in base_model_name:
    if script_args.load_in_8bit:
        quantization_config=BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            device_map=f'cuda:{gpu_id}', 
            quantization_config=quantization_config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            torch_dtype=torch.bfloat16,
            device_map=gpu_id, 
        )
else:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        device_map=gpu_id, 
    )
    
model.resize_token_embeddings(len(tokenizer))

if exp_type == 'assistant':
    dataset = build_dataset(hhrlhf_dataset_path, tokenizer, split='train') 
    response_template_ids = tokenizer.encode(Instructions.response_split, add_special_tokens=False)[1:]  # TODO:?
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer, mlm=False)
else:
    dataset = build_dataset_summary(summary_dataset_path, tokenizer, split='train')
    response_template_ids = tokenizer.encode(Instructions_summary.response_split, add_special_tokens=False)[1:]  # TODO:?
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer, mlm=False)
train_dataset = dataset.shuffle()
print(f"Size of the train set: {len(train_dataset)}")
# Size of the train set: 145728

# ac

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=lora_config if 'gpt2' not in base_model_name else None,
    packing=False,
    dataset_text_field="query",
    data_collator=collator,
)

print("Training........")
trainer.train()

if process_id == 0:
    print("Saving last checkpoint of the model")
    save_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'model')
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    

