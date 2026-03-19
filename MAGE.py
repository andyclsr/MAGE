from datasets import load_dataset
from transformers import AutoTokenizer

import os
import gc
from dataclasses import dataclass, field, asdict
from typing import List
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
import time, logging

import numpy as np
import pandas as pd
from utils import get_clean_data, load_main_tokenizer, save_configs, print_trainable_parameters, \
                  merge_weights_with_preference, Instructions, Instructions_summary, build_dataset_eval, build_dataset_summary_eval, efficient_merge_weights_with_preference, \
                  efficient_merge_value_model, efficient_merge_explicit_token_rm, efficient_merge_weights_with_preference_bf16
from multi_reward_models import RewardModels

import torch.distributed as dist


### ivg decoding
from inference_tools.utils import set_seeds, extract_responses
from inference_tools.decoder.decoder import EFTPosthocGenerationMixin, TokenWiseSamplingMixin, BeamTuningWithEFTPosthocGenerationMixin
from inference_tools.scorer import ImplicitRewardScorer, ExplicitRewardScorer, NotImplementedScorer
from inference_tools.model import PrefixPreTrainedWrapper
from inference_tools.modeling_llama import LlamaModelForTokenScore
from inference_tools.modeling_gpt2 import GPT2ModelForTokenScore
# from scripts.alpaca_leaderboard.src.utils import get_chat_prompt_template, get_scorer
###

tqdm.pandas()

class ModelManager:
    def __init__(self):
        self.models = {}
        self.device_map = gpu_id
        self.torch_dtype = torch.float16

    def load_model(self, model_name, model_type, **kwargs):

        if model_name not in self.models:
            model = model_type.from_pretrained(
                model_name,
                # torch_dtype=self.torch_dtype,
                # torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map=self.device_map,
                **kwargs,
            )
            self.models[model_name] = model
        return self.models[model_name]

# define paths for two datasets
# hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
# summary_dataset_path = 'openai/summarize_from_feedback'

# hhrlhf_dataset_path = '/home/pretrainx/dataset/hh-rlhf'
hhrlhf_dataset_path = 'dataset/hh-rlhf'
summary_dataset_path = 'dataset/summarize_from_feedback'
tokenizer_path = 'models/llama2-7b-hf'

@dataclass
class BeamTuningGenConfig:
    max_length: Optional[int] = 128
    temperature: Optional[float] = None 
    top_p: Optional[float] = 0.7
    top_k: Optional[int] = 0 #
    # new
    num_beams: Optional[int] = 1
    num_candidates: Optional[int] = 1
    block_len: Optional[int] = 12000
    # scorer
    beta: Optional[float] = 1
    average_log_prob: Optional[bool] = False
    reference_free: Optional[bool] = False
    sft_free: Optional[bool] = False
    scorer_type: Optional[str] = "explicit" # options: implicit, explicit， none
    def __post_init__(self):
        valid_types = {"implicit", "explicit", "none"}
        if self.scorer_type not in valid_types:
            raise ValueError(f"Invalid scorer type: {self.scorer_type}. Valid options are: {valid_types}")


@dataclass
class TokenWiseSamplingGenConfig:
    beta_list: List[float] = field(default_factory=lambda: [2.0, 2.0])
    token_wise_sampling_type: Optional[str] = "tokenrm"
    def __post_init__(self):
        valid_types = {"eft", "tokenrm", "none"}
        if self.token_wise_sampling_type not in valid_types:
            raise ValueError(f"Invalid token_wise sampling type: {self.token_wise_sampling_type}. Valid options are: {valid_types}")



@dataclass
class ScriptArguments:
    save_directory: Optional[str] = field(default='./logs_rewardedsoups_summary_eval')
    base_model_path_1: Optional[str] = field(default='models/summary_sft_merged')
    base_model_path_2: Optional[str] = field(default='models/summary_sft_merged')
    base_model_path_3: Optional[str] = field(default='models/summary_sft_merged')
    scorer_name1: Optional[str] = field(default='models/summary_ppo_faith_merged')
    scorer_name2: Optional[str] = field(default='models/summary_ppo_prefer1_merged')
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "minibatch size for eval"})
    run_name: Optional[str] = field(default='eval_pposoups_klreg0.2_harmless_helpful', metadata={"help": "Name for this experiment"})
    logging_path: Optional[str] = field(default='./logs_Q')
    reward_names:Optional[str] = field(default='harmless,helpful')
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load model in 8bit"})
    frac: Optional[float] = field(default=1.0, metadata={"help": "fraction of the dataset to use"})
    is_validate: Optional[bool] = field(default=False, metadata={"help": "validate the model"})
    weight1: Optional[float] = field(default=0.5, metadata={"help": "weight for the first model"})
    weight2: Optional[float] = field(default=0.5, metadata={"help": "weight for the second model"})
    skipk: Optional[bool] = field(default=True, metadata={"help": "skip the k search"})
    is_log: Optional[bool] = field(default=True, metadata={"help": "whether to log the information"})
    ref_model_path: Optional[str] = field(default='models/summary_sft_merged')
    
    tokenwisesampling_generation_configs: TokenWiseSamplingGenConfig = field(default_factory=lambda: TokenWiseSamplingGenConfig())
    gen: BeamTuningGenConfig = field(default_factory=lambda: BeamTuningGenConfig())
    value_model_path: Optional[str] = field(default=None)
    test_option: Optional[bool] = field(default=False)
    candiate_len: Optional[int] = field(default=100)
    
    merge_explicit_token_rm: Optional[bool] = field(default=False)
    get_topk: Optional[bool] = field(default=False)
    begin: Optional[int] = field(default=0)
    merge_eft: Optional[bool] = field(default=True)
    interval_to_guide: Optional[int] = field(default=1)
    entropy_judge: Optional[bool] = field(default=False)
    
    is_beam_search: Optional[bool] = field(default=False)
    window_size: Optional[int] = field(default=10)
    k_std: Optional[float] = field(default=1.0)
    
    merge_base_model: Optional[bool] = field(default=True)
    no_expo: Optional[bool] = field(default=False)
    
    merge_base_bf16: Optional[bool] = field(default=False)
    
    self_guided: Optional[bool] = field(default=False, metadata={"help": "self guided"})

    
# TODO:here change the skipk to False

# parser = HfArgumentParser(ScriptArguments)
script_args = tyro.cli(ScriptArguments)
base_model_name_1 = script_args.base_model_path_1
base_model_name_2 = script_args.base_model_path_2
base_model_name_3 = script_args.base_model_path_3
# script_args = parser.parse_args_into_dataclasses()[0]
exp_type = script_args.exp_type
tokenier_name = tokenizer_path



reward_names = [x.strip() for x in script_args.reward_names.split(',')]
reward_path_tokenizer_dict = {
    'harmless': ['Ray2333/gpt2-large-harmless-reward_model'],
    'helpful': ['Ray2333/gpt2-large-helpful-reward_model'],
    'deberta': ['OpenAssistant/reward-model-deberta-v3-large-v2'],
    'preference2': ['OpenAssistant/reward-model-deberta-v3-large-v2'],
    'summary': ['Tristan/gpt2_reward_summarization'],
    'faithful':['CogComp/bart-faithful-summary-detector'],
    'humor': ['mohameddhiab/humor-no-humor'],
}
# add prefix in reward_path
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


accelerator = Accelerator()
process_id = Accelerator().local_process_index 
gpu_id = process_id
reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id) 

if accelerator.is_main_process:
    if not os.path.exists(script_args.save_directory+'/'+script_args.run_name):
        os.makedirs(script_args.save_directory+'/'+script_args.run_name)
        



set_seed(8888)
current_device = Accelerator().local_process_index

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
def evaluate_model_bs(model, tokenizer, valid_dataset):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    valid_data_loader = DataLoader(valid_dataset, batch_size=script_args.mini_batch_size, drop_last=True, collate_fn=data_collator)
    model.resize_token_embeddings(tokenizer.vocab_size) #TODO:?????

    accelerator = Accelerator()
    valid_data_loader = accelerator.prepare(valid_data_loader)
    model.to(accelerator.device)
    model.eval()
    # model, valid_data_loader = accelerator.prepare(model, valid_data_loader)

    generation_kwargs = {
        "max_new_tokens": 128 if exp_type == 'assistant' else 48, 
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 0.9, 
        "do_sample": True,
    }

    full_responses = []
    full_prompts = []
    pbar = tqdm(total=len(valid_dataset) // script_args.mini_batch_size // accelerator.num_processes)
    with torch.no_grad():
        for i, batch in enumerate(valid_data_loader):
            # mention here
            response_tensors = accelerator.unwrap_model(model).generate(batch["input_ids"], **generation_kwargs) #length_sampler=output_length_sampler, 
            full_responses.extend(response_tensors)
            full_prompts.extend(batch['input_ids'])
            pbar.update(1)
    
    full_responses = tokenizer.batch_decode(full_responses)
    full_prompts = tokenizer.batch_decode(full_prompts)
    # clean data
    full_prompts, full_responses = get_clean_data(full_responses, full_prompts)

    queries_responses = [
        (instructions.get_input(text), instructions.get_response(text))
        for text in full_responses
    ]
    if hasattr(instructions, 'get_post'):
        rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
    else:
        rewards_list = reward_models.get_reward_model_scores(queries_responses)
    
    ### error here may because of old version of transformers/accelerate/peft
    all_rewards = []
    for i in range(reward_models.num_rewards):
        all_rewards.append(accelerator.gather(torch.tensor(rewards_list[i], dtype=torch.float32, device=accelerator.device)))
        # all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))
    # all_full_prompts = accelerator.gather_for_metrics(full_prompts)
    # all_full_responses = accelerator.gather_for_metrics(full_responses)
    # return all_rewards, all_full_prompts, all_full_responses
    return all_rewards


def evaluate_model(merged_model, tokenizer, valid_dataset, preference_weight, merged_value_model=None):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    valid_data_loader = DataLoader(valid_dataset, batch_size=script_args.mini_batch_size, drop_last=True, collate_fn=data_collator)

    valid_data_loader = accelerator.prepare(valid_data_loader)
    merged_model.eval()
    # TODO: consider .prepare the model
    
    full_responses = []
    full_prompts = []
    pbar = tqdm(total=len(valid_dataset) // script_args.mini_batch_size // accelerator.num_processes)
    with torch.no_grad():
        for i, batch in enumerate(valid_data_loader):
        # prompt = prompt_template.format(raw_prompt=raw_prompt)
            prompt_tokenized = batch
            raw_prompt = tokenizer.batch_decode(prompt_tokenized['input_ids'], skip_special_tokens=True)[0]

            if script_args.tokenwisesampling_generation_configs.token_wise_sampling_type == "eft":
                if not script_args.merge_eft:
                    for value_model in merged_value_model:
                        value_model.set_raw_prompt(raw_prompt)
                model = EFTPosthocGenerationMixin(
                    base=PrefixPreTrainedWrapper(merged_model, tokenizer, raw_prompt),
                    tune_r=PrefixPreTrainedWrapper(merged_value_model, tokenizer, raw_prompt) if script_args.merge_eft else merged_value_model,
                    base_r=PrefixPreTrainedWrapper(implict_ref_model, tokenizer, raw_prompt),
                    w=script_args.tokenwisesampling_generation_configs.beta_list[0] if script_args.merge_eft else script_args.tokenwisesampling_generation_configs.beta_list, # XXX beta selection is important
                    preference_weight=preference_weight[:2] if not script_args.merge_eft else [1.0],
                    get_topk=script_args.get_topk,
                    is_main_process = accelerator.is_main_process,
                    interval_to_guide = script_args.interval_to_guide,
                    entropy_judge = script_args.entropy_judge,
                    window_size = script_args.window_size,
                    k_std = script_args.k_std,
                )
                max_prefix_input_ids_length= max(model.base.get_prefix_input_ids_length(), model.tune_r[0].get_prefix_input_ids_length(), model.base_r.get_prefix_input_ids_length())
                # max_new_tokens = script_args.gen.max_length - max_prefix_input_ids_length # TODO: mention here
                max_new_tokens = 128 if exp_type == 'assistant' else 48
                max_length = max_new_tokens + max_prefix_input_ids_length
            elif script_args.tokenwisesampling_generation_configs.token_wise_sampling_type == "tokenrm":
                
                for token_reward_model in token_reward_model_list:
                    token_reward_model.set_raw_prompt(raw_prompt)
                # TODO: here the input of the reward model seems to be in wrong format
                
                model = TokenWiseSamplingMixin(
                    base=PrefixPreTrainedWrapper(merged_model, tokenizer, raw_prompt),
                    token_reward_model_list=token_reward_model_list,
                    beta_list=script_args.tokenwisesampling_generation_configs.beta_list,
                    w=preference_weight,# TODO here
                    candiate_len = script_args.candiate_len,
                    interval_to_guide = script_args.interval_to_guide,
                    entropy_judge = script_args.entropy_judge,
                    window_size = script_args.window_size,
                    k_std = script_args.k_std,
                )
                max_new_tokens = 128 if exp_type == 'assistant' else 48
                max_length = max_new_tokens + model.get_prefix_input_ids_length()
            else :
                model = PrefixPreTrainedWrapper(merged_model, tokenizer, prompt_template.format(raw_prompt=raw_prompt))
                max_new_tokens = script_args.gen.max_length - model.get_prefix_input_ids_length()
    
            bt_model = BeamTuningWithEFTPosthocGenerationMixin(model, tokenizer)
            
            outputs = bt_model.bon_beam_sample(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                scorer=None if not script_args.is_beam_search else scorer,
                split_by_prompt_text=False,
                # 
                max_length=max_length, # TODO: mention we use max_length instead of max_new_tokens
                # temperature=script_args.gen.temperature,
                top_p=script_args.gen.top_p,
                # top_k=script_args.gen.top_k,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                # 
                num_beams=script_args.gen.num_beams,
                num_candidates=script_args.gen.num_candidates,
                block_len=script_args.gen.block_len, # block_len = inf to enable sequence wise bon
                # return_dict_in_generate=True,
            )
            
            # response = extract_responses(outputs, tokenizer, prompt_len=prompt_tokenized["input_ids"].size(1))[0]
            full_responses.extend(outputs)
            full_prompts.extend(batch['input_ids'])
            response = tokenizer.batch_decode(outputs)
            if exp_type == 'summary':
                log_info("response",i, f"{bt_model.base.entropy_count/bt_model.base.step:.2f}",  response[0].split('### Response: ')[-1])
            else:
                log_info("response",i, f"{bt_model.base.entropy_count/bt_model.base.step:.2f}",  response[0].split('Assistant: ')[-1])
            pbar.update(1)
            
        full_prompts = tokenizer.batch_decode(full_prompts)
        full_responses = tokenizer.batch_decode(full_responses)
        # clean data
        full_prompts, full_responses = get_clean_data(full_responses, full_prompts)

        queries_responses = [
            (instructions.get_input(text), instructions.get_response(text))
            for text in full_responses
        ]
        if hasattr(instructions, 'get_post'):
            rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
        else:
            rewards_list = reward_models.get_reward_model_scores(queries_responses)
        
        # for k in range(len(rewards_list)):
        #     if exp_type == 'summary':
        #         log_info("response",k, f"{rewards_list[0][k]:.2f}, {rewards_list[1][k]:2f}",  full_responses[k].split('### Response: ')[-1])
        #     else:
        #         log_info("response",k, f"{rewards_list[0][k]:.2f}, {rewards_list[1][k]:2f}",  full_responses[k].split('Assistant: ')[-1])

        # for i in range(reward_models.num_rewards):
        #     all_rewards.append(accelerator.gather(torch.tensor(rewards_list[i], dtype=torch.float32, device=accelerator.device)))
            
        all_rewards = []
        for i in range(reward_models.num_rewards):
            all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))
        all_full_prompts = accelerator.gather_for_metrics(full_prompts)
        all_full_responses = accelerator.gather_for_metrics(full_responses)
        return all_rewards, all_full_prompts, all_full_responses


if script_args.frac < 1.0:
    Tfrac = script_args.frac
else:
    Tfrac = None

tokenizer = load_main_tokenizer(tokenier_name)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token


if script_args.merge_base_model is False:
    merged_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_3,
        device_map=gpu_id
    )
    merged_model.resize_token_embeddings(tokenizer.vocab_size)
    

if exp_type == 'assistant':
    if script_args.is_validate:
        valid_dataset = build_dataset_eval(hhrlhf_dataset_path, tokenizer, reward_models.rm_tokenizers, split='train', frac=0.02) 
    else:
        valid_dataset = build_dataset_eval(hhrlhf_dataset_path, tokenizer, reward_models.rm_tokenizers, split='test', frac=Tfrac) 
    small_valid_dataset = build_dataset_eval(hhrlhf_dataset_path, tokenizer, reward_models.rm_tokenizers, split='train', frac=0.01) 
    instructions = Instructions()
else:
    if script_args.is_validate:
        valid_dataset = build_dataset_summary_eval(summary_dataset_path, tokenizer, reward_models.rm_tokenizers, split='train', frac=0.1)
    else:
        valid_dataset = build_dataset_summary_eval(summary_dataset_path, tokenizer, reward_models.rm_tokenizers, split='test', frac=Tfrac)
    small_valid_dataset = build_dataset_summary_eval(summary_dataset_path, tokenizer, reward_models.rm_tokenizers, split='train', frac=0.1) 
    instructions = Instructions_summary()
log_info(f"Size of the validation set: {len(valid_dataset)}")

# only remove when the key is in the dataset
if 'prompt' in valid_dataset.column_names:
    valid_dataset = valid_dataset.remove_columns('prompt')
    small_valid_dataset = small_valid_dataset.remove_columns('prompt')
if 'query' in valid_dataset.column_names:
    valid_dataset = valid_dataset.remove_columns('query')
    small_valid_dataset = small_valid_dataset.remove_columns('query')

for key in ['key', 'text', 'response']:
    if key in valid_dataset.column_names:
        valid_dataset = valid_dataset.remove_columns(key)
        small_valid_dataset = small_valid_dataset.remove_columns(key)

############################################################################################
############################################################################################
#XXX: the code below is adapted from ivg

explicit_token_rm_paths = {
    "faithful": "RiC/ppo/logs_vas/token_rm_summary_faith_llama/token_rm_summary_faith_llama_b2_new_best",
    "summary": "RiC/ppo/logs_vas/token_rm_summary_pref1_llama/token_rm_summary_pref1_llama_b2_new_best",
    "harmless": "RiC/ppo/logs_vas/token_rm_assist_harmless_llama/token_rm_assist_harmless_llama_b2_best",
    "helpful": "RiC/ppo/logs_vas/token_rm_assist_helpful_llama/token_rm_assist_helpful_llama_b2_best",
    "humor": "RiC/ppo/logs_vas/token_rm_assist_humor_llama/token_rm_assist_humor_llama_b2_best",
    #TODO: humor
}

if script_args.value_model_path is not None:
    explicit_token_rm_paths['summary'] = script_args.value_model_path

if script_args.tokenwisesampling_generation_configs.token_wise_sampling_type == "eft":
    implict_ref_model = AutoModelForCausalLM.from_pretrained(
        script_args.ref_model_path,
        load_in_8bit=True,
        device_map=gpu_id
    )
    # if implict_ref_model.config.vocab_size > tokenizer.vocab_size:
    implict_ref_model.resize_token_embeddings(tokenizer.vocab_size)

model_manager = ModelManager()

# prompt_template = get_chat_prompt_template(script_args.model_name, tokenizer) # TODO: chekc this

# base.generation_config.eos_token_id = None # TODO: check here


def get_scorer(name, model_type="explicit"):
    if model_type == "explicit":
        model_path = explicit_token_rm_paths[name]
    else:
        model_path = name
    # token_reward_model_template = "\n\nHuman: {raw_prompt}\n\nAssitant:"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,
        device_map=gpu_id
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenier_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if model_type == "explicit":
        explicit_reward_scorer = ExplicitRewardScorer(
            token_reward_model=model,
            tokenizer=tokenizer,
            # beta=beta, # TODO: check how to adjust beta
        )
        return explicit_reward_scorer
    else:
        implicit_reward_scorer = ImplicitRewardScorer(
            model=model,
            ref_model=implict_ref_model,
            tokenizer=tokenizer,
        )
        return implicit_reward_scorer


scorer= None
### TODO: !!! we will not include this part into the code for now.
if script_args.is_beam_search:
    pass
    if script_args.gen.scorer_type == "implicit":
        scorer = []
        implicit_name_list = [script_args.scorer_name1, script_args.scorer_name2]
        for i in range(len(implicit_name_list)):
            scorer.append(get_scorer(implicit_name_list[i], model_type="implicit"))
        
    elif script_args.gen.scorer_type == "explicit":
        scorer = []
        for i in range(len(reward_names)):
            scorer.append(get_scorer(reward_names[i], model_type="explicit"))

    else :
        scorer = NotImplementedScorer()


## mention here
# if script_args.tokenwisesampling_generation_configs.token_wise_sampling_type == "eft":
#     if script_args.gen.scorer_type != "implicit":
#         eft_scorer = get_scorer(
#             scorer_name=script_args.scorer_name,
#             load_in_4bit=True,
#             beta=script_args.gen.beta,
#             average_log_prob=script_args.gen.average_log_prob,
#             reference_free=script_args.gen.reference_free,
#         )
#     else :
#         eft_scorer = scorer

if script_args.tokenwisesampling_generation_configs.token_wise_sampling_type == "tokenrm" and not script_args.merge_explicit_token_rm:
    # token_reward_model_template = "\n\nHuman: {raw_prompt}\n\nAssitant:"
    token_reward_model_tokenizer_list = []
    token_reward_model_list = []
    for idx, name in enumerate(reward_names):
        token_reward_model = model_manager.load_model(
            explicit_token_rm_paths[name],
            LlamaModelForTokenScore,
        )
        token_reward_model_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        token_reward_model_tokenizer.pad_token = token_reward_model_tokenizer.bos_token # TODO: check here
        token_reward_model_tokenizer.padding_side = "left"
        
        token_reward_model_list.append(PrefixPreTrainedWrapper(token_reward_model, token_reward_model_tokenizer, ''))
        token_reward_model_tokenizer_list.append(token_reward_model_tokenizer)
        
# elif script_args.tokenwisesampling_generation_configs.token_wise_sampling_type == "none":
#     pass
# else :
#     raise ValueError(f"Invalid token_wise_sampling_type: {script_args.tokenwisesampling_generation_configs.token_wise_sampling_type}")

log_info("Evaluating........")
## preference list
M = 10
preferences = np.zeros((M+1, 3))
preferences[:, 0] = np.arange(0,1+ 1/M,1 / M)
preferences[:, 1] = 1 - preferences[:, 0]
preferences = np.round(preferences, 1)

preferences_org = np.arange(0,1+ 1/M,1 / M)

if script_args.test_option:
    pass
    # preferences[2] = preferences[-1]
    # preferences_org[2] = preferences_org[-1]
# swap the 2nd and the last one in preferences
# preferences[1], preferences[-1] = preferences[-1], preferences[1]
# preferences_org[1], preferences_org[-1] = preferences_org[-1], preferences_org[1]

w11 = script_args.weight1
w12 = 1 - w11
w21 = script_args.weight2
w22 = 1 - w21

a = [w11, w21]
b = [w12, w22]
solution1 = np.linalg.solve(np.array([a, b]), [1,0])
solution2 = np.linalg.solve(np.array([a, b]), [0,1])


k1 = 1/0.5
k2 = 1/0.5

    
if script_args.skipk == False and script_args.no_expo == False:
    a_candidates= [0.1,0.2,0.3,0.4,0.5]
    result1 = []
    max_reward = -1000
    k1 = 0.1
    for a in a_candidates:
        k = 1/(1-a)
        x1 = solution1[0]*k
        x2 = solution1[1]*k
        x3 = 1-k
        temp_save_path = '/home/pretrainx/temp_models/{}'.format(script_args.run_name)
        preference = [x1, x2, x3]
        base_model_list = [base_model_name_1, base_model_name_2, base_model_name_3]
        merged_model = efficient_merge_weights_with_preference(base_model_list, preference, temp_save_path)

        accelerator.wait_for_everyone()
        gc.collect()
        torch.cuda.empty_cache()
        log_info(a, k, preference)
        all_rewards = evaluate_model_bs(merged_model, tokenizer, small_valid_dataset)
        # all_rewards, all_full_prompts, all_full_responses = evaluate_model_bs(merged_model, tokenizer, small_valid_dataset)
        gc.collect()
        torch.cuda.empty_cache()

        evaluation_result = {}
        temp_list =[]
        for i in range(reward_models.num_rewards):
            evaluation_result['obtained_score{}'.format(i+1)] = all_rewards[i].cpu().numpy()
            log_info('total average obtained score {}: {}'.format(i+1, np.mean(evaluation_result['obtained_score{}'.format(i+1)])))
            temp_list.append(np.mean(evaluation_result['obtained_score{}'.format(i+1)]))
        result1.append(temp_list)
        if temp_list[0] > max_reward:
            max_reward = temp_list[0]
            k1 = k
        del merged_model
        del all_rewards
        gc.collect()
        torch.cuda.empty_cache()
        
    if process_id == 0:
        # save results to pickle
        import pickle
        with open(os.path.join(script_args.save_directory, script_args.run_name, 'result1.pkl'), 'wb') as f:
            pickle.dump(result1, f)

    a_candidates= [0.1,0.2,0.3,0.4,0.5]
    result2 = []
    k2= 0.1
    max_reward = -1000
    for a in a_candidates:
        k = 1/(1-a)
        x1 = solution2[0]*k
        x2 = solution2[1]*k
        x3 = 1-k
        temp_save_path = '/home/pretrainx/temp_models/{}'.format(script_args.run_name)
        preference = [x1, x2, x3]
        base_model_list = [base_model_name_1, base_model_name_2, base_model_name_3]
        merged_model = efficient_merge_weights_with_preference(base_model_list, preference, temp_save_path)

        accelerator.wait_for_everyone()
        gc.collect()
        torch.cuda.empty_cache()
        log_info(a, k, preference)
        all_rewards = evaluate_model_bs(merged_model, tokenizer, small_valid_dataset)
        # all_rewards, all_full_prompts, all_full_responses = evaluate_model_bs(merged_model, tokenizer, small_valid_dataset)
        gc.collect()
        torch.cuda.empty_cache()

        evaluation_result = {}
        temp_list = []
        for i in range(reward_models.num_rewards):
            evaluation_result['obtained_score{}'.format(i+1)] = all_rewards[i].cpu().numpy()
            log_info('total average obtained score {}: {}'.format(i+1, np.mean(evaluation_result['obtained_score{}'.format(i+1)])))
            temp_list.append(np.mean(evaluation_result['obtained_score{}'.format(i+1)]))
        result2.append(temp_list)
        if temp_list[1] > max_reward:
            max_reward = temp_list[1]
            k2 = k
        del merged_model
        del all_rewards
        gc.collect()
        torch.cuda.empty_cache()
        
    if process_id == 0:
        # save results to pickle
        import pickle
        with open(os.path.join(script_args.save_directory, script_args.run_name, 'result2.pkl'), 'wb') as f:
            pickle.dump(result2, f)

    log_info('k1:', k1)
    log_info('k2:', k2)
    
import copy
# sample
results = []

if script_args.no_expo:
    k1 = 1.0
    k2 = 1.0

if script_args.test_option:
    bg, ed = 4, 6
else:
    bg, ed = script_args.begin, len(preferences)

for k in range(bg, ed):
    # if k>2 and script_args.test_option:
        # break
    preference = preferences[k]
    preference_vector = copy.deepcopy(preference)
    preference_org = preferences_org[k]

    x1 = preference[0]*solution1[0]*k1 + preference[1] * solution2[0]*k2
    x2 = preference[0]*solution1[1]*k1 + preference[1] * solution2[1]*k2
    x3 = (1-k1)*preference[0] + (1-k2)*preference[1]

    if process_id == 0:
        log_info('preference:', preference)
        log_info('x1, x2, x3:', x1, x2, x3)
        log_info('solution1:', solution1)
        log_info('solution2:', solution2)
        log_info("org preference:", preference_org)

    preference[0], preference[1], preference[2] = x1, x2, x3

    temp_save_path = '/home/pretrainx/temp_models/{}'.format(script_args.run_name)
    # if process_id == 0:
    if len(preference) == 3:
        base_model_list = [base_model_name_1, base_model_name_2, base_model_name_3]
    else:
        base_model_list = [base_model_name_1, base_model_name_2]
    # merged_model = efficient_merge_value_model(base_model_list, preference, temp_save_path, gpu_id) # TODO: orginal is efficient_merge_weights_with_preference
    # mask this line in 2025-2-18. remember to check the approach for merging.
    if script_args.merge_base_model:
        if script_args.merge_base_bf16:
            merged_model = efficient_merge_weights_with_preference_bf16(base_model_list, preference, temp_save_path)
        else:
            merged_model = efficient_merge_weights_with_preference(base_model_list, preference, temp_save_path)
        merged_model.resize_token_embeddings(tokenizer.vocab_size) # TODO: check here
        merged_model.to(current_device) # TODO: here restore
    
    
    # accelerator.wait_for_everyone()
    log_info(k, preference)
    
    if script_args.tokenwisesampling_generation_configs.token_wise_sampling_type == "eft":
        if script_args.merge_eft:
            to_merge_value_list = [script_args.scorer_name1, script_args.scorer_name2]
            
            if script_args.self_guided:
                merged_value_model = copy.deepcopy(merged_model)
            else:
                merged_value_model = efficient_merge_value_model(to_merge_value_list, [preference_org,1-preference_org], temp_save_path, gpu_id)
            # if merged_value_model.config.vocab_size > tokenizer.vocab_size:
            merged_value_model.resize_token_embeddings(tokenizer.vocab_size)
        else:
            to_merge_value_list = [script_args.scorer_name1, script_args.scorer_name2]
            merged_value_model = []
            for name in to_merge_value_list:
                implicit_scorer = AutoModelForCausalLM.from_pretrained(name, device_map=gpu_id, load_in_8bit= True)
                implicit_scorer.resize_token_embeddings(tokenizer.vocab_size)
                merged_value_model.append(PrefixPreTrainedWrapper(implicit_scorer, tokenizer, None))
    else:
        merged_value_model = None
    
    if script_args.merge_explicit_token_rm:
        to_merge_token_rm_list = []
        for name in reward_names:
            to_merge_token_rm_list.append(explicit_token_rm_paths[name])
            
        merged_token_rm = efficient_merge_explicit_token_rm(to_merge_token_rm_list, preference_vector, temp_save_path, gpu_id)
        token_reward_model_list = []
        token_reward_model_tokenizer_list = []
        
        token_reward_model_list.append(PrefixPreTrainedWrapper(merged_token_rm, tokenizer, ''))
        token_reward_model_tokenizer_list.append(tokenizer)
        preference_vector = [1.0]
    
    if script_args.is_beam_search:
        assert script_args.merge_explicit_token_rm or script_args.merge_eft
        if script_args.merge_explicit_token_rm:
            scorer = ExplicitRewardScorer(
                token_reward_model=token_reward_model_list[0],
                tokenizer=tokenizer,
                # beta=beta, # TODO: check how to adjust beta
            )
        else:
            scorer = ImplicitRewardScorer(
                model=merged_value_model,
                ref_model=implict_ref_model,
                tokenizer=tokenizer,
            )

    all_rewards, all_full_prompts, all_full_responses = evaluate_model(merged_model, tokenizer, valid_dataset, preference_vector, merged_value_model=merged_value_model)
    # gc.collect()
    # torch.cuda.empty_cache()

    if process_id == 0:
        evaluation_result = {
            'prompt': all_full_prompts,
            'response': all_full_responses,
        }
        for i in range(reward_models.num_rewards):
            evaluation_result['obtained_score{}'.format(i+1)] = all_rewards[i]
            log_info('total average obtained score {}: {}'.format(i+1, np.mean(evaluation_result['obtained_score{}'.format(i+1)])))

        dataframe = pd.DataFrame(evaluation_result)

        # when the following is error, conduct another logit
        try:
            dataframe.to_csv(os.path.join(script_args.save_directory, script_args.run_name,'eval_data_pref{}_{}.csv'.format(round(preference_org,1), round(1-preference_org,1))))
        except:
            # drop prompt and response
            dataframe = dataframe.drop(columns=['prompt', 'response'])
            dataframe.to_csv(os.path.join(script_args.save_directory, script_args.run_name,'eval_data_pref{}_{}.csv'.format(round(preference_org,1), round(1-preference_org,1))))
        del evaluation_result
    del merged_value_model
