from dataclasses import dataclass
import warnings
from typing import Dict, List, Any, Optional, Union
import time
import os
import GPUtil


import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationMixin
from transformers.generation.logits_process import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxNewTokensCriteria,
    StoppingCriteriaList,
)

from inference_tools.scorer import BaseScorer
from inference_tools.utils import StopOnStringCriteria, extract_responses, get_truncated_responses
from inference_tools.model import PrefixPreTrainedWrapper
from inference_tools.utils import set_seeds, StopOnStringCriteria
from inference_tools.decoder.eft import EFTPosthocGenerationMixin
from inference_tools.decoder.ivg import TokenWiseSamplingMixin


__all__ = [
    "BeamTuningWithEFTPosthocGenerationMixin",
]


"""
BeamTuningWithEFTPosthocGenerationMixin is a mixin class that implements the beam search decoding strategy with EFTPosthocGenerationMixin, TokenWiseSamplingMixin, and PrefixPreTrainedWrapper.
Attention that other PosthocGenerationMixin classes are not supported because the reordering of the input_ids and attention_mask is not implemented.
"""

@dataclass
class BeamTuningWithEFTPosthocGenerationMixin(GenerationMixin):
    base: PreTrainedModel | EFTPosthocGenerationMixin | TokenWiseSamplingMixin | PrefixPreTrainedWrapper
    tokenizer: PreTrainedTokenizer

    length2time = {}
    length2gpu_mem = {}

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.base, name)

    def prepare_inputs_for_generation(self, input_ids, **model_kwargs):
        return self.base.prepare_inputs_for_generation(input_ids, **model_kwargs)

    def _reorder_cache(self, past_key_values, beam_idx):
        if isinstance(self.base, EFTPosthocGenerationMixin):
            if past_key_values['base']: 
                past_key_values['base'] = self.base.base._reorder_cache(past_key_values['base'], beam_idx)
            if past_key_values['tune_r']:
                past_key_values['tune_r'] = [model._reorder_cache(past_key_values['tune_r'][i], beam_idx) for i, model in enumerate(self.base.tune_r)]
            if past_key_values['base_r']:
                past_key_values['base_r'] = self.base.base_r._reorder_cache(past_key_values['base_r'], beam_idx)
        elif isinstance(self.base, TokenWiseSamplingMixin):
            if past_key_values['base']:
                past_key_values['base'] = self.base.base._reorder_cache(past_key_values['base'], beam_idx)
            if past_key_values['token_reward']:
                past_key_values['token_reward'] = self.base.token_reward_model._reorder_cache(past_key_values['token_reward'], beam_idx)
        elif isinstance(self.base, PrefixPreTrainedWrapper):
            past_key_values = self.base._reorder_cache(past_key_values, beam_idx)
        return past_key_values
    
    # if any model is a PrefixPreTrainedWrapper, reorder the input_ids and attention_mask
    # althought it is not necessary because the past_key_values and last token( in input_ids ) have been reordered
    def _reorder_input_ids(self, beam_idx):
        if isinstance(self.base, EFTPosthocGenerationMixin):
            if isinstance(self.base.base, PrefixPreTrainedWrapper):
                self.base.base.input_ids = self.base.base.input_ids[beam_idx]
                self.base.base.attention_mask = self.base.base.attention_mask[beam_idx]
            if all([isinstance(model, PrefixPreTrainedWrapper) for model in self.base.tune_r]):
                for i, model in enumerate(self.base.tune_r):
                    model.input_ids = model.input_ids[beam_idx]
                    model.attention_mask = model.attention_mask[beam_idx]
            if isinstance(self.base.base_r, PrefixPreTrainedWrapper):
                self.base.base_r.input_ids = self.base.base_r.input_ids[beam_idx]
                self.base.base_r.attention_mask = self.base.base_r.attention_mask[beam_idx]
        elif isinstance(self.base, TokenWiseSamplingMixin):
            if isinstance(self.base.base, PrefixPreTrainedWrapper):
                self.base.base.input_ids = self.base.base.input_ids[beam_idx]
                self.base.base.attention_mask = self.base.base.attention_mask[beam_idx]
            if isinstance(self.base.token_reward_model, PrefixPreTrainedWrapper):
                self.base.token_reward_model.input_ids = self.base.token_reward_model.input_ids[beam_idx]
                self.base.token_reward_model.attention_mask = self.base.token_reward_model.attention_mask[beam_idx]
        elif isinstance(self.base, PrefixPreTrainedWrapper):
            self.base.input_ids = self.base.input_ids[beam_idx]
            self.base.attention_mask = self.base.attention_mask[beam_idx]
        

    @torch.no_grad()
    def bon_beam_sample(
        self,
        input_ids: torch.LongTensor,
        scorer: BaseScorer,
        max_new_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[float] = 50,
        top_p: Optional[float] = None,
        eos_strings: Optional[int] = None,
        split_by_prompt_text: Optional[bool] = True,
        **kwargs,
    ):
        logits_warper = []
        if temperature: logits_warper.append(TemperatureLogitsWarper(temperature))
        if top_k: logits_warper.append(TopKLogitsWarper(top_k))
        if top_p: logits_warper.append(TopPLogitsWarper(top_p))
        logits_warper = LogitsProcessorList(logits_warper)
        
        stopping_criteria = []
        if eos_strings:
            stopping_criteria.extend([StopOnStringCriteria(input_ids.size(1), eos_string, self.tokenizer) for eos_string in eos_strings])
        assert not (max_new_tokens and max_length)
        if max_length: stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
        if max_new_tokens: stopping_criteria.append(MaxNewTokensCriteria(start_length=input_ids.size(1), max_new_tokens=max_new_tokens))
        stopping_criteria = StoppingCriteriaList(stopping_criteria)
        self.generation_config.pad_token_id = kwargs['eos_token_id']
        kwargs.update({"use_cache": True})
        return self._bon_beam_sample(
            input_ids,
            scorer,
            stopping_criteria=stopping_criteria,
            logits_warper=logits_warper,
            eos_strings=eos_strings,
            split_by_prompt_text=split_by_prompt_text,
            **kwargs
        )
    
    def time_record(self, step, start):
        def get_total_gpu_mem():
            used_gpu_ids = os.getenv('CUDA_VISIBLE_DEVICES')
            gpu_ids = list(map(int, used_gpu_ids.split(',')))
            gpus = GPUtil.getGPUs()
            total_memory_util = 0
            for gpu in gpus:
                if gpu.id in gpu_ids:
                    total_memory_util += gpu.memoryUtil
            return total_memory_util

        if step % 10 == 0:
            self.length2time[step] = time.time() - start
            self.length2gpu_mem[step] = get_total_gpu_mem()
        return self.length2time, self.length2gpu_mem
    
    @torch.no_grad()
    def _bon_beam_sample(
        self,
        input_ids: torch.LongTensor,
        scorer: BaseScorer,
        num_beams: Optional[int] = 4,
        num_candidates: Optional[int] = 4,
        block_len: Optional[int] = 10,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        return_dict_in_generate: Optional[bool] = None,
        eos_strings: Optional[int] = None,
        split_by_prompt_text: Optional[bool] = True,
        **model_kwargs,
    ) -> Union[Dict, torch.LongTensor]:
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        assert not (eos_strings and eos_token_id)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

        # repeat input_ids and attention_mask
        input_ids = input_ids.repeat(num_beams * num_candidates, 1)
        model_kwargs["attention_mask"] = model_kwargs["attention_mask"].repeat(num_beams * num_candidates, 1)

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        blocks_to_use = block_len
        prompt, prompt_len = self.tokenizer.decode(input_ids[0]), input_ids.size(1)

        this_peer_finished = False  # used by synced_gpus only
        
        
        step = 0
        start = time.time()

        while True:
            
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            
            outputs = self.__call__(
                **model_inputs,
                return_dict=True,
            )
            
            next_token_logits = outputs.logits[:, -1, :].float()
            next_token_logits = logits_processor(input_ids, next_token_logits)
            next_token_logits = logits_warper(input_ids, next_token_logits)
            
            probs = nn.functional.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1).to(torch.int64)
            blocks_to_use -= 1

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            # if stopping_criteria(input_ids, None):
            #     this_peer_finished = True
            # !!! 2025-9-4 we replace the content above in following lines.
            criteria_result = stopping_criteria(input_ids, None)
            if isinstance(criteria_result, bool): # 兼容返回单个布尔值的情况
                if criteria_result:
                    this_peer_finished = True
            elif criteria_result.all(): # 如果返回的是张量，则检查是否所有元素都为True
                this_peer_finished = True

            del outputs
            if blocks_to_use <= 0 or this_peer_finished == True:
                if this_peer_finished == True and blocks_to_use > 10000:
                    break # TODO: the entrance to exit the loop
                # print('warning we use _reorder_input_ids')
                blocks_to_use = block_len

                if split_by_prompt_text:
                    responses = extract_responses(input_ids, self.tokenizer, prompt=prompt)
                else:
                    responses = extract_responses(input_ids, self.tokenizer, prompt_len=prompt_len)
                if eos_strings:
                    responses, unfinished_sequences = get_truncated_responses(responses, eos_strings)
                beam_scores = scorer(
                    {
                        "response": responses,
                        "eos": unfinished_sequences == 0,
                    },
                )

                _, beam_idx = torch.topk(
                    beam_scores, 
                    num_beams, dim=0, largest=True, sorted=True
                )

                # repeat beam_idx by candidate numbers
                beam_idx = beam_idx.repeat(num_candidates)

                # reorder states
                input_ids = input_ids[beam_idx, :] # TODO:here has been reordered then why need self._reorder_input_ids
                unfinished_sequences = unfinished_sequences[beam_idx]

                if unfinished_sequences.max().item() == 0:
                    this_peer_finished = True 

                if model_kwargs["past_key_values"] is not None:
                    model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

                # reorder input_ids of PrefixPreTrainedWrapper; 
                # although it is not necessary because the past_key_values and last token( in input_ids ) have been reordered
                self._reorder_input_ids(beam_idx)
                if this_peer_finished == True:
                    break
            step += 1
            self.time_record(step, start)

        if return_dict_in_generate:
            return {
                "output_ids": input_ids[:num_beams],
                "scores": beam_scores[:num_beams],
            }
        else:
            return input_ids[0, None]


"""
Unit test for BeamTuningWithEFTPosthocGenerationMixin.

"""

def unit_test():
    import os
    from dataclasses import dataclass, field
    from typing import Optional, Text

    import tyro
    import tqdm
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
    from datasets import Dataset

    from inference_tools.utils import set_seeds, extract_responses, StopOnStringCriteria
    from inference_tools.decoder.decoder import EFTPosthocGenerationMixin, BeamTuningWithEFTPosthocGenerationMixin
    from inference_tools.scorer import ImplicitRewardScorer, ExplicitRewardScorer, NotImplementedScorer
    from inference_tools.model import PrefixPreTrainedWrapper
    from inference_tools.modeling_llama import LlamaModelForTokenScore
    from inference_tools.modeling_gpt2 import GPT2ModelForTokenScore, GPT2ForScore

    from utils import load_instruction_dataset
    from datasets import load_dataset
    from utils import get_local_model_name

    DEFAULT_PROMPT_TEMPLATE = """SUBREDDIT: r/relationships\nTITLE: Me [21F] with my boyfriend [19M] of 2months has a gaming habit.\nPOST: I tried to start a convo with the boyfriend everyday but it seems to be making me a little depress because he's always playing video games than paying attention to me. I'm not trying to be an attention but it's seems to be a bad habit of his. I don't know what to do or how to even confront him about it. Any IDEAS?
TL;DR: Boyfriend has a gaming habit and I don't know how to confront him about it. I'm not trying to be an attention whore but it's making me feel bad.

SUBREDDIT: r/relationships\nTITLE: Me [21 M] with my...idk [19 F] Just not sure what to do.\nPOST: Went on vacation 1 1/2 years ago\n\nMet an amazing girl\n\nSpent a lot of time together\n\nHad to leave\n\nWe had agreed it would be ok to see other people\n\nBut we keep in contact and talk about how much we miss each other all the time\n\nStill have feelings for her\n\nShe just entered a relationship recently\n\nIt bothers me\n\nIdk if I should tell her how I feel or if I am just idealizing something we had and should move on.
TL;DR: Had an amazing time with this girl before we had to leave for summer vacation 1 1/2 years ago. Still have feelings for her and want to pursue relationship w/ her. Don't know whether to tell her or not.

{raw_prompt}
TL;DR:"""


    class ModelManager:
        def __init__(self):
            self.models = {}
            self.device_map = "auto"
            self.torch_dtype = torch.bfloat16

        def load_model(self, model_name, model_type, **kwargs):
            if model_name not in self.models:
                model = model_type.from_pretrained(
                    model_name,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device_map,
                    **kwargs,
                )
                self.models[model_name] = model
            return self.models[model_name]
    model_manager = ModelManager()
    set_seeds(0)

    def get_eft_scorer(beta, average_log_prob, reference_free, sft_free):
        model = model_manager.load_model(
            get_local_model_name("gpt2-summarize-dpo"),
            AutoModelForCausalLM,
        )
        ref_model = model_manager.load_model(
            get_local_model_name("gpt2-summarize") if not sft_free else "openai-community/gpt2",
            AutoModelForCausalLM,
        )

        tokenizer = AutoTokenizer.from_pretrained(get_local_model_name("gpt2-summarize"))
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        implicit_reward_scorer = ImplicitRewardScorer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            model_prompt_template="{raw_prompt}TL;DR: ",
            ref_model_prompt_template="{raw_prompt}TL;DR: ",
            beta=beta,
            average_log_prob=average_log_prob,
            reference_free=reference_free,
        )
        return implicit_reward_scorer

    def get_tokenrm_scorer(beta, ):
        model_path = get_local_model_name("gpt2-summarize-token-rm")
        model = model_manager.load_model(
            model_path,
            GPT2ModelForTokenScore,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        explicit_reward_scorer = ExplicitRewardScorer(
            token_reward_model=model,
            tokenizer=tokenizer,
            model_prompt_template="{raw_prompt}TL;DR: ",
            beta=beta,
        )
        return explicit_reward_scorer

    def get_args_scorer(beta):
        model_path = get_local_model_name("gpt2-summarize-sequence-rm")
        model = model_manager.load_model(
            model_path,
            GPT2ForScore,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        args_reward_scorer = ExplicitRewardScorer(
            token_reward_model=model,
            tokenizer=tokenizer,
            model_prompt_template="{raw_prompt}TL;DR: ",
            beta=beta,
        )
        return args_reward_scorer



    base_model = model_manager.load_model("/mnt/hwfile/llm-safety/models/gpt2-large", AutoModelForCausalLM)
    tokenizer = AutoTokenizer.from_pretrained("/mnt/hwfile/llm-safety/models/gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model.generation_config.eos_token_id = None # because of the conflict with the eos_strings

    # token wise sampling strategy
    tune_gpt2_small = model_manager.load_model(
        "/mnt/hwfile/llm-safety/models/gpt2-summarize-dpo-v2",
        AutoModelForCausalLM,
    )
    base_gpt2_small = model_manager.load_model(
        "/mnt/hwfile/llm-safety/models/gpt2-summarize-v2",
        AutoModelForCausalLM,
    )

    dataset = load_dataset(
            "chadlzx/openai_summarize_comparisons_relabel", split="test"
        ).shuffle(seed=42).select_columns("prompt").rename_columns({"prompt":"raw_prompt"}).select(range(20))
    
    # scorer = get_eft_scorer(beta=1, average_log_prob=False, reference_free=False, sft_free=False)
    # scorer = get_tokenrm_scorer(beta=1) 
    scorer = get_tokenrm_scorer(beta=1)
    # scorer = NotImplementedScorer()

    results = []
    for raw_prompt in tqdm.tqdm(dataset["raw_prompt"]):
        model = EFTPosthocGenerationMixin(
            base=PrefixPreTrainedWrapper(base_model, tokenizer, DEFAULT_PROMPT_TEMPLATE.format(raw_prompt=raw_prompt)),
            tune_r=PrefixPreTrainedWrapper(tune_gpt2_small, tokenizer, f'{raw_prompt}TL;DR: '),
            base_r=PrefixPreTrainedWrapper(base_gpt2_small, tokenizer, f'{raw_prompt}TL;DR: '),
            w=0.25,
        )
        # model = PrefixPreTrainedWrapper(base_model, tokenizer, DEFAULT_PROMPT_TEMPLATE.format(raw_prompt=raw_prompt))
        bt_model = BeamTuningWithEFTPosthocGenerationMixin(model, tokenizer)
        prompt = ""
        prompt_tokenized = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        eos_strings = ["\n", "<|endoftext|>"]
        outputs = bt_model.bon_beam_sample(
            input_ids=prompt_tokenized["input_ids"].cuda(),
            attention_mask=prompt_tokenized["attention_mask"].cuda(),
            scorer=scorer.set_raw_prompt(raw_prompt),
            top_k=None,
            max_new_tokens=128,
            temperature=0.7,
            eos_strings=eos_strings,
            num_beams=4,
            num_candidates=4,
            block_len=5,
        )
        response = extract_responses(outputs, tokenizer, prompt=prompt)[0]
        for eos_string in eos_strings:
            response = response.split(eos_string)[0]
        results.append({
            "prompt": raw_prompt,
            "response": response,
        })
    
    del model
    del scorer
    del tune_gpt2_small
    del base_gpt2_small

    # breakpoint()

    # evaluation
    from scripts.summarize_from_feedback.rm import LlamaModelForScore
    from inference_tools.utils import prepare_input

    generation = results
    rm = LlamaModelForScore.from_pretrained(
        get_local_model_name("gpt2-summarize-golden-rm"),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_flash_attention_2=True,
    )
    rm.eval()
    rm_tokenizer = AutoTokenizer.from_pretrained(get_local_model_name("gpt2-summarize-golden-rm"))

    results = []
    with torch.no_grad():
        for sample in tqdm.tqdm(generation):
            text = sample["prompt"] + "TL;DR: " + sample["response"] + rm_tokenizer.eos_token
            inputs  = prepare_input(rm_tokenizer(text, return_tensors="pt"))
            outputs = rm(**inputs)
            score   = outputs.end_scores.squeeze(0)[0].item()
            results.append({
                "prompt": sample["prompt"],
                "response": sample["response"],
                "score": score,
            })
        
    mean_score = sum([result["score"] for result in results]) / len(results)
    print(mean_score)

    import json
    with open("cbs_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # 20 sample:
    # base model: -1.8373291015625
    # tokenrm scorer(4,4,5): 1.4162109375

if __name__ == "__main__":
    unit_test()