from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import os
import time

import GPUtil
import torch
from torch import nn
from transformers import GenerationMixin, PreTrainedModel, PreTrainedTokenizer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation.stopping_criteria import MaxLengthCriteria, StoppingCriteriaList

from inference_tools.decoder.eft import EFTPosthocGenerationMixin
from inference_tools.decoder.ivg import TokenWiseSamplingMixin
from inference_tools.model import PrefixPreTrainedWrapper
from inference_tools.scorer import BaseScorer
from inference_tools.utils import StopOnStringCriteria, extract_responses, get_truncated_responses


__all__ = ["BeamTuningWithEFTPosthocGenerationMixin"]


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
            if past_key_values["base"]:
                past_key_values["base"] = self.base.base._reorder_cache(past_key_values["base"], beam_idx)
            if past_key_values["tune_r"]:
                past_key_values["tune_r"] = [
                    model._reorder_cache(past_key_values["tune_r"][i], beam_idx)
                    for i, model in enumerate(self.base.tune_r)
                ]
            if past_key_values["base_r"]:
                past_key_values["base_r"] = self.base.base_r._reorder_cache(past_key_values["base_r"], beam_idx)
        elif isinstance(self.base, TokenWiseSamplingMixin):
            if past_key_values["base"]:
                past_key_values["base"] = self.base.base._reorder_cache(past_key_values["base"], beam_idx)
            if past_key_values["token_reward"]:
                past_key_values["token_reward"] = self.base.token_reward_model._reorder_cache(
                    past_key_values["token_reward"], beam_idx
                )
        elif isinstance(self.base, PrefixPreTrainedWrapper):
            past_key_values = self.base._reorder_cache(past_key_values, beam_idx)
        return past_key_values

    def _reorder_input_ids(self, beam_idx):
        if isinstance(self.base, EFTPosthocGenerationMixin):
            if isinstance(self.base.base, PrefixPreTrainedWrapper):
                self.base.base.input_ids = self.base.base.input_ids[beam_idx]
                self.base.base.attention_mask = self.base.base.attention_mask[beam_idx]
            if all(isinstance(model, PrefixPreTrainedWrapper) for model in self.base.tune_r):
                for model in self.base.tune_r:
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
        do_sample: Optional[bool] = True,
        temperature: Optional[float] = None,
        top_k: Optional[int] = 0,
        top_p: Optional[float] = None,
        eos_strings: Optional[int] = None,
        split_by_prompt_text: Optional[bool] = True,
        **kwargs,
    ):
        logits_warper = []
        if do_sample and temperature and temperature != 1.0:
            logits_warper.append(TemperatureLogitsWarper(temperature))
        if do_sample and top_k and top_k > 0:
            logits_warper.append(TopKLogitsWarper(top_k))
        if do_sample and top_p and top_p < 1.0:
            logits_warper.append(TopPLogitsWarper(top_p))
        logits_warper = LogitsProcessorList(logits_warper)

        stopping_criteria = []
        if eos_strings:
            stopping_criteria.extend(
                [StopOnStringCriteria(input_ids.size(1), eos_string, self.tokenizer) for eos_string in eos_strings]
            )
        assert not (max_new_tokens and max_length)
        if max_length:
            stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
        if max_new_tokens:
            stopping_criteria.append(MaxLengthCriteria(max_length=input_ids.size(1) + max_new_tokens))
        stopping_criteria = StoppingCriteriaList(stopping_criteria)
        self.generation_config.pad_token_id = kwargs["eos_token_id"]
        kwargs.update({"use_cache": True})
        return self._bon_beam_sample(
            input_ids,
            scorer,
            stopping_criteria=stopping_criteria,
            logits_warper=logits_warper,
            eos_strings=eos_strings,
            split_by_prompt_text=split_by_prompt_text,
            do_sample=do_sample,
            **kwargs,
        )

    def time_record(self, step, start):
        def get_total_gpu_mem():
            used_gpu_ids = os.getenv("CUDA_VISIBLE_DEVICES")
            gpu_ids = list(map(int, used_gpu_ids.split(",")))
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
        do_sample: Optional[bool] = True,
        **model_kwargs,
    ) -> Union[Dict, torch.LongTensor]:
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        assert not (eos_strings and eos_token_id)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

        input_ids = input_ids.repeat(num_beams * num_candidates, 1)
        model_kwargs["attention_mask"] = model_kwargs["attention_mask"].repeat(num_beams * num_candidates, 1)

        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
        blocks_to_use = block_len
        prompt, prompt_len = self.tokenizer.decode(input_ids[0]), input_ids.size(1)

        this_peer_finished = False
        step = 0
        start = time.time()

        while True:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self.__call__(**model_inputs, return_dict=True)

            next_token_logits = outputs.logits[:, -1, :].float()
            next_token_logits = logits_processor(input_ids, next_token_logits)
            next_token_logits = logits_warper(input_ids, next_token_logits)

            if do_sample:
                probs = nn.functional.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1).to(torch.int64)
            blocks_to_use -= 1

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False
            )

            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            criteria_result = stopping_criteria(input_ids, None)
            if isinstance(criteria_result, bool):
                if criteria_result:
                    this_peer_finished = True
            elif criteria_result.all():
                this_peer_finished = True

            del outputs
            if blocks_to_use <= 0 or this_peer_finished is True:
                if this_peer_finished is True and blocks_to_use > 10000:
                    break
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

                _, beam_idx = torch.topk(beam_scores, num_beams, dim=0, largest=True, sorted=True)
                beam_idx = beam_idx.repeat(num_candidates)

                input_ids = input_ids[beam_idx, :]
                unfinished_sequences = unfinished_sequences[beam_idx]

                if unfinished_sequences.max().item() == 0:
                    this_peer_finished = True

                if model_kwargs["past_key_values"] is not None:
                    model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

                self._reorder_input_ids(beam_idx)
                if this_peer_finished is True:
                    break
            step += 1
            self.time_record(step, start)

        if return_dict_in_generate:
            return {
                "output_ids": input_ids[:num_beams],
                "scores": beam_scores[:num_beams],
            }
        return input_ids[0, None]
