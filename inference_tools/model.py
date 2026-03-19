from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel


class ModelWrapper(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.model, name)


@dataclass
class PrefixPreTrainedWrapper(ModelWrapper):
    """
    Wrap a language model so that generations will be concatenated to the prefix string. 
    This is useful when multiple language models are decoded at the same time, but not sharing \
        the same template or context. In this case, please use the wrapper together with PosthocGenerationMixin.

    Example: 
        PrefixPreTrainedWrapper(
            model, 
            tokenizer,
            prefix="1234567"
        ).generate() -> "891011...."
    """
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    prefix: str
    add_special_tokens: Optional[bool] = False # XXX important change here 2025-2-26

    def __post_init__(self):
        self.tokenizer = deepcopy(self.tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.input_ids = None
        self.attention_mask = None
        self.cache_position = None
    
    def set_raw_prompt(self, prefix):
        self.prefix = prefix
        self.input_ids = None
        self.attention_mask = None
        self.cache_position = None
        
    
    def get_prefix_input_ids_length(self):
        return len(self.tokenizer(self.prefix)["input_ids"])

    def _concat_prefix_to_batch(self, input_ids):
        if not self.prefix:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
            return input_ids, attention_mask
        inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        # prefix_inputs = [self.prefix + input for input in inputs]
        prefix_inputs = [self.prefix + input for input in inputs]
        batch = self.tokenizer(
            text=prefix_inputs,
            # padding=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        ).to(input_ids.device)
        return batch["input_ids"], batch["attention_mask"]

    def __call__(
        self,
        input_ids,
        attention_mask,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_ids=None,
        **kwargs,
    ):
        # here we add the judgment of whether the self.input_ids is None or not
        if not past_key_values and (self.input_ids is None or self.input_ids.shape[1] == 0):
            old_input_ids = input_ids
            if not self.prefix:
                self.input_ids = old_input_ids
                self.attention_mask = attention_mask
            else:
                self.input_ids, self.attention_mask = self._concat_prefix_to_batch(old_input_ids)
            # if input_ids.device.index == 0:
            #     print('input_ids ', input_ids.shape, input_ids)
            #     print('self.input_ids', self.input_ids.shape, self.input_ids)
            if hasattr(self.model, "_get_initial_cache_position"):
                self.cache_position = self.model._get_initial_cache_position(
                    self.input_ids,
                    {"attention_mask": self.attention_mask, "use_cache": use_cache},
                )["cache_position"]
            else:
                self.cache_position = cache_position
        else:
            self.input_ids = torch.cat([self.input_ids, input_ids[:, -1, None]], dim=-1)
            if cache_position is not None:
                self.cache_position = cache_position

        prepare_kwargs = dict(kwargs)
        if position_ids is not None:
            prepare_kwargs["position_ids"] = position_ids
        if self.cache_position is not None:
            prepare_kwargs["cache_position"] = self.cache_position

        model_inputs = self.model.prepare_inputs_for_generation(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **prepare_kwargs,
        )
        model_outputs = self.model(**model_inputs)
        self.attention_mask = torch.cat(
            [self.attention_mask, self.attention_mask.new_ones((self.attention_mask.shape[0], 1))], dim=-1
        )
        if self.cache_position is not None:
            self.cache_position = self.cache_position[-1:] + 1
        return model_outputs

