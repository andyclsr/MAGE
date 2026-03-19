from dataclasses import dataclass
import warnings
from typing import Dict, List, Any, Optional, Union

import torch.nn.functional as F

import torch
from torch import nn
from transformers import PreTrainedModel, GenerationMixin
from collections import deque

class SlidingWindowEntropyMonitor:
    def __init__(self, window_size=10, k_std=1.0):
        self.window_size = window_size
        self.k_std = k_std
        self.entropy_window = deque(maxlen=window_size)

    def compute_entropy(self, logits):
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12))
        return entropy.item()

    def update(self, logits):
        current_entropy = self.compute_entropy(logits)
        self.entropy_window.append(current_entropy)
        
        if len(self.entropy_window) < self.window_size:
            return current_entropy, False, None, None
        
        window_tensor = torch.tensor(list(self.entropy_window))
        mean_entropy = torch.mean(window_tensor)
        std_entropy = torch.std(window_tensor)
        
        threshold = mean_entropy + self.k_std * std_entropy
        intervention_needed = current_entropy > threshold
        
        return current_entropy, intervention_needed, mean_entropy.item(), std_entropy.item()
    
    
__all__ = ['TokenWiseSamplingMixin']
@dataclass(kw_only=True)
class TokenWiseSamplingMixin(GenerationMixin):
    base: PreTrainedModel
    token_reward_model_list: List[PreTrainedModel]
    beta_list: List[float]
    w: float
    candiate_len: int
    get_topk: Optional[bool] = False
    step: Optional[int] = 0
    interval_to_guide: Optional[int] = 1
    entropy_judge: Optional[bool] = False
    entropy_count: Optional[int] = 0
    window_size: Optional[int] = 10
    k_std: Optional[float] = 1.0
    
    def __post_init__(self):
        if self.entropy_judge is True:
            self.entropy_monitor = SlidingWindowEntropyMonitor(self.window_size, self.k_std)

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.base, name)

    def prepare_inputs_for_generation(self, input_ids, **model_kwargs):
        if 'past_key_values' in model_kwargs:
            past_key_values = model_kwargs['past_key_values']
            model_kwargs['past_key_values'] = model_kwargs['past_key_values']['base']
        result = self.base.prepare_inputs_for_generation(input_ids, **model_kwargs)
        if 'past_key_values' in model_kwargs:
            result['past_key_values'] = past_key_values
        return result

    @torch.no_grad()
    def __call__(self, *args, past_key_values=None, **kwargs):
        if not past_key_values:
            past_key_values = {'base': None, 'token_reward': [None] * len(self.token_reward_model_list)}

        base_outputs = self.base(*args, past_key_values=past_key_values['base'], **kwargs)
        base_logits  = base_outputs.logits[:, -1, :]
        logits = base_logits
        if self.entropy_judge is True:
            current_entropy, intervention_needed, mean_entropy, std_entropy = self.entropy_monitor.update(logits)
        else:
            intervention_needed = True
        
        candiate_len = self.candiate_len
        topk_values = torch.topk(base_logits, candiate_len, dim=-1)
        indices = topk_values.indices
        self.step += 1

        if intervention_needed :
            self.entropy_count += 1
        # print('entropy count', self.entropy_count, self.step)

        new_past_key_values = {'base': base_outputs.past_key_values, 'token_reward': []}
        for idx, token_reward_model in enumerate(self.token_reward_model_list):
            token_reward_outputs = token_reward_model(*args, past_key_values=past_key_values['token_reward'][idx], **kwargs)
            token_reward_logits  = token_reward_outputs.logits[:, -1, :]
            if intervention_needed is True or (self.entropy_judge is not True and self.step % self.interval_to_guide == 0):
                min_reward = torch.min(token_reward_logits, dim=-1).values
                max_reward = torch.max(torch.abs(token_reward_logits-min_reward), dim=-1).values
                safe_denominator = torch.where(max_reward > 0, max_reward, torch.ones_like(max_reward))
                if self.get_topk is True:
                    for kk in range(len(logits)):
                        logits[kk, indices[kk]] += (
                            self.w[idx]
                            * (token_reward_logits[kk, indices[kk]] - min_reward[kk])
                            * self.beta_list[idx]
                            / safe_denominator[kk]
                        )
                else:
                    logits += self.w[idx] * (token_reward_logits - min_reward) * self.beta_list[idx] / safe_denominator
            
            new_past_key_values['token_reward'].append(token_reward_outputs.past_key_values)
        
        outputs = base_outputs
        # topk_values = torch.topk(logits, candiate_len, dim=-1)
        # indices = topk_values.indices
        # topk_soft_values = torch.topk(F.softmax(logits, dim=-1), candiate_len, dim=-1)
        # if logits.device == torch.device('cuda:0'):
            # print('final base', topk_values.values, indices)
            # print('final base soft', topk_soft_values.values, topk_soft_values.indices)
            # print('---'*20)
        
        outputs.logits = logits.unsqueeze(-2)
        outputs.past_key_values = new_past_key_values
        return outputs



