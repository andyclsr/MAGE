from dataclasses import dataclass
import warnings
from typing import Dict, List, Any, Optional, Union

import torch
from torch import nn
from transformers import PreTrainedModel, GenerationMixin

from collections import deque
from torch.nn import functional as F

__all__ = ['EFTPosthocGenerationMixin']




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
        
        # print('window:' , len(self.entropy_window), self.window_size)
        if len(self.entropy_window) < self.window_size:
            return current_entropy, False, None, None
        
        window_tensor = torch.tensor(list(self.entropy_window))
        mean_entropy = torch.mean(window_tensor)
        std_entropy = torch.std(window_tensor)
        
        threshold = mean_entropy + self.k_std * std_entropy
        intervention_needed = current_entropy > threshold
        
        return current_entropy, intervention_needed, mean_entropy.item(), std_entropy.item()
    
    
@dataclass(kw_only=True)
class EFTPosthocGenerationMixin(GenerationMixin):
    """
    args: 
        `base`: 
            the base language model to be steered at inference time.
        `tune_r`, `base_r`: 
            `tune_r` are a list of N language models fine-tuned from `ref_r`, \
            Each model from `tune_r` can be combined with `ref_r` to define a implicit reward models r_i = logp_{tune_r[i]} - logp_{base_r}.
        `w`: 
            the linear weights for combining the implicit reward models r = sum_{i=1}^N r_i.

    The resulting sampling distribution is proportioal to 
          logp_{base} * r
        = logp_{base} * sum_{i=1}^{N}(w_i*r_i), 
    where r_i = logp_{tune_r[i]} - logp_{base_r}.
    """
    base:     PreTrainedModel
    tune_r:   List[PreTrainedModel] | PreTrainedModel
    base_r:   Optional[PreTrainedModel] = None
    w:        List[float] | float
    preference_weight: List[float] = None
    get_topk: Optional[bool] = False
    is_main_process: Optional[bool] = False
    candiate_len: Optional[int] = 100
    entropy_judge: Optional[bool] = False
    entropy_count: Optional[int] = 0
    step: Optional[int] = 0
    window_size: Optional[int] = 10
    k_std: Optional[float] = 1.0
    interval_to_guide: Optional[int] = 1

    def __post_init__(self):
        if not isinstance(self.tune_r, list):  self.tune_r = [self.tune_r] # TODO: here change the "not isinstance(self.base, list)" to "not isinstance(self.tune_r, list)"
        if not isinstance(self.w, list): self.w = [self.w]
        if not self.base_r: self.base_r = self.base
        if self.entropy_judge is True:
            self.entropy_monitor = SlidingWindowEntropyMonitor(self.window_size, self.k_std)
        assert len(self.tune_r) == len(self.w) 
        assert len(self.preference_weight) == len(self.w)

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

    def _parallel_decode(self, *args, past_key_values_list, **kwargs):
        models_outputs_list = []
        for i, model in enumerate(self.tune_r):
            outputs = model(*args, past_key_values=past_key_values_list[i], **kwargs)
            models_outputs_list.append(outputs)
        return models_outputs_list

    @torch.no_grad()
    def __call__(self, *args, past_key_values=None, **kwargs):
        if not past_key_values:
            past_key_values = {'base': None, 'tune_r': [None] * len(self.tune_r), 'base_r': None}

        base_outputs = self.base(*args, past_key_values=past_key_values['base'], **kwargs)
        base_logits  = base_outputs.logits[:, -1, :]

        if self.entropy_judge is True:
            current_entropy, intervention_needed, mean_entropy, std_entropy = self.entropy_monitor.update(base_logits)
        else:
            intervention_needed = True

        self.step+=1

        ##### here the part of using r #####
        if self.base == self.base_r:
            base_r_outputs = base_outputs
            base_r_logits  = base_logits
        else:
            base_r_outputs = self.base_r(*args, past_key_values=past_key_values['base_r'], **kwargs)
            base_r_logits  = base_r_outputs.logits[:, -1, :]

        tune_r_outputs_list = self._parallel_decode(*args, past_key_values_list=past_key_values['tune_r'], **kwargs)
        tune_r_logits_list  = [outputs.logits[:, -1, :] for outputs in tune_r_outputs_list]

        tune_r_logits = torch.stack(tune_r_logits_list, dim=-1)
        w = torch.tensor(self.w).to(tune_r_logits.device)
        preference_weight = torch.tensor(self.preference_weight).to(tune_r_logits.device)
        r = (preference_weight * w * (tune_r_logits - base_r_logits.unsqueeze(-1))).sum(-1)
        ###################################
        ###################################
        
        if self.entropy_judge is True and intervention_needed is False:
            logits = base_logits
        else:
            if self.step % self.interval_to_guide == 0:
                self.entropy_count += 1
                if self.get_topk:
                    base_topk = torch.topk(base_logits, self.candiate_len, dim=-1)
                    logits = base_logits
                    for i in range(len(logits)):
                        logits[i, base_topk.indices[i]] += r[i, base_topk.indices[i]]
                else:
                    logits = base_logits + r
            else:
                logits = base_logits

        # print('entropy' ,self.step, self.entropy_count)

        outputs = base_outputs
        outputs.logits = logits.unsqueeze(-2)
        outputs.past_key_values = {
            'base':   base_outputs.past_key_values,
            'tune_r': [outputs.past_key_values for outputs in tune_r_outputs_list],
            'base_r': base_r_outputs.past_key_values,
        }
        return outputs
