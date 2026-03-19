from datasets import load_dataset
from transformers import AutoTokenizer

import torch
import torch.nn as nn

from transformers import LlamaModel, LlamaPreTrainedModel, PreTrainedModel, AutoModelForSequenceClassification, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.utils.doc import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.utils.generic import ModelOutput
from transformers.models.llama.modeling_llama import _CONFIG_FOR_DOC, LLAMA_INPUTS_DOCSTRING
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)

from typing import List, Optional, Tuple, Union
from typing import Any, ClassVar, Mapping

from dataclasses import dataclass


@dataclass
class TokenScoreModelOutputWithLoss(ModelOutput):
    """
    Output of the score model.
    """
    scores: torch.Tensor | None = None  # size = (B, L, D)
    end_scores: torch.Tensor | None = None  # size = (B, D)
    logits: torch.Tensor | None = None 
    past_key_values: list[torch.FloatTensor] | None = None

@dataclass
class ScoreModelOutput(ModelOutput):
    """
    Output of the score model.

    Args:
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, score_dim, sequence_length)`):
            Prediction scores of the score model.
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, score_dim)`):
            Prediction scores of the end of the sequence.
    """

    scores: torch.Tensor | None = None  # size = (B, L, D)
    end_scores: torch.Tensor | None = None  # size = (B, D)
    
@dataclass
class ScoreModelOutputWithPastkeyvalues(ModelOutput):
    """
    Output of the score model.

    Args:
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, score_dim, sequence_length)`):
            Prediction scores of the score model.
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, score_dim)`):
            Prediction scores of the end of the sequence.
        past_key_values (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
    """

    scores: torch.Tensor | None = None  # size = (B, L, D)
    end_scores: torch.Tensor | None = None  # size = (B, D)
    past_key_values: list[torch.FloatTensor] | None = None



class LlamaModelForTokenScore(LlamaPreTrainedModel):
    # _keys_to_ignore_on_load_missing = [r"lm_head.weight", ]

    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__(config)

        self.model = LlamaModel(config)
        config.architectures = [self.__class__.__name__]
        self.score_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    
    def get_end_hidden_states(self, hidden_states, attention_mask, shift_index=0):
        end_hidden_states = []
        end_indexs = []
        for i in range(hidden_states.size(0)):
            end_index = attention_mask[i].nonzero()[-1].item() 
            end_indexs.append(end_index)
            end_hidden_states.append(hidden_states[i, end_index + shift_index])
        return torch.stack(end_hidden_states, dim=0)
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        coefficients_labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        assert attention_mask is not None
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = self.score_head(hidden_states)

        if attention_mask.shape[1] != hidden_states.shape[1]:
            # end_logits = logits[..., -1, :] # shape: (B, D)
            # end_labels = input_ids[..., -1].to(end_logits.device) # shape: (B)
            # end_scores = end_logits.gather(-1, end_labels.unsqueeze(-1)).squeeze(-1) # shape: (B)
            return TokenScoreModelOutputWithLoss(
                # end_scores=end_scores,
                logits=logits,
                past_key_values=outputs.past_key_values,
            )
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous().to(shift_logits.device)
        scores = shift_logits.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1) # size: (B, L)
        end_score = self.get_end_hidden_states(scores, attention_mask, shift_index=-1) # shift_index = -1
        
        if not return_dict:
            return scores, end_score
        
        return TokenScoreModelOutputWithLoss(
            scores=scores,
            end_scores=end_score, 
            logits=logits, # size: (B, L, D)
            past_key_values=outputs.past_key_values,
        )

class LlamaModelForScore(LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = ['lm_head.weight']

    def __init__(self, config: PretrainedConfig, **kwargs: Any) -> None:
        super().__init__(config)
        self.model = LlamaModel(config)

        config.architectures = [self.__class__.__name__]
        # self.init_score_head(config, hidden_size=config.hidden_size, **kwargs)
        config.score_dim = kwargs.pop('score_dim', getattr(config, 'score_dim', 1))
        self.score_head = nn.Linear(config.hidden_size, config.score_dim, bias=config.bias)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> None:
        return None

    def set_decoder(self, decoder: PreTrainedModel) -> None:
        self.model = decoder

    def get_decoder(self) -> PreTrainedModel:
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ScoreModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(  # pylint: disable=too-many-arguments
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        labels: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | ScoreModelOutput | ScoreModelOutputWithPastkeyvalues:
        """
        Args:

        Returns:

        Examples:

        ```python
        >>> from safe_rlhf.models.llama.modeling_llama import LlamaModelForScore
        >>> from transformers import LlamaTokenizer

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        # got score
        >>> outputs = model(**inputs)
        >>> scores = outputs.scores
        >>> scores
        tensor([[[0.0000]]])
        ```
        """
        assert attention_mask is not None
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # size = (B, L, E)
        scores = self.score_head(hidden_states)  # size = (B, L, D)

        if attention_mask.shape[1] != hidden_states.shape[1]: # always happen during generation
            return ScoreModelOutputWithPastkeyvalues(
                scores=scores,
                end_scores=scores[:, -1, :],
                past_key_values=outputs.past_key_values,
            )
        
        end_score = []
        for i in range(hidden_states.size(0)):
            end_index = attention_mask[i].nonzero()[-1].item()
            end_score.append(scores[i, end_index])  # size = (D,)
        end_scores = torch.stack(end_score, dim=0)  # size = (B, D)

        
        return ScoreModelOutputWithPastkeyvalues(
            scores=scores,
            end_scores=end_scores,
            past_key_values=outputs.past_key_values,
        ) 


def unit_test_llama_for_token_scorer():
    from datasets import load_dataset, load_from_disk
    import random
    from tqdm import tqdm
    dataset = load_from_disk("/dataset/ultrafeedback_binarized")
    # dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized",split='test_prefs')

    import sys
    sys.path.insert(0, "..")
    from utils import get_local_model_name

    acc = []
    dataset = [d for d in dataset]
    dataset = random.sample(dataset, 1000)

    # model_path = get_local_model_name("llama-instruction-following-token-rm")
    model_path = "/models/llama2-7b-hf"
    # model = LlamaModelForTokenScore.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", )
    model = LlamaModelForTokenScore.from_pretrained(model_path, device_map="cuda:0", load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    for d in tqdm(dataset):
        response_chosen = d['chosen']
        response_rejected = d['rejected']
        format_response_chosen = "\n\nHuman: {prompt} \n\nAssistant: {response}".format(prompt = d['prompt'], response = response_chosen[1]['content'])
        format_response_rejected = "\n\nHuman: {prompt} \n\nAssistant: {response}".format(prompt = d['prompt'], response = response_rejected[1]['content'])
        responses = [format_response_chosen, format_response_rejected]
        input_ids = tokenizer(responses, return_tensors="pt", padding=True, truncation=True, max_length=128)["input_ids"]
        attention_mask = input_ids != tokenizer.pad_token_id
        output = model(input_ids.cuda(), attention_mask=attention_mask.cuda(), return_dict=True)
        end_scores = output.end_scores.to(torch.float32)
        acc.append(end_scores[0] > end_scores[1])
        print(end_scores[0] > end_scores[1])
    print(sum(acc)/len(acc))
    # 0.7410 prompt

if __name__ == "__main__":
    unit_test_llama_for_token_scorer()