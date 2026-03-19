

from __future__ import annotations

import warnings
from typing import Any, ClassVar

import torch
from transformers import GPT2Model, GPT2PreTrainedModel, PreTrainedModel, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.models.gpt2.modeling_gpt2 import (
    DEPARALLELIZE_DOCSTRING,
    GPT2_INPUTS_DOCSTRING,
    GPT2_START_DOCSTRING,
    PARALLELIZE_DOCSTRING,
)
from transformers.utils.doc import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.utils.generic import ModelOutput

from dataclasses import dataclass

from torch import nn
from typing import List, Optional, Tuple, Union


@dataclass
class TokenScoreModelOutputWithLoss(ModelOutput):
    """
    Output of the score model.
        scores: torch.Tensor | None = None  # size = (B, L, D)
        end_scores: torch.Tensor | None = None  # size = (B, D)
        logits: torch.Tensor | None # size = (B, L, V)
        past_key_values: list[torch.FloatTensor] | None = None

    """
    scores: torch.Tensor | None = None 
    end_scores: torch.Tensor | None = None 
    logits: torch.Tensor | None = None 
    past_key_values: list[torch.FloatTensor] | None = None

@dataclass
class ScoreModelOutputWithPastkeyvalues(ModelOutput):
    """
    Output of the score model.
        scores: torch.Tensor | None = None  # size = (B, L, D)
        end_scores: torch.Tensor | None = None  # size = (B, D)
        past_key_values: list[torch.FloatTensor] | None = None
    """
    scores: torch.Tensor | None = None  # size = (B, L, D)
    end_scores: torch.Tensor | None = None  # size = (B, D)
    past_key_values: list[torch.FloatTensor] | None = None

@add_start_docstrings(
    """
    The GPT2 Model transformer with a score head on top (linear layer with weights tied to the input embeddings).
    The score head share the same architecture with the transformer causal head.
    """,
)
class GPT2ModelForTokenScore(GPT2PreTrainedModel):
    # _keys_to_ignore_on_load_missing = [r"lm_head.weight", ]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [
        'attn.masked_bias',
        'attn.bias',
        'lm_head.weight',
    ]

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs


    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__(config)

        self.transformer = GPT2Model(config)
        config.architectures = [self.__class__.__name__]
        self.score_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.model_parallel = False
        # self.device_map = None

        self.post_init()

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map: str | None = None) -> None:
        warnings.warn(
            '`GPT2LMHeadModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load'
            " your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'transformer.h.0':"
            " 0, 'transformer.h.1': 1, ...}",
            FutureWarning,
            stacklevel=1,
        )
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.score_head = self.score_head.to(self.transformer.first_device)
        self.model_parallel = True
                

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self) -> None:
        warnings.warn(
            'Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.',
            FutureWarning,
            stacklevel=1,
        )
        self.transformer.deparallelize()
        self.transformer = self.transformer.to('cpu')
        self.score_head = self.score_head.to('cpu')
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self) -> None:
        return None

    def set_decoder(self, decoder: PreTrainedModel) -> None:
        self.transformer = decoder

    def get_decoder(self) -> PreTrainedModel:
        return self.transformer
    
    def get_end_hidden_states(self, hidden_states, attention_mask):
        end_hidden_states = []
        end_indexs = []
        for i in range(hidden_states.size(0)):
            end_index = attention_mask[i].nonzero()[-1].item() # attention_mask[0].nonzero()[-1].item()
            end_indexs.append(end_index)
            end_hidden_states.append(hidden_states[i, end_index])
        # print("end_indexs:", end_indexs)
        return torch.stack(end_hidden_states, dim=0)



    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    def forward(  # pylint: disable=too-many-arguments
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | TokenScoreModelOutputWithLoss:
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

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        scores = self.score_head(hidden_states) 

        logits = scores.contiguous()

        if attention_mask.shape[1] != hidden_states.shape[1]:
            return TokenScoreModelOutputWithLoss(
                scores=scores,
                logits=logits,
                past_key_values=transformer_outputs.past_key_values,
            )

        zero_vector = torch.zeros((scores.shape[0], 1, scores.shape[2]), dtype=scores.dtype, device=scores.device)
        scores = torch.cat([zero_vector, scores[:, :-1, :]], dim=1)
        scores = torch.gather(scores, -1, input_ids.unsqueeze(-1).to(scores.device)).squeeze(-1)
        end_score = self.get_end_hidden_states(scores, attention_mask.to(scores.device))

        if not return_dict:
            return scores, end_score
        
        return TokenScoreModelOutputWithLoss(
            scores=scores,
            end_scores=end_score,  # size = (B, D)
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
        )

@add_start_docstrings(
    """
    The GPT2 Model transformer with a score head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT2_START_DOCSTRING,
)
class GPT2ForScore(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [
        'attn.masked_bias',
        'attn.bias',
        'lm_head.weight',
    ]

    def __init__(self, config: PretrainedConfig, **kwargs: Any) -> None:
        super().__init__(config)
        self.transformer = GPT2Model(config)

        config.architectures = [self.__class__.__name__]
        # self.init_score_head(config, hidden_size=config.hidden_size, **kwargs)
        config.score_dim = kwargs.pop('score_dim', getattr(config, 'score_dim', 1))
        self.score_head = nn.Linear(config.hidden_size, config.score_dim, bias=config.bias)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()
    
    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs


    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map: str | None = None) -> None:
        warnings.warn(
            '`GPT2LMHeadModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load'
            " your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'transformer.h.0':"
            " 0, 'transformer.h.1': 1, ...}",
            FutureWarning,
            stacklevel=1,
        )
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.score_head = self.score_head.to(self.transformer.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self) -> None:
        warnings.warn(
            'Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.',
            FutureWarning,
            stacklevel=1,
        )
        self.transformer.deparallelize()
        self.transformer = self.transformer.to('cpu')
        self.score_head = self.score_head.to('cpu')
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self) -> None:
        return None

    def set_decoder(self, decoder: PreTrainedModel) -> None:
        self.transformer = decoder

    def get_decoder(self) -> PreTrainedModel:
        return self.transformer

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    def forward(  # pylint: disable=too-many-arguments
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] :
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]  # size = (B, L, E)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
        scores = self.score_head(hidden_states)  # size = (B, L, D)

        if attention_mask.shape[1] != hidden_states.shape[1]:
            return ScoreModelOutputWithPastkeyvalues(
                scores=scores,
                end_scores=scores[:, -1, :],
                past_key_values=transformer_outputs.past_key_values,
            )


        end_score = []
        for i in range(hidden_states.size(0)):
            end_index = attention_mask[i].nonzero()[-1].item()
            end_score.append(scores[i, end_index])  # size = (D,)
        end_scores = torch.stack(end_score, dim=0)  # size = (B, D)

        
        return ScoreModelOutputWithPastkeyvalues(
            scores=scores,
            end_scores=end_scores,
            past_key_values=transformer_outputs.past_key_values,
        ) 


def unit_test_imdb_model():
    from datasets import load_dataset
    import random
    from tqdm import tqdm
    import sys
    sys.path.insert(0, "..")

    from utils import get_local_model_name
    dataset = load_dataset("ZHZisZZ/imdb_preference",split='test')

    acc = []
    dataset = [d for d in dataset]
    dataset = random.sample(dataset, 1000)

    model_path = get_local_model_name("gpt2-imdb-token-rm")
    model = GPT2ModelForTokenScore.from_pretrained(model_path, torch_dtype=torch.bfloat16, ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for d in tqdm(dataset):
        prompt = d['prompt']
        responses = [ prompt + response for response in d['responses'] ]
        is_better = d['chosen']
        input_ids = tokenizer(responses, return_tensors="pt", padding=True, truncation=True, max_length=1024,)["input_ids"]
        attention_mask = input_ids != tokenizer.pad_token_id
            
        output = model(input_ids.cuda(), attention_mask=attention_mask.cuda(), return_dict=True)
        end_scores = output.end_scores.to(torch.float32)
        acc.append((end_scores[0] < end_scores[1]) == is_better)
    print(sum(acc)/len(acc)) # 0.8110

def unit_test_summarize_model():
    from datasets import load_dataset
    import random
    from tqdm import tqdm
    import sys
    sys.path.insert(0, "..")
    from utils import get_local_model_name

    dataset = load_dataset("chadlzx/openai_summarize_comparisons_relabel",split='test')

    acc = []
    dataset = [d for d in dataset]
    dataset = random.sample(dataset, 1000)
    model_path = get_local_model_name("gpt2-summarize-token-rm")
    model = GPT2ModelForTokenScore.from_pretrained(model_path, torch_dtype=torch.bfloat16, ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for d in tqdm(dataset):
        prompt = d['prompt']
        responses = [prompt + d['chosen'], prompt+d['rejected'],]
        is_better = d['new_chosen_label']
        input_ids = tokenizer(responses, return_tensors="pt", padding=True, truncation=True, max_length=1024,)["input_ids"]
        attention_mask = input_ids != tokenizer.pad_token_id
        output = model(input_ids.cuda(), attention_mask=attention_mask.cuda(), return_dict=True)
        end_scores = output.end_scores.to(torch.float32)
        acc.append((end_scores[0] < end_scores[1]) == is_better)
    print(sum(acc)/len(acc))  # 0.6170



if __name__ == "__main__":
    # test_summarize_model()
    unit_test_imdb_model()
    unit_test_summarize_model()