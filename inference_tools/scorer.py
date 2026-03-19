from dataclasses import dataclass, asdict
from email import policy
from typing import Text, List, Dict, Optional
from abc import ABC, abstractclassmethod

import torch
from accelerate import Accelerator
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from inference_tools.utils import (
    SFTDataMapFunc, 
    SFTDataCollatorWithPadding,
    prepare_input,
    get_batch_logps,
)

__all__ = [
    "ImplicitRewardScorer",
    "ExplicitRewardScorer",
    "ClusterRewardScorer",
    "NotImplementedScorer",
]

DEFAULT_PROMPT_TEMPLATE = "{raw_prompt}"


@dataclass
class ScorerInput:
    response: List[str]
    eos: List[bool]


@dataclass
class BaseScorer(ABC):
    
    @abstractclassmethod
    def __call__(self, input: ScorerInput) -> torch.Tensor:
        raise NotImplementedError

@dataclass
class NotImplementedScorer(BaseScorer):

    def __init__(self):
        print("NotImplementedScorer is used, please implement your own scorer!")
        super().__init__()

    def set_raw_prompt(self, raw_prompt):
        self.raw_prompt = raw_prompt
        return self

    @torch.no_grad()
    def __call__(self, input: ScorerInput) -> torch.Tensor:
        return torch.zeros(len(input["response"]))

@dataclass
class ImplicitRewardScorer(BaseScorer):
    model: PreTrainedModel
    ref_model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    add_special_tokens: Optional[bool] = False
    model_prompt_template: Optional[str] = DEFAULT_PROMPT_TEMPLATE
    ref_model_prompt_template: Optional[str] = DEFAULT_PROMPT_TEMPLATE
    raw_prompt: Optional[str] = ''
    beta: Optional[bool] = 1.0
    average_log_prob: Optional[bool] = False
    reference_free: Optional[bool] = False
    label_pad_token_id: Optional[int] = -100

    def set_raw_prompt(self, raw_prompt):
        self.raw_prompt = raw_prompt
        return self

    @torch.no_grad()
    def __call__(self, input: ScorerInput) -> torch.Tensor:
        policy_all_logps = self.forward(
            self.model, 
            self.model_prompt_template, 
            input
        )
        if self.reference_free: return self.beta * policy_all_logps
        ref_all_logps = self.forward(
            self.ref_model, 
            self.ref_model_prompt_template, 
            input
        )
        return self.beta * (policy_all_logps - ref_all_logps)

    @torch.no_grad()
    def forward(
        self, 
        model: PreTrainedModel, 
        prompt_template: Text, 
        input: ScorerInput | Dict
    ) -> torch.Tensor:
        input = asdict(input) if isinstance(input, ScorerInput) else input
        input["prompt"] = [self.raw_prompt] * len(input["response"])

        tokens = SFTDataMapFunc(tokenizer=self.tokenizer, add_special_tokens=self.add_special_tokens)(input)
        batch  = SFTDataCollatorWithPadding(tokenizer=self.tokenizer)(
            [{k:v[i] for k,v in tokens.items()} for i in range(len(input["response"]))]
        )
        batch = prepare_input(batch)

        all_logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.to(torch.float32)
        all_logps = get_batch_logps(
            all_logits,
            batch["labels"],
            average_log_prob=self.average_log_prob,
            label_pad_token_id=self.label_pad_token_id,
        )
        return all_logps


@dataclass
class ExplicitRewardScorer(BaseScorer):
    token_reward_model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    add_special_tokens: Optional[bool] = True
    model_prompt_template: Optional[str] = DEFAULT_PROMPT_TEMPLATE
    raw_prompt: Optional[str] = ''
    beta: Optional[bool] = 1.0

    def set_raw_prompt(self, raw_prompt):
        self.raw_prompt = raw_prompt
        return self
    
    @torch.no_grad()
    def __call__(self, input: ScorerInput) -> torch.Tensor:
        # print(input)
        all_end_scores = self.forward(self.token_reward_model, self.model_prompt_template, input)
        return self.beta * all_end_scores
        
    @torch.no_grad()
    def forward(
        self, 
        model: PreTrainedModel, 
        prompt_template: Text, 
        input: ScorerInput | Dict
    ) -> torch.Tensor:
        input = asdict(input) if isinstance(input, ScorerInput) else input
        prompt = self.raw_prompt
        input["prompt"] = [prompt] * len(input["response"])
        # input["prompt"] = [prompt_template.format(raw_prompt=self.raw_prompt)] * len(input["response"])

        tokens = SFTDataMapFunc(tokenizer=self.tokenizer, add_special_tokens=self.add_special_tokens)(input)
        batch  = SFTDataCollatorWithPadding(tokenizer=self.tokenizer)(
            [{k:v[i] for k,v in tokens.items()} for i in range(len(input["response"]))]
        )
        batch = prepare_input(batch)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        all_end_scores = outputs.end_scores.to(torch.float32)
        if len(all_end_scores.shape) != 1:
            all_end_scores = all_end_scores.squeeze()
        
        # According to the number of response tokens, we could calculate the average score
        # all_end_scores = all_end_scores / ((self.tokenizer(input["response"], return_tensors="pt", padding=True )["attention_mask"].sum(dim=1).to(torch.float32) + 1).to(all_end_scores.device))
        
        return all_end_scores

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

@dataclass
class ClusterRewardScorer(BaseScorer):
    emb_model: PreTrainedModel
    add_special_tokens: Optional[bool] = True
    model_prompt_template: Optional[str] = DEFAULT_PROMPT_TEMPLATE
    raw_prompt: Optional[str] = ''
    beta: Optional[bool] = 1.0

    def set_raw_prompt(self, raw_prompt):
        self.raw_prompt = raw_prompt
        return self
    
    @torch.no_grad()
    def __call__(self, input: ScorerInput) -> torch.Tensor:
        all_end_scores = self.forward(self.emb_model, input)
        return self.beta * all_end_scores
        
    @torch.no_grad()
    def forward(
        self, 
        model: PreTrainedModel, 
        input: ScorerInput | Dict
    ) -> torch.Tensor:
        input = asdict(input) if isinstance(input, ScorerInput) else input
        responses = input["response"]
        prompt = self.raw_prompt
        response_embs = model.encode(responses, prompt_name="s2s_query")
        print(response_embs.shape)
        similarities = model.similarity(response_embs, response_embs)
        clustering = AgglomerativeClustering(n_clusters=4, linkage='complete')
        cluster_labels = clustering.fit_predict(1 - similarities.numpy())
        cluster_sizes = torch.tensor([list(cluster_labels).count(i) for i in cluster_labels])
        return cluster_sizes.float()


def unit_test_implict_reward_scorer():
    import sys
    sys.path.insert(0, ".")
    from utils import get_local_model_name

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from accelerate import Accelerator

    model = AutoModelForCausalLM.from_pretrained(
        get_local_model_name("gpt2-imdb-dpo"),
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().local_process_index},
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        get_local_model_name("gpt2-imdb"),
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().local_process_index},
    )

    tokenizer = AutoTokenizer.from_pretrained(get_local_model_name("gpt2-imdb"))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    implicit_reward = ImplicitRewardScorer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    implicit_reward.set_raw_prompt("I think this movie is ")

    result = implicit_reward({"response": ["exciting", "boring"], "eos": [False, False]})

    print(result)

def unit_test_explicit_reward_scorer():
    import sys
    sys.path.insert(0, ".")
    from utils import get_local_model_name

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from accelerate import Accelerator
    from inference_tools.modeling_llama import LlamaModelForTokenScore

    model = LlamaModelForTokenScore.from_pretrained(
        get_local_model_name("llama-instruction-following-token-rm"),
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().local_process_index},
    )

    tokenizer = AutoTokenizer.from_pretrained(get_local_model_name("llama-instruction-following-token-rm"))
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    explicit_reward = ExplicitRewardScorer(
        token_reward_model=model,
        tokenizer=tokenizer,
        add_special_tokens=True,
    )

    explicit_reward.set_raw_prompt("\n\nHuman: How to make a delicious lunch quickly? \n\nAssistant: ")

    response = [
        """Firstly, consider making a wrap. """,
        """You could quickly whip up a sandwich """,
        """For a delicious and quick lunch, try """,
    ]

    result = explicit_reward({"response": response, "eos": [False, False, False]})

    print(result)

def unit_test_cluster_reward_scorer():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True).cuda()

    cluster_reward = ClusterRewardScorer(
        emb_model=model,
    )

    cluster_reward.set_raw_prompt("I think this movie is ")

    response = [
        """Firstly, consider making a wrap. """,
        """You could quickly whip up a sandwich """,
        """For a delicious and quick lunch, try """,
        """Let's go to the cinema to watch a movie.""",
        """I think this movie is interesting.""",
        """I think this movie is boring.""",
        """I think this movie is exciting.""",
        """I think this movie is scary.""",
    ]

    result = cluster_reward({"response": response, "eos": [False, False, False]})

    print(result)

if __name__ == "__main__":
    unit_test_implict_reward_scorer()
    unit_test_explicit_reward_scorer()
    unit_test_cluster_reward_scorer()
