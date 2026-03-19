## MAGE

MAGE is a training-and-inference pipeline for multi-objective alignment built around:

- SFT initialization
- single-reward PPO experts
- MORLHF training
- token-level reward modeling
- inference-time value guidance with `MAGE.py`


## Project Layout

- `MAGE.py`: main inference-time value-guided decoding script
- `sft/`: SFT training and LoRA merge
- `ppo.py`: single-reward PPO training
- `morlhf.py`: MORLHF training
- `train_token_rm.py`: token-level reward model training
- `vas_create_dataset.py`: build token-RM training data
- `vas_create_valid_dataset.py`: build half-response validation data
- `create_valid_dataset_rm_judge.py`: build RM-judge validation data
- `inference_tools/`: decoding, scorer, and wrapper utilities used by `MAGE.py`
- `token_rm/`: token reward model definitions
- `model_merging_methods/`: model merging helpers imported by `utils.py`

## Environment

The original codebase assumes Python with `torch`, `transformers`, `accelerate`, `trl`, `datasets`, `peft`, `numpy`, and `pandas`.

Example:

```bash
pip install torch transformers accelerate trl datasets peft numpy pandas tyro
```

Most scripts also assume multi-GPU training/evaluation through `accelerate`.

## Data And Models

Before running the pipeline, prepare:

- a base causal LM for SFT
- preference datasets such as HH-RLHF or summarization-from-feedback
- reward models for PPO/MORLHF evaluation
- merged SFT checkpoints if using LoRA

Many scripts still contain local hardcoded paths from the original research environment. For open-source use, update those paths or override them with CLI arguments where supported.

## Pipeline

### 1. Train SFT

Train the supervised fine-tuned model that initializes PPO and MORLHF.

```bash
cd sft
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch sft.py \
  --base_model_name "meta-llama/Llama-2-7b-hf" \
  --exp_type "assistant" \
  --wandb_name "sft_assistant"
```

For summarization:

```bash
cd sft
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch sft.py \
  --base_model_name "meta-llama/Llama-2-7b-hf" \
  --exp_type "summary" \
  --wandb_name "sft_summary"
```

### 2. Merge SFT LoRA

If SFT was trained with LoRA adapters, merge them into a standalone checkpoint before downstream training.

```bash
cd sft
python merge_sft_lora.py \
  --base_model_name "meta-llama/Llama-2-7b-hf" \
  --lora_name "/path/to/sft/lora" \
  --save_directory "/path/to/sft_merged"
```

### 3. Train PPO Experts

Train one PPO model per reward objective. These PPO experts are later merged or compared at inference time.

Example for `harmless`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch ppo.py \
  --base_model_name "/path/to/sft_merged" \
  --reward_name "harmless" \
  --exp_type "assistant" \
  --wandb_name "ppo_harmless"
```

Example for `helpful`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch ppo.py \
  --base_model_name "/path/to/sft_merged" \
  --reward_name "helpful" \
  --exp_type "assistant" \
  --wandb_name "ppo_helpful"
```

### 4. Train MORLHF

Train a single multi-objective policy using a preference-weighted combination of rewards.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch morlhf.py \
  --base_model_name "/path/to/sft_merged" \
  --reward_names "harmless,helpful" \
  --preference 0.5 \
  --exp_type "assistant" \
  --wandb_name "morlhf_assistant"
```

For more than two rewards, see the argument handling in `morlhf.py`.

### 5. Build Token-RM Training Dataset

Construct generation data for token-level reward modeling.

Assistant-style example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python vas_create_dataset.py \
  --base_model_name "/path/to/sft_merged" \
  --reward_names "harmless,helpful" \
  --exp_type "assistant" \
  --output_dataset_path "/path/to/tokenrm_train.csv"
```

### 6. Build Token-RM Validation Datasets

Create the half-response validation set:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python vas_create_valid_dataset.py \
  --base_model_name "/path/to/sft_merged" \
  --reward_names "harmless,helpful" \
  --exp_type "assistant" \
  --output_dataset_path "/path/to/tokenrm_valid_half.csv"
```

Create the RM-judge validation set:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python create_valid_dataset_rm_judge.py \
  --base_model_name "/path/to/sft_merged" \
  --reward_names "harmless,helpful" \
  --exp_type "assistant" \
  --output_dataset_path "/path/to/tokenrm_valid_judge.csv"
```

### 7. Train Token RM

Train the explicit token-level reward model used by value-guided decoding.

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train_token_rm.py \
  --base_model_name "/path/to/sft_merged" \
  --reward_name "harmless" \
  --dataset_path "/path/to/tokenrm_train.csv" \
  --run_name "token_rm_harmless"
```

Run again for each reward dimension you want to guide with, for example `helpful`.

### 8. Validate Token RM

Use the built-in validation flow during training, or run the auxiliary validation script:

```bash
python rm_valid.py
```

In practice you will usually want to edit the paths in `rm_valid.py` or adapt it to your own checkpoints and generated validation files.

### 9. Run MAGE Inference

`MAGE.py` is the main inference-time script for value-guided decoding. It can combine:

- merged base policies
- token-level reward models
- EFT-style implicit reward guidance

Example with explicit value (token-level RM) guidance:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch MAGE.py \
  --base_model_path_1 "/path/to/ppo_harmless" \
  --base_model_path_2 "/path/to/ppo_helpful" \
  --base_model_path_3 "/path/to/sft_merged" \
  --reward_names "harmless,helpful" \
  --tokenwisesampling_generation_configs.token_wise_sampling_type tokenrm \
  --run_name "mage_tokenrm"
```

Example with implicit value guidance:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch MAGE.py \
  --base_model_path_1 "/path/to/ppo_harmless" \
  --base_model_path_2 "/path/to/ppo_helpful" \
  --base_model_path_3 "/path/to/sft_merged" \
  --scorer_name1 "/path/to/value_model_1" \
  --scorer_name2 "/path/to/value_model_2" \
  --reward_names "harmless,helpful" \
  --tokenwisesampling_generation_configs.token_wise_sampling_type eft \
  --run_name "mage_eft"
```

## Typical End-to-End Workflow

1. Train SFT with `sft/sft.py`
2. Merge the SFT LoRA adapter with `sft/merge_sft_lora.py`
3. Train single-reward PPO experts with `ppo.py`
4. Optionally train MORLHF with `morlhf.py`
5. Build token-RM train/valid data with:
   `vas_create_dataset.py`, `vas_create_valid_dataset.py`, `create_valid_dataset_rm_judge.py`
6. Train one token RM per reward with `train_token_rm.py`
7. Run inference-time merging and decoding with `MAGE.py`

## Notes

- The exported code is intentionally focused on the original MAGE workflow rather than Qwen3-specific variants.
- Several scripts still use research-environment defaults and hardcoded paths. Expect to edit those paths for a clean public release.
- The import style assumes you run commands from the project root so that local modules like `utils.py`, `multi_reward_models.py`, and `inference_tools/` are importable.
