# RM Training

## Overview

Train a scalar reward model from chat-format preference pairs. The training implementations include validation, early stopping, best-checkpoint selection, and optional test metrics. All commands below run from the repository root. Replace `/path/to/...` with your local model and data paths.

## Files

| Python file | Purpose |
| --- | --- |
| `train_pairwise.py` | Train a pretrained sequence-classification backbone with a scalar reward head using chosen/rejected preference pairs. Supports validation or a holdout split, regularization, early stopping, and checkpoint resume. |
| `train_pairwise_zero3.py` | Provide the same training workflow with a ZeRO-3-aware score-head initialization path: gather partitioned head parameters, initialize on rank 0, then redistribute them. |

Supporting files:

| File | Purpose |
| --- | --- |
| `train.sh` | Launch the standard trainer, or the ZeRO-3 trainer when `USE_DEEPSPEED=1`. |
| `configs/deepspeed_zero3.json` | Configure BF16, ZeRO stage 3, and automatic batch-size settings for DeepSpeed. |

Standalone evaluation, candidate ranking, tests, and release-validation tools are not included in this directory.

## Requirements

Recommended versions are pinned to the current `vllm19` environment: Python 3.11.15, PyTorch 2.10.0 with CUDA 12.8, and Transformers 4.57.6. Install the training dependencies, then the matching FlashAttention version:

```bash
pip install -r requirements-training.txt
pip install --no-build-isolation flash-attn==2.8.3
```

Provide a CUDA-capable GPU environment, sufficient model memory, and compatible PyTorch/Transformers versions. Both scripts request BF16 and `flash_attention_2`, so a compatible FlashAttention installation is required. Multi-GPU ZeRO-3 training also needs Accelerate and DeepSpeed configured for your machine.

The default optimizer is `paged_adamw_32bit`; the training requirements include the reference environment's `bitsandbytes==0.49.2`. The examples below explicitly select `--optim adamw_torch` instead. No GPU training is performed by the `--help` commands.

## Inputs and Outputs

Both Python files accept the same preference-data format. Each JSONL record contains `chosen` and `rejected` chat-message arrays:

```json
{"chosen": [{"role": "user", "content": "Case details and instructions"}, {"role": "assistant", "content": "Preferred answer"}], "rejected": [{"role": "user", "content": "Case details and instructions"}, {"role": "assistant", "content": "Less preferred answer"}]}
```

Set `--model_name` to your pretrained model and supply the required `--train_set_path` preference data. Supply either `--eval_set_path` or `--holdout_ratio` for validation. Although `--model_name` has a default local path, the examples set it explicitly for the target environment. An optional `--test_set_path` adds test metrics during evaluation.

| Output | Location and contents |
| --- | --- |
| Training checkpoints | Under `--output_path`, managed by the trainer and checkpoint settings. |
| Best model and tokenizer | `<output_path>/best_checkpoint/`. |
| Selection metadata | `<output_path>/best_checkpoint_info.json`, including the selected metric, data counts, and training settings. |

The model backbone is loaded from pretrained weights in both scripts. `--random_init true` reinitializes the score head rather than the entire backbone.

## Usage

### train_pairwise.py

Run the standard single-process training entry point:

```bash
python rm_training/train_pairwise.py \
    --model_name /path/to/Qwen3-8B \
    --train_set_path /path/to/train.jsonl \
    --eval_set_path /path/to/validation.jsonl \
    --output_path outputs/rm_pairwise \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --optim adamw_torch --report_to none
```

To create validation data from the training file instead, omit `--eval_set_path` and add `--holdout_ratio 0.05`. Add `--resume_from_checkpoint /path/to/checkpoint` to resume a run.

### train_pairwise_zero3.py

Launch the ZeRO-3 implementation on eight GPUs and explicitly reinitialize the score head:

```bash
accelerate launch --multi_gpu --num_processes 8 \
    rm_training/train_pairwise_zero3.py \
    --model_name /path/to/Qwen3-8B \
    --train_set_path /path/to/train.jsonl \
    --eval_set_path /path/to/validation.jsonl \
    --output_path outputs/rm_pairwise_zero3 \
    --deepspeed rm_training/configs/deepspeed_zero3.json \
    --random_init true \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --optim adamw_torch --report_to none
```

Set `--num_processes` to the GPU count you intend to use. Omit `--random_init true` to retain the loaded score head.

### train.sh

Use the launcher for the standard implementation:

```bash
MODEL=/path/to/Qwen3-8B \
TRAIN_FILE=/path/to/train.jsonl \
VALIDATION_FILE=/path/to/validation.jsonl \
OUTPUT_DIR=outputs/rm_launcher \
bash rm_training/train.sh --optim adamw_torch
```

Select the ZeRO-3 implementation through environment variables:

```bash
MODEL=/path/to/Qwen3-8B \
TRAIN_FILE=/path/to/train.jsonl \
VALIDATION_FILE=/path/to/validation.jsonl \
OUTPUT_DIR=outputs/rm_launcher_zero3 \
NUM_PROCESSES=8 USE_DEEPSPEED=1 \
bash rm_training/train.sh --optim adamw_torch --random_init true
```

`MODEL`, `TRAIN_FILE`, and `VALIDATION_FILE` are required by the launcher. `TEST_FILE` and `OUTPUT_DIR` are optional. Additional command-line arguments are forwarded to the selected Python trainer. The launcher supplies the DeepSpeed configuration automatically when `USE_DEEPSPEED=1`.

## Notes

- Prepare training and validation files from the constructed preference data before using the launcher. The direct Python entry points also support `--holdout_ratio`.
- `--report_to none` disables experiment-reporting backends in the direct examples; the launcher sets it by default.
- `--disable_early_stopping true` disables early stopping. `--preference_label_smoothing`, `--reward_l2_coef`, and `--score_head_init_std` expose the corresponding regularization and initialization controls.
- Examples describe the current CLI and have not been used to start GPU training as part of this documentation update.
- Inspect all available arguments without loading a model or starting training:

```bash
python rm_training/train_pairwise.py --help
python rm_training/train_pairwise_zero3.py --help
```
