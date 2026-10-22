
# Locate the release root for package imports during direct execution.
import sys as _release_sys
from pathlib import Path as _ReleasePath
_release_root = _ReleasePath(__file__).resolve().parents[1]
if str(_release_root) not in _release_sys.path:
    _release_sys.path.insert(0, str(_release_root))

########################
# This script is modified from the TRL package https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/reward_modeling.py
# This script is designed for the reward modeling with Gemma model but can also be applied to any models with a chat template and an official pad token
# If you have any question, feel free to send me an email via wx13@illinois.edu
########################
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
# from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy

import pdb


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"})

    deepspeed: Optional[str] = field(
        # default="dp3.json",
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=8)
    learning_rate: Optional[float] = field(default=1e-5)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="models/Qwen3-8B",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    random_init: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, only the score head is randomly initialized (backbone loads pretrained weights).",
        },
    )
    score_head_init_std: Optional[float] = field(
        default=None,
        metadata={"help": "If set, reinitialize the score head with this std."},
    )
    preference_label_smoothing: Optional[float] = field(
        default=0.0,
        metadata={"help": "Symmetric label smoothing for pairwise preference loss."},
    )
    reward_l2_coef: Optional[float] = field(
        default=0.0,
        metadata={"help": "L2 penalty coefficient on chosen/rejected rewards."},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_set_path: Optional[str] = field(
        default=None,
        metadata={"help": "The dir of the subset of the training data to use"},
    )
    eval_set_path: Optional[str] = field(
        default=None,
        metadata={"help": "The dir of the subset of the eval data to use"},
    )
    test_set_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional held-out test set logged alongside every eval."},
    )
    output_path: Optional[str] = field(
        default='outputs/reward_model',
        metadata={"help": "The dir for output model"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        # default="adamw_hf",
        default="paged_adamw_32bit",
        # default="adamw_torch_fused",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=4096)

    save_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Save the model every x steps (disabled by default)."},
    )
    eval_every_steps: Optional[int] = field(
        default=20,
        metadata={"help": "Evaluate and checkpoint every x optimizer steps."},
    )
    torch_compile: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable torch.compile for faster training."},
    )
    report_to: Optional[str] = field(
        default="none",
        metadata={"help": "Reporting backend: wandb, none, etc."},
    )
    eval_max_samples: Optional[int] = field(
        default=256,
        metadata={"help": "Deterministic maximum number of natural validation pairs."},
    )
    test_max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Optional deterministic maximum number of test pairs."},
    )
    holdout_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help": "If set, split this fraction of the train set as a holdout eval set "
            "(ignores eval_set_path). e.g. 0.05 -> 5%% of train used for early-stop eval."
        },
    )
    train_max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Optional deterministic training cap for smoke tests."},
    )
    early_stopping_patience: Optional[int] = field(default=3)
    early_stopping_threshold: Optional[float] = field(default=0.001)
    disable_early_stopping: Optional[bool] = field(
        default=False,
        metadata={"help": "Disable the EarlyStoppingCallback and always run all epochs."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Trainer checkpoint directory from which to resume."},
    )
    save_total_limit: Optional[int] = field(default=3)
    logging_steps: Optional[int] = field(default=5)
    seed: Optional[int] = field(default=42)
    run_name: Optional[str] = field(default="qwen3_8b_hard_negative_rm_earlystop")
    metric_for_best_model: Optional[str] = field(default="eval_accuracy")
    greater_is_better: Optional[bool] = field(default=True)

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
if not script_args.train_set_path:
    parser.error("--train_set_path is required; training datasets are not bundled")
if not script_args.eval_set_path and script_args.holdout_ratio is None:
    parser.error("provide --eval_set_path or --holdout_ratio for validation")
if not 0.0 <= script_args.preference_label_smoothing < 0.5:
    raise ValueError("preference_label_smoothing must be in [0, 0.5)")
if script_args.reward_l2_coef < 0.0:
    raise ValueError("reward_l2_coef must be non-negative")

# Load the value-head model and tokenizer.
tokenizer_name = script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Adjusted according to the base model
# Need to do this for the models that don't have an official pad token.
tokenizer.truncation_side = "left"
tokenizer.model_max_length = script_args.max_length
tokenizer.bos_token = tokenizer.eos_token

tokenizer.pad_token = tokenizer.eos_token

# Get the dataset
train_path = script_args.train_set_path
eval_path = script_args.eval_set_path
test_path = script_args.test_set_path
output_name = script_args.output_path

def build_dataset(tokenizer, train_path, eval_path, test_path=None):

    def tokenize(sample):
        
        sample['positive'] = tokenizer.apply_chat_template(
            sample['chosen'], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
        sample['negative'] = tokenizer.apply_chat_template(
            sample['rejected'], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
        
        tokenized_pos = tokenizer(sample['positive'], truncation=True)
        tokenized_neg = tokenizer(sample['negative'], truncation=True)
        sample["input_ids_j"] = tokenized_pos["input_ids"]
        sample["attention_mask_j"] = tokenized_pos["attention_mask"]
        sample["input_ids_k"] = tokenized_neg["input_ids"]
        sample["attention_mask_k"] = tokenized_neg["attention_mask"]
        return sample
    
    train_dataset = load_dataset(
        "json", data_files={"train": train_path}, split="train"
    ).shuffle(seed=script_args.seed)
    if script_args.train_max_samples is not None:
        train_dataset = train_dataset.select(
            range(min(script_args.train_max_samples, len(train_dataset)))
        )
    if script_args.holdout_ratio is not None:
        n = len(train_dataset)
        n_eval = int(round(n * script_args.holdout_ratio))
        eval_dataset = train_dataset.select(range(n - n_eval, n))
        train_dataset = train_dataset.select(range(0, n - n_eval))
    else:
        eval_dataset = load_dataset(
            "json", data_files={"eval": eval_path}, split="eval"
        ).shuffle(seed=script_args.seed)
    if script_args.eval_max_samples is not None:
        eval_dataset = eval_dataset.select(
            range(min(script_args.eval_max_samples, len(eval_dataset)))
        )
    test_dataset = None
    if test_path is not None:
        test_dataset = load_dataset(
            "json", data_files={"test": test_path}, split="test"
        )
        if script_args.test_max_samples is not None:
            test_dataset = test_dataset.select(
                range(min(script_args.test_max_samples, len(test_dataset)))
            )
    train_dataset = train_dataset.map(tokenize, num_proc=16)
    eval_dataset = eval_dataset.map(tokenize, num_proc=16)
    if test_dataset is not None:
        test_dataset = test_dataset.map(tokenize, num_proc=16)

    return train_dataset, eval_dataset, test_dataset


train_dataset, eval_dataset, test_dataset = build_dataset(
    tokenizer, train_path, eval_path, test_path
)
print("Training set: ", len(train_dataset))
if script_args.holdout_ratio is not None:
    print(f"Holdout validation set (ratio={script_args.holdout_ratio}): ", len(eval_dataset))
else:
    print("Natural validation set: ", len(eval_dataset))
if test_dataset is not None:
    print("Held-out test set: ", len(test_dataset))

# Define the trainer
training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=script_args.eval_every_steps,
    save_steps=script_args.save_every_steps,
    save_total_limit=script_args.save_total_limit,
    load_best_model_at_end=True,
    metric_for_best_model=script_args.metric_for_best_model,
    greater_is_better=script_args.greater_is_better,
    eval_on_start=True,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=script_args.logging_steps,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_ratio=0.03,
    torch_compile=script_args.torch_compile,
    report_to=script_args.report_to,
    run_name=script_args.run_name,
    seed=script_args.seed,
    data_seed=script_args.seed,
)

# enable if you want to train with lora
# peft_config = LoraConfig(
#     task_type=TaskType.SEQ_CLS,
#     inference_mode=False,
#     r=8,
#     lora_alpha=32,
#     lora_dropout=0.1,
# )

# Load the model (always from pretrained weights).
# random_init=True only reinitializes the score head, not the backbone.
model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

if script_args.random_init or script_args.score_head_init_std is not None:
    print("=" * 60)
    print("RANDOM INIT: Reinitializing the score head (backbone pretrained)")
    print("=" * 60)
    init_std = script_args.score_head_init_std or 0.01
    if hasattr(model, 'score'):
        score_parameters = [model.score.weight]
        if model.score.bias is not None:
            score_parameters.append(model.score.bias)
        # TrainingArguments initializes HfDeepSpeedConfig before model loading.
        # Under ZeRO-3 these parameters are already partitioned here, so a plain
        # nn.init call mutates only the one-element placeholder (the source of
        # the all-ones head seen in the broken LexChain runs). Gather, initialize
        # once on rank 0, then let DeepSpeed repartition and broadcast it.
        if hasattr(model.score.weight, "ds_id"):
            import deepspeed
            import torch.distributed as dist
            with deepspeed.zero.GatheredParameters(score_parameters, modifier_rank=0):
                rank = dist.get_rank() if dist.is_initialized() else 0
                if rank == 0:
                    nn.init.normal_(model.score.weight, std=init_std)
                    if model.score.bias is not None:
                        nn.init.zeros_(model.score.bias)
                    values = model.score.weight.detach().float()
                    print(
                        "  ZERO3_GATHERED_SCORE_INIT "
                        f"shape={tuple(values.shape)} mean={values.mean().item():.9g} "
                        f"std={values.std().item():.9g} norm={values.norm().item():.9g}",
                        flush=True,
                    )
        else:
            nn.init.normal_(model.score.weight, std=init_std)
            if model.score.bias is not None:
                nn.init.zeros_(model.score.bias)
            values = model.score.weight.detach().float()
            print(
                "  STANDARD_SCORE_INIT "
                f"shape={tuple(values.shape)} mean={values.mean().item():.9g} "
                f"std={values.std().item():.9g} norm={values.norm().item():.9g}",
                flush=True,
            )
        print(f"  Reinitialized model.score: {model.score}")
    else:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                print(f"  Reinitialized {name}: {module}")
                break
# model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = not script_args.gradient_checkpointing
num_proc = 24  # Can adjust to be higher if you have more processors.
original_columns = train_dataset.column_names


# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
        }
        return batch


# Define the trainer
def compute_metrics(eval_pred):
    pos_predictions_scores = np.asarray(eval_pred.predictions[0]).reshape(-1)
    neg_predictions_scores = np.asarray(eval_pred.predictions[1]).reshape(-1)
    margins = pos_predictions_scores - neg_predictions_scores
    return {
        "accuracy": float(np.mean(margins > 0)),
        "reward_chosen": float(np.mean(pos_predictions_scores)),
        "reward_rejected": float(np.mean(neg_predictions_scores)),
        "reward_margin": float(np.mean(margins)),
        "reward_rms": float(
            np.sqrt(
                np.mean(
                    np.concatenate([pos_predictions_scores, neg_predictions_scores]) ** 2
                )
            )
        ),
    }


class RewardTrainer(Trainer):
    def __init__(self, *args, test_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_dataset = test_dataset
        self._train_reward_chosen = []
        self._train_reward_rejected = []
        self._train_rank_loss = []
        self._train_reward_l2_loss = []

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        rewards = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        margins = rewards_j - rewards_k
        smoothing = script_args.preference_label_smoothing
        rank_loss = (
            -(1.0 - smoothing) * nn.functional.logsigmoid(margins)
            - smoothing * nn.functional.logsigmoid(-margins)
        ).mean()
        reward_l2_loss = script_args.reward_l2_coef * 0.5 * (
            rewards_j.float().pow(2) + rewards_k.float().pow(2)
        ).mean()
        loss = rank_loss + reward_l2_loss
        if model.training:
            self._train_reward_chosen.append(float(rewards_j.detach().float().mean()))
            self._train_reward_rejected.append(float(rewards_k.detach().float().mean()))
            self._train_rank_loss.append(float(rank_loss.detach().float()))
            self._train_reward_l2_loss.append(float(reward_l2_loss.detach().float()))
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss

    def log(self, logs, start_time=None):
        logs = dict(logs)
        if "loss" in logs and self._train_reward_chosen:
            chosen = float(np.mean(self._train_reward_chosen))
            rejected = float(np.mean(self._train_reward_rejected))
            logs.update({
                "train_reward_chosen": chosen,
                "train_reward_rejected": rejected,
                "train_reward_margin": chosen - rejected,
                "train_rank_loss": float(np.mean(self._train_rank_loss)),
                "train_reward_l2_loss": float(np.mean(self._train_reward_l2_loss)),
            })
            self._train_reward_chosen.clear()
            self._train_reward_rejected.clear()
            self._train_rank_loss.clear()
            self._train_reward_l2_loss.clear()
        return super().log(logs, start_time=start_time)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        if metric_key_prefix == "eval" and self.test_dataset is not None:
            test_metrics = super().evaluate(
                eval_dataset=self.test_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix="test",
            )
            metrics.update(test_metrics)
            # TrainingArguments automatically prefixes metric_for_best_model
            # with "eval_". Expose an alias so test_accuracy can be used for
            # checkpoint selection while retaining the readable test_* logs.
            if "test_accuracy" in test_metrics:
                metrics["eval_test_accuracy"] = test_metrics["test_accuracy"]
        return metrics


# Train the model, woohoo.
callbacks = []
if not script_args.disable_early_stopping:
    callbacks.append(
        EarlyStoppingCallback(
            early_stopping_patience=script_args.early_stopping_patience,
            early_stopping_threshold=script_args.early_stopping_threshold,
        )
    )

trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    test_dataset=test_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(
        tokenizer=tokenizer, max_length=script_args.max_length),
    callbacks=callbacks,
)


trainer.train(resume_from_checkpoint=script_args.resume_from_checkpoint)

print(f"Best trainer checkpoint: {trainer.state.best_model_checkpoint}")
print(f"Best validation metric: {trainer.state.best_metric}")
print("Saving loaded best model")
best_output = output_name + "/best_checkpoint"
trainer.save_model(best_output)
tokenizer.save_pretrained(best_output)
Path(output_name, "best_checkpoint_info.json").write_text(
    __import__("json").dumps(
        {
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "metric_for_best_model": training_args.metric_for_best_model,
            "natural_eval_set": str(eval_path),
            "natural_eval_samples": len(eval_dataset),
            "test_set": str(test_path) if test_path is not None else None,
            "test_samples": len(test_dataset) if test_dataset is not None else 0,
            "holdout_ratio": script_args.holdout_ratio,
            "preference_label_smoothing": script_args.preference_label_smoothing,
            "reward_l2_coef": script_args.reward_l2_coef,
            "score_head_init_std": script_args.score_head_init_std,
            "early_stopping_disabled": script_args.disable_early_stopping,
            "resumed_from_checkpoint": script_args.resume_from_checkpoint,
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
