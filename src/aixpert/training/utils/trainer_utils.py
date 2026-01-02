"""
Utility functions for Original-DPO training.

Includes dataset loading, model setup, LoRA configuration, and construction
of a TRL DPO trainer.
"""

import json
from typing import Any, Tuple

import pandas as pd
import torch
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel


def load_dataset_for_dpo(jsonl_path: str) -> Dataset:
    """Load a JSONL file containing prompt/chosen/rejected triples."""
    rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    ds = Dataset.from_pandas(df, preserve_index=False)

    return ds.map(
        lambda x: {
            "prompt": x["prompt"],
            "chosen": x["chosen"],
            "rejected": x["rejected"],
        }
    )


def load_model_and_tokenizer(
    model_name: str, max_seq_length: int, load_in_4bit: bool = True
) -> Tuple[Any, Any]:
    """Load an Unsloth QLoRA model and tokenizer enabled."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,
        device_map=None,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_seq_length
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    model.config.use_flash_attention_2 = True

    return model, tokenizer


def apply_lora(model: Any, hp: dict) -> Any:
    """Apply LoRA adapters to the model using hyperparameters inside config.yaml."""
    return FastLanguageModel.get_peft_model(
        model,
        r=hp["lora_r"],
        lora_alpha=hp["lora_alpha"],
        lora_dropout=hp["lora_dropout"],
        use_gradient_checkpointing=True,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def build_dpo_trainer(
    model: Any,
    tokenizer: Any,
    train_data: Dataset,
    eval_data: Dataset,
    cfg: dict,
    output_dir: str,
) -> DPOTrainer:
    """Build and return a TRL DPOTrainer for Original-DPO."""
    training_args = DPOConfig(
        output_dir=output_dir,
        beta=cfg["beta"],
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accumulation"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        save_steps=cfg["save_steps"],
        logging_steps=cfg["logging_steps"],
        seed=cfg["seed"],
        remove_unused_columns=False,
        max_length=cfg["max_seq_length"],
        max_prompt_length=cfg["max_seq_length"] // 2,
        padding_value=tokenizer.pad_token_id,
        report_to="none",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )

    return DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
    )
