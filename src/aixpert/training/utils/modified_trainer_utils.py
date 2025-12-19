"""
Utility functions for loading datasets, preparing Unsloth models, and building
DPO/Factual-DPO++ training configurations.
"""

import json

import torch
from datasets import Dataset
from trl import DPOConfig
from unsloth import FastLanguageModel


def load_and_clean_jsonl(path: str) -> Dataset:
    """Loads JSONL factual dataset and converts it into a HF Dataset with required fields."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            ex = json.loads(line)
            rows.append(
                {
                    "prompt": ex.get("prompt", ""),
                    "chosen": ex.get("chosen", ""),
                    "rejected": ex.get("rejected", ""),
                    "h_w": float(ex.get("h_w", 0)),
                    "h_l": float(ex.get("h_l", 0)),
                }
            )
    return Dataset.from_list(rows)


def load_unsloth_model(model_name: str, max_seq_length: int):
    """Loads a 4-bit Unsloth QLoRA model + tokenizer with FlashAttention2 and LoRA applied."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        device_map=None,
    )

    model.config.use_flash_attention_2 = True

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_seq_length
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    # LoRA configuration
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing=True,
    )
    return model, tokenizer


def build_dpo_config(hp, tokenizer, delta: float, output_dir: str):
    """Builds TRL DPOConfig with Safe-DPO factual margin Δ added."""
    cfg = DPOConfig(
        output_dir=output_dir,
        beta=hp["beta"],
        num_train_epochs=hp["num_train_epochs"],
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        per_device_eval_batch_size=hp["per_device_eval_batch_size"],
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        learning_rate=hp["learning_rate"],
        warmup_ratio=hp["warmup_ratio"],
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        save_strategy="steps",
        save_steps=hp["save_steps"],
        logging_steps=hp["logging_steps"],
        remove_unused_columns=False,
        max_length=hp["max_seq_length"],
        max_prompt_length=hp["max_seq_length"] // 2,
        padding_value=tokenizer.pad_token_id,
        ddp_find_unused_parameters=False,
        report_to=["wandb"],
        resume_from_checkpoint=True,
    )

    cfg.delta = delta

    return cfg
