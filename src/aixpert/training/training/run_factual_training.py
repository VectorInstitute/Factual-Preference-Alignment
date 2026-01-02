"""Run Factual-DPO training for a single model and a single Δ value."""

import argparse
import os

import wandb
from training.factualdpo_trainer import DataCollatorForPreference
from training.factualdpo_trainer import FactualDPOTrainer as DPOTrainer
from unsloth import PatchDPOTrainer
from utils.config_loader import load_config
from utils.factual_trainer_utils import (
    build_dpo_config,
    load_and_clean_jsonl,
    load_unsloth_model,
)


PatchDPOTrainer()


def train_one_model(model_id: str, short: str, delta: float) -> None:
    """Load config, datasets, model, then run training for a single (model, Δ) pair."""
    cfg = load_config()
    mod = cfg["factual_dpo"]

    train_file = mod["paths"]["train_file"]
    eval_file = mod["paths"]["eval_file"]
    output_root = mod["paths"]["output_root"]

    hp = mod["hyperparams"]
    wandb_cfg = mod["wandb"]

    output_dir = os.path.join(output_root, f"{short}_delta{delta}")
    os.makedirs(output_dir, exist_ok=True)

    train_dataset = load_and_clean_jsonl(train_file).shuffle(seed=42)
    eval_dataset = load_and_clean_jsonl(eval_file).shuffle(seed=42)

    print(f"Train size: {len(train_dataset)}")
    print(f"Eval size : {len(eval_dataset)}")

    wandb.init(
        project=wandb_cfg["project"],
        entity=wandb_cfg["entity"],
        name=f"{wandb_cfg['run_prefix']}_{short}_delta{delta}",
        config={
            "model_name": model_id,
            "delta": delta,
            "epochs": hp["num_train_epochs"],
        },
    )

    model, tokenizer = load_unsloth_model(model_id, hp["max_seq_length"])

    dpo_cfg = build_dpo_config(hp, tokenizer, delta, output_dir)
    collator = DataCollatorForPreference(tokenizer.pad_token_id)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_cfg,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print(f"\nStarting FactualDPO Training: {model_id} (Δ={delta})")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved model to: {output_dir}")
    wandb.finish()


def main() -> None:
    """Parse CLI arguments and launch training for a single model–Δ combination."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--short", type=str, required=True)
    parser.add_argument("--delta", type=float, required=True)
    args = parser.parse_args()

    train_one_model(
        model_id=args.model_id,
        short=args.short,
        delta=args.delta,
    )


if __name__ == "__main__":
    main()
