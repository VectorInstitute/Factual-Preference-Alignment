"""Run baseline Original-DPO training."""

from utils.config_loader import load_config
from utils.trainer_utils import (
    apply_lora,
    build_dpo_trainer,
    load_dataset_for_dpo,
    load_model_and_tokenizer,
)


def train_single_model(model_name: str) -> None:
    """Load config, dataset, model, and run Original-DPO training for one model."""
    cfg = load_config()
    hp = cfg["original_dpo"]["hyperparams"]
    paths = cfg["original_dpo"]["paths"]

    print(f"Training model: {model_name}")

    output_dir = f"{paths.output_root}/{model_name.replace('/', '_')}_OriginalDPO"

    train_data = load_dataset_for_dpo(paths.train)
    eval_data = load_dataset_for_dpo(paths.eval)

    model, tokenizer = load_model_and_tokenizer(
        model_name,
        hp.max_seq_length,
        hp.load_in_4bit,
    )

    model = apply_lora(model, hp)

    trainer = build_dpo_trainer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        eval_data=eval_data,
        cfg=hp,
        output_dir=output_dir,
    )

    print("Starting training...")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Finished training {model_name}. Output saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    train_single_model(args.model)
