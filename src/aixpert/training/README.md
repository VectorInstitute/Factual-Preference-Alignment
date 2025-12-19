# AIXpert Preference Alignment — Training Pipeline
Original-DPO & Factual-DPO++ Fine-Tuning
Vector Institute — AI Engineering Template Compatible

This directory contains the full training pipeline for:

1. **Original Direct Preference Optimization (DPO)** — Baseline alignment
2. **Modified Factual-DPO++** — A SafeDPO-style variant with a factual margin Δ
3. **Multi-model training orchestration** driven entirely by `config.yaml`

All training is configured from:

```
src/aixpert/config/config.yaml
```

---

## 📌 Configuration Overview (Training)

All training behavior—models, hyperparameters, dataset paths, LoRA configuration, and output directories—is defined centrally in:

```
src/aixpert/config/config.yaml
```

The configuration contains **three core blocks**:

---

## 1️⃣ Model Registry

The `models:` section defines all base LLMs to be fine-tuned.

Each entry includes:

- **id** → HuggingFace model identifier
- **short** → Filesystem-friendly shorthand used for checkpoints

Example:

```yaml
models:
  - id: "google/gemma-2-9b-it"
    short: "gemma2-9b"
  - id: "Qwen/Qwen2.5-14B-Instruct"
    short: "qwen2.5-14b"
  - id: "meta-llama/Llama-3.2-1B-Instruct"
    short: "llama3.2-1b"
```

This registry enables **automatic multi-model training** for both Original-DPO and Factual-DPO++.

---

## 2️⃣ Original-DPO Training Configuration

Defined under:

```yaml
original_dpo:
```

Includes:

### Dataset Paths

```yaml
paths:
  train: "src/aixpert/training/data/original/train_finallast.jsonl"
  eval:  "src/aixpert/training/data/original/eval_final.jsonl"
  output_root: "src/aixpert/training/data/original/Models"
```

### Hyperparameters (LoRA + TRL DPO)

```yaml
hyperparams:
  max_seq_length: 2048
  load_in_4bit: true
  lora_r: 32
  lora_alpha: 64
  lora_dropout: 0.05
  batch_size: 2
  grad_accumulation: 16
  num_epochs: 3
  learning_rate: 1.8e-6
  warmup_ratio: 0.25
  beta: 0.1
  save_steps: 100
  logging_steps: 20
  seed: 3407
```

These govern **baseline DPO training**, implemented in:

```
src/aixpert/training/run_dpo_training.py
```

### Output directory structure

```
src/aixpert/training/data/original/Models/<short>_OriginalDPO/
```

---

## 3️⃣ Modified Factual-DPO++ Configuration

Defined under:

```yaml
modified_dpo:
```

This block extends Original-DPO with factuality-aware training using SafeDPO-style Δ-margin.

### Δ-Values

```yaml
deltas: [0, 2, 4, 6, 8, 10, 20, 30, 50, 100]
```

Each Δ produces a **separate fine-tuned model**.

---

### Factual Dataset Paths

```yaml
paths:
  train_file: "src/aixpert/training/data/modified/train_final_flipped.jsonl"
  eval_file:  "src/aixpert/training/data/modified/eval_final_flipped.jsonl"
  output_root: "src/aixpert/training/data/modified/Models"
```

These datasets include:

- factuality indicators (`h_w`, `h_l`)
- corrected orientation for factual supervision

---

### Hyperparameters (TRL-compatible)

```yaml
hyperparams:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
  num_train_epochs: 3
  learning_rate: 1.8e-6
  warmup_ratio: 0.25
  save_steps: 100
  logging_steps: 20
  max_seq_length: 2048
  lora_r: 32
  lora_alpha: 64
  lora_dropout: 0.05
```

---

### Weights & Biases Logging

```yaml
wandb:
  project: "aixpert"
  entity: "vector-institute-aieng"
  run_prefix: "FactualDPO"
```

Each training run is tagged:

```
FactualDPO_<short>_delta<value>
```

---

## 🚀 Training Workflows

### 1️⃣ Original-DPO Training

```bash
python -m aixpert.training.run_dpo_training --model "google/gemma-2-9b-it"
```

This:

- Loads config
- Loads datasets
- Applies QLoRA
- Trains TRL DPO
- Saves to:

```
src/aixpert/training/data/original/Models/<short>_OriginalDPO/
```

---

### 2️⃣ Modified Factual-DPO++ Training

Train a model with Δ=10:

```bash
python -m aixpert.training.run_modified_training     --model_id "google/gemma-2-9b-it"     --short "gemma2-9b"     --delta 10
```

Saves to:

```
src/aixpert/training/data/modified/Models/<short>_delta10/
```

---

## 📌 Notes on TRL Files

The following files are **copied directly from TRL GitHub** and intentionally excluded from linting:

```
training/trl/
training/modifieddpo_trainer.py
```

They contain internal SafeDPO logic required for Δ-margin training.

---

## ✅ Summary

This training pipeline supports:

- Multi-model Original-DPO training
- Factual-DPO++ training with configurable Δ
- Config-driven reproducibility
- Full WandB integration
- Unsloth QLoRA optimization
