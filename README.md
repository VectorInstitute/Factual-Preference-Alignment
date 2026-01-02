# Reducing Hallucinations in LLMs via Factuality-Aware Preference Learning
### A Modular Benchmark & Training Framework for Factual-Aware DPO

<p align="center">
  <img src="https://github.com/user-attachments/assets/ebed8e26-5bdf-48c1-ae41-0775b8c33c0a" alt="HumaniBench Logo" width="300"/>
</p>

<p align="center">
  <b>🌐 Website:</b> <a href="https://vectorinstitute.github.io/humanibench/">vectorinstitute.github.io/humanibench</a>  
  &nbsp;|&nbsp;
  <b>📄 Paper:</b> <a href="https://arxiv.org/abs/2505.11454">arxiv.org/abs/2505.11454</a>  
  &nbsp;|&nbsp;
  <b>📊 Dataset:</b> <a href="https://huggingface.co/datasets/vector-institute/Factuality_Alignment">Hugging Face</a>
</p>

---

## 🧭 About

**Factual Preference Alignment** is a **research and engineering framework** for studying and improving **factual alignment in preference-optimized Large Language Models (LLMs)**.

The project introduces **Factual-DPO**, a factuality-aware extension of **Direct Preference Optimization (DPO)** that incorporates:

* Explicit factuality supervision
* Synthetic hallucination inversion
* Margin-based factual penalties

The repository provides **end-to-end infrastructure** for:

* Dataset construction
* Multi-model preference fine-tuning
* Automated factuality evaluation

All components are **config-driven**, reproducible, and aligned with the **Vector Institute AI Engineering Template**.

---

## ✨ Key Contributions

* 🔍 Binary factuality supervision integrated into preference learning
* 🧪 Synthetic hallucination inversion pairs
* 📐 Δ-margin factual penalties for controllable hallucination suppression
* ⚙️ Fully config-driven data, training, and evaluation pipelines
* 📊 Multi-model × multi-Δ benchmarking at scale

---

## 📦 Repository Structure

```
aixpert/
│
├── src/aixpert/
│   ├── config/                  # Central config.yaml
│   ├── data_construction/       # 8-stage factual dataset pipeline
│   ├── training/                # Original-DPO & Factual-DPO training
│   ├── evaluation/              # GPT-4o-mini judge evaluation
│   └── utils/                   # Shared helpers
│
├── README.md
└── pyproject.toml
```

---

## 🧠 What Is Factual-DPO?

Standard DPO aligns models to **human preferences**, but does not explicitly discourage **hallucinated yet preferred responses**.

**Factual-DPO** introduces a factuality-aware margin:

* Each preference tuple includes `(h_w, h_l)` factuality indicators
* A penalty λ is applied when the preferred response is less factual
* Optimization pressure shifts toward **factually correct preferences**

➡️ Result: **Lower hallucination rates without sacrificing preference alignment**

---

## 🔬 Skywork → Factual-DPO Data Construction Pipeline

This repository contains a complete **eight-stage pipeline** for converting the **Skywork Reward-Preference-80K** dataset into **balanced, factual-aware DPO datasets**.

### Pipeline Stages

| Stage | Description                             |
| ----- | --------------------------------------- |
| 1     | Skywork extraction & de-duplication     |
| 2     | Preference pair conversion              |
| 3     | Binary factuality scoring (GPT-4o-mini) |
| 4     | Canonical DPO transformation            |
| 5     | Synthetic hallucination generation      |
| 6     | Dataset merging                         |
| 7     | Balanced bucket construction            |
| 8     | Optional preference flipping            |

All paths and parameters are defined in:

```
src/aixpert/config/config.yaml
```

---

## ⚙️ Configuration-Driven Design

Every component — **datasets, models, hyperparameters, outputs, and evaluation** — is controlled via:

```
src/aixpert/config/config.yaml
```

Loaded using:

```python
from utils.config_loader import load_config
cfg = load_config()
```

This enables:

* Full reproducibility
* Multi-model automation
* Zero hard-coded paths

---

## 🏋️ Training Pipelines

### 1️⃣ Original-DPO (Baseline)

```bash
python -m aixpert.training.run_dpo_training \
  --model "google/gemma-2-9b-it"
```

Trains standard DPO using Skywork preferences.

---

### 2️⃣ Factual-DPO (Δ-Margin Training)

```bash
python -m aixpert.training.run_factual_training \
  --model_id "google/gemma-2-9b-it" \
  --short "gemma2-9b" \
  --delta 10
```

Each Δ value produces a **separate fine-tuned model**.

---

## 📊 Evaluation Pipeline

Evaluation is performed using **GPT-4o-mini as an LLM-as-a-Judge**.

### Metrics

| Metric      | Meaning                   |
| ----------- | ------------------------- |
| factuality  | Mean factual score        |
| halluc_rate | % outputs below threshold |
| win_rate    | Δ-model vs baseline       |
| count       | Prompts evaluated         |

Run evaluation:

```bash
python -m aixpert.evaluation.evaluations.run_all_evaluations
```

Outputs:

```
eval_results.json
```

---

## 🧪 Supported Models

* Gemma-2 (2B, 9B)
* Qwen-2.5 / Qwen-3
* LLaMA-3.x
* Any TRL-compatible causal LLM

Models are registered centrally in `config.yaml`.

---

## 🧰 Frameworks & Tooling

* **Hugging Face TRL** — DPO reference implementation
* **Unsloth** — QLoRA optimization
* **BitsAndBytes** — 4-bit quantization
* **Flash-Attention-2**
* **Weights & Biases** — experiment tracking
* **Accelerate** — multi-GPU orchestration

---

## 📚 Dataset Attribution & Credits

This project **builds upon and extends** the **Skywork Reward-Preference-80K** dataset.

> **We do not claim ownership of the Skywork dataset.**
> All credit belongs to the original authors.

If you use this repository, **please cite Skywork**:

```bibtex
@article{liu2024skywork,
  title={Skywork-Reward: Bag of Tricks for Reward Modeling in LLMs},
  author={Liu, Chris Yuhao and Zeng, Liang and Liu, Jiacai and Yan, Rui and He, Jujie and Wang, Chaojie and Yan, Shuicheng and Liu, Yang and Zhou, Yahui},
  journal={arXiv preprint arXiv:2410.18451},
  year={2024}
}
```

For dataset-related concerns, please contact the **Skywork authors** via their paper or Hugging Face repository.

---

## 📖 Citation (Factual Preference Alignment)

If you find this code or dataset useful for your research, please consider citing:

```bibtex
@article{chaduvula2026factualdpo,
  title={Reducing Hallucinations in LLMs via Factuality-Aware Preference Learning},
  author={Sindhuja Chaduvula, Azib Farooq, Ahmed Radwan, Shaina Raza},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
---

## 📬 Contact

For questions, collaborations, or issues:

* Open a GitHub Issue
* Or contact the maintainers via the Vector Institute

---

### ⚡ Factual DPO promotes in reducing hallucinations and increase factualness

**We invite researchers and practitioners to build upon this framework.**
