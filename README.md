# Reducing Hallucinations in LLMs via Factuality-Aware Preference Learning
### A Modular Training Framework for Factuality-Aware Direct Preference Optimization(F-DPO)

<p align="center" style="margin-top: -10px; margin-bottom: -10px;">
  <img src="docs/assets/factualDPO.png" width="320"/>
</p>

<p align="center">
  <b>🎉 Accepted at ACL 2026 Findings</b>
</p>

<p align="center">
  <b>🌐 Website:</b> <a href="https://vectorinstitute.github.io/Factual-Preference-Alignment/">vectorinstitute.github.io/Factual-Preference-Alignment</a>  
  &nbsp;|&nbsp;
  <b>📄 Paper:</b> <a href="https://arxiv.org/abs/2601.03027">arxiv.org/abs/2601.03027</a>  
  &nbsp;|&nbsp;
  <b>📊 Dataset:</b> <a href="https://huggingface.co/datasets/vector-institute/Factuality_Alignment">Hugging Face</a>
</p>

---

## 🧭 About

**Factuality-aware Direct Preference Optimization** is a **research and engineering framework** for studying and improving **factual alignment in preference-optimized Large Language Models (LLMs)**.

The project introduces **F-DPO**, a factuality-aware extension of **Direct Preference Optimization (DPO)** that incorporates:

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
│   ├── training/                # Original-DPO & F-DPO training
│   ├── evaluation/              # GPT-4o-mini judge evaluation
│   └── utils/                   # Shared helpers
│
├── README.md
└── pyproject.toml
```

---

## 🧠 What Is F-DPO?

Standard DPO aligns models to **human preferences**, but does not explicitly discourage **hallucinated yet preferred responses**.

**F-DPO** introduces a factuality-aware margin:

* Each preference tuple includes `(h_w, h_l)` factuality indicators
* A penalty λ is applied when the preferred response is less factual
* Optimization pressure shifts toward **factually correct preferences**

➡️ Result: **Lower hallucination rates without sacrificing preference alignment**

---

## 🔬 Skywork → F-DPO Data Construction Pipeline

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

### 2️⃣ F-DPO (Δ-Margin Training)

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

## 📖 Citation (Factuality-aware Direct Preference Optimization)

If you find this code or dataset useful for your research, please consider citing:

```bibtex
@inproceedings{FactualAlignment2026,
  title={Reducing Hallucinations in LLMs via Factuality-Aware Preference Learning},
  author={Sindhuja Chaduvula and Ahmed Radwan and Azib Farooq and Yani Ioannou and Shaina Raza},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  year={2026}
}
```
---

## 📬 Contact

For questions, collaborations, or issues:

* **Corresponding author:** [Shaina Raza](mailto:shaina.raza@vectorinstitute.ai) (shaina.raza@vectorinstitute.ai)
* Open a GitHub Issue
* Or contact the maintainers via the Vector Institute

---

### ⚡ Factuality-aware Direct Preference Optimization promotes in reducing hallucinations and increase factualness

---

## 🙏 Acknowledgments
Resources used in preparing this research were provided, in part, by the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the [Vector Institute](http://www.vectorinstitute.ai/#partners). This research was funded by the European Union's Horizon Europe research and innovation programme under the AIXPERT project (Grant Agreement No. 101214389), which aims to develop an agentic, multi-layered, GenAI-powered framework for creating explainable, accountable, and transparent AI systems. 

---

**We invite researchers and practitioners to build upon this framework.**
