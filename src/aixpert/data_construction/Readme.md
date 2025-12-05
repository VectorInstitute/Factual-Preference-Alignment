# Skywork → Factual-DPO Data Construction Pipeline

This repository contains a complete, modular, and type-safe data-construction pipeline for generating **factual-aware DPO datasets** from the **Skywork Reward-Preference-80K** dataset.

The pipeline supports:
- Direct Preference Optimization (DPO)
- Factual-DPO
- Synthetic hallucination inversion pairs
- Balanced and flipped datasets

## Configuration

All configuration is centralized in:

```bash
src/aixpert/config/config.yaml
```
Loaded dynamically using:
```python
utils/config_loader.load_config()
