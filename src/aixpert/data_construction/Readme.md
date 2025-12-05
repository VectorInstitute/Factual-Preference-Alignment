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
```
## Configuration Summary (`config.yaml`)

### Model Settings
- **model.name:** `gpt-4o-mini`
- **model.temperature:** `0.8`

---

### Paths (All datasets + intermediate outputs)

The configuration tracks every stage of the data pipeline, including:

- Cleaned **train / eval / test** splits
- **Preference pairs** (DPO-style)
- **Factual-scored** outputs
- **Synthetic inversion** samples (train + eval)
- **Merged** intermediate datasets
- **Balanced** final datasets
- **Flipped** datasets for ablation

**Examples:**
```yaml
skywork_train_cleaned: "src/.../skywork_extracted_77k.jsonl"
skywork_train_pairs:   "src/.../skywork_preference_pairs_77k.jsonl"
skywork_train_factual: "src/.../skywork_binary_factual_train.jsonl"
final_train_out:       "src/.../train_balanced.jsonl"
```

## Pipeline Stages — Summary

Below is a concise overview of all eight stages in the Skywork → Factual-DPO data pipeline.

---

### ** Stage 1 — Skywork Extraction**
**Scripts:**
- `dataextraction_train.py`
- `dataextraction_eval.py`
- `dataextraction_test.py`(These samples are directly used in evaluation)

**Tasks:**
- Load slices from Skywork Preference dataset
- Extract:
  - **prompt** (first user message)
  - **chosen** (assistant reply)
  - **rejected** (assistant reply)
- Remove exact duplicates
- Save cleaned JSONL files

---

### ** Stage 2 — Preference Pair Conversion**
**Scripts:**
- `dataconversion_train.py`
- `dataconversion_eval.py`

**Tasks:**
- Convert `(prompt, chosen, rejected)` → **DPO-style preference pairs**
- Produce:
  - `response_0`, `response_1`
  - `better_response_id`
- Random symmetric assignment for unbiased supervision

---

### ** Stage 3 — Binary Factuality Evaluation**
**Scripts:**
- `dataset_train.py`
- `dataset_val.py`

**Components:**
Uses `utils.factual_utils` to evaluate factual correctness using **GPT-4o-mini**.

**Outputs:**
- Binary hallucination flags:
  - `h0`, `h1` (aliases for `factual_flag_0`, `factual_flag_1`)

**Features:**
- Resume-safe incremental scoring
- Async concurrency
- Retry logic

---

### ** Stage 4 — DPO Transformation**
**Scripts:**
- `data_transform_train.py`
- `data_transform_val.py`

**Tasks:**
Transform factual-scored items into canonical DPO format:

- `prompt`, `chosen`, `rejected`
- `h_w`, `h_l`
- `response_0`, `response_1`
- `flipped=False`

---

### ** Stage 5 — Synthetic Hallucination Generation**
**Scripts:**
- `data_synthetic_train.py`
- `data_synthetic_val.py`

**Tasks:**
- Select samples where winner is factual (`h_w=0`) and loser is incorrect (`h_l=1`)
- Use **GPT-4o-mini** to generate hallucinated corruptions
- Build synthetic inversion pairs

**Outputs:**
- **10,000** synthetic train samples
- **400** synthetic eval samples

---

### ** Stage 6 — Merging**
**Scripts:**
- `merge_train.py`
- `merge_eval.py`

**Tasks:**
- Merge Skywork transformed data with synthetic inversion pairs
- Bucket by `(h_w, h_l)`
- Sample subsets
- Shuffle and save merged datasets

---

### ** Stage 7 — Balanced Dataset Construction**
**Scripts:**
- `balance_train.py`
- `build_final_eval.py`

**Train Balancing:**
Use `balance_targets` to create balanced buckets:

- `(0,1)` — 10,000
- `(1,0)` — 10,000
- `(0,0)` — 15,000
- `(1,1)` — 10,000

**Eval Construction:**
Combine:
- Skywork eval transformed
- 400 synthetic eval inversion samples
- 1500 clean `(1,1)` samples (unused in train)
- 1500 clean `(0,0)` samples (unused in train)

---

### ** Stage 8 — Flipping (Optional)**
**Scripts:**
- `data_flipped_train.py`
- `data_flipped_val.py`

**Tasks:**
- Flip all `(1,0)` samples → `(0,1)`
- Swap `chosen` ↔ `rejected`
- Produce alternate dataset for inversion or ablation studies

---

This structured overview provides a clear high-level map of the complete Factual-DPO data construction workflow.

## Utilities Summary

### `utils/config_loader.py`
- Centralized configuration loader
- All stages call `load_config()` to read `config.yaml`

---

### `utils/data_utils.py`
Core data-processing helpers:
- `extract_prompt()` — first user message
- `extract_answer()` — first assistant reply
- `filter_duplicates()` — removes exact matches
- `create_preference_pairs()` — builds DPO response pairs
- `bucket_by_flags()` — groups by (h_w, h_l)
- `flip_sample()` — converts (1,0) → (0,1)
- JSONL read/write utilities

---

### `utils/factual_utils.py`
- Async binary factuality scoring using GPT-4o-mini
- Concurrency + retry logic
- Resume-safe checkpointing
- Produces `h0`, `h1` hallucination flags

---

### `utils/dpo_transform_utils.py`
- Converts factual-scored items into final DPO format:
  - `prompt`, `chosen`, `rejected`, `h_w`, `h_l`, `response_0`, `response_1`, `flipped=False`

---

### `utils/synthetic_utils.py`
- GPT-based corruption generator
- Creates synthetic inversion pairs (hallucinated → correct)

---

### `utils/prompt_templates.py`
Provides all system/user prompts:
- Strict factuality judge prompt
- Hallucination corruption prompts

---

## Running the Pipeline

Example sequence for **training pipeline**:

```bash
python src/aixpert/data_construction/stage_1_extraction/dataextraction_train.py
python src/aixpert/data_construction/stage_2_conversion/dataconversion_train.py
python src/aixpert/data_construction/stage_3_factuality/dataset_train.py
python src/aixpert/data_construction/stage_4_transformation/data_transform_train.py
python src/aixpert/data_construction/stage_5_syntheticdata/data_synthetic_train.py
python src/aixpert/data_construction/stage_6_merging/merge_train.py
python src/aixpert/data_construction/stage_7_balancing/balance_train.py
python src/aixpert/data_construction/stage_8_flipping/data_flipped_train.py
```
