import os
import json
import random
import re
import numpy as np
from datasets import Dataset, DatasetDict

random.seed(2021)

# ===============================
# Helper: Safe JSON parsing
# ===============================
def safe_json_loads(s):
    try:
        return json.loads(s)
    except:
        return None


# ===============================
# Evaluate function (English)
# ===============================
def evaluate(output_str):
    """Parse model outputs to extract better_answer = 1 or 2."""
    output_str = output_str.strip().replace('”,', '",').replace('”', '"')

    # Case 1: JSON block
    if "```json" in output_str:
        match = re.search(r'```json(.*?)```', output_str, re.DOTALL)
        if match:
            data = safe_json_loads(match.group(1))
            if data:
                return data.get("better_answer") or data.get("better answer")

    # Case 2: direct JSON
    if output_str.startswith("{"):
        data = safe_json_loads(output_str)
        if data:
            return data.get("better_answer") or data.get("better answer")

    # Case 3: embedded JSON
    if "{" in output_str and "}" in output_str:
        match = re.search(r'{(.*?)}', output_str, re.DOTALL)
        if match:
            data = safe_json_loads("{" + match.group(1) + "}")
            if data:
                return data.get("better_answer") or data.get("better answer")

    # Case 4: regex fallback
    match = re.search(r'"better[_ ]answer"\s*:\s*(1|2)', output_str)
    if match:
        return int(match.group(1))

    return None


# ===============================
# Load dataset from .jsonl file
# ===============================
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ===============================
# Split positive/negative per test_id
# ===============================
def split_positive_negative(dataset):
    grouped = {}
    none_rate = []

    for item in dataset:
        prompt = item.get("prompt", "")
        meta = item.get("meta", {})
        test_id = item.get("test_id", item.get("prompt_idx", None))
        correct_answer = meta.get("chosen", item.get("chosen", None))
        tag = meta.get("tag", item.get("tag", "default"))

        if test_id not in grouped:
            grouped[test_id] = {"positive": [], "negative": [], "unknown": []}

        outputs = item.get("outputs", [])
        for gen_str in outputs:
            eval_result = evaluate(gen_str)
            sample = {
                "prompt": prompt,
                "gen": gen_str,
                "chosen": correct_answer,  # ground-truth label (1/2)
                "test_id": test_id,
                "tag": tag,
            }

            if eval_result == correct_answer:
                grouped[test_id]["positive"].append(sample)
                none_rate.append(0)
            elif eval_result is None:
                grouped[test_id]["unknown"].append(sample)
                none_rate.append(1)
            else:
                grouped[test_id]["negative"].append(sample)
                none_rate.append(0)

    print(f"Unknown rate: {np.mean(none_rate) if none_rate else 0:.3f}")
    return grouped


# ===============================
# Construct DPO pairs (per test_id)
# ===============================
def construct_dpo_pairs(guide_split, guide_rev_split, output_split):
    pairs = {
        "conversations": [],
        "chosen": [],
        "rejected": [],
        "pair_type": [],
        "test_id": [],
        "tag": [],
        "chosen_id": []
    }

    num_stats = {
        "best_of_n": 0,
        "best_of_n_positive2unknown": 0,
        "preamble": 0,
        "preamble2unknown": 0,
        "num_lost": 0,
    }

    lost_ids = []

    for test_id in output_split.keys():
        data = output_split[test_id]
        pos, neg, unk = data["positive"], data["negative"], data["unknown"]
        random.shuffle(pos)
        random.shuffle(neg)
        random.shuffle(unk)

        # --- Best-of-n positive vs negative
        if len(pos) > 0 and len(neg) > 0:
            for i in range(min(len(pos), len(neg), 3)):
                pairs["conversations"].append([{"from": "human", "value": pos[i]["prompt"]}])
                pairs["chosen"].append(pos[i]["gen"])
                pairs["rejected"].append(neg[i]["gen"])
                pairs["pair_type"].append("best_of_n")
                pairs["test_id"].append(test_id)
                pairs["tag"].append(pos[i].get("tag", "default"))
                pairs["chosen_id"].append(pos[i].get("chosen", None))
                num_stats["best_of_n"] += 1

        # --- Positive vs Unknown (fallback)
        elif len(pos) > 0 and len(unk) > 0:
            pairs["conversations"].append([{"from": "human", "value": pos[0]["prompt"]}])
            pairs["chosen"].append(pos[0]["gen"])
            pairs["rejected"].append(unk[0]["gen"])
            pairs["pair_type"].append("best_of_n_positive2unknown")
            pairs["test_id"].append(test_id)
            pairs["tag"].append(pos[0].get("tag", "default"))
            pairs["chosen_id"].append(pos[0].get("chosen", None))
            num_stats["best_of_n_positive2unknown"] += 1

        # --- Preamble (guide vs guide_reverse)
        if test_id in guide_split and test_id in guide_rev_split:
            g_pos = guide_split[test_id]["positive"]
            g_rev_neg = guide_rev_split[test_id]["negative"]
            if len(g_pos) > 0 and len(g_rev_neg) > 0:
                pairs["conversations"].append([{"from": "human", "value": g_pos[0]["prompt"]}])
                pairs["chosen"].append(g_pos[0]["gen"])
                pairs["rejected"].append(g_rev_neg[0]["gen"])
                pairs["pair_type"].append("preamble")
                pairs["test_id"].append(test_id)
                pairs["tag"].append(g_pos[0].get("tag", "default"))
                pairs["chosen_id"].append(g_pos[0].get("chosen", None))
                num_stats["preamble"] += 1
            elif len(g_pos) > 0 and len(guide_rev_split[test_id]["unknown"]) > 0:
                pairs["conversations"].append([{"from": "human", "value": g_pos[0]["prompt"]}])
                pairs["chosen"].append(g_pos[0]["gen"])
                pairs["rejected"].append(guide_rev_split[test_id]["unknown"][0]["gen"])
                pairs["pair_type"].append("preamble2unknown")
                pairs["test_id"].append(test_id)
                pairs["tag"].append(g_pos[0].get("tag", "default"))
                pairs["chosen_id"].append(g_pos[0].get("chosen", None))
                num_stats["preamble2unknown"] += 1
        else:
            num_stats["num_lost"] += 1
            lost_ids.append(test_id)

    print("\n=== Pairing Summary ===")
    for k, v in num_stats.items():
        print(f"{k}: {v}")

    if lost_ids:
        print(f"\n❌ Lost {len(lost_ids)} preamble test_ids: {len(lost_ids)}")
    else:
        print("\n✅ No lost preamble test_ids!")

    return pairs


# ===============================
# Domain Split
# ===============================
def domain_split(dataset, tag_field="tag"):
    tags = list(set(dataset[tag_field]))
    print(f"\n🔹 Domains detected: {tags}")
    domain_splits = {}
    for tag in tags:
        sub = dataset.filter(lambda x: x[tag_field] == tag)
        train_test = sub.train_test_split(test_size=0.1, seed=42)
        domain_splits[tag] = train_test
    return domain_splits


# ===============================
# Main script
# ===============================
if __name__ == "__main__":
    # Input paths
    guide = load_jsonl("/outputs/template_key/guide.jsonl")
    guide_rev = load_jsonl("/outputs/template_key/guide_reverse.jsonl")
    outputs = load_jsonl("/outputs/output.jsonl")

    # Step 1: Split positive/negative
    guide_split = split_positive_negative(guide)
    guide_rev_split = split_positive_negative(guide_rev)
    output_split = split_positive_negative(outputs)

    # Step 2: Construct DPO pairs
    dpo_pairs = construct_dpo_pairs(guide_split, guide_rev_split, output_split)

    # Step 3: Convert to HF Dataset
    dataset = Dataset.from_dict(dpo_pairs)

    # Step 4: Metadata summary
    meta_info = {
        "total_pairs": len(dataset),
        "domains": list(set(dataset["tag"])),
        "pair_types": list(set(dataset["pair_type"])),
    }

    # Step 5: Train/validation split
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    dataset_dict = DatasetDict({
        "train": split_dataset["train"],
        "validation": split_dataset["test"]
    })

    # Step 6: Domain splits
    domain_splits = domain_split(dataset)

    # Step 7: Save
    save_dir = "dpo_dataset"
    os.makedirs(save_dir, exist_ok=True)
    dataset_dict.save_to_disk(save_dir)
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    for domain, split in domain_splits.items():
        domain_path = os.path.join(save_dir, f"domain_{domain}")
        os.makedirs(domain_path, exist_ok=True)
        split["train"].save_to_disk(os.path.join(domain_path, "train"))
        split["test"].save_to_disk(os.path.join(domain_path, "validation"))

    # Summary
    print(f"\n✅ DPO dataset saved successfully with {len(dataset)} total pairs.")
    print(f"Train samples: {len(dataset_dict['train'])}")
    print(f"Validation samples: {len(dataset_dict['validation'])}")
    print(f"Metadata saved at: {save_dir}/metadata.json")
