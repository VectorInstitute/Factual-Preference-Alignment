import jsonlines
import os

def renumber_sequential(input_path):
    """Ensure guide and guide_reverse files go 0–999."""
    records = [r for r in jsonlines.open(input_path)]
    records.sort(key=lambda x: x.get("prompt_idx", 0))
    for i, r in enumerate(records[:1000]):  # only first 1000 if excess
        r["test_id"] = i
    with jsonlines.open(input_path, "w") as w:
        for r in records[:1000]:
            w.write(r)
    print(f"[✓] Renumbered 0–{len(records[:1000]) - 1} → {input_path}")

import jsonlines
import os

def clean_output_jsonl_keep_meta(input_path):
    """
    Cleans output.jsonl:
      - Keeps only meta.test_id (no top-level test_id)
      - Removes meta.gen if empty or present
      - Renumbers meta.test_id sequentially per prompt_idx group (0–999)
    """
    from collections import defaultdict

    # Step 1: Group all records by prompt_idx
    grouped = defaultdict(list)
    with jsonlines.open(input_path) as reader:
        for rec in reader:
            pid = rec.get("prompt_idx")
            if pid is not None:
                grouped[pid].append(rec)

    # Step 2: Sequential renumbering (0–999 per unique prompt_idx)
    cleaned_records = []
    for new_tid, pid in enumerate(sorted(grouped.keys())):
        for rec in grouped[pid]:
            # Remove top-level test_id
            rec.pop("test_id", None)

            # Ensure meta exists
            meta = rec.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}

            # Remove "gen" if present
            meta.pop("gen", None)

            # Assign renumbered test_id into meta
            meta["test_id"] = new_tid
            rec["meta"] = meta

            cleaned_records.append(rec)

    # Step 3: Write cleaned file
    tmp_path = input_path + ".tmp"
    with jsonlines.open(tmp_path, "w") as writer:
        for rec in cleaned_records:
            writer.write(rec)

    os.replace(tmp_path, input_path)
    print(f"[✓] Cleaned {len(cleaned_records)} records into {len(grouped)} groups (meta.test_id 0–{len(grouped)-1}) → {input_path}")


# Paths
guide = "/projects/aixpert/users/sindhu/Con-J/src/outputs/template_key/guide.jsonl"
guide_rev = "/projects/aixpert/users/sindhu/Con-J/src/outputs/template_key/guide_reverse.jsonl"
output = "/projects/aixpert/users/sindhu/Con-J/src/outputs/output.jsonl"

# Apply sequential test_id renumbering
renumber_sequential(guide)
renumber_sequential(guide_rev)
clean_output_jsonl_keep_meta(output)