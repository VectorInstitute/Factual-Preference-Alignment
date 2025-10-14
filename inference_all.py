#!/usr/bin/env python3
import os
import re
import json
import glob
from pathlib import Path

import torch
import jsonlines
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==============================
# English Templates
# ==============================
templates = {
    "guide": [
        "As an evaluation expert, given a question and its two possible answers, "
        "please analyze the performance of both in terms of coherence, accuracy, "
        "coverage, and the overall quality defined above.\nQuestion:",
        "\nAnswer 1:",
        "\nAnswer 2:",
        "\nKnown answer ",
        " is better, please explain the reason:"
    ],
    "guide_reverse": [
        "As an evaluation expert, given a question and its two possible answers, "
        "please analyze the performance of both in terms of coherence, accuracy, "
        "coverage, and the overall quality defined above.\nQuestion:",
        "\nAnswer 1:",
        "\nAnswer 2:",
        "\nKnown answer ",
        " is better, please explain the reason:"
    ],
}

# ==============================
# JSON Output Templates
# ==============================
modification_templates = {
    "guide": [
        '{"reason":"',
        '","better answer": ',
        '}'
    ],
    "guide_reverse": [
        '{"reason":"',
        '","better answer": ',
        '}'
    ]
}

# ==============================
# reshape_output → creates structured JSON text
# ==============================
def reshape_output(output, template_key):
    if "guide_reverse" in template_key:
        template = modification_templates[template_key]
        return template[0] + output["reason"] + template[1] + str(3 - output["chosen"]) + template[2]
    elif "guide" in template_key:
        template = modification_templates[template_key]
        return template[0] + output["reason"] + template[1] + str(output["chosen"]) + template[2]
    else:
        return output["reason"]

# ==============================
# Checkpoint helpers (per-template)
# ==============================
def save_checkpoint(scenes, task_name, template_key, next_id, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    fname = os.path.join(checkpoint_dir, f"ckpt_{task_name}_{template_key}_{next_id}.json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)
    print(f"[ckpt] saved → {fname}")

def _suffix_num(p):
    try:
        return int(Path(p).stem.split("_")[-1])
    except Exception:
        return -1

def load_checkpoint(task_name, template_key, checkpoint_dir):
    pattern = os.path.join(checkpoint_dir, f"ckpt_{task_name}_{template_key}_*.json")
    files = glob.glob(pattern)
    if not files:
        return [], 0
    latest = max(files, key=_suffix_num)
    try:
        with open(latest, "r", encoding="utf-8") as f:
            scenes = json.load(f)
        last_id = max((s.get("prompt_idx", 0) for s in scenes), default=0)
        print(f"[ckpt] loaded ← {latest} (last prompt_idx={last_id})")
        return scenes, last_id
    except Exception as e:
        print(f"[ckpt][warn] failed to load {latest}: {e}")
        return [], 0

# ==============================
# Prompt preprocessing
# ==============================
def process_prompt_data(chat_messages, tokenizer):
    """Apply chat template so model starts generating as assistant."""
    try:
        prompt = tokenizer.apply_chat_template(
            chat_messages, add_generation_prompt=True, tokenize=False
        )
    except Exception:
        safe_msgs = []
        for m in chat_messages:
            role = "user" if m.get("role") == "human" else m.get("role", "")
            safe_msgs.append({"role": role, "content": m.get("content", "")})
        prompt = tokenizer.apply_chat_template(
            safe_msgs, add_generation_prompt=True, tokenize=False
        )
    return prompt

def process_data(rec, tokenizer):
    """Extract prompt & meta and return model-ready string."""
    meta = {k: rec[k] for k in rec.keys() if k != "prompt"}
    chat = [{"role": "human", "content": rec["prompt"]}]
    prompt_str = process_prompt_data(chat, tokenizer)
    return meta, prompt_str

# ==============================
# Data transformation (guide + reverse)
# ==============================
def change_data_templates(dataset, template_key, reverse=False):
    template = templates[template_key]
    new_records = []

    pattern = r"Question:\s*(.*?)\s*Answer 1:\s*(.*?)\s*Answer 2:\s*(.*)"

    for item in dataset:
        it = dict(item)
        it["chosen"] = it["chosen_id"] if "chosen_id" in it else it.get("chosen", None)

        if not all(k in it for k in ("q", "r1", "r2")):
            if "prompt" not in it:
                raise ValueError("Item missing both (q, r1, r2) and 'prompt' for parsing.")
            m = re.search(pattern, it["prompt"], re.DOTALL)
            if not m:
                raise ValueError("Failed to parse q/r1/r2 from 'prompt'. Expected 'Question:', 'Answer 1:', 'Answer 2:'.")
            it["q"]  = m.group(1).strip()
            it["r1"] = m.group(2).strip()
            it["r2"] = m.group(3).strip()

        prompt = template[0] + it["q"] + template[1] + it["r1"] + template[2] + it["r2"]

        # ✅ reverse logic for guide_reverse
        if "guide" in template_key:
            chosen_for_hint = it["chosen"] if not reverse else (3 - it["chosen"])
            gen = template[3] + str(chosen_for_hint) + template[4]
        else:
            gen = ""

        full_prompt = prompt.rstrip()
        if gen:
            full_prompt += "\n\n" + gen.lstrip()

        new_records.append({
            **{k: v for k, v in it.items() if k != "prompt"},
            "prompt": full_prompt,
        })

    return new_records

# ==============================
# IO Utils
# ==============================
def load_jsonl_records(path):
    return [rec for rec in jsonlines.open(path) if ("prompt" in rec or "q" in rec)]

def write_jsonl_append(records, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mode = "a" if os.path.exists(path) else "w"
    with jsonlines.open(path, mode) as w:
        for r in records:
            w.write(r)

# ==============================
# Multi-sample generation
# ==============================
def generate_for_template(input_path, output_path, checkpoint_dir,
                          model, tokenizer, task_name, template_key, reverse=False,
                          n=1, checkpoint_every=10):
    raw = load_jsonl_records(input_path)
    transformed = change_data_templates(raw, template_key=template_key, reverse=reverse)

    scenes, last_idx = load_checkpoint(task_name, template_key, checkpoint_dir)
    next_start = last_idx + 1
    total = len(transformed)
    print(f"[run] {template_key}: {total} prompts; resuming from idx={last_idx}")

    for i in tqdm(range(next_start, total + 1), desc=f"Generating ({template_key})"):
        rec = transformed[i - 1]
        prompt_text = rec["prompt"]

        meta, prompt_str = process_data({"prompt": prompt_text}, tokenizer)
        inputs = tokenizer(prompt_str, return_tensors="pt", truncation=True, max_length=6400)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        expanded = {k: v.repeat(n, 1) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **expanded,
                max_new_tokens=512,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                top_k=50,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        in_len = int(inputs["attention_mask"].sum(dim=1)[0].item())
        gens = [tokenizer.decode(out[j, in_len:], skip_special_tokens=True).strip()
                for j in range(out.size(0))]

        # 🔹 Call reshape_output here to format as JSON
        json_outputs = [
            reshape_output({"reason": g, "chosen": rec["chosen"]}, template_key)
            for g in gens
        ]

        scene = {
            "prompt_idx": i,
            "template_key": template_key,
            "prompt": prompt_text,
            "outputs": json_outputs,
        }
        for k in ("tag", "test_id", "chosen", "chosen_id", "q", "r1", "r2"):
            if k in rec:
                scene[k] = rec[k]

        write_jsonl_append([scene], output_path)
        scenes.append(scene)

        if i % checkpoint_every == 0:
            save_checkpoint(scenes, task_name, template_key, next_id=i + 1, checkpoint_dir=checkpoint_dir)

    save_checkpoint(scenes, task_name, template_key, next_id=total + 1, checkpoint_dir=checkpoint_dir)
    print(f"[done] {template_key} → {output_path}")

# ==============================
# Main
# ==============================
def main():
    input_path = "/outputs/input.jsonl"
    out_dir = "/outputs/template_key"
    ckpt_dir = "/outputs/checkpoints_template_key"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    out_guide = os.path.join(out_dir, "guide.jsonl")
    out_grev = os.path.join(out_dir, "guide_reverse.jsonl")

    task_name = "Sky"

    model_id = "Qwen/Qwen2-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    except Exception as e:
        print(f"[warn] flash_attention_2 not available ({e}); using SDPA.")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    try:
        model = torch.compile(model, mode="max-autotune")
    except Exception as e:
        print(f"[warn] torch.compile failed ({e}); continuing without compile.")

    model.eval()

    # guide (normal)
    generate_for_template(
        input_path=input_path,
        output_path=out_guide,
        checkpoint_dir=ckpt_dir,
        model=model,
        tokenizer=tokenizer,
        task_name=task_name,
        template_key="guide",
        reverse=False,
        n=1,                
        checkpoint_every=10
    )

    # guide_reverse (flipped labels)
    generate_for_template(
        input_path=input_path,
        output_path=out_grev,
        checkpoint_dir=ckpt_dir,
        model=model,
        tokenizer=tokenizer,
        task_name=task_name,
        template_key="guide_reverse",
        reverse=True,
        n=1,                
        checkpoint_every=10
    )

if __name__ == "__main__":
    main()