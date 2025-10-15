import os
import json
import glob
from pathlib import Path
import torch
import jsonlines
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
# torch.backends.cuda.sdp_kernel(enable_flash=True,
#                                enable_math=False,
#                                enable_mem_efficient=False)


# ==============================
# Checkpoint helpers
# ==============================
def save_checkpoint(scenes, task_name, next_id, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    fname = os.path.join(checkpoint_dir, f"ckpt_{task_name}_{next_id}.json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)
    print(f"[ckpt] saved → {fname}")

def _suffix_num(p):
    try:
        return int(Path(p).stem.split("_")[-1])
    except Exception:
        return -1

def load_checkpoint(task_name, checkpoint_dir):
    pattern = os.path.join(checkpoint_dir, f"ckpt_{task_name}_*.json")
    files = glob.glob(pattern)
    if not files:
        return [], -1
    latest = max(files, key=_suffix_num)
    try:
        with open(latest, "r", encoding="utf-8") as f:
            scenes = json.load(f)
        last_idx = max((s.get("prompt_idx", -1) for s in scenes), default=-1)
        print(f"[ckpt] loaded ← {latest} (last prompt_idx={last_idx})")
        return scenes, last_idx
    except Exception as e:
        print(f"[ckpt][warn] failed to load {latest}: {e}")
        return [], -1

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
        # fallback rename
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
# Core generation
# ==============================
def generate_jsonl(
    input_path,
    output_path,
    checkpoint_dir,
    model,
    tokenizer,
    n=8,
    checkpoint_every=5,
    max_new_tokens=512,
    prompt_max_len=6400,
    temperature=1.2,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.0,
    checkpoint=True,
    resume=True,
    task_name="ConJ_Task"
):
    # Load input data
    records = [rec for rec in jsonlines.open(input_path) if "prompt" in rec]
    print(f"[data] loaded {len(records)} prompts")

    # Resume if requested
    scenes, last_idx = (load_checkpoint(task_name, checkpoint_dir)
                        if (checkpoint and resume) else ([], -1))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = jsonlines.open(output_path, "w")

    for i, rec in enumerate(tqdm(records, desc="Generating")):
        if i <= last_idx:
            continue

        meta, prompt_str = process_data(rec, tokenizer)

        # Tokenize & repeat for N generations
        inputs = tokenizer(prompt_str, return_tensors="pt", truncation=True,
                           max_length=prompt_max_len)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        expanded = {k: v.repeat(n, 1) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            out = model.generate(
                **expanded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=1,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Slice prompt tokens, keep new text
        in_len = int(inputs["attention_mask"].sum(dim=1)[0].item())
        gens = [tokenizer.decode(out[j, in_len:], skip_special_tokens=True).strip()
                for j in range(out.size(0))]

        scene = {"prompt_idx": i, "prompt": prompt_str, "outputs": gens, "meta": meta}
        scenes.append(scene)

        # Periodic checkpoint
        if checkpoint and (len(scenes) % checkpoint_every == 0):
            save_checkpoint(scenes, task_name, len(scenes), checkpoint_dir)

    # Write output file
    for s in scenes:
        writer.write(s)
    writer.close()

    print(f"[done] scenes → {output_path}")
    if checkpoint:
        print(f"[done] checkpoints saved in → {checkpoint_dir}")

# ==============================
# Main
# ==============================
def main():
    input_path = "/projects/aixpert/users/sindhu/Con-J/src/outputs/input.jsonl"
    output_path = "/projects/aixpert/users/sindhu/Con-J/src/outputs/output.jsonl"
    checkpoint_dir = "/projects/aixpert/users/sindhu/Con-J/src/outputs/checkpoint_output/"

    model_id = "Qwen/Qwen2-7B-Instruct"

    # Model setup
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
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        )
    print("Attention backend:", model.config._attn_implementation)

    # except Exception as e:
    #     print(f"[warn] flash_attention_2 not available ({e}); using SDPA.")
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_id,
    #         device_map="auto",
    #         torch_dtype=torch.bfloat16,
    #         attn_implementation="sdpa",
    #     )

    try:
        model = torch.compile(model, mode="max-autotune")
    except Exception as e:
        print(f"[warn] torch.compile not available/failed ({e}); continuing without compile.")

    model.eval()

    generate_jsonl(
        input_path=input_path,
        output_path=output_path,
        checkpoint_dir=checkpoint_dir,
        model=model,
        tokenizer=tokenizer,
        n=8,  # number of generations per prompt
        checkpoint_every=5,  # save after every 25 prompts
        checkpoint=True,
        resume=True,
        task_name="ConJ_Task",
    )


if __name__ == "__main__":
    main()