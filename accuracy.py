import re
import json
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Config
# -----------------------------
base_model_name = "Qwen/Qwen2-7B-Instruct"
dpo_model_path = "/projects/aixpert/users/sindhu/Con-J/src/outputs/qwen2_dpo_full_epoch_5"
dataset_path = "/projects/aixpert/users/sindhu/Con-J/src/dpo_dataset"
max_prompt_tokens = 32768   # full context window for Qwen2-7B
max_new_tokens = 256

# -----------------------------
# Load tokenizer & models
# -----------------------------
print("🚀 Loading models...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def load_model(path):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    return model

base_model = load_model(base_model_name)
dpo_model = load_model(dpo_model_path)

# -----------------------------
# Load validation data
# -----------------------------
dataset = load_from_disk(dataset_path)
test_data = dataset["validation"]
print(f"✅ Loaded validation dataset with {len(test_data)} samples.")

# -----------------------------
# Helper: extract Q/A pairs
# -----------------------------
def extract_qa(conv_text):
    """Extract Question, Answer 1, and Answer 2 from conversation text."""
    q = re.search(r"[Qq]uestion\s*:\s*(.*?)(?:Answer\s*1\s*:|$)", conv_text, re.DOTALL)
    a1 = re.search(r"Answer\s*1\s*:\s*(.*?)(?:Answer\s*2\s*:|$)", conv_text, re.DOTALL)
    a2 = re.search(r"Answer\s*2\s*:\s*(.*)", conv_text, re.DOTALL)
    q = q.group(1).strip() if q else ""
    a1 = a1.group(1).strip() if a1 else ""
    a2 = a2.group(1).strip() if a2 else ""

    # 🧹 Clean up unwanted trailing phrases
    cleanup_pattern = r"Known answer\s*\d+\s*is better,?\s*please explain( the reason)?[:]*"
    q = re.sub(cleanup_pattern, "", q, flags=re.IGNORECASE).strip()
    a1 = re.sub(cleanup_pattern, "", a1, flags=re.IGNORECASE).strip()
    a2 = re.sub(cleanup_pattern, "", a2, flags=re.IGNORECASE).strip()

    return q, a1, a2

# -----------------------------
# Helper: safe JSON parsing
# -----------------------------
def safe_parse_json(output_str):
    match = re.search(r"\{.*\}", output_str, re.DOTALL)
    if not match:
        return {"reason": "No JSON detected", "better_answer": None}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        repaired = match.group().replace("'", '"')
        try:
            return json.loads(repaired)
        except:
            return {"reason": "Malformed JSON", "better_answer": None}

# -----------------------------
# Helper: model inference
# -----------------------------
def run_inference(model, q, a1, a2, name="model"):
    """Generate the model’s judgment between Answer 1 and Answer 2."""
    prompt = (
        f"As an evaluation expert, given a question and its two possible answers, "
        f"please select which answer best meets the criteria of coherence, accuracy, coverage, "
        f"and overall quality as defined above.\n"
        f"Please output your judgment in JSON format, where 'reason' is your explanation, "
        f"and 'better_answer' is an integer value of 1 or 2.\n"
        f"For example: {{'reason': 'your explanation', 'better_answer': 1}}.\n\n"
        f"Question: {q}\n"
        f"Answer 1: {a1}\n"
        f"Answer 2: {a2}\n"
        f"Output JSON:"
    )

    # Tokenize (no truncation)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False,
        padding=False,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]
    print(f"  🔹 [{name}] Input length: {input_len} tokens")

    if input_len > max_prompt_tokens:
        print(f"⚠️ [{name}] Input exceeds {max_prompt_tokens} tokens! Consider chunking.")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return decoded

# -----------------------------
# Main loop: evaluate all samples
# -----------------------------
results = []
base_correct = 0
dpo_correct = 0
total = 0

print("\n🔍 Starting full validation evaluation (no truncation)...\n")
for i, sample in enumerate(test_data):
    conv_text = sample["conversations"][0]["value"]
    chosen_id = sample.get("chosen_id", None)
    q, a1, a2 = extract_qa(conv_text)

    if not q or not a1 or not a2:
        continue

    base_out = run_inference(base_model, q, a1, a2, "Base")
    dpo_out = run_inference(dpo_model, q, a1, a2, "DPO")

    base_json = safe_parse_json(base_out)
    dpo_json = safe_parse_json(dpo_out)

    base_pred = base_json.get("better_answer")
    dpo_pred = dpo_json.get("better_answer")

    total += 1
    if base_pred == chosen_id:
        base_correct += 1
    if dpo_pred == chosen_id:
        dpo_correct += 1

    results.append({
        "index": i + 1,
        "chosen_id": chosen_id,
        "question": q,
        "base_output": base_json,
        "dpo_output": dpo_json,
        "base_pred": base_pred,
        "dpo_pred": dpo_pred
    })

# -----------------------------
# Compute Accuracy
# -----------------------------
base_acc = round(base_correct / total * 100, 2) if total > 0 else 0
dpo_acc = round(dpo_correct / total * 100, 2) if total > 0 else 0

print(f"\n✅ Evaluation complete on {total} samples")
print(f"📊 Base Model Accuracy: {base_acc}%")
print(f"📈 DPO Model Accuracy: {dpo_acc}%")

# -----------------------------
# Save results
# -----------------------------
output_path = "dpo_eval_results_full_no_trunc.json"
with open(output_path, "w") as f:
    json.dump({
        "base_accuracy": base_acc,
        "dpo_accuracy": dpo_acc,
        "results": results
    }, f, indent=2)

print(f"\n📁 Results saved to {output_path}")
