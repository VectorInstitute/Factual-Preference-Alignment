import re
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Config
# -----------------------------
base_model_name = "Qwen/Qwen2-7B-Instruct"
dpo_model_path = "/outputs/qwen2_dpo_full"
dataset_path = "/dpo_dataset"
max_new_tokens = 512
sample_limit = 2  # number of samples to test

# -----------------------------
# Load tokenizer & models
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Base model (pre-trained)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"  # 🔥 Use FA2 if available
)

# DPO fine-tuned model
dpo_model = AutoModelForCausalLM.from_pretrained(
    dpo_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

base_model.eval()
dpo_model.eval()

# -----------------------------
# Load test data
# -----------------------------
dataset = load_from_disk(dataset_path)
test_data = dataset["validation"]  # ✅ 'validation' not 'test'

if sample_limit:
    test_data = test_data.select(range(sample_limit))

# -----------------------------
# Helper: extract Q/A pairs
# -----------------------------
def extract_qa(conv_text):
    """Extract Question, Answer 1, and Answer 2 from conversation text."""
    q = re.search(r"Question:(.*?)(?:Answer 1:|$)", conv_text, re.DOTALL)
    a1 = re.search(r"Answer 1:(.*?)(?:Answer 2:|$)", conv_text, re.DOTALL)
    a2 = re.search(r"Answer 2:(.*)", conv_text, re.DOTALL)
    return (
        q.group(1).strip() if q else "",
        a1.group(1).strip() if a1 else "",
        a2.group(1).strip() if a2 else "",
    )

# -----------------------------
# Helper: run model inference
# -----------------------------
def run_inference(model, q, a1, a2):
    """Run generation for a single sample using the given model."""
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
    f"Output JSON {{'reason': '...', 'better_answer': 1 or 2}}"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded[len(prompt):].strip()
    return response

# -----------------------------
# Run inference and compare
# -----------------------------
with torch.inference_mode():
    for i, sample in enumerate(test_data):
        conv_text = sample["conversations"][0]["value"]
        q, a1, a2 = extract_qa(conv_text)
        print(f"\n=== SAMPLE {i + 1} ===")
        print(f"Q: {q[:120]}...\n")
        base_out = run_inference(base_model, q, a1, a2)
        dpo_out = run_inference(dpo_model, q, a1, a2)
        print("🧠 Base Model →", base_out)
        print("🎯 DPO Model  →", dpo_out)
