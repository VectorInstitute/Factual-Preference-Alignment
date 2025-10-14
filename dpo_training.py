# =========================================
# train_dpo_full.py  ✅ final version
# =========================================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model

# ==========================================
# Config
# ==========================================
model_name = "Qwen/Qwen2-7B-Instruct"
dataset_path = "/dpo_dataset"
output_dir = "/outputs/qwen2_dpo_full"

beta = 0.1                    # DPO regularization strength
sft_loss_lambda = 0.001       # auxiliary SFT loss weight
lr = 5e-7
num_epochs = 3
batch_size = 4
grad_accum = 2

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load quantized model
policy_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
)

# Apply LoRA adapter
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj"],
)

policy_model = get_peft_model(policy_model, peft_config)

# Reference model (frozen)
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

# ==========================================
# Load and preprocess dataset
# ==========================================
dataset = load_from_disk(dataset_path)

def process_dpo_dataset(dataset):
    """Extracts prompt, chosen, rejected + metadata."""
    def make_prompt(example):
        convs = example["conversations"]
        if isinstance(convs, list) and len(convs) > 0 and "value" in convs[0]:
            prompt = convs[0]["value"]
        else:
            prompt = str(convs)
        return {
            "prompt": prompt,
            "chosen": example["chosen"],
            "rejected": example["rejected"],
            "tag": example.get("tag", ""),
            "test_id": example.get("test_id", ""),
            "pair_type": example.get("pair_type", "")
        }
    return dataset.map(make_prompt, remove_columns=[])

train_dataset = process_dpo_dataset(dataset["train"])
eval_dataset = process_dpo_dataset(dataset["validation"])

dpo_config = DPOConfig(
    beta=beta,                             # DPO regularization (η)
    learning_rate=lr,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    logging_steps=25,
    save_strategy="steps",
    save_steps=200,
    warmup_ratio=0.05,
    bf16=True,
    remove_unused_columns=False,
    dataloader_num_workers=4,
    max_grad_norm=1.0,
    output_dir=output_dir,
    report_to="none",
    loss_type=["sigmoid", "sft"],          # combine DPO + SFT
    loss_weights=[1.0, 0.001],        # L = L_DPO + α·L_SFT
)
# ==========================================
# Trainer setup
# ==========================================
trainer = DPOTrainer(
    model=policy_model,
    ref_model=None,
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,  # processing_class also accepted, tokenizer still works in TRL 0.10
    peft_config=peft_config,
)

# ==========================================
# Train and save
# ==========================================
trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ DPO training complete. Model saved to {output_dir}")