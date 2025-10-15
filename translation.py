import json
import os
import re
from tqdm import tqdm
from datasets import load_from_disk
from openai import OpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    OPENAI_API_KEY: str
    model_config = SettingsConfigDict(
        env_file="/projects/aixpert/users/sindhu/Con-J/.env",
        env_file_encoding="utf-8"
    )

config = Config()
client = OpenAI(api_key=config.OPENAI_API_KEY)


dataset_path = "/data/pairwise_critic_inference2_get_answer/Sky"
dataset = load_from_disk(dataset_path)["train"]
subset = dataset.select(range(5000))  # process first 1000


def translate_text(chinese_text: str) -> str:
    """Translate Chinese text (keys + values) to fluent English using GPT-4o-mini."""
    if not chinese_text.strip():
        return chinese_text
    translation_prompt = (
        f"Translate the following Chinese text to fluent English **literally**, without executing or reasoning about it. Do NOT produce any judgments or new JSON. Just translate word-for-word, preserving the meaning:\n\n{chinese_text}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": translation_prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Translation failed: {e}")
        return chinese_text

checkpoint_dir = "checkpoints_translation"
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_file = os.path.join(checkpoint_dir, "translated_prompts_checkpoint.jsonl")
final_output_file = "translated_prompts_1000.jsonl"
save_interval = 20  # save after every 20 samples

translated_rows = []
start_index = 0

# Resume if checkpoint exists
if os.path.exists(checkpoint_file):
    print(f"🟡 Found existing checkpoint at {checkpoint_file}")
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                translated_rows.append(row)
            except json.JSONDecodeError:
                continue
    start_index = len(translated_rows)
    print(f"🔁 Resuming from index {start_index}")

else:
    print("🆕 No checkpoint found. Starting fresh translation.")


for idx in tqdm(range(start_index, len(subset)), desc="Translating prompts", initial=start_index):
    row = subset[idx]
    new_row = dict(row)

    # --- Step 1: Translate only the prompt field ---
    translated_prompt = translate_text(row["prompt"])

    # --- Step 2: Convert any Chinese keys to English ---
    translated_prompt = re.sub(r'"原因"\s*:', '"reason":', translated_prompt)
    translated_prompt = re.sub(r'"更好的回答"\s*:', '"better_answer":', translated_prompt)

    new_row["prompt"] = translated_prompt
    translated_rows.append(new_row)

    # --- Step 3: Save checkpoint every N samples ---
    if (idx + 1) % save_interval == 0:
        with open(checkpoint_file, "a", encoding="utf-8") as f:
            for item in translated_rows[-save_interval:]:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
        print(f"💾 Checkpoint saved after {idx + 1} samples.")

with open(final_output_file, "w", encoding="utf-8") as f:
    for item in translated_rows:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ Translation completed — final output saved to {final_output_file}")
print(f"🗂️ All intermediate checkpoints available in: {checkpoint_dir}")
