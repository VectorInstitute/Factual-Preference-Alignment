import random
import numpy as np
import torch
import json
import tqdm
import copy
from datasets import Dataset, interleave_datasets, load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer
import os
seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

PARQUET_PATH = "/projects/aixpert/users/sindhu/Con-J/src/data/train-00000-of-00001.parquet"
SAVE_DIR = "/projects/aixpert/users/sindhu/Con-J/src/data/pairwise_critic_inference2_get_answer/Sky"

def Sky():
    # Load from a single parquet file instead of load_from_disk()
    ds_dict = load_dataset(
        "parquet",
        data_files={"train": PARQUET_PATH}
    )
    dataset = ds_dict["train"]  # use the "train" split created by parquet loader
    template = ['''作为一个评价专家，给定一个问题和它的两个可能的回答，请选出哪一个回答在连贯性、准确性、覆盖度和上述定义的整体质量方面最为符合。请用JSON格式输出你的判断, 其中"原因"是你提供的解释，"更好的回答"是整数类型的1或2，例如{"原因": "你的解释", "更好的回答": 1}。以下是问题和候选回答的内容：
    \n问题：''',
    '\n回答1：',
    '\n回答2：',]
    #tokenizer = AutoTokenizer.from_pretrained("../../open_models/Qwen2-7B-Instruct/")
    i = 0
    all_data = {}
    length_cut = []
    
    # Build a new train split
    new_data = {
        'prompt':[], 'gen':[], 'tag':[], 'test_id':[],
        'chosen':[], 'q':[], 'r1':[], 'r2':[]
    }
    for item in dataset:
        if random.random() < 0.5:
            chosen_key, rejected_key = 'chosen', 'rejected'
        else:
            chosen_key, rejected_key = 'rejected', 'chosen'

        # question comes from the first message in the *preferred* thread
        item_prompt = item[chosen_key][0]['content']
        # answers are the last messages of each thread
        ans_chosen  = item[chosen_key][-1]['content']
        ans_reject  = item[rejected_key][-1]['content']

        prompt = template[0] + item_prompt + template[1] + ans_chosen + template[2] + ans_reject

        new_data['q'].append(item_prompt)
        new_data['prompt'].append(prompt)
        new_data['gen'].append('')
        new_data['tag'].append('reward-bench')
        new_data['test_id'].append(i)
        new_data['chosen'].append(1 if chosen_key == 'chosen' else 2)  # label is 1 if the original preferred is first
        new_data['r1'].append(ans_chosen)
        new_data['r2'].append(ans_reject)

        i += 1
    all_data['train'] = Dataset.from_dict(new_data)

    print('length_cut:', np.mean(length_cut) if length_cut else 0)

    all_data = DatasetDict(all_data)
    os.makedirs(SAVE_DIR, exist_ok=True)
    all_data.save_to_disk(SAVE_DIR)

if __name__ == '__main__':
    Sky()
