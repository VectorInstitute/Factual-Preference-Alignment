from datasets import load_from_disk
dataset = load_from_disk("dpo_dataset")
print(dataset)
print(dataset['train'][0])

print("_______")
print(dataset['validation'][0])

from collections import Counter
pair_type_counts = Counter(dataset["train"]["pair_type"])
print("\nPair type distribution:\n", pair_type_counts)

pair_type_counts = Counter(dataset["validation"]["pair_type"])
print("\nPair type distribution:\n", pair_type_counts)
