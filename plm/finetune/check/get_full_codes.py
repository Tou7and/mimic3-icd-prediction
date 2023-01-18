"""
KEPT Paper(https://arxiv.org/pdf/2210.03304v2.pdf) claims there are 8,692 labels.

Counting:
In all, there are 8,921 labels, after nan is removed.
  - In train set, there are 8,686 labels.
  - In train+test, there are 8,858 labels.
  - In train+dev, there are 8,759 labels.

Maybe only training set is used and some labels removed for their label counting.

2022.12.06, James.
"""
from datasets import load_dataset
# from config import LABEL_LIST

FULL_CODES = []

train_dataset = load_dataset("json", data_files="exp/mimic3_full.json", field="train", cache_dir="exp/")
train_set = train_dataset["train"]

dev_dataset = load_dataset("json", data_files="exp/mimic3_full.json", field="dev", cache_dir="exp/")
dev_set = dev_dataset["train"]

test_dataset = load_dataset("json", data_files="exp/mimic3_full.json", field="test", cache_dir="exp/")
test_set = test_dataset["train"]

for example in train_set:
    for label in example['labels']:
        if label != 'nan':
            FULL_CODES.append(label)

for example in dev_set:
    for label in example['labels']:
        if label != 'nan':
            FULL_CODES.append(label)

for example in test_set:
    for label in example['labels']:
        if label != 'nan':
            FULL_CODES.append(label)

FULL_CODES_SET = list(set(FULL_CODES))
FULL_CODES_SET.sort()

print(len(FULL_CODES_SET))

with open("exp/FULL_CODES.csv", 'w') as writer:
    writer.write("\n".join(FULL_CODES_SET))
