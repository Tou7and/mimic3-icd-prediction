"""
Results (FLAN-T5-bese-FULL):
    {'macro': {'f1': 0.0618, 'precision': 0.0837, 'recall': 0.0617}, 
    'micro': {'f1': 0.3819, 'precision': 0.5652, 'recall': 0.2884}}

Results (FLAN-T5-base-50):
    {'macro': {'f1': 0.3634, 'precision': 0.508, 'recall': 0.3339}, 
    'micro': {'f1': 0.5063, 'precision': 0.6439, 'recall': 0.4172}}
"""

import json
from metrics import get_f1

def get_report(file_path):
    with open(file_path, 'r') as reader:
        info = json.load(reader)
    ref_list = []
    hyp_list = []
    for example in info:
        ref_list.append(example['refs'])
        hyp_list.append(example['hyps'])
    print(len(ref_list))
    print(get_f1(ref_list, hyp_list))

get_report("exp/test_full.json")
# get_report("exp/test_50.json")
