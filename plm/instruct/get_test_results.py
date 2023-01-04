"""
Test instruction-tuned FLAN-T5 model using MIMIC3-FULL.

2022.12.30, JamesH.
"""
import os
import json
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LongT5ForConditionalGeneration
from tqdm import tqdm
from metrics import get_report

# google/t5-v1_1-base
# exp/google/flan-t5-base_full
class T5ForICD:
    def __init__(self, model_id, tokenizer_id):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    
    def predict(self, text, prompt="Translate English to ICD: "):
        input_text = prompt + text
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_length=100)
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        icd_codes = output_text.strip().split(" ")
        unique_icds = list(set(icd_codes))
        return unique_icds

class LongT5ForICD:
    def __init__(self, model_id, tokenizer_id):
        self.model = LongT5ForConditionalGeneration.from_pretrained(model_id).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    
    def predict(self, text, prompt="Translate English to ICD: "):
        input_text = prompt + text
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

        sequences = self.model.generate(input_ids, max_length=100) # .sequences
        summary = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)

        icd_codes = summary[0].strip().split(" ")
        unique_icds = list(set(icd_codes))
        return unique_icds

def main_full():
    print("loading model...")
    # model = FlanT5ICD("exp/google/flan-t5-base_full", "google/flan-t5-base_full")
    # model = T5ForICD("exp/google/t5-v1_1-base_full", "google/t5-v1_1-base")
    model = LongT5ForICD("exp/google/long-t5-tglobal-base_full", "google/long-t5-tglobal-base")
    output_dir = "exp/results_longt5base"
    output_path = os.path.join(output_dir, "test_full.json")
    os.makedirs(output_dir, exist_ok=True)

    print("loading data...")
    test_set = load_dataset(
        "json", data_files="exp/mimic3_full.json", 
        cache_dir="exp/cache", field="test")['train']

    results = []
    for idx, example in enumerate(test_set):
        print("refs: ", example['labels'])
        hyps = model.predict(example['text'])
        print("hyps: ", hyps)
        results.append({"idx": idx, "refs": example['labels'], "hyps": hyps})
        print("-"*100)

    with open(output_path, 'w') as writer:
        json.dump(results, writer, indent=4)

    results = get_report(output_path)
    return results

def main_50():
    print("loading model...")
    model = FlanT5ICD("exp/google/flan-t5-base_50")

    print("loading data...")
    test_set = load_dataset(
        "json", data_files="exp/mimic3_50.json", 
        cache_dir="exp/cache", field="test")['train']

    results = []
    for idx, example in enumerate(tqdm(test_set)):
        hyps = model.predict(example['text'])
        results.append({"idx": idx, "refs": example['labels'], "hyps": hyps})

    with open("exp/test_50.json", 'w') as writer:
        json.dump(results, writer, indent=4)

if __name__ == "__main__":
    # main_50()
    main_full()
