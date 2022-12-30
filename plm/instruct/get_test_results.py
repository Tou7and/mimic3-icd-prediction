"""
Test instruction-tuned FLAN-T5 model using MIMIC3-FULL.

2022.12.30, JamesH.
"""
import json
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

# exp/google/flan-t5-base_full
class FlanT5ICD:
    def __init__(self, model_path):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    
    def predict(self, text, prompt="Translate English to ICD: "):
        input_text = prompt + text
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_length=100)
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        icd_codes = output_text.strip().split(" ")
        unique_icds = list(set(icd_codes))
        return unique_icds

def main_full():
    print("loading model...")
    model = FlanT5ICD("exp/google/flan-t5-base_full")

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

    with open("exp/test_full.json", 'w') as writer:
        json.dump(results, writer, indent=4)

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
