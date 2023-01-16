"""
Results:
    results/full/README.md

Last update:
    2023.01.16
"""
import sys
from datasets import load_dataset
from tqdm import tqdm
from metrics import get_f1
from predict import IcdModel
from labels import LABEL_LIST_50, LABEL_LIST_FULL

def main(model_path, the_threshold=0.1):
    the_model = IcdModel(
        model_path,
        LABEL_LIST_FULL, seq_length=4096)

    test_set = load_dataset("json",
        data_files="exp/mimic3_full.json", cache_dir="exp/cache_full", field="test")
    
    test_set = test_set['train']
 
    refs = []
    preds = []
    for text, ref in tqdm(zip(test_set["text"], test_set["labels"]), total=len(test_set["text"])):
        # print(sample)
        pred, scores = the_model.predict(text, thr=the_threshold)
        preds.append(pred)
        refs.append(ref)

    f1 = get_f1(refs, preds)
    print(f"{model_path}, thr: {the_threshold}")
    print(f1)

if __name__ == "__main__":
    model_path = sys.argv[1]
    threshold = sys.argv[2]
    main(model_path, threshold)
