""" Inference Pipeline for FULL

""" 
import sys
import re
import json
import warnings
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.special import expit
import torch
from tqdm import tqdm
from glob import glob
from labels import LABEL_LIST_50, LABEL_LIST_FULL
warnings.filterwarnings("ignore")

class IcdModel:
    def __init__(self, model_path, label_list, seq_length=512, device_name="auto"):
        if device_name == "auto":
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device(device_name) # "cuda", "cpu", "cuda:0", "cuda:1", ...

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, output_attentions=True).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.seq_length = seq_length
        self.label_list = label_list

    def predict(self, texts, thr=0.5):
        """
        Given text, return a list of ICD codes 

        Args:
            texts(str): input text
            thr(float): threshold

        Returns:
            results(list): list of labels
            scores(list): list of corresponding scores(logits after sigmoid)
        """
        encoding = self.tokenizer(
            texts,
            max_length=self.seq_length, # 1024
            padding="max_length",
            truncation=True,
            return_tensors='pt',
        )
        inputs = encoding.input_ids.to(self.device)
        outputs = self.model(inputs)
        logits = outputs.logits
        # attentions = outputs.attentions
        preds = expit(logits)

        results = []
        scores = []
        for idx, x in enumerate(preds[0]):
            if x > float(thr):
                results.append(self.label_list[idx])
                scores.append(float(x))
        return results, scores

def main(model_path, text_list):
    the_model = IcdModel(
        model_path,
        LABEL_LIST_FULL,
        seq_length=512
    )

    hyp_list = []
    score_list = []
    for text in text_list:
        hyp, scores = the_model.predict(text, thr=0.1)
        hyp_list.append(hyp)
        score_list.append(scores)
    return hyp_list, score_list

if __name__ == "__main__":
    text_list = ["admission date discharge date date of birth sex m service med blumga for content of this discharge summary please refer to the discharge summary dictated by myself with discharge date of for content dr first name stitle first name3 lf dictated by last name namepattern1 medquist36 d t job job number"]
    hyp_list = main("exp/bert-base-cased_full", text_list)

    for hyp in hyp_list:
        print(hyp)
