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
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class IcdModel:
    def __init__(self, model_path, label_list, seq_length=512):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, output_attentions=True).to(device)
        self.model.eval()
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if "bert-base-cased" in model_path:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        elif "":

        else:
            raise ValueError("")
        self.seq_length = seq_length
        self.label_list = label_list

    def predict(self, texts, thr=0.5):
        """ Given text, return a list of ICD codes """
        encoding = self.tokenizer(
            texts,
            max_length=self.seq_length, # 1024
            padding="max_length",
            truncation=True,
            return_tensors='pt',
        )
        inputs = encoding.input_ids.to(device)
        outputs = self.model(inputs)
        logits = outputs.logits
        # attentions = outputs.attentions
        preds = expit(logits)

        results = []
        for idx, x in enumerate(preds[0]):
            if x > thr:
                results.append([self.label_list[idx], float(x)])
        return results

def main(model_path, text_list):
    the_model = IcdModel(
        model_path,
        LABEL_LIST_FULL,
        seq_length=512
    )

    hyp_list = []
    for text in text_list:
        # text = clean_text_op(text)
        hyp = the_model.predict(text, thr=0.1)
        hyp_list.append(hyp)

    return hyp_list

if __name__ == "__main__":
    text_list = [
        "nan intravenous chemotherapy hour nan nan nan spirometry demonstrates moderate obstructive ventilatory defect . fev fvc pred fev p l fvc l lung volumes are consistent with air trapping . rv p following the inhalation of a post bronchodilator , there is no significant change in airway obstruction . nan",
        "nan hemodialysis upper gi panendoscopy nan nan nan abdominal sonography diagnosis liver cirrhosis , score with moderate ascites hccs with bilateral pvtts , suspected viable gallstones and gallbladder wall secondary change renal parenchyma disease suggestion follow up and supportive care esophagogastroduodenoscopy diagnostic impression hemorrhagic esophagitis ph gastropathy superficial gastritis shallow gus and dus suggestion of management ppis therapy endoscopist attending physician nan"
    ]

    text_list = ["admission date discharge date date of birth sex m service med blumga for content of this discharge summary please refer to the discharge summary dictated by myself with discharge date of for content dr first name stitle first name3 lf dictated by last name namepattern1 medquist36 d t job job number"]
    hyp_list = main("exp/bert-base-cased_full", text_list)

    for hyp in hyp_list:
        print(hyp)
