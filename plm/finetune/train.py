"""
A simple PLM finetuning pipeline for multi-label tasks using Transformer and Trainer.

Use clinical longformer to try out:
    yikuan8/Clinical-Longformer
    Epoch: 10
    token size: 4096
    learning rate: 5e-5

KEPT Paper's results on MIMIC-50:
    Macro-F1: 58.61
    Micro-F1: 67.22

Reference:
    https://huggingface.co/docs/transformers/training
    https://arxiv.org/pdf/2210.03304v2.pdf (KEPT paper)
    https://huggingface.co/whaleloops/keptlongformer (KEPT public release)
    https://huggingface.co/yikuan8/Clinical-Longformer (Clinical Longformer public release)

2022.12.06, JamesH.
"""
import argparse
# import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, set_seed
from transformers import DataCollatorWithPadding, EvalPrediction
from transformers.utils import logging
from scipy.special import expit
from sklearn.metrics import f1_score
from multilabel_trainer import MultilabelTrainer
from labels import LABEL_LIST_50, LABEL_LIST_FULL

logging.set_verbosity_warning()

def compute_metrics(p: EvalPrediction):
    """ Compute Micro and Macro-F1 """
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = (expit(logits) > 0.5).astype('int32')
    macro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
    return {'macro-f1': macro_f1, 'micro-f1': micro_f1}

def main(modelname, dataset, maxlen, batchsize, learnrate, epoch):
    set_seed(123) # Helper function for reproducible behavior by fixing random

    if dataset == "full":
        label_list = LABEL_LIST_FULL
        json_path = "exp/mimic3_full.json"
    else:
        label_list = LABEL_LIST_50
        json_path = "exp/mimic3_50.json"

    print("Loading dataset...")
    train_set = load_dataset("json", data_files=json_path, cache_dir="exp/cache", field="train")
    train_set = train_set['train']
    test_set = load_dataset("json", data_files=json_path, cache_dir="exp/cache", field="test")
    test_set = test_set['train']

    print("Prepare tokenizer and tokenized dataset...")
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    model = AutoModelForSequenceClassification.from_pretrained(
        modelname, num_labels=len(label_list))

    def preprocess_function(examples):
        # Tokenize the texts
        batch = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=maxlen # 512-4096
        )
        batch["labels"] = [
            [1 if label in labels else 0 for label in label_list] for labels in examples["labels"]
        ]
        return batch

    tokenized_trainset = train_set.map(preprocess_function, batched=True)
    tokenized_testset = test_set.map(preprocess_function, batched=True)

    print("Training...")
    training_args = TrainingArguments(
        output_dir=f"exp/{modelname}_{dataset}",
        per_device_train_batch_size=batchsize, # 2-16
        per_device_eval_batch_size=batchsize, # 2-16
        learning_rate=learnrate,
        num_train_epochs=epoch, # 2-10
        evaluation_strategy="epoch",
        save_strategy='epoch',
        save_total_limit=2,
    )

    trainer = MultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_trainset,
        eval_dataset=tokenized_testset,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model()

    eval_metrics = trainer.evaluate()
    print("------- Eval Metrics -------")
    for k, v in eval_metrics.items():
        print(k, v)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('modelname')
    parser.add_argument('--dataset', nargs='?', default="50", type=str, help="50 or full")
    parser.add_argument('--maxlen', nargs='?', default=512, type=int, help="max token length")
    parser.add_argument('--batchsize', nargs='?', default=8, type=int, help="batch size on each GPU")
    parser.add_argument('--learnrate', nargs='?', default=5e-5, type=float, help="learning rate")
    parser.add_argument('--epoch', nargs='?', default=10, type=int, help="number of epoch")
    args = parser.parse_args()
    print(args.modelname, args.dataset, args.maxlen, args.learnrate, args.epoch)
    print("-"*100)
    main(args.modelname, args.dataset, args.maxlen, args.batchsize, args.learnrate, args.epoch)
