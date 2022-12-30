"""
Reference:
    https://www.philschmid.de/fine-tune-flan-t5
"""
import argparse
import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

nltk.download("punkt")
metric = evaluate.load("rouge")

def main(model_id, dataset, maxlen, n_epoch):
    """
    model_id: "google/flan-t5-base"
    dataset: "50" or "full"
    maxlen: 512 ~ 4096
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    def preprocess_function(sample, padding="max_length"):
        # add prefix to the input for t5
        inputs = ["Translate English to ICD: " + item for item in sample["text"]]

        # tokenize inputs
        model_inputs = tokenizer(inputs, max_length=maxlen, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        label_strings = [" ".join(x) for x in sample['labels']]
        labels = tokenizer(
            text_target=label_strings, max_length=maxlen, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100
        # when we want to ignore padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def postprocess_text(preds, labels):
        """
        helper function to postprocess text.
        """
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(sent_tokenize(label)) for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    if dataset == "full":
        # label_list = LABEL_LIST_FULL
        json_path = "exp/mimic3_full.json"
    else:
        # label_list = LABEL_LIST_50
        json_path = "exp/mimic3_50.json"

    print("Loading dataset...")
    train_set = load_dataset("json", data_files=json_path, cache_dir="exp/cache", field="train")
    train_set = train_set['train']
    test_set = load_dataset("json", data_files=json_path, cache_dir="exp/cache", field="test")
    test_set = test_set['train']

    tokenized_trainset = train_set.map(preprocess_function, batched=True)
    tokenized_testset = test_set.map(preprocess_function, batched=True)
    
    # print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")
    # we want to ignore tokenizer pad token in the loss
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"exp/{model_id}_{dataset}",
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        predict_with_generate=True,
        fp16=False, # Overflows with fp16
        learning_rate=5e-5,
        num_train_epochs=n_epoch,
        logging_dir=f"exp/{model_id}_{dataset}/log",
        logging_strategy="steps",
        logging_steps=5000,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
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
    parser.add_argument('--epoch', nargs='?', default=10, type=int, help="number of epoch")
    args = parser.parse_args()
    print(args.modelname, args.dataset, args.maxlen, args.epoch)
    print("-"*100)
    main(args.modelname, args.dataset, args.maxlen, args.epoch)
