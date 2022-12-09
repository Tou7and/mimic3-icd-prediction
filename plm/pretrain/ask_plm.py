"""
Apply MLM from an existing PLM.
(domain adaption? or knowledge injection?)

https://huggingface.co/course/chapter7/3?fw=pt
"""
import torch
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import pipeline

# model_checkpoint = "distilbert-base-uncased"
# model_checkpoint = "bert-base-uncased"
model_checkpoint = "exp/bert-mlm-imdb"
# model_checkpoint = "exp/bert-wwm-imdb"

model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def ask_plm(text="This is a great [MASK]."):
    # Try to predict the mask in a sentence
    inputs = tokenizer(text, return_tensors="pt")
    token_logits = model(**inputs).logits

    # Find the location of [MASK] and extract its logits
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    # Pick the [MASK] candidates with the highest logits
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    for token in top_5_tokens:
        print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

def ask_plm2(text="This is a great [MASK]."):
    mask_filler = pipeline("fill-mask", model="exp/bert-mlm-imdb")
    preds = mask_filler(text)
    for pred in preds:
        print(f">>> {pred['sequence']}")
    return

if __name__ == "__main__":
    # ask_plm()
    # ask_plm("This movie is terrible. I will give it a [MASK] out of ten.")
    ask_plm("This movie is so good. I will give it a [MASK] out of ten.")
