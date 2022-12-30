"""
With a few manual tests, I thought small model seem to be too weak.
It can know the intent, but the answer content it generated is just too shallow.
Let's try large model.

"""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("exp/google/flan-t5-base_50").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to("cuda")
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl").to("cuda")
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

def ask(text):
    # inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=150)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

