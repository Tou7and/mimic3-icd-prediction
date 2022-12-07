# Finetune BERT for MIMIC-3 ICD prediction
Hypothesis:
Is it possible to get a BERT finetuned model with better performance than CAML(macro-f1=8.8%, micro-f1=58.9%)?

Check [here](results/README.md) for detailed testing results.

Macro and Micro F1 of Clinical Longformer in MIMIC3-50:
- result in KEPT paper: 58.61%, 67.22%
- try to reproduce: 0.5874, 0.6768
  - (python train.py yikuan8/Clinical-Longformer --epoch 10 --maxlen 4096 --batchsize 2)

Reference:
- [KEPTLongformer](https://arxiv.org/pdf/2210.03304v2.pdf)

