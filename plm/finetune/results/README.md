# Results


# MIMIC3-50
Use mimic3-50 for quick experiments. <br>
```
(Fix epoch to 10)
BERT:
  eval_macro-f1: 0.2715
  eval_micro-f1: 0.4247 

Clinical_BERT:
  eval_macro-f1 0.3267
  eval_micro-f1 0.4814

Clinical Longformer, token-size=1024:
  eval_macro-f1 0.4898
  eval_micro-f1 0.5986

Clinical Longformer, token-size=4096:
  eval_macro-f1 0.5807
  eval_micro-f1 0.6754
# close to paper results (KEPT-Longformer w/o HSAP & Prompt)


python train.py yikuan8/Clinical-Longformer --epoch 20 --maxlen 4096 --batchsize 2
------- Eval Metrics -------
eval_loss 0.2244482934474945
eval_macro-f1 0.6172008100512372
eval_micro-f1 0.6827600285801776
eval_runtime 133.4915
eval_samples_per_second 12.952
eval_steps_per_second 3.244
epoch 20.0
```

# MIMIC3-FULL
Select only a few settings from results on 50 since mimic3-full will be more resource consuming.



