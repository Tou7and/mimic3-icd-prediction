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
-------------------------------------
"epoch": 10.0,
"eval_loss": 0.20092199742794037,                                                                         
"eval_macro-f1": 0.5874442444680102,
"eval_micro-f1": 0.6768977237614585,
"step": 20170

python train.py whaleloops/keptlongformer --epoch 10 --maxlen 4096 --batchsize 2
-------------------------------------
"epoch": 10.0,
"eval_loss": 0.20039880275726318,
"eval_macro-f1": 0.5509869716707443,
"eval_micro-f1": 0.6616715913887717,
"eval_runtime": 138.3993,
"step": 20170
```

# MIMIC3-FULL
Select only a few settings from results on 50 since mimic3-full will be more resource consuming.



