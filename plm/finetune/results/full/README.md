# Results on MIMIC3-FULL
Select only a few settings from results on 50 since mimic3-full will be more resource consuming.

```
exp/bert-base-cased_full 
- macro-f1: 0.0018 
- micro-f1: 0.1854

exp/yikuan8/Clinical-Longformer_full
- (thr=0.05) macro-f1: 0.0028, micro-f1: 0.164
- (thr=0.1)  macro-f1: 0.0015, micro-f1: 0.1842
- (thr=0.2)  macro-f1: 0.0007, micro-f1: 0.1496

yikuan8/Clinical-Longformer --dataset full --epoch 4 --maxlen 4096 --batchsize 2 
- (thr=0.1) macro-f1: 0.0022, micro-f1: 0.2252
```

