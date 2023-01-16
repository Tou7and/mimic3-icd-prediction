# Results on MIMIC3-FULL
Select only a few settings from results on 50 since mimic3-full will be more resource consuming.

```
exp/bert-base-cased_full {'macro': {'f1': 0.0018, 'precision': 0.0013, 'recall': 0.0052}, 'micro': {'f1': 0.1854, 'precision': 0.1756, 'recall': 0.1962}}

exp/yikuan8/Clinical-Longformer_full, thr: 0.05
{'macro': {'f1': 0.0028, 'precision': 0.0016, 'recall': 0.015}, 'micro': {'f1': 0.164, 'precision': 0.1062, 'recall': 0.3601}}

exp/yikuan8/Clinical-Longformer_full, thr: 0.1
{'macro': {'f1': 0.0015, 'precision': 0.0009, 'recall': 0.0056}, 'micro': {'f1': 0.1842, 'precision': 0.1641, 'recall': 0.2098}}

exp/yikuan8/Clinical-Longformer_full, thr: 0.2
{'macro': {'f1': 0.0007, 'precision': 0.0005, 'recall': 0.0017}, 'micro': {'f1': 0.1496, 'precision': 0.267, 'recall': 0.1039}}
```

