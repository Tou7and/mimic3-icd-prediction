# Finetune BERT for MIMIC-3 ICD prediction
Hypothesis:
Is it possible to get a BERT finetuned model with better performance than CAML(macro-f1=8.8%, micro-f1=58.9%)?

# MIMIC3-50
Use mimic3-50 for quick experiments. <br>
Select only a few settings for mimic3-full, which is more resource consuming.

```
BERT, EP10:
  eval_macro-f1: 0.2715199793067581
  eval_micro-f1: 0.42479525933817164 

Clinical_BERT, EP10:
  eval_macro-f1 0.3267354954657719
  eval_micro-f1 0.4814141540776031

Clinical Longformer, EP10, token-size=1024:
  eval_macro-f1 0.48988258762700626
  eval_micro-f1 0.5986001555382735

Clinical Longformer, EP10, token-size=4096:
  eval_macro-f1 0.580707539211979
  eval_micro-f1 0.6754994522977413
--> this is close to paper results (KEPT-Longformer w/o HSAP & Prompt)

```

