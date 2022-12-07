# Logistic Regression Baseline
Try to reproduce results of logistic regression on [Papers-with-code MIMIC3 Benchmarks](https://paperswithcode.com/sota/medical-code-prediction-on-mimic-iii) with [Scikit-Learn](https://scikit-learn.org/stable/).

Full-code testing results:
- LR in CAML paper: micro-f1=27.2%, macro-f1=1.1%
- [LR-50](train_lr_50.py): micro-f1=0.2720, macro-f1=0.0115

Top-50 testing results:
- LR in CAML paper: micro-f1=53.3%, macro-f1=47.7%
- [LR-Full](train_lr_full.py): micro-f1=0.5275, macro-f1=0.4666

Reference:
- [CAML paper](https://arxiv.org/pdf/1802.05695.pdf)
- [CAML: logictic regression](https://github.com/jamesmullenbach/caml-mimic/blob/master/log_reg.py)
  - The logistic regression model consists of X binary one-vs-rest classifiers acting on unigram bag-of-words features for all labels present in the training data (description from CAML paper).
- [CAML: evaluation](https://github.com/jamesmullenbach/caml-mimic/blob/master/evaluation.py)
  - Use the same evaluation metrics as CAML.

