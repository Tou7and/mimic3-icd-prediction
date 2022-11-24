"""
Train a Logistic Regression model for MIMIC3 Medical Code Prediction.
Since it is a multi-lable classification task, we can use the OVR strategy to deal with it.

2022.11.22.
"""
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from prepare_mimic3 import prepare_mimic3_50, prepare_mimic3_full
import evaluation
import warnings
warnings.filterwarnings("ignore")

dataset = prepare_mimic3_full()

xx_tr = dataset['train'][0]
yy_tr = dataset['train'][1]

# use the same setting as https://github.com/jamesmullenbach/caml-mimic/blob/master/log_reg.py?
clf = OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=20, solver='sag'), n_jobs=-1)

# Similar mimic3-50 results when iter reaches 200
# clf = OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=200, solver='sag'), n_jobs=-1)

# Check data shape
for k, v in dataset.items():
    print(k, v[0].shape, v[1].shape)

# train
print("training...")
clf.fit(xx_tr, yy_tr)

print("evaluating...")
print("DEV:")
y_true = dataset['dev'][1]
y_pred = clf.predict(dataset['dev'][0])
y_pred_raw = clf.predict_proba(dataset['dev'][0])
y_true = y_true.toarray()
y_pred = y_pred.toarray()
metrics = evaluation.all_metrics(y_pred, y_true, k=[5, 8, 15], yhat_raw=y_pred_raw)
evaluation.print_metrics(metrics)

# pickle.dump(y_true, open("exp/y_true.pkl", 'wb'))
# pickle.dump(y_pred, open("exp/y_pred.pkl", 'wb'))
# pickle.dump(y_pred_raw, open("exp/y_pred_raw.pkl", 'wb'))

print("TEST:")
y_true = dataset['test'][1]
y_pred = clf.predict(dataset['test'][0])
y_pred_raw = clf.predict_proba(dataset['test'][0])
y_true = y_true.toarray()
y_pred = y_pred.toarray()
metrics = evaluation.all_metrics(y_pred, y_true, k=[5, 8, 15], yhat_raw=y_pred_raw)
evaluation.print_metrics(metrics)
