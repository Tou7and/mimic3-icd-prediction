import pickle
import numpy as np
import evaluation
from sklearn.metrics import f1_score
from prepare_mimic3 import prepare_mimic3_full

dataset = prepare_mimic3_full()

yy_tr = dataset['train'][1]
print(yy_tr.shape)

def union_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)

def intersect_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)

# y_true = pickle.load(open('exp/y_true.pkl', 'rb'))
# y_pred = pickle.load(open('exp/y_pred.pkl', 'rb'))
# y_pred_raw = pickle.load(open('exp/y_pred_raw.pkl', 'rb'))

# y_true = y_true.toarray()
# y_pred = y_pred.toarray()

# print(union_size(y_true, y_predict.any(), 1))
# print(np.logical_and(y_true, y_predict))
# metrics = evaluation.all_metrics(y_pred, y_true, k=[8, 15], yhat_raw=y_pred_raw)
# evaluation.print_metrics(metrics)
