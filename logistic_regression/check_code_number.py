"""
Check the number of uniqe codes in training set and testing set.

train : 8686
dev : 3009
test : 4075
all : 8922
"""
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

labels_list_all = []
for split in ['train', 'dev', 'test']:
    df_tmp = pd.read_csv(f"/media/volume1/Corpus/mimic3/mimicdata/mimic3/{split}_full.csv")
    text_list = df_tmp['TEXT'].tolist()
    raw_labels_list = df_tmp['LABELS'].tolist()
    labels_list = [str(x).split(";") for x in raw_labels_list]

    tmp_labeler = MultiLabelBinarizer().fit(labels_list)
    print(split, ":", len(tmp_labeler.classes_))

    labels_list_all += labels_list

tmp_labeler = MultiLabelBinarizer().fit(labels_list_all)
print("all", ":", len(tmp_labeler.classes_))
