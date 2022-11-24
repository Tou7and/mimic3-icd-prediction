"""
After downloading MIMIC3 dataset, put these CSV tables together:
    DIAGNOSES_ICD.csv
    NOTEEVENTS.csv
    PROCEDURES_ICD.csv

Then preprocess them following CAML recipes:
    https://github.com/jamesmullenbach/caml-mimic/tree/master/notebooks

Files we got after preprocessing: (ls /media/volume1/Corpus/mimic3/mimicdata/mimic3)
    ALL_CODES.csv           DIAGNOSES_ICD.csv      notes_labeled.csv       TOP_50_CODES.csv
    ALL_CODES_filtered.csv  disch_dev_split.csv    PROCEDURES_ICD.csv      train_50.csv
    dev_50.csv              disch_full.csv         test_50.csv             train_50_hadm_ids.csv
    dev_50_hadm_ids.csv     disch_test_split.csv   test_50_hadm_ids.csv    train_full.csv
    dev_full.csv            disch_train_split.csv  test_full.csv           train_full_hadm_ids.csv
    dev_full_hadm_ids.csv   NOTEEVENTS.csv         test_full_hadm_ids.csv  vocab.csv

Check the header and an example from train_full.csv:
    SUBJECT_ID,HADM_ID,TEXT,LABELS,length
    158,169433,admission date discharge date ...,532.40;493.20;V45.81;412;401.9;44.43,51

Check number:
    Full-label setting: 47,724 discharge summaries for training, 1,632 and 3,372 for validation and testing.
    Top 50: 8,067 summaries for training, 1,574 for validation, and 1,730 for testing.

Now start the final preprocess for LR training.

2022.11.22, JamesH.
"""
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

def build_unigram_vectorizer(train_texts, save_local=False):
    """
    Use count-vectorizer for unigram BOW extraction.

    Args:
        train_texts(list): list of training corpus.
    """
    vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=3) # token_pattern=regex, min_df=min_freq
    vectorizer.fit(train_texts)

    # set save-local to True if we will use the vectorizer later.
    if save_local:
        pickle.dump(vectorizer, open("exp/vectorizer.pkl", 'wb'))
    return vectorizer

def build_label_encoder(list_labels, save_local=False):
    """
    Use MultiLabelBinarizer for multi-label discretizing.

    Args:
        list_labels(list): list of target ICD labels.
    """
    label_encoder = MultiLabelBinarizer(sparse_output=True)
    label_encoder.fit(list_labels)

    # set save-local to True if we will use the label-encoder later.
    if save_local:
        pickle.dump(vectorizer, open("exp/labelencoder.pkl", 'wb'))
    return label_encoder

def prepare_mimic3_full():
    """
    Return mimic3 full-code dataset.
    """
    dataset = dict()
    for split in ['train', 'dev', 'test']:
        df_tmp = pd.read_csv(f"/media/volume1/Corpus/mimic3/mimicdata/mimic3/{split}_full.csv")
        text_list = df_tmp['TEXT'].tolist()
        raw_labels_list = df_tmp['LABELS'].tolist()
        labels_list = [str(x).split(";") for x in raw_labels_list]

        if split == 'train':
            vectorizer = build_unigram_vectorizer(text_list)
            label_encoder = build_label_encoder(labels_list)
        data_x = vectorizer.transform(text_list)
        data_y = label_encoder.transform(labels_list)
        dataset[split] = [data_x, data_y]
    return dataset

def prepare_mimic3_50():
    """
    Return mimic3 top-50 dataset:
        {'train': [X, Y], 'dev': [X, Y], 'test': [X, Y]}
    X and Y are discretized text and labels.
    """
    dataset = dict()
    for split in ['train', 'dev', 'test']:
        df_tmp = pd.read_csv(f"/media/volume1/Corpus/mimic3/mimicdata/mimic3/{split}_50.csv")

        text_list = df_tmp['TEXT'].tolist()
        raw_labels_list = df_tmp['LABELS'].tolist()
        labels_list = [x.split(";") for x in raw_labels_list]

        if split == 'train':
            vectorizer = build_unigram_vectorizer(text_list)
            label_encoder = build_label_encoder(labels_list)
        data_x = vectorizer.transform(text_list)
        data_y = label_encoder.transform(labels_list)
        dataset[split] = [data_x, data_y]
    return dataset

if __name__ == "__main__":
    dataset = prepare_mimic3_50()

    for k, v in dataset.items():
        print(k, v[0].shape, v[1].shape)
