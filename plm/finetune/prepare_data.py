"""
Prepare mimic3 data to datasets formats.

"""
import json
import pandas as pd

def prepare_50():
    """
    /media/volume1/Corpus/mimic3/mimicdata/mimic3/TOP_50_CODES.csv
    /media/volume1/Corpus/mimic3/mimicdata/mimic3/train_50.csv
    /media/volume1/Corpus/mimic3/mimicdata/mimic3/test_50.csv
    /media/volume1/Corpus/mimic3/mimicdata/mimic3/dev_50.csv
    """
    mimic50_dict = {
        "train": {'text': [], 'labels': []},
        "test": {'text': [], 'labels': []},
        "dev": {'text': [], 'labels': []}
    }

    for key, val in mimic50_dict.items():
        df = pd.read_csv(f"/media/volume1/Corpus/mimic3/mimicdata/mimic3/{key}_50.csv")
        for idx, row in df.iterrows():
            text = row['TEXT']
            labels = row['LABELS'].split(";")
            mimic50_dict[key]['text'].append(text)
            mimic50_dict[key]['labels'].append(labels)

    with open("exp/mimic3_50.json", 'w') as writer:
        json.dump(mimic50_dict, writer, indent=4)
    return

def prepare_full():
    """
    /media/volume1/Corpus/mimic3/mimicdata/mimic3/ALL_CODES.csv
    /media/volume1/Corpus/mimic3/mimicdata/mimic3/train_full.csv
    /media/volume1/Corpus/mimic3/mimicdata/mimic3/test_full.csv
    /media/volume1/Corpus/mimic3/mimicdata/mimic3/dev_full.csv
    """
    mimic_dict = {
        "train": {'text': [], 'labels': []},
        "test": {'text': [], 'labels': []},
        "dev": {'text': [], 'labels': []}
    }

    for key, val in mimic_dict.items():
        df = pd.read_csv(f"/media/volume1/Corpus/mimic3/mimicdata/mimic3/{key}_full.csv")
        for idx, row in df.iterrows():
            text = row['TEXT']
            labels = str(row['LABELS']).split(";")
            mimic_dict[key]['text'].append(text)
            mimic_dict[key]['labels'].append(labels)

    with open("exp/mimic3_full.json", 'w') as writer:
        json.dump(mimic_dict, writer, indent=4)
    return

if __name__ == "__main__":
    prepare_full()
