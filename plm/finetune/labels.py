with open("/media/volume1/Corpus/mimic3/mimicdata/mimic3/TOP_50_CODES.csv", "r") as reader:
    LABEL_LIST_50 = reader.read().strip().split("\n")

with open("/media/volume1/Corpus/mimic3/mimicdata/mimic3/FULL_CODES.csv", "r") as reader:
    LABEL_LIST_FULL = reader.read().strip().split("\n")

