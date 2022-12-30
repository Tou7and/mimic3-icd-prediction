from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.preprocessing import MultiLabelBinarizer
# from pecos.utils import smat_util

import warnings
warnings.filterwarnings("ignore")

def get_f1(refs, hyps):
    """ compute F1 socrs.

    refs = [
        ['cat', 'dog'],
        ['dog'],
        ['wolf'],
    ]

    hyps = [
        ['cat', 'wolf'],
        ['dog'],
        ['cat'],
    ]
    """
    mlb = MultiLabelBinarizer()
    mlb.fit(refs)

    ref_bag = mlb.transform(refs)
    hyp_bag = mlb.transform(hyps)

    micro_f1 = f1_score(ref_bag, hyp_bag, average="micro")
    micro_recall = recall_score(ref_bag, hyp_bag, average="micro")
    micro_precision = precision_score(ref_bag, hyp_bag, average="micro")
    micro = {
        'f1': round(micro_f1, 4), 
        'precision': round(micro_precision, 4), 
        'recall': round(micro_recall, 4)
    } 

    macro_f1 = f1_score(ref_bag, hyp_bag, average="macro")
    macro_recall = recall_score(ref_bag, hyp_bag, average="macro")
    macro_precision = precision_score(ref_bag, hyp_bag, average="macro")
    macro = {
        'f1': round(macro_f1, 4), 
        'precision': round(macro_precision, 4), 
        'recall': round(macro_recall, 4)
    } 
    return {"macro": macro, "micro": micro}

def keep_target_labels(input_list, target_labels):
    output_list = []
    for input_labels in input_list:
        output_labels = []
        for label in input_labels:
            if label in target_labels:
                output_labels.append(label)
        output_list.append(output_labels)
    return output_list

if __name__ == "__main__":
    refs = [
        ['cat', 'dog', 'rat'],
        ['dog'],
        ['rat'],
    ]
    hyps = [
        ['cat'],
        ['dog'],
        ['rat'],
    ]

    print(get_f1(refs, hyps))

    refs = [
        ['cat', 'dog', 'rat'],
        ['dog'],
        ['rat'],
    ]
    hyps = [
        ['cat', 'dog', 'rat'],
        ['dog', 'cat', 'rat'],
        ['rat', 'cat', 'dog'],
    ]
    print(get_f1(refs, hyps))

    refs_keep = keep_target_labels(refs, ["rat"])
    hyps_keep = keep_target_labels(hyps, ["rat"])
    print(get_f1(refs_keep, hyps_keep))
