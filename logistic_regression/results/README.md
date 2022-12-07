## MIMIC3-full
There are 8922 unique codes in the whole dataset. <br>
But after partition, the training set contain only 8686 codes, <br>
which means the model trained on training set can not predict some of the codes in testing set.

With iter equal to 20:
```
python train_lr_full.py

TEST Results:
[MACRO] accuracy, precision, recall, f-measure, AUC
0.0066, 0.0255, 0.0074, 0.0115, 0.5627
[MICRO] accuracy, precision, recall, f-measure, AUC
0.1574, 0.6796, 0.1701, 0.2720, 0.9382
rec_at_5: 0.2060
prec_at_5: 0.6260
rec_at_8: 0.2768
prec_at_8: 0.5412
rec_at_15: 0.3802
prec_at_15: 0.4109
```

Increase iter to 200:
```
TEST Results:
[MACRO] accuracy, precision, recall, f-measure, AUC
0.0186, 0.0517, 0.0222, 0.0310, 0.7126
[MICRO] accuracy, precision, recall, f-measure, AUC
0.2257, 0.6367, 0.2591, 0.3683, 0.9620
rec_at_8: 0.3155
prec_at_8: 0.6045
rec_at_15: 0.4337
prec_at_15: 0.4636
```

## MIMIC3-50
2022.11.22: <br>
Unable to run the recipe CAML provides in my Python3.9.6 environment on Jupyter, DGX-2. <br>
Maybe we can write our own recipe and try to reproduce the results with scikit-learn and their dataset partitions.

With LR max-iter equal to 200:
```
python train_lr_50.py

TEST Results:
[MACRO] accuracy, precision, recall, f-measure, AUC
0.3175, 0.6144, 0.3762, 0.4666, 0.8438
[MICRO] accuracy, precision, recall, f-measure, AUC
0.3582, 0.6826, 0.4298, 0.5275, 0.8771
rec_at_5: 0.5316
prec_at_5: 0.5549
```
