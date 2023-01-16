#!/bin/bash

# python train.py bert-base-cased --dataset full --epoch 2 --maxlen 512 --batchsize 2

# yikuan8/Clinical-Longformer
python train.py yikuan8/Clinical-Longformer  --dataset full --epoch 10 --maxlen 4096 --batchsize 2
# python train.py yikuan8/Clinical-Longformer --epoch 20 --maxlen 4096 --batchsize 2

# whaleloops/keptlongformer
# python train.py whaleloops/keptlongformer --epoch 10 --maxlen 4096 --batchsize 2

