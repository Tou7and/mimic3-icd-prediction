#!/bin/bash

# 1st
# "google/flan-t5-base"
# python train.py google/flan-t5-base --dataset full --epoch 10 --maxlen 1024

# 2nd
# google/t5-v1_1-base
# python train.py google/t5-v1_1-base --dataset full --epoch 10 --maxlen 1024

# TODO: verify if the origin training recipe can work
# TODO: if not, make special training scripts for long-t5 models 
# 3rd
# google/long-t5-tglobal-base
# https://huggingface.co/google/long-t5-tglobal-base
# https://huggingface.co/google/long-t5-local-base
# global better than local?
python train.py google/long-t5-tglobal-base --dataset full --epoch 10 --maxlen 2048
# python train.py google/long-t5-tglobal-base --dataset full --epoch 10 --maxlen 1024
# python train_long.py google/long-t5-tglobal-base --dataset full --epoch 10 --maxlen 1024

# 4th
# Stancld/longt5-tglobal-large-16384-pubmed-3k_steps
# https://huggingface.co/Stancld/longt5-tglobal-large-16384-pubmed-3k_steps
# python train_long.py Stancld/longt5-tglobal-large-16384-pubmed-3k_steps --dataset full --epoch 10 --maxlen 1024

# 5th
# google/long-t5-tglobal-large
# (This is to compare with 4th model)

