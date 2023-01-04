# Tuning Text-to-text models via instructions
WIP

# Setup
```
pip install -r requirements.txt
./install.sh
```

# Records of Results (MIMIC3-FULL, macro-f1, micro-f1)
- FLAN-T5-bese-FULL: 6.18%, 38.19%
- T511-base-FULL: 4.61%, 35.24%
- LONG-T5-base-FULL (1024): 5.03%, 36.11%

{'macro': {'f1': 0.0503, 'precision': 0.0691, 'recall': 0.0514}, 'micro': {'f1': 0.3611, 'precision': 0.4849, 'recall': 0.2877}}
