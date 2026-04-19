# Hardened Dataset Ablation

## Goal
Compare original dataset (with source shortcuts) vs hardened dataset
(contrastive pairs + style normalization) to show:

1. Contrastive augmentation breaks source-only shortcuts
2. Knowledge distillation provides larger gains on harder data
3. Weak models benefit more from distillation on harder tasks

## Results

| Dataset | Model | Distilled | Acc | F1 | AUROC |
|---------|-------|-----------|-----|----|----- |
| original | Majority class | - | 0.5000 | 0.3333 | 0.5000 |
| original | Source-only LR (base source) | - | 0.3902 | 0.3541 | 0.5922 |
| original | TF-IDF BoW LR | - | 0.9924 | 0.9924 | 0.9999 |
| original | BERT-tiny hard | No | 0.9924 | 0.9924 | 0.9999 |
| original | BERT-tiny distilled | Yes | 0.9937 | 0.9937 | 0.9999 |
| original | DeBERTa hard | No | 0.9975 | 0.9975 | 1.0000 |
| original | DeBERTa distilled | Yes | 0.9962 | 0.9962 | 1.0000 |
| hardened | Majority class | - | 0.5000 | 0.3333 | 0.5000 |
| hardened | Source-only LR (base source) | - | 0.5424 | 0.5170 | 0.4515 |
| hardened | TF-IDF BoW LR | - | 0.9254 | 0.9254 | 0.9718 |
| hardened | BERT-tiny hard | No | 0.9641 | 0.9645 | 0.9920 |
| hardened | BERT-tiny distilled | Yes | 0.9622 | 0.9625 | 0.9879 |
| hardened | DeBERTa hard | No | 0.9917 | 0.9917 | 0.9997 |
| hardened | DeBERTa distilled | Yes | 0.9954 | 0.9954 | 0.9985 |