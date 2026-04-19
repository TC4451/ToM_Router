# Distillation Ablation Results

## Research Question
Does knowledge distillation from an OLMo-3-7B teacher improve student router performance, especially for weaker student models?

## Results

| Model | Params | Distilled | Test Acc | Test F1 | AUROC | Brier |
|-------|--------|-----------|----------|---------|-------|-------|
| Majority class | - | - | 0.5000 | 0.3333 | 0.5000 | - |
| Source-only LR | - | - | **0.9975** | **0.9975** | **1.0000** | - |
| Length-only LR | - | - | 0.5278 | 0.5121 | 0.7086 | - |
| TF-IDF BoW LR | - | - | 0.9924 | 0.9924 | 0.9999 | - |
| BERT-tiny (4M) — hard only | 4M | No | 0.9924 | 0.9924 | 0.9999 | 0.0046 |
| BERT-tiny (4M) — distilled | 4M | Yes | 0.9937 | 0.9937 | 0.9999 | 0.0118 |
| DistilRoBERTa (82M) — hard only | 82M | No | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| DistilRoBERTa (82M) — distilled | 82M | Yes | 1.0000 | 1.0000 | 1.0000 | 0.0077 |
| DeBERTa-v3-base (86M) — hard only | 184M | No | 0.9975 | 0.9975 | 1.0000 | 0.0019 |
| DeBERTa-v3-base (86M) — distilled | 184M | Yes | 0.9987 | 0.9987 | 1.0000 | 0.0099 |

## Key Findings

### 1. Source-style shortcut dominates (critical finding)
A logistic regression using **only the source dataset name** as a feature achieves 99.75% accuracy — matching or exceeding all neural models. This means the current dataset composition has a near-perfect correlation between source and label: ToM samples come from SimpleToM/tomi_nli/theory_of_mind, while Non-ToM samples come from social_iqa/cicero. The models are largely learning source-specific surface patterns, not genuine ToM reasoning detection.

### 2. BoW baseline is nearly as strong as neural models
TF-IDF bag-of-words logistic regression achieves 99.24% accuracy. This confirms that lexical cues alone (vocabulary differences between ToM and non-ToM datasets) are sufficient for near-perfect classification.

### 3. Distillation provides marginal gains
- BERT-tiny (4M): +0.13% accuracy with distillation
- DeBERTa-v3-base: +0.12% accuracy with distillation
- DistilRoBERTa: no gain (already perfect)

The gains from distillation are real but tiny because the task is already saturated. Distillation's value would be more visible on a harder, shortcut-free dataset.

### 4. Even 4M-param models nearly solve this task
BERT-tiny with only 2 layers and 128-dim hidden states achieves 99.24% — the same as the BoW baseline. This further confirms the task is solvable from surface features alone.

### 5. Length is not a strong shortcut
Length-only LR achieves only 52.78%, meaning context length alone doesn't predict the label well. The shortcut is lexical/stylistic, not structural.

## Implications for Future Work

The near-perfect source-shortcut baseline (99.75%) means the router's high accuracy does not validate that it has learned to distinguish ToM from non-ToM reasoning. To make the ablation meaningful, the dataset needs:

1. **Cross-source label mixing**: ensure ToM and non-ToM samples come from the same source datasets
2. **Adversarial non-ToM samples**: generate non-ToM questions about the same stories used for ToM questions
3. **Style normalization**: paraphrase samples to remove source-specific vocabulary patterns
4. **KokoMind-only evaluation**: KokoMind is the only source with both ToM and non-ToM samples from the same contexts — evaluating on KokoMind-only would be a more valid test
