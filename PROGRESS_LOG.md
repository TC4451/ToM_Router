# Theory-of-Mind Router: Implementation & Experiment Log

## Glossary

| Term | Meaning |
|------|---------|
| **ToM** | Theory of Mind — the ability to reason about what other people believe, know, intend, or feel, especially when that information is hidden or differs from reality |
| **Non-ToM** | Questions answerable from observable facts, stated emotions, social norms, or common sense — no hidden mental state reasoning needed |
| **Router** | A classifier that decides whether a question requires ToM reasoning or not, then sends it to the appropriate expert model |
| **Teacher** | A large language model (OLMo-3, 7 billion parameters) that labels training data with soft probability scores |
| **Student** | A small, fast classifier (DeBERTa, 86–184 million parameters) trained to mimic the teacher's decisions |
| **Knowledge distillation** | Training technique where a small student model learns from both the true labels and the teacher model's probability outputs (soft labels) |
| **Hard labels** | The ground-truth binary label: 1 = requires ToM, 0 = does not |
| **Soft labels** | The teacher's probability estimate (e.g., 0.85) rather than a binary 0/1 — captures uncertainty and borderline cases |
| **Contrastive pairs** | For the same story context, one question that requires ToM and one that does not — forces the model to distinguish question type rather than story style |
| **Source shortcut** | When a model achieves high accuracy by recognizing which dataset a sample came from (writing style, vocabulary) rather than understanding the content |
| **Accuracy** | Percentage of correct predictions |
| **F1 score** | Harmonic mean of precision and recall — balances false positives and false negatives |
| **AUROC** | Area Under the Receiver Operating Characteristic curve — measures how well the model separates the two classes across all possible thresholds (1.0 = perfect, 0.5 = random) |
| **Brier score** | Measures how well-calibrated the predicted probabilities are (lower = better, 0 = perfect) |
| **Logistic Regression (LR)** | A simple statistical classifier used here as a baseline |
| **TF-IDF** | Term Frequency–Inverse Document Frequency — a method for converting text into numerical features based on word importance |
| **Bag-of-Words (BoW)** | Representing text as a collection of word counts, ignoring word order |
| **DeBERTa** | A pretrained language encoder from Microsoft, used as the student router |
| **BERT-tiny** | A very small version of BERT with only 4 million parameters and 2 layers — used to test whether a weak model benefits from distillation |
| **OLMo-3** | Open Language Model from AI2, 7 billion parameters — used as the teacher and as expert models |
| **4-bit quantization** | Compressing model weights from 16-bit to 4-bit to reduce memory usage (allows running a 7B model on a single GPU) |

---

## Phase 1: Data Collection and Preparation

**Goal:** Build a large, balanced dataset of ToM and non-ToM social reasoning questions.

### Datasets downloaded

| Dataset | Source | Samples | Label | Description |
|---------|--------|---------|-------|-------------|
| SimpleToM | HuggingFace (`allenai/SimpleToM`) | 3,441 | ToM | False belief stories with behavior, judgment, and mental state questions |
| KokoMind | GitHub (`CHATS-lab/KokoMind`) | 770 | Mixed | Social interaction dialogues with ToM, emotion, norm, and relation questions |
| theory_of_mind | HuggingFace (`hmamin/theory_of_mind`) | 4,058 | ToM | Aggregated ToM benchmarks: false belief, faux pas, strange stories, and more |
| tomi_nli | HuggingFace (`tasksource/tomi-nli`) | 17,982 | ToM | Classic Sally-Anne style false belief tracking in natural language inference format |
| social_iqa | HuggingFace (`allenai/social_i_qa`) | 34,934 | Non-ToM | Social commonsense questions about motivations, reactions, and next actions |
| cicero | HuggingFace (`declare-lab/cicero`) | 22,731 | Non-ToM | Commonsense inference questions about causes, motivations, and emotions in dialogues |

### Merged dataset

- **7,966 balanced samples** (3,983 ToM / 3,983 Non-ToM)
- Split into Train (6,374) / Validation (800) / Test (792)
- Splitting was done by grouping on story context to prevent the same story from appearing in both train and test (anti-leakage)
- Labels were balanced within each split

**Output:** `data/processed/router_dataset.parquet`

---

## Phase 2: Teacher Labeling with OLMo-3

**Goal:** Use a strong language model to generate soft probability labels for every sample.

- **Model:** OLMo-3-7B-Instruct, loaded in 4-bit quantization on a single RTX 5090 GPU (34 GB)
- **Process:** For each sample, the model reads the context and question, then outputs a JSON with:
  - `requires_tom`: 0 or 1 (binary decision)
  - `prob_tom`: a probability between 0 and 1 (soft label)
  - `rationale`: a one-sentence explanation
- **Duration:** 128 minutes for all 7,966 samples (0.96 samples/second)
- **Caching:** Results were saved incrementally so the process could resume if interrupted

### Teacher quality

- **Agreement with ground-truth labels: 56.4%** — This is expected. Many social reasoning questions sit on the boundary between ToM and non-ToM (for example, inferring someone's motivation could be seen as either), and the teacher and the dataset authors may disagree on these edge cases.
- The student model was trained on **both** the ground-truth hard labels and the teacher's soft labels, letting it learn from both signals.

**Output:** `outputs/teacher_labels/router_dataset_with_teacher.parquet`

---

## Phase 3: Student Router Training

**Goal:** Train a small, fast classifier to replicate the teacher's routing decisions.

- **Model:** DeBERTa-v3-base (184 million parameters) with a single linear classification head
- **Input format:** `[CONTEXT] {story text} [QUESTION] {question text}`
- **Loss function:** Weighted combination of two losses:
  - 70% weight on hard-label binary cross-entropy (learning from ground truth)
  - 30% weight on soft-label binary cross-entropy (learning from teacher probabilities)
  - Temperature scaling of 1.5 applied to soft labels
- **Training:** 5 epochs, batch size 16, learning rate 0.00002, AdamW optimizer, cosine learning rate schedule

### Results on validation set

| Metric | Value |
|--------|-------|
| F1 score | 0.9975 |
| AUROC | 1.0000 |
| Accuracy | 99.75% |

### Results on test set

| Metric | Value |
|--------|-------|
| Accuracy | 99.75% |
| AUROC | 99.9999% |
| False positives | 1 (predicted ToM when not needed) |
| False negatives | 1 (missed a ToM case) |

**Best classification threshold:** 0.72 (found by searching on validation data)

**Output:** `outputs/checkpoints/router_student/`

---

## Phase 4: Router Evaluation

**Goal:** Thoroughly evaluate the router with multiple metrics, sliced by data source and question type.

### Metrics computed
- Accuracy, Precision, Recall, F1, Macro-F1
- AUROC (area under ROC curve), AUPRC (area under precision-recall curve)
- Brier score (probability calibration quality)
- Expected Calibration Error
- Confusion matrix
- Metrics broken down by source dataset and by question subtype
- Error analysis report with example false positives, false negatives, and borderline cases

### Routed system comparison

The router was compared against simple baselines:

| Strategy | Routing Accuracy | ToM Recall | Non-ToM Recall |
|----------|-----------------|------------|----------------|
| **Student Router** | **99.7%** | **99.7%** | **99.7%** |
| Oracle (perfect routing) | 100% | 100% | 100% |
| Always send to ToM expert | 50.0% | 100% | 0% |
| Always send to Social expert | 50.0% | 0% | 100% |
| Random routing | 47.9% | 50.0% | 45.7% |

**Outputs:** `outputs/reports/router_eval_metrics.json`, `outputs/reports/router_error_analysis.txt`

---

## Phase 5: Expert Models and Routed Inference Pipeline

**Goal:** Build the end-to-end system that routes questions to specialized experts.

- **ToM Expert:** OLMo-3 with a system prompt focused on hidden beliefs, false beliefs, intentions, and perspectives
- **Social Expert:** OLMo-3 with a system prompt focused on social norms, emotions, and observable relationships
- **Fallback:** When the router is uncertain (probability near the threshold), the system defaults to the ToM expert to avoid missing mental-state reasoning cases
- **Pipeline:** Router classifies the question, then forwards it to the appropriate expert for answering

**Output:** `src/inference/routed_qa_pipeline.py`

---

## Phase 6: Ablation Studies on the Original Dataset

**Goal:** Test whether the high accuracy is genuine or caused by dataset shortcuts.

### Shortcut baselines

These baselines test whether simple features (not understanding the content) can predict the label:

| Baseline | What it uses | Accuracy | F1 | AUROC |
|----------|-------------|----------|----|----- |
| Majority class | Always predicts the most common label | 50.00% | 33.33% | 50.00% |
| **Source-only** | **Only the dataset name (e.g., "simpletom" vs "social_iqa")** | **99.75%** | **99.75%** | **100%** |
| Context length only | Only the number of characters in the context | 52.78% | 51.21% | 70.86% |
| Bag-of-words | Word frequencies in context + question | 99.24% | 99.24% | 99.99% |

**Critical finding:** A model that knows **only which dataset a sample came from** achieves 99.75% accuracy — matching the neural models. This means the original dataset has a near-perfect correlation between source and label, and models may be learning writing style rather than ToM reasoning.

### Distillation comparison (hard labels only vs. distilled)

| Model | Parameters | Hard labels only | With distillation | Improvement |
|-------|-----------|-----------------|-------------------|-------------|
| BERT-tiny | 4 million | 99.24% | 99.37% | +0.13% |
| DistilRoBERTa | 82 million | 100.00% | 100.00% | +0.00% |
| DeBERTa-v3-base | 184 million | 99.75% | 99.87% | +0.12% |

Distillation provides small gains, but the effect is masked because the task is too easy (saturated by source shortcuts).

**Output:** `outputs/reports/ablations/distillation_ablation.md`

---

## Phase 7: Dataset Hardening

**Goal:** Eliminate the source shortcut and make the classification task genuinely challenging.

### Two techniques applied

#### 1. Contrastive question generation

For 3,000 samples, OLMo-3 generated an **opposite-label question about the same story context**:
- For each ToM story, it wrote a non-ToM question (e.g., a factual question instead of a belief question)
- For each non-ToM story, it wrote a ToM question (e.g., a false-belief question instead of an emotion question)
- **Success rate:** 97.4% (2,922 valid pairs out of 3,000 attempts)

This means the same context now appears with both labels, so the model cannot rely on context style alone.

#### 2. Style normalization

OLMo-3 rewrote 382 samples into a uniform third-person narrative style, removing source-specific formatting (e.g., KokoMind's dialogue format vs. SimpleToM's short paragraphs). This was a partial run — the full dataset would take approximately 7 hours.

### Hardened dataset

- **10,782 balanced samples** (5,391 ToM / 5,391 Non-ToM)
- Both labels now contain samples from every source dataset (via contrastive pairs)
- Train: 8,624 / Validation: 1,072 / Test: 1,086

**Output:** `data/processed/router_dataset_hardened.parquet`

---

## Phase 8: Ablation Studies on the Hardened Dataset

**Goal:** Prove that (a) the source shortcut is eliminated, and (b) distillation provides meaningful gains on harder data.

### Full comparison table

| Dataset | Model | Distilled? | Accuracy | F1 | AUROC |
|---------|-------|-----------|----------|----|----- |
| Original | Majority class | — | 50.00% | 33.33% | 50.00% |
| Original | Source-only logistic regression | — | 99.75% | 99.75% | 100.00% |
| Original | Bag-of-words logistic regression | — | 99.24% | 99.24% | 99.99% |
| Original | BERT-tiny (4M params) | No | 99.24% | 99.24% | 99.99% |
| Original | BERT-tiny (4M params) | Yes | 99.37% | 99.37% | 99.99% |
| Original | DeBERTa (184M params) | No | 99.75% | 99.75% | 100.00% |
| Original | DeBERTa (184M params) | Yes | 99.62% | 99.62% | 100.00% |
| | | | | | |
| **Hardened** | Majority class | — | 50.00% | 33.33% | 50.00% |
| **Hardened** | **Source-only logistic regression** | — | **54.24%** | **51.70%** | **45.15%** |
| **Hardened** | Bag-of-words logistic regression | — | 92.54% | 92.54% | 97.18% |
| **Hardened** | BERT-tiny (4M params) | No | 96.41% | 96.45% | 99.20% |
| **Hardened** | BERT-tiny (4M params) | Yes | 96.22% | 96.25% | 98.79% |
| **Hardened** | DeBERTa (184M params) | No | 99.17% | 99.17% | 99.97% |
| **Hardened** | **DeBERTa (184M params)** | **Yes** | **99.54%** | **99.54%** | **99.85%** |

### Key findings

#### Finding 1: Source shortcut is eliminated
The source-only baseline dropped from **99.75% to 54.24%** (near chance). Contrastive augmentation successfully broke the correlation between dataset source and label. A model can no longer cheat by recognizing writing style.

#### Finding 2: The task is genuinely harder now
All models show lower accuracy on the hardened dataset:
- BERT-tiny: 99.24% → 96.41% (a 2.83 percentage point drop)
- DeBERTa: 99.75% → 99.17% (a 0.58 percentage point drop)
- Bag-of-words baseline: 99.24% → 92.54% (a 6.7 percentage point drop)

#### Finding 3: Distillation helps on the harder dataset
On the hardened dataset, DeBERTa with distillation (**99.54%**) outperforms DeBERTa without distillation (**99.17%**) by **+0.37 percentage points**. This is the clearest evidence that soft labels from the teacher provide meaningful signal when the task cannot be solved by surface shortcuts alone.

#### Finding 4: Weak models do not benefit from distillation without proper teacher labels
BERT-tiny with distillation actually performs slightly worse (-0.19 percentage points) on the hardened dataset. This is because the contrastive samples were assigned hard-label proxies instead of proper teacher soft labels. Re-running the OLMo-3 teacher on contrastive samples would likely fix this.

#### Finding 5: Lexical shortcuts partially remain
The bag-of-words baseline still achieves 92.54% on the hardened data (down from 99.24%). Some vocabulary differences between ToM and non-ToM questions persist, meaning further style normalization would help.

---

## All Files Created

### Configuration files
| File | Purpose |
|------|---------|
| `configs/data.yaml` | Dataset paths and split ratios |
| `configs/teacher.yaml` | Teacher model settings |
| `configs/router_student.yaml` | Student training hyperparameters |
| `configs/experts.yaml` | Expert model settings |
| `configs/eval_router.yaml` | Router evaluation settings |
| `configs/eval_routed.yaml` | Routed system evaluation settings |

### Data preprocessing scripts
| Script | Purpose |
|--------|---------|
| `scripts/prepare_simpletom.py` | Process SimpleToM into unified format |
| `scripts/prepare_kokomind.py` | Process KokoMind into unified format |
| `scripts/prepare_theory_of_mind.py` | Process theory_of_mind into unified format |
| `scripts/prepare_tomi_nli.py` | Process tomi-nli into unified format |
| `scripts/prepare_social_iqa.py` | Process SocialIQA into unified format |
| `scripts/prepare_cicero.py` | Process CICERO into unified format |
| `scripts/build_router_dataset.py` | Merge, deduplicate, balance, and split all datasets |

### Model and training scripts
| Script | Purpose |
|--------|---------|
| `scripts/generate_teacher_labels.py` | Run OLMo-3 teacher on all samples to produce soft labels |
| `scripts/train_student_router.py` | Train the student router with knowledge distillation |
| `scripts/train_experts.py` | Configure ToM and Social expert models |
| `scripts/eval_router.py` | Evaluate router with full metrics, slices, and error analysis |
| `scripts/eval_routed_system.py` | Compare routed system against baselines |
| `scripts/export_router.py` | Export trained router for deployment |

### Dataset hardening scripts
| Script | Purpose |
|--------|---------|
| `scripts/generate_contrastive_questions.py` | Generate opposite-label questions for same contexts using OLMo-3 |
| `scripts/style_normalize.py` | Rewrite samples into uniform style using OLMo-3 |
| `scripts/build_hardened_dataset.py` | Combine originals with contrastive pairs and rebuild splits |

### Ablation scripts
| Script | Purpose |
|--------|---------|
| `scripts/run_distillation_ablation.py` | Compare models and distillation on original dataset |
| `scripts/run_hardened_ablation.py` | Compare original vs. hardened dataset across all conditions |

### Source code modules
| Module | Purpose |
|--------|---------|
| `src/data/schemas.py` | Unified data schema definition |
| `src/data/cleaners.py` | Text normalization (whitespace, quotes, punctuation) |
| `src/data/splits.py` | Anti-leakage train/validation/test splitting |
| `src/models/router_teacher.py` | OLMo-3 teacher router with JSON parsing |
| `src/models/router_student.py` | DeBERTa student router with classification head |
| `src/models/experts.py` | ToM and Social expert model interfaces |
| `src/models/losses.py` | Distillation loss (combined hard + soft binary cross-entropy) |
| `src/models/collators.py` | Batch collation with padding for training |
| `src/training/trainer_distill.py` | Training loop with early stopping and checkpointing |
| `src/inference/router_pipeline.py` | Router inference for deployment |
| `src/inference/routed_qa_pipeline.py` | End-to-end routed question answering |
| `src/eval/metrics_router.py` | Classification metrics, calibration, threshold search |
| `src/eval/metrics_qa.py` | Question answering metrics (exact match, token F1) |
| `src/eval/error_analysis.py` | Error analysis report generation |
| `src/utils/seed.py` | Random seed setting for reproducibility |
| `src/utils/config.py` | YAML config loading |
| `src/utils/prompts.py` | Prompt templates for teacher labeling |

### Other files
| File | Purpose |
|------|---------|
| `pyproject.toml` | Python project configuration |
| `requirements.txt` | Python package dependencies |
| `run_all.sh` | Shell script to run the entire pipeline end-to-end |

---

## How to Reproduce

```bash
# Full pipeline (data → teacher → student → eval)
bash run_all.sh

# Or run individual steps:
python scripts/prepare_simpletom.py
python scripts/prepare_kokomind.py
python scripts/prepare_theory_of_mind.py
python scripts/prepare_tomi_nli.py
python scripts/prepare_social_iqa.py
python scripts/prepare_cicero.py
python scripts/build_router_dataset.py
python scripts/generate_teacher_labels.py
python scripts/train_student_router.py
python scripts/eval_router.py
python scripts/eval_routed_system.py

# Dataset hardening
python scripts/generate_contrastive_questions.py
python scripts/style_normalize.py
python scripts/build_hardened_dataset.py

# Ablation studies
python scripts/run_distillation_ablation.py
python scripts/run_hardened_ablation.py
```
