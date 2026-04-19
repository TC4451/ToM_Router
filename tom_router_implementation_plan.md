# Theory-of-Mind Router with Knowledge Distillation

## Full Implementation Plan

This document is written as an implementation-ready specification for a coding agent working inside an IDE. The goal is to build a system that decides when a query requires **Theory of Mind (ToM)** reasoning versus ordinary social reasoning, then routes the query to the appropriate expert model.

The initial dataset combination is:
- **SimpleToM** as the primary ToM-positive dataset
- **KokoMind** as the mixed social reasoning dataset, from which we derive both ToM and non-ToM samples

The router should be trained using **knowledge distillation**:
- A stronger **teacher router** makes the ToM vs Non-ToM decision
- A smaller **student router** is distilled from the teacher and used at inference time

---

## 1. Problem Definition

We want to learn a function:

```text
router(context, question) -> P(requires_tom)
```

where:
- `context` is a short social scenario, story, or dialogue
- `question` is a query about the scenario
- output is a probability that answering the question requires reasoning about hidden beliefs, intentions, false beliefs, perspective mismatch, or other latent mental states

This is **not** the same as answering the question itself. The router only decides whether the problem should go to:
- a **ToM expert**, or
- a **normal social reasoning expert**

At inference time:

```text
if P(requires_tom) >= threshold:
    send to ToM expert
else:
    send to Social expert
```

---

## 2. High-Level System

Build the project in six stages:

1. **Data ingestion**
   - Download or load SimpleToM and KokoMind
   - Convert both into one unified schema

2. **Label construction**
   - Label all SimpleToM samples as `requires_tom = 1`
   - Label KokoMind samples by category:
     - `ToM -> 1`
     - `emotion -> 0`
     - `norm -> 0`
     - `relation -> 0`

3. **Dataset curation**
   - Clean text
   - Standardize field names
   - Balance class distribution
   - Split into train/val/test with anti-leakage rules

4. **Teacher router**
   - Use a larger model or API-based LLM to classify whether ToM is needed
   - Generate soft labels and optionally rationales

5. **Student router via knowledge distillation**
   - Train a smaller classifier on hard labels + teacher soft targets
   - Student is the deployed router

6. **Expert models + routed inference**
   - Train or plug in a ToM expert and a non-ToM social expert
   - Build a routing pipeline and evaluate both router quality and downstream QA quality

---

## 3. Deliverables

The coding agent should produce:

- A reproducible Python project
- Scripts for dataset preprocessing and unification
- Teacher labeling pipeline
- Student distillation training pipeline
- Evaluation scripts
- Routed inference pipeline
- Clear configs and experiment tracking

At minimum, the repo should support:

```bash
python scripts/prepare_simpletom.py
python scripts/prepare_kokomind.py
python scripts/build_router_dataset.py
python scripts/generate_teacher_labels.py --config configs/teacher.yaml
python scripts/train_student_router.py --config configs/router_student.yaml
python scripts/train_experts.py --config configs/experts.yaml
python scripts/eval_router.py --config configs/eval_router.yaml
python scripts/eval_routed_system.py --config configs/eval_routed.yaml
```

---

## 4. Recommended Tech Stack

Use:
- **Python 3.10+**
- **PyTorch**
- **Transformers** from Hugging Face
- **Datasets** from Hugging Face if available
- **Pandas** and **pyarrow** for preprocessing
- **scikit-learn** for splitting and metrics
- **OmegaConf** or **Hydra** for configs
- **Weights & Biases** or **MLflow** for experiment logging

Suggested model stack:
- **Teacher router**:
  - Option A: a large instruction model accessed by API or local inference
  - Option B: a large encoder classifier such as DeBERTa-v3-large
- **Student router**:
  - DistilRoBERTa, DeBERTa-v3-base, or MiniLM-based classifier
- **Experts**:
  - Start simple: encoder-decoder or instruction model
  - If answering multiple-choice only, a classifier is sufficient
  - If open-ended answers are needed, use a generative model

---

## 5. Repository Structure

Use the following layout:

```text
project_root/
  README.md
  requirements.txt
  pyproject.toml
  configs/
    data.yaml
    teacher.yaml
    router_student.yaml
    experts.yaml
    eval_router.yaml
    eval_routed.yaml
  data/
    raw/
      simpletom/
      kokomind/
    interim/
    processed/
  scripts/
    prepare_simpletom.py
    prepare_kokomind.py
    build_router_dataset.py
    generate_teacher_labels.py
    train_student_router.py
    train_experts.py
    eval_router.py
    eval_routed_system.py
    export_router.py
  src/
    data/
      schemas.py
      loaders.py
      cleaners.py
      builders.py
      splits.py
    models/
      router_teacher.py
      router_student.py
      experts.py
      losses.py
      collators.py
    training/
      trainer_router.py
      trainer_distill.py
      trainer_experts.py
    inference/
      router_pipeline.py
      routed_qa_pipeline.py
    eval/
      metrics_router.py
      metrics_qa.py
      error_analysis.py
    utils/
      io.py
      logging.py
      prompts.py
      seed.py
      config.py
  notebooks/
    eda_router_dataset.ipynb
    router_error_analysis.ipynb
  outputs/
    teacher_labels/
    checkpoints/
    reports/
```

---

## 6. Unified Data Schema

All samples for the router should be converted into one schema.

Use a row-based JSONL or Parquet format with this schema:

```json
{
  "sample_id": "string",
  "source_dataset": "simpletom|kokomind",
  "context": "string",
  "question": "string",
  "answer": "string or null",
  "requires_tom": 0,
  "subtype": "belief|emotion|norm|relation|other",
  "original_category": "string or null",
  "split": "train|val|test",
  "teacher_prob_tom": 0.0,
  "teacher_label": 0,
  "teacher_rationale": "string or null",
  "metadata": {
    "raw_id": "string",
    "question_type": "string or null"
  }
}
```

Mandatory fields for the router:
- `sample_id`
- `source_dataset`
- `context`
- `question`
- `requires_tom`
- `subtype`

`answer` is not required for router-only training but should be preserved when available for later expert training and routed evaluation.

---

## 7. Data Mapping Rules

### 7.1 SimpleToM mapping

Map every sample to:
- `requires_tom = 1`
- `subtype = belief` unless finer metadata is available

If SimpleToM has multiple question types or annotations, preserve them under `metadata.question_type`.

### 7.2 KokoMind mapping

Map categories as:

```text
ToM      -> requires_tom = 1, subtype = belief
emotion  -> requires_tom = 0, subtype = emotion
norm     -> requires_tom = 0, subtype = norm
relation -> requires_tom = 0, subtype = relation
```

If KokoMind contains multiple QA formats, normalize them into a common `(context, question, answer)` representation.

### 7.3 Text normalization

Implement a cleaning utility that:
- strips excessive whitespace
- standardizes quotes and punctuation
- removes duplicate spaces and line breaks
- trims leading instruction artifacts if present
- preserves semantics exactly

Do **not** paraphrase dataset text during preprocessing.

---

## 8. Data Curation and Split Strategy

This part is important because the router can otherwise learn dataset shortcuts instead of reasoning requirements.

### 8.1 Primary risk
If all SimpleToM examples are positive and most non-ToM examples come only from KokoMind, the router may cheat by identifying source-specific style rather than true ToM need.

### 8.2 Mitigations

The coding agent should implement all of the following:

#### A. Balance labels
Create near-balanced train/val/test splits:
- target 50% ToM
- target 50% Non-ToM

#### B. Preserve source diversity
Ensure both labels include at least some KokoMind samples.

Recommended positive label composition:
- SimpleToM positives
- KokoMind ToM positives

Recommended negative label composition:
- KokoMind emotion, norm, relation negatives

#### C. Add source-aware analysis
Always report metrics by source:
- overall
- SimpleToM subset
- KokoMind subset

#### D. Duplicate and leak control
Split by stable scenario ID if available.
If not available, compute a hash of normalized context and ensure near-duplicate contexts do not cross train/val/test.

#### E. Length control
Check whether ToM examples are systematically longer or shorter.
If there is a severe imbalance, either stratify by length bins or downsample to reduce length-based shortcuts.

### 8.3 Splits
Use:
- Train: 80%
- Validation: 10%
- Test: 10%

All random operations must be seeded.

---

## 9. Teacher Router Design

The teacher router is a stronger model that provides:
- a hard decision: ToM or Non-ToM
- a soft score: probability that ToM is required
- optionally a short rationale

### 9.1 Teacher choices
Use one of these:

#### Preferred teacher option
A strong instruction-following LLM with a carefully controlled classification prompt.

#### Alternative teacher option
A large encoder classifier trained first on the hard labels, then used to generate soft probabilities over the whole dataset.

The implementation should support both.

### 9.2 Teacher prompt for LLM-based labeling
Use a prompt that is narrow and operational, not philosophical.

Example prompt:

```text
You are labeling social reasoning tasks.

Given a context and a question, decide whether answering the question requires Theory of Mind reasoning.

Definition:
Theory of Mind is required when the answer depends on reasoning about a person's hidden beliefs, false beliefs, intentions, knowledge state, perspective, or information that is not directly observable in the scene.

Non-Theory-of-Mind means the question can be answered from observable facts, explicit emotional content, social norms, or relationship knowledge without inferring hidden mental states.

Return JSON with fields:
- requires_tom: 0 or 1
- prob_tom: float between 0 and 1
- rationale: one short sentence

Context: {context}
Question: {question}
```

### 9.3 Teacher label generation protocol
For each sample:
1. Run teacher prompt or teacher model
2. Store `teacher_label`
3. Store `teacher_prob_tom`
4. Store `teacher_rationale` if available

### 9.4 Teacher quality control
Create a manually reviewed validation subset of around 200 examples and inspect:
- cases where teacher disagrees with dataset hard label
- ambiguous boundary cases
- obvious teacher mistakes

Do not blindly trust teacher outputs. Hard labels should remain in the dataset, and the student should use both hard labels and teacher soft labels.

---

## 10. Student Router via Knowledge Distillation

The deployed router should be a compact classifier.

### 10.1 Student input format
Use concatenated text:

```text
[CONTEXT] {context} [QUESTION] {question}
```

For encoder models, standard single-sequence tokenization is sufficient.

### 10.2 Student model
Recommended starting point:
- `microsoft/deberta-v3-base`

Lighter fallback:
- `distilroberta-base`
- MiniLM classifier

Architecture:
- pretrained encoder
- pooling token representation
- linear classification head outputting one logit

### 10.3 Distillation targets
The student should learn from:
1. **hard labels** from dataset mapping
2. **soft labels** from teacher probabilities

### 10.4 Distillation loss
Use a weighted sum:

```text
L_total = alpha * L_hard + beta * L_soft
```

Where:
- `L_hard` = binary cross entropy with hard label `requires_tom`
- `L_soft` = distillation loss from teacher probability

For binary distillation, implement:

```text
teacher_prob = p_t
student_logit = z_s
student_prob = sigmoid(z_s / T)

L_soft = BCE(student_prob, p_t)
```

Where:
- `T` is temperature, usually 1 to 2 for binary probability distillation
- Start with `T = 1.5`

Suggested starting weights:
- `alpha = 0.7`
- `beta = 0.3`

Expose these in config.

### 10.5 Optional hidden-state distillation
Not required in v1, but the codebase should make it possible later to distill intermediate embeddings from teacher to student if both are encoder models.

---

## 11. Training Procedure

### 11.1 Phase A: dataset preparation
Implement scripts:
- `prepare_simpletom.py`
- `prepare_kokomind.py`
- `build_router_dataset.py`

Outputs:
- normalized per-dataset files
- merged router dataset in Parquet or JSONL
- train/val/test splits

### 11.2 Phase B: teacher labeling
Implement `generate_teacher_labels.py` to:
- load merged dataset
- query teacher
- cache outputs incrementally
- resume if interrupted
- write labeled dataset with `teacher_prob_tom`

Use robust logging and retry behavior if calling an API.

### 11.3 Phase C: student training
Implement `train_student_router.py`:
- load merged dataset with teacher fields
- tokenize input
- train student with hard + soft loss
- early stopping on validation macro-F1 and AUROC
- save best checkpoint

### 11.4 Phase D: calibration
After student training, fit probability calibration on validation set if needed.

Recommended:
- Platt scaling or isotonic regression

Export:
- raw model checkpoint
- calibrated threshold
- calibration object if separate

---

## 12. Metrics for the Router

The evaluation script must report at least:

### Main metrics
- Accuracy
- Precision
- Recall
- F1
- Macro-F1
- AUROC
- AUPRC

### Calibration metrics
- Expected Calibration Error
- Brier score

### Slice metrics
Report metrics by:
- source dataset
- subtype
- text length bin
- label

### Error categories
The script should summarize:
- false positives: predicted ToM when not needed
- false negatives: missed ToM cases

False negatives are especially important because they represent failure to engage latent-state reasoning when needed.

---

## 13. Threshold Selection

The router outputs a probability, so thresholding matters.

Do not hardcode `0.5` without analysis.

Implement threshold search on validation data for multiple goals:
- maximize F1
- maximize recall for ToM class
- maximize downstream routed QA accuracy

Save at least three candidate thresholds and evaluate them all.

Recommended defaults:
- `threshold_f1_best`
- `threshold_high_recall_tom`
- `threshold_downstream_best`

---

## 14. Experts

The first version does not need highly optimized expert models, but the project should support them.

### 14.1 ToM expert
Train or configure a model on:
- SimpleToM
- KokoMind ToM subset

This expert should specialize in:
- false belief
- hidden intention
- perspective mismatch
- latent knowledge state

### 14.2 Social expert
Train or configure a model on:
- KokoMind emotion subset
- KokoMind norm subset
- KokoMind relation subset

This expert should specialize in:
- social norms
- explicit emotions
- relationship reasoning
- observable social content

### 14.3 Expert interfaces
Implement a common inference interface:

```python
class BaseExpert:
    def predict(self, context: str, question: str) -> dict:
        ...
```

Return structure:

```json
{
  "answer": "string",
  "confidence": 0.0,
  "metadata": {}
}
```

---

## 15. Routed Inference Pipeline

Implement `src/inference/routed_qa_pipeline.py`.

Pseudo-code:

```python
router_out = router.predict_proba(context, question)
prob_tom = router_out["prob_tom"]

if prob_tom >= threshold:
    route = "tom"
    expert_out = tom_expert.predict(context, question)
else:
    route = "social"
    expert_out = social_expert.predict(context, question)

return {
    "route": route,
    "prob_tom": prob_tom,
    "answer": expert_out["answer"],
    "expert_confidence": expert_out.get("confidence")
}
```

Also support an optional fallback mode:

```text
if router uncertainty is high:
    either query both experts
    or default to ToM expert
```

Router uncertainty options:
- probability near 0.5
- high entropy
- low margin to threshold

---

## 16. Downstream Evaluation

You should evaluate not only router classification but also final task performance.

### 16.1 Systems to compare
Implement evaluation for:

1. **Single general expert**
   - no routing

2. **Oracle router + experts**
   - use ground truth ToM label to route
   - this is the upper bound

3. **Teacher router + experts**
   - optional diagnostic

4. **Student router + experts**
   - actual system

5. **Always ToM expert**
   - useful baseline

6. **Always Social expert**
   - useful baseline

### 16.2 Metrics
Depending on answer format:
- exact match
- accuracy
- multiple-choice accuracy
- token F1 if generative

Also compute:
- routing accuracy
- end-to-end answer accuracy
- answer accuracy conditional on route
- route confusion matrix

---

## 17. Data Analysis and Sanity Checks

Before training, the coding agent should generate a short report with:
- label counts by source
- subtype distribution
- average context length by label
- average question length by label
- train/val/test distribution
- source mix by split

Also run these sanity checks:

1. Can a model predict label from **source_dataset only**?
2. Can a model predict label from **context length only**?
3. Can a model predict label from **question keywords only**?

These baselines are important. If they are too strong, the dataset may contain shortcuts.

Implement at least these weak baselines:
- majority class
- logistic regression over bag-of-words
- source-only heuristic

---

## 18. Prompt-Based Teacher Agreement Analysis

Teacher and hard labels may disagree. That disagreement can be useful.

Implement an analysis table with these groups:
- hard=1, teacher=1
- hard=1, teacher=0
- hard=0, teacher=1
- hard=0, teacher=0

Randomly sample examples from disagreement buckets for manual inspection.

Possible causes:
- dataset annotation ambiguity
- prompt misinterpretation
- edge cases where emotion and ToM overlap
- latent intention questions mislabeled as non-ToM

Preserve disagreement info in the final dataset for future experiments.

---

## 19. Hard Negative Mining

This project will benefit from non-ToM examples that look superficially similar to ToM questions.

Implement a future-ready hard-negative mining hook:

- embed all samples with a sentence encoder
- for each ToM-positive sample, retrieve semantically similar non-ToM samples
- prioritize these in training batches

This is not required in the first version, but the code should be structured to allow this extension.

---

## 20. Contrastive Extension (Optional but Recommended)

After the first working version, add a contrastive auxiliary loss.

Goal:
- same or similar social contexts
- separate ToM-required questions from non-ToM questions in embedding space

Basic formulation:
- use student encoder embedding
- positive pair: two ToM samples or two Non-ToM samples with similar context
- hard negative pair: similar context, opposite label

This is a likely quality improvement but should remain optional in v1.

---

## 21. Config Design

All key hyperparameters must live in config files.

### Example `configs/router_student.yaml`

```yaml
seed: 42
model_name: microsoft/deberta-v3-base
max_length: 512
batch_size: 16
lr: 2e-5
weight_decay: 0.01
epochs: 5
warmup_ratio: 0.1
scheduler: cosine
alpha_hard: 0.7
beta_soft: 0.3
distill_temperature: 1.5
threshold_mode: f1_best
use_calibration: true
output_dir: outputs/checkpoints/router_student
```

### Example `configs/teacher.yaml`

```yaml
teacher_type: llm_api
provider: openai_or_other
model_name: strong_teacher_model
max_retries: 5
save_every: 100
resume: true
prompt_template: tom_router_v1
output_path: outputs/teacher_labels/router_teacher_labels.parquet
```

---

## 22. Script-by-Script Requirements

### `scripts/prepare_simpletom.py`
Responsibilities:
- load raw SimpleToM data
- normalize columns
- create unified rows
- save cleaned dataset to `data/interim/simpletom.parquet`

### `scripts/prepare_kokomind.py`
Responsibilities:
- load raw KokoMind data
- map categories to labels
- normalize fields
- save cleaned dataset to `data/interim/kokomind.parquet`

### `scripts/build_router_dataset.py`
Responsibilities:
- merge interim files
- apply balancing rules
- deduplicate
- split train/val/test
- save processed dataset to `data/processed/router_dataset.parquet`

### `scripts/generate_teacher_labels.py`
Responsibilities:
- load processed dataset
- query teacher model
- cache results after every N examples
- merge teacher outputs back into dataset

### `scripts/train_student_router.py`
Responsibilities:
- tokenize dataset
- train with hard + soft distillation loss
- evaluate on validation split
- save best model and metrics

### `scripts/eval_router.py`
Responsibilities:
- evaluate saved student router on test split
- generate plots and confusion matrices
- save detailed per-example predictions

### `scripts/train_experts.py`
Responsibilities:
- train or configure ToM and social experts
- save checkpoints and metadata

### `scripts/eval_routed_system.py`
Responsibilities:
- run routed inference on test data
- compare routed system to baselines
- save route decisions and answer outcomes

---

## 23. Training Details

### Tokenization
Use truncation strategy:
- prioritize keeping the full question
- if needed, truncate context first

Recommended formatting:

```text
Context: {context}
Question: {question}
```

### Optimizer
Use AdamW.

### Mixed precision
Enable fp16 or bf16 if available.

### Early stopping
Stop based on validation macro-F1 or AUROC after patience of 2 epochs.

### Checkpointing
Save:
- best by macro-F1
- best by AUROC
- last checkpoint

### Random seeds
Set seeds in Python, NumPy, and PyTorch.

---

## 24. Probability Calibration

Since the router controls system behavior, calibration matters.

Implement:
- validation-time calibration fitting
- calibrated inference option

Compare raw vs calibrated scores using:
- Brier score
- reliability diagrams
- threshold stability for routed QA

---

## 25. Error Analysis Requirements

The coding agent should generate a markdown or HTML report after evaluation with:
- top false negatives
- top false positives
- examples near decision boundary
- examples where teacher and student disagree
- examples where correct route still leads to wrong answer

For each example, include:
- source dataset
- subtype
- context
- question
- hard label
- teacher prob
- student prob
- predicted route
- answer correctness if available

---

## 26. Minimal Viable Version

If time is limited, implement this smallest correct version first:

1. Normalize SimpleToM and KokoMind
2. Build merged binary ToM dataset
3. Train large teacher classifier or use strong LLM teacher
4. Distill to DeBERTa-base student
5. Evaluate router classification only
6. Add simple routed inference using placeholder experts

Placeholder experts can initially be simple classifiers or even the same LLM with two different prompting styles.

---

## 27. Version 2 Improvements

After v1 works, implement these in order:

1. Better teacher prompt and label auditing
2. Better hard-negative sampling
3. Contrastive auxiliary loss
4. Calibrated uncertainty fallback
5. Expert specialization improvements
6. Multi-task learning with answer prediction and routing jointly

---

## 28. Suggested Development Milestones

### Milestone 1: data foundation
Success criteria:
- both datasets load and normalize correctly
- merged dataset saved with correct labels
- train/val/test split and report generated

### Milestone 2: teacher labeling
Success criteria:
- teacher labels generated for all samples
- disagreement report produced

### Milestone 3: student router
Success criteria:
- distilled student trains successfully
- metrics saved and reproducible

### Milestone 4: routed experts
Success criteria:
- end-to-end inference pipeline works
- routed system benchmark produced

### Milestone 5: analysis package
Success criteria:
- error report, calibration report, and route analysis report generated

---

## 29. Risks and Failure Modes

### Risk 1: dataset shortcut learning
Mitigation:
- source-aware metrics
- simple baselines
- balanced composition

### Risk 2: ambiguous boundary between emotion and ToM
Mitigation:
- preserve subtypes
- inspect disagreement cases
- possibly introduce a third label later such as `mixed`

### Risk 3: teacher noise
Mitigation:
- combine hard and soft labels
- do manual audit
- keep teacher weight modest initially

### Risk 4: poor calibration
Mitigation:
- fit calibration
- evaluate threshold robustness

### Risk 5: experts do not differ enough
Mitigation:
- ensure data specialization
- inspect per-route answer quality
- compare against single-expert baseline

---

## 30. Recommended Experimental Table Set

The evaluation should produce tables for a paper or report.

### Table A: Router classification
Columns:
- model
- hard labels only or distilled
- accuracy
- F1
- AUROC
- ECE

### Table B: Router by subset
Rows:
- overall
- SimpleToM
- KokoMind ToM
- KokoMind emotion
- KokoMind norm
- KokoMind relation

### Table C: Routed QA system
Rows:
- single expert
- always ToM expert
- always social expert
- oracle router
- student router

Columns:
- route accuracy
- QA accuracy
- ToM subset QA accuracy
- non-ToM subset QA accuracy

---

## 31. Pseudocode for Student Distillation Trainer

```python
for batch in train_loader:
    inputs = tokenizer(batch["context_question_text"], ...)
    hard_labels = batch["requires_tom"].float()
    teacher_probs = batch["teacher_prob_tom"].float()

    logits = student(**inputs).logits.squeeze(-1)

    loss_hard = bce_with_logits(logits, hard_labels)

    student_probs_temp = torch.sigmoid(logits / temperature)
    loss_soft = binary_cross_entropy(student_probs_temp, teacher_probs)

    loss = alpha_hard * loss_hard + beta_soft * loss_soft

    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

---

## 32. Example Task for a Coding Agent

You can pass the following instruction block directly into a coding agent:

```text
Build a Python project for a Theory-of-Mind router trained on SimpleToM and KokoMind.

Requirements:
1. Normalize both datasets into one unified schema with fields:
   sample_id, source_dataset, context, question, answer, requires_tom, subtype, original_category, split, teacher_prob_tom, teacher_label, teacher_rationale, metadata.
2. Map labels as follows:
   - all SimpleToM samples -> requires_tom=1
   - KokoMind category ToM -> 1
   - KokoMind categories emotion, norm, relation -> 0
3. Build train/val/test splits with deduplication and balanced label distribution.
4. Implement a teacher labeling pipeline that stores teacher hard label, probability, and rationale.
5. Implement a student router using DeBERTa-v3-base.
6. Train student with combined hard-label BCE and soft-label distillation BCE.
7. Implement evaluation with Accuracy, F1, AUROC, AUPRC, Brier score, ECE, confusion matrix, and metrics by dataset slice.
8. Implement a routed inference pipeline that chooses between a ToM expert and a social expert.
9. Keep code modular, config-driven, and reproducible.
10. Generate reports and save per-example predictions for error analysis.
```

---

## 33. Final Recommended Execution Order

The coding agent should follow this exact order:

1. Create repo skeleton
2. Implement schemas and data loaders
3. Implement SimpleToM preprocessing
4. Implement KokoMind preprocessing
5. Implement merged router dataset builder and split logic
6. Run EDA and sanity reports
7. Implement teacher labeling pipeline
8. Generate teacher labels on train/val/test
9. Implement student router model and distillation trainer
10. Train and evaluate the student router
11. Add calibration and threshold search
12. Implement expert wrappers
13. Implement routed inference
14. Benchmark routed system against baselines
15. Generate error analysis reports

---

## 34. Definition of Done

This project is done when all of the following are true:

- Both datasets are normalized into one processed router dataset
- Teacher soft labels exist for all train/val/test examples
- Student router trains reproducibly and beats simple baselines
- Router metrics are reported overall and by slice
- Threshold and calibration are analyzed
- Routed inference works with two experts
- End-to-end routed system is compared to baseline systems
- Reports and saved predictions make it easy to inspect mistakes

---

## 35. Recommended First Checkpoint Goal

Aim first for this concrete milestone:

- merged dataset built correctly
- teacher soft labels generated
- student router achieves strong AUROC and reasonable F1
- a basic routed prototype runs end-to-end

That is enough to validate the research direction before investing in more sophisticated experts or contrastive extensions.

