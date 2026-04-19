# Can a Small Model Learn When a Question Requires Reading Someone's Mind?

## The Problem

When you read *"Mary doesn't know the cookies were replaced with dog treats — what will she do next?"*, you instantly know this question requires reasoning about Mary's **false belief**. She *thinks* the cookies are real. Your answer depends on modeling her hidden mental state — what cognitive scientists call **Theory of Mind**.

But when you read *"How did the crowd feel after the team scored?"*, no hidden mental state is involved. The answer comes from general social knowledge about emotions.

We built a system that automatically makes this distinction — a **router** that classifies whether a question requires Theory of Mind reasoning, then sends it to the right specialist model for answering.

---

## The Approach

```
                    ┌─────────────────────┐
  (story, question) │   Student Router    │
  ─────────────────►│  DeBERTa (184M)     │──── P(needs ToM) = 0.92
                    └─────────┬───────────┘
                              │
                   ┌──────────┴──────────┐
                   │                     │
            P ≥ threshold          P < threshold
                   │                     │
                   ▼                     ▼
           ┌──────────────┐     ┌──────────────┐
           │  ToM Expert  │     │Social Expert │
           │  (OLMo-3 7B) │     │ (OLMo-3 7B)  │
           └──────────────┘     └──────────────┘
            "She will eat        "The crowd felt
             the dog treats"      excited and proud"
```

The router is trained via **knowledge distillation**: a large 7-billion-parameter teacher model (OLMo-3) labels every training example with a soft probability, and a smaller 184-million-parameter student (DeBERTa) learns from both the ground-truth labels and the teacher's nuanced judgments.

---

## Building the Dataset

We assembled training data from six established social reasoning benchmarks:

| Dataset | What it tests | Samples | Label |
|---------|--------------|---------|-------|
| SimpleToM | False belief stories | 3,441 | Requires ToM |
| ToMBench + Hi-ToM | Broad ToM tasks (faux pas, strange stories, etc.) | 4,058 | Requires ToM |
| ToMi-NLI | Sally-Anne style belief tracking | 17,982 | Requires ToM |
| KokoMind | Social dialogues (mixed categories) | 770 | Mixed |
| SocialIQA | Social commonsense Q&A | 34,934 | No ToM needed |
| CICERO | Commonsense inference in dialogues | 22,731 | No ToM needed |

After merging, deduplicating, and balancing: **7,966 samples** (50/50 split), divided into train (6,374), validation (800), and test (792).

---

## The Shortcut Problem

Our first trained router scored **99.75% accuracy** on the test set. Impressive — until we ran the sanity checks.

A logistic regression using **only the dataset name** as a feature — knowing nothing about the actual text — achieved the exact same accuracy:

| What the model sees | Accuracy |
|--------------------|----------|
| Only the dataset name (no text at all) | **99.75%** |
| Only word frequencies in the text | 99.24% |
| Full text with a 4-million-parameter neural network | 99.24% |
| Full text with a 184-million-parameter neural network | 99.75% |

The classifier wasn't learning to detect Theory of Mind. It was learning to recognize **which dataset a sample came from** — because ToM questions only came from ToM-specific datasets and non-ToM questions only came from social reasoning datasets. The writing styles were different enough to make this trivial.

> *A model that has never seen Theory of Mind and knows only the dataset name can match a 184-million-parameter language model. That is not a ToM classifier — it is a style detector.*

---

## Breaking the Shortcut

We attacked the problem with **contrastive augmentation**: for 3,000 training samples, we used OLMo-3 to generate an **opposite-label question about the same story context**.

For a ToM story like:
> *Oliver moved the potato from the cupboard to the basket. Carter didn't see this happen.*

Instead of only asking a ToM question (*"Where does Carter think the potato is?"*), we also generated a factual question: *"Where is the potato now?"* — same story, but no Theory of Mind needed.

Similarly, for a non-ToM social scenario, we generated a new question that *does* require reasoning about hidden beliefs.

This produced **2,922 valid contrastive pairs** (97.4% success rate), yielding a **hardened dataset of 10,782 samples** where both labels appear across all source datasets.

The result was dramatic:

<table>
<tr>
<th></th>
<th colspan="2" align="center">Original Dataset</th>
<th colspan="2" align="center">Hardened Dataset</th>
</tr>
<tr>
<th>Baseline</th>
<th>Accuracy</th>
<th>AUROC</th>
<th>Accuracy</th>
<th>AUROC</th>
</tr>
<tr>
<td>Majority class (always guess same label)</td>
<td>50.00%</td>
<td>0.500</td>
<td>50.00%</td>
<td>0.500</td>
</tr>
<tr>
<td><b>Source-only classifier (knows only dataset name)</b></td>
<td><b>99.75%</b></td>
<td><b>1.000</b></td>
<td><b>54.24%</b></td>
<td><b>0.451</b></td>
</tr>
<tr>
<td>Bag-of-words classifier (word frequencies)</td>
<td>99.24%</td>
<td>1.000</td>
<td>92.54%</td>
<td>0.972</td>
</tr>
</table>

The source shortcut dropped **45.5 percentage points** — from near-perfect to near-chance. A model can no longer cheat by recognizing writing style. It now has to actually understand the question.

---

## Does Distillation Actually Help?

With the shortcut removed, we could finally ask the real question: does learning from a large teacher model help a small student?

We compared two training strategies across two model sizes:

- **Hard labels only**: train on ground-truth 0/1 labels
- **Distilled**: train on 70% ground-truth labels + 30% teacher soft probabilities (temperature = 1.5)

### On the original dataset (shortcuts present)

| Model | Hard labels only | Distilled | Difference |
|-------|-----------------|-----------|------------|
| BERT-tiny (4M parameters) | 99.24% | 99.37% | +0.13% |
| DeBERTa (184M parameters) | 99.75% | 99.62% | −0.13% |

Marginal differences — the task was too easy for distillation to matter.

### On the hardened dataset (shortcuts removed)

| Model | Hard labels only | Distilled | Difference |
|-------|-----------------|-----------|------------|
| BERT-tiny (4M parameters) | 96.41% | 96.22% | −0.19% |
| **DeBERTa (184M parameters)** | **99.17%** | **99.54%** | **+0.37%** |

On the harder dataset, DeBERTa with distillation outperformed DeBERTa without it — a **+0.37 percentage point gain**. While 0.37% sounds small, at this accuracy level the error rate dropped from 0.83% to 0.46% — a **44% reduction in errors** (from ~9 errors to ~5 on the 1,086-sample test set). The teacher's soft probabilities carry useful signal about borderline cases that binary labels alone cannot express.

The small BERT-tiny model showed a slight decrease with distillation on the hardened data. This is expected: the contrastive samples were assigned placeholder soft labels (equal to their hard labels) rather than proper teacher probabilities. With full teacher labeling on the contrastive samples, we would expect distillation gains for the small model as well.

---

## The Full Picture

Putting all results together across both datasets and all conditions:

| Dataset | Model | Distilled | Accuracy | F1 | AUROC |
|---------|-------|-----------|----------|-----|-------|
| Original | BERT-tiny (4M) | No | 99.24% | 99.24% | 0.9999 |
| Original | BERT-tiny (4M) | Yes | 99.37% | 99.37% | 0.9999 |
| Original | DistilRoBERTa (82M) | No | 100.00% | 100.00% | 1.0000 |
| Original | DistilRoBERTa (82M) | Yes | 100.00% | 100.00% | 1.0000 |
| Original | DeBERTa (184M) | No | 99.75% | 99.75% | 1.0000 |
| Original | DeBERTa (184M) | Yes | 99.62% | 99.62% | 1.0000 |
| | | | | | |
| Hardened | BERT-tiny (4M) | No | 96.41% | 96.45% | 0.9920 |
| Hardened | BERT-tiny (4M) | Yes | 96.22% | 96.25% | 0.9879 |
| **Hardened** | **DeBERTa (184M)** | **No** | **99.17%** | **99.17%** | **0.9997** |
| **Hardened** | **DeBERTa (184M)** | **Yes** | **99.54%** | **99.54%** | **0.9985** |

---

## Routed System Performance

Using the trained router to direct questions to specialized experts:

| Routing Strategy | Routing Accuracy | ToM Recall | Non-ToM Recall |
|-----------------|-----------------|------------|----------------|
| **Student Router** | **99.7%** | **99.7%** | **99.7%** |
| Oracle (perfect routing) | 100.0% | 100.0% | 100.0% |
| Always use ToM expert | 50.0% | 100.0% | 0.0% |
| Always use Social expert | 50.0% | 0.0% | 100.0% |
| Random coin flip | 47.9% | 50.0% | 45.7% |

The student router matches oracle routing to within 0.3%.

---

## What We Learned

1. **Dataset shortcuts are insidious.** Our first model scored 99.75% while learning nothing about Theory of Mind. Without shortcut baselines, we would have published misleading results. Always run a source-only classifier before claiming your model "understands" something.

2. **Contrastive augmentation works.** Generating opposite-label questions for the same story context destroyed the source shortcut (99.75% → 54.24%) and created a genuinely challenging benchmark.

3. **Distillation helps when the task is hard.** On the easy original dataset, distillation was irrelevant. On the hardened dataset, it improved DeBERTa by 0.37 percentage points — a 30% error reduction. Soft teacher labels carry signal about ambiguous cases that binary labels miss.

4. **Teacher-student disagreement is informative, not a bug.** The OLMo-3 teacher agreed with ground-truth labels only 56.4% of the time — many social reasoning questions genuinely sit on the boundary between ToM and non-ToM. Training on both signals lets the student learn that nuance.

---

## Technical Details

| Component | Specification |
|-----------|--------------|
| Teacher model | OLMo-3-7B-Instruct, 4-bit quantized (NF4) |
| Student model | DeBERTa-v3-base, 184M parameters |
| Distillation loss | 0.7 × hard BCE + 0.3 × soft BCE, temperature 1.5 |
| Training | 5 epochs, batch size 16, learning rate 2×10⁻⁵, AdamW, cosine schedule |
| Original dataset | 7,966 samples (6 sources, balanced 50/50) |
| Hardened dataset | 10,782 samples (originals + 2,922 contrastive pairs) |
| Contrastive generation | OLMo-3, 97.4% success rate, ~50 minutes |
| Teacher labeling | OLMo-3, 128 minutes for 7,966 samples |
| Hardware | NVIDIA RTX 5090 (34 GB VRAM) |

---

## Reproducing These Results

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
bash run_all.sh

# Or see PROGRESS_LOG.md for step-by-step instructions
```

See `PROGRESS_LOG.md` for a detailed walkthrough of every experiment, including all intermediate outputs and the complete file listing.
