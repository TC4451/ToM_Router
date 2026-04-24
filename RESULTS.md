# Can a Small Model Learn When a Question Requires Reading Someone's Mind?

## The Problem

When you read *"Mary doesn't know the cookies were replaced with dog treats — what will she do next?"*, you instantly know this question requires reasoning about Mary's **false belief**. She *thinks* the cookies are real. Your answer depends on modeling her hidden mental state — what cognitive scientists call **Theory of Mind**.

But when you read *"How did the crowd feel after the team scored?"*, no hidden mental state is involved. The answer comes from general social knowledge about emotions.

We built a system that automatically makes this distinction — a **router** that classifies whether a question requires Theory of Mind reasoning, then sends it to the right specialist model for answering.

### Contributions

1. **A shortcut-free ToM routing dataset** — 10,782 samples from 6 benchmarks, with contrastive augmentation that eliminates the source-style shortcut found in naive multi-source datasets (source-only classifier: 99.75% → 54.24%). Every sample includes soft probability labels from an OLMo-3-7B teacher. (*See [Dataset Card](data/processed/DATASET_CARD.md)*)

2. **A knowledge distillation pipeline** — An OLMo-3-7B teacher labels each sample with a ToM probability, and a 4M-parameter BERT-tiny student is distilled on both hard labels and teacher soft labels, achieving a 23% error reduction over hard-label-only training.

3. **A downstream adaptive routing agent** — A multi-turn dialogue agent that uses the trained router to decide per-turn which expert to call, reaching 84% routing accuracy on mixed conversations (vs. the 50% ceiling of fixed policies) at 27% lower cost.

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

## Curating the Dataset

No single dataset captures the full spectrum of social reasoning. We assembled ours from **six established benchmarks**, spanning both Theory of Mind and non-ToM social reasoning. Building this dataset — and then discovering and fixing its hidden flaws — turned out to be the most important part of the project.

### Step 1: Sourcing and Unifying Six Datasets

Each source dataset has a different format, schema, and focus. We normalized all of them into one unified representation: a `(context, question, answer, requires_tom)` tuple.

#### ToM-positive sources (questions that require reasoning about hidden mental states)

**SimpleToM** (3,441 samples) — Short stories where a character is unaware of a critical fact. Three question types test whether a model can apply that gap in knowledge:
> **Context:** *"The bag of potato chips has moldy chips in it. Mary picks up the bag in the supermarket and walks to the cashier."*
> **Question:** *"Is Mary likely to be aware that the bag of potato chips has moldy chips in it?"*

Mary doesn't know — you need to model her ignorance to answer correctly.

**ToMBench + Hi-ToM** (4,058 samples) — An aggregation of classic ToM test batteries: false belief tasks (600), faux pas recognition (560), strange stories (407), unexpected outcomes (300), hidden emotions (80), higher-order belief reasoning (1,200), and more. This is the broadest single collection of ToM evaluation items.

**ToMi-NLI** (17,982 samples) — Sally-Anne style belief tracking recast as natural language inference. Stories describe agents moving objects while other agents are absent, then test whether a belief statement follows:
> **Context:** *"Isla entered the porch. The cucumber is in the red bathtub. Isla moved the cucumber to the red bottle. Chloe exited the porch."*
> **Question:** *"Does this follow? Chloe will look for the cucumber in the red bathtub."*

#### Non-ToM sources (questions answerable without hidden mental state reasoning)

**SocialIQA** (34,934 samples) — Commonsense questions about social situations — motivations, emotional reactions, likely next actions — that can be answered from general social knowledge:
> **Context:** *"Casey was heading to the coffee shop before work. Casey bought Bailey a drink while there."*
> **Question:** *"How would Casey feel afterwards?"*

No hidden beliefs or false beliefs are involved. The answer follows from common sense about buying gifts.

**CICERO** (22,731 samples) — Commonsense inference grounded in dialogue — causes, prerequisites, motivations, and emotional reactions extracted from conversations:
> **Context:** *"A: I was wondering whether you could ship the tennis racket overseas to Taiwan. B: I'm sorry, we don't ship overseas."*
> **Question:** *"What is the prerequisite of the target statement?"*

**KokoMind** (770 samples) — The only dataset with **both labels from the same source**. Each social interaction dialogue has questions spanning six categories: Theory of Mind (395), Emotion Recognition (81), Social Norms (64), Social Relations (84), Counterfactual reasoning (71), and Social Advice (75). We mapped ToM → requires ToM, everything else → does not require ToM.

#### The unification process

Every sample, regardless of source, was converted into the same schema:

```json
{
  "sample_id": "simpletom_potato_chip_food_sev1_action",
  "source_dataset": "simpletom",
  "context": "The bag of potato chips has moldy chips in it...",
  "question": "What will Mary likely do next?",
  "answer": "pay for the chips",
  "requires_tom": 1,
  "subtype": "belief",
  "original_category": "behavior-qa"
}
```

Key normalization steps:
- **KokoMind**: The `text` field combines context and question in one string separated by `**Briefly answer these question**:`. We split these apart and removed the `Read the following context:` prefix.
- **ToMi-NLI**: The `premise` became the context; the `hypothesis` was wrapped into a question: *"Does this follow? {hypothesis}"*
- **SocialIQA**: The correct answer was resolved from the label index (`1`/`2`/`3` mapping to `answerA`/`B`/`C`).
- **CICERO**: Dialogue turns stored as lists were joined into readable text. Correct answers were resolved from index arrays.
- **All sources**: Unicode normalization, whitespace collapsing, quote standardization, and trailing artifact removal.

### Step 2: Merging and Balancing

From a combined pool of 83,916 samples (25,876 ToM / 58,040 non-ToM), we subsampled to **4,000 per class** using proportional sampling — each source dataset contributed proportionally to its original representation within each class. This preserves source diversity rather than letting the largest datasets dominate.

| Source | ToM samples | Non-ToM samples | Total |
|--------|------------|----------------|-------|
| tomi_nli | 2,769 | — | 2,769 |
| social_iqa | — | 2,398 | 2,398 |
| cicero | — | 1,560 | 1,560 |
| theory_of_mind | 623 | — | 623 |
| simpletom | 530 | — | 530 |
| kokomind | 61 | 25 | 86 |
| **Total** | **3,983** | **3,983** | **7,966** |

### Step 3: Anti-Leakage Splitting

A naive random split would allow the same story to appear in both training and test sets — the model could memorize stories rather than learn reasoning patterns. We prevented this with **context-hash grouping**: every sample's context was hashed, and all samples sharing the same context hash were assigned to the same split. The split was stratified by label to maintain balance.

| Split | ToM | Non-ToM | Total |
|-------|-----|---------|-------|
| Train | 3,187 | 3,187 | 6,374 |
| Validation | 400 | 400 | 800 |
| Test | 396 | 396 | 792 |

We verified **zero context overlap** between train, validation, and test sets.

### Step 4: Discovering the Shortcut

Before training any neural model, we ran the sanity checks recommended in our implementation plan — and found a critical flaw.

A logistic regression trained on **only the dataset name** as input (a single integer feature) achieved **99.75% accuracy**. The reason was clear from the label-source table above: ToM samples come exclusively from SimpleToM, theory_of_mind, and tomi_nli, while non-ToM samples come exclusively from social_iqa and cicero. Each dataset has its own writing style, vocabulary, and formatting. A model can learn to distinguish these styles trivially — without ever understanding Theory of Mind.

We also checked other potential shortcuts:

| Shortcut | Accuracy | Verdict |
|----------|----------|---------|
| Source dataset name only | 99.75% | **Fatal shortcut** |
| Bag-of-words (word frequencies) | 99.24% | **Strong shortcut** (vocabulary differs across sources) |
| Context length only | 52.78% | Not a shortcut |
| Question length only | — | Not a shortcut (ToM: avg 79 chars, non-ToM: avg 41 chars — correlated but not separable) |

### Step 5: Contrastive Augmentation

To break the source shortcut, we needed both labels to appear within the same writing style. We used **OLMo-3-7B-Instruct** to generate contrastive questions — for each story context, an opposite-label question about that same context.

**ToM story → generated non-ToM question:**

| | |
|---|---|
| **Context** | *Isla entered the porch. The cucumber is in the red bathtub. Isla moved the cucumber to the red bottle. Chloe exited the porch.* |
| **Original question (ToM)** | *Where does Chloe think the cucumber is?* |
| **Generated question (non-ToM)** | *Where was the cucumber after Isla moved it?* |

The original question requires modeling Chloe's false belief (she left before Isla moved the cucumber). The generated question is purely factual — the cucumber is in the red bottle, observable to anyone.

**Non-ToM story → generated ToM question:**

| | |
|---|---|
| **Context** | *"A: I was wondering whether you could ship the tennis racket overseas. B: I'm sorry, we don't ship overseas."* |
| **Original question (non-ToM)** | *What is the prerequisite of the target statement?* |
| **Generated question (ToM)** | *What does speaker A likely believe about the store's shipping policy before asking the question?* |

The original is a factual commonsense question. The generated question requires reasoning about A's (incorrect) belief that overseas shipping might be possible.

We generated contrastive questions for 3,000 samples (1,500 per class), achieving a **97.4% success rate** (2,922 valid pairs). The generation took approximately 50 minutes on a single GPU.

### Step 6: Style Normalization (Partial)

As a complementary measure, we used OLMo-3 to rewrite samples into a uniform third-person narrative style, removing source-specific formatting cues (e.g., KokoMind's parenthetical stage directions, ToMi's telegraphic agent-action lists). We completed 382 samples before prioritizing the contrastive approach — enough to demonstrate feasibility.

### Step 7: Building the Hardened Dataset

We combined the original 7,966 samples (with style normalization where available) with the 2,922 contrastive pairs, deduplicated, and re-split with the same anti-leakage strategy:

| Source | ToM | Non-ToM | Total |
|--------|-----|---------|-------|
| tomi_nli | 2,769 | — | 2,769 |
| social_iqa | — | 2,377 | 2,377 |
| cicero | — | 1,543 | 1,543 |
| **tomi_nli_contrastive** | — | **997** | 997 |
| **social_iqa_contrastive** | **885** | — | 885 |
| theory_of_mind | 623 | — | 623 |
| simpletom | 530 | — | 530 |
| **cicero_contrastive** | **512** | — | 512 |
| **theory_of_mind_contrastive** | — | **243** | 243 |
| **simpletom_contrastive** | — | **191** | 191 |
| kokomind | 61 | 24 | 85 |
| kokomind_contrastive | 11 | 16 | 27 |
| **Total** | **5,391** | **5,391** | **10,782** |

The critical difference: **every source dataset now contributes to both labels**. ToMi stories have both belief-tracking questions (ToM) and factual questions (non-ToM). SocialIQA scenarios have both commonsense questions (non-ToM) and generated belief-reasoning questions (ToM). A model can no longer distinguish the classes by recognizing the source.

| Split | ToM | Non-ToM | Total |
|-------|-----|---------|-------|
| Train | 4,312 | 4,312 | 8,624 |
| Validation | 536 | 536 | 1,072 |
| Test | 543 | 543 | 1,086 |

### The Result: Shortcuts Destroyed

| Baseline | Original Dataset | Hardened Dataset | Change |
|----------|-----------------|-----------------|--------|
| Source-only classifier | 99.75% | **54.24%** | **−45.5 pp** |
| Bag-of-words classifier | 99.24% | 92.54% | −6.7 pp |

The source shortcut is gone. The bag-of-words shortcut is weakened but persists — some vocabulary differences between ToM and non-ToM questions remain (e.g., words like "think", "believe", "aware" naturally appear more in ToM questions). Full style normalization of all samples would reduce this further.

---

## The Shortcut Problem

Our first trained router scored **99.75% accuracy** on the original test set. Impressive — until we ran the sanity checks.

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

### On the hardened dataset (shortcuts removed, real teacher labels on all samples)

| Model | Hard labels only | Distilled | Difference | Error Reduction |
|-------|-----------------|-----------|------------|-----------------|
| **BERT-tiny (4M parameters)** | **96.41%** | **97.24%** | **+0.83%** | **23.1%** |
| DeBERTa (184M parameters) | 99.17% | 99.08% | −0.09% | — |

On the harder dataset, distillation improved BERT-tiny by **+0.83 percentage points** — a **23.1% reduction in errors**. The tiny model with only 4 million parameters and 2 transformer layers learned to leverage the teacher's nuanced probability judgments to handle borderline cases it couldn't resolve from hard labels alone.

DeBERTa showed no gain from distillation (−0.09%, within noise). This makes sense: DeBERTa at 184 million parameters is powerful enough to nearly solve this task with hard labels alone (99.17%). It doesn't need the teacher's help. **Distillation's value is proportional to the gap between the student's capacity and the task's difficulty.**

This is the key finding: knowledge distillation is most valuable when deployed on **weaker, cheaper models** — exactly the deployment scenario where you want a small, fast router running at inference time.

---

## The Full Picture

Putting all results together across both datasets and all conditions:

| Dataset | Model | Distilled | Accuracy | F1 | AUROC |
|---------|-------|-----------|----------|-----|-------|
| Original | BERT-tiny (4M) | No | 99.24% | 99.24% | 0.9999 |
| Original | BERT-tiny (4M) | Yes | 99.37% | 99.37% | 0.9999 |
| Original | DeBERTa (184M) | No | 99.75% | 99.75% | 1.0000 |
| Original | DeBERTa (184M) | Yes | 99.62% | 99.62% | 1.0000 |
| | | | | | |
| Hardened | BERT-tiny (4M) | No | 96.41% | 96.45% | 0.9920 |
| **Hardened** | **BERT-tiny (4M)** | **Yes** | **97.24%** | **97.28%** | **0.9933** |
| Hardened | DeBERTa (184M) | No | 99.17% | 99.17% | 0.9998 |
| Hardened | DeBERTa (184M) | Yes | 99.08% | 99.08% | 0.9976 |

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

## Downstream Agent: Adaptive Routing in Multi-Turn Dialogue

To demonstrate practical value, we built a multi-turn dialogue agent that uses the router online — deciding on every turn whether the user's question needs deep ToM reasoning or surface social reasoning.

### Three Reasoning Policies

| Policy | Strategy | When to use |
|--------|----------|-------------|
| **Always-ToM** | Every turn uses the ToM expert (deep belief reasoning) | Maximum quality, maximum cost |
| **General-Social** | Every turn uses the Social expert (surface reasoning) | Minimum cost, misses ToM cases |
| **Adaptive Router** | Trained router decides per turn which expert to call | Best cost-quality tradeoff |

### Evaluation Setup

We constructed **50 multi-turn dialogue scenarios** (360 total turns) from the test set:
- 10 pure ToM conversations (6 turns each)
- 10 pure social conversations (6 turns each)
- 10 mixed conversations (8 turns, alternating ToM and non-ToM)
- 10 social-to-ToM transitions (8 turns, shift mid-conversation)
- 10 ToM-to-social transitions (8 turns, shift mid-conversation)

### Results

| Policy | Routing Accuracy | Tokens/Turn | Cost vs. Always-ToM | ToM Usage |
|--------|-----------------|-------------|---------------------|-----------|
| Always-ToM | 49.4% | 367 | 100% | 100% |
| General-Social | 50.6% | 140 | 38% | 0% |
| **Adaptive Router** | **76.1%** | **267** | **73%** | **56%** |

The fixed policies are stuck at ~50% — Always-ToM gets all ToM turns right but all social turns wrong, and vice versa. The adaptive router outperforms both by **26 percentage points** because it actually reads each question and decides per turn. It does this while using **only 56% ToM calls** instead of 100%, saving **27% in token cost**.

### Breakdown by Scenario Type

| Scenario | Always-ToM | General-Social | Adaptive Router |
|----------|-----------|----------------|-----------------|
| Pure ToM | 100% | 0% | **77%** |
| Pure Social | 0% | 100% | **70%** |
| **Mixed (alternating)** | **50%** | **50%** | **84%** |
| Social → ToM transition | 46% | 54% | **71%** |
| ToM → Social transition | 51% | 49% | **78%** |

The standout result is **mixed dialogues: 84% routing accuracy**. When ToM and non-ToM questions alternate every turn — the hardest setting for any fixed policy — the adaptive router correctly identifies 84% of turns. One mixed scenario achieved 100% perfect routing across all 8 turns.

Key observations:
- On **mixed dialogues**, the adaptive router's advantage is largest (+34 percentage points over the 50% ceiling that fixed policies face)
- On **transition scenarios**, the router correctly detects mid-conversation shifts from social to ToM reasoning (71%) and vice versa (78%)
- The router uses the expensive ToM expert only when needed (56% of turns vs 100%), reducing cost by 27%

### Cost-Quality Tradeoff

The core value proposition: **the adaptive router is the only policy that can handle mixed conversations**. Fixed policies are fundamentally limited to ~50% accuracy on any conversation with both ToM and non-ToM turns. The adaptive router breaks this ceiling at 76% overall, reaching 84% on mixed scenarios, while using 27% fewer tokens than always calling the expensive expert.

---

## What We Learned

1. **Dataset shortcuts are insidious.** Our first model scored 99.75% while learning nothing about Theory of Mind. Without shortcut baselines, we would have published misleading results. Always run a source-only classifier before claiming your model "understands" something.

2. **Contrastive augmentation works.** Generating opposite-label questions for the same story context destroyed the source shortcut (99.75% → 54.24%) and created a genuinely challenging benchmark.

3. **Distillation helps weak models most.** On the hardened dataset, distillation improved BERT-tiny (4M params) by 0.83 percentage points — a 23% error reduction. DeBERTa (184M params) showed no gain, because it's powerful enough to solve the task without help. This matches the deployment scenario: you want distillation for the small, fast router that runs at inference time.

4. **Teacher-student disagreement is informative, not a bug.** The OLMo-3 teacher agreed with ground-truth labels only 56.4% of the time — many social reasoning questions genuinely sit on the boundary between ToM and non-ToM. Training on both signals lets the student learn that nuance.

5. **Adaptive routing is the only viable strategy for mixed conversations.** Fixed policies (always-ToM or always-social) hit a 50% ceiling on any conversation mixing both types. The adaptive router reaches 84% on mixed dialogues and 76% overall, while using 27% fewer tokens — it calls the expensive ToM expert only when the question actually requires it.

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

# Dataset hardening
python scripts/generate_contrastive_questions.py
python scripts/build_hardened_dataset.py
python scripts/label_contrastive_teacher.py

# Downstream agent evaluation
python scripts/build_dialogue_scenarios.py
python scripts/eval_dialogue_agent.py

# Ablation studies
python scripts/run_distillation_ablation.py
python scripts/run_hardened_ablation.py
python scripts/run_extended_ablations.py

# Generate all figures
python scripts/generate_visualizations.py
```

See `PROGRESS_LOG.md` for a detailed walkthrough of every experiment.
