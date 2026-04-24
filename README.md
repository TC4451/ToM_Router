# Theory-of-Mind Router with Knowledge Distillation

A system that decides whether a social reasoning question requires **Theory of Mind (ToM)** — reasoning about hidden beliefs, false beliefs, intentions, or knowledge states — and routes it to the appropriate expert model.

The router is trained using **knowledge distillation**: a large teacher model (OLMo-3-7B) labels training data with soft probabilities, and a smaller student model (DeBERTa-v3-base) is trained on both ground-truth labels and the teacher's soft predictions.

## Key Results

### Routing Accuracy (hardened dataset, source shortcuts removed)

| Model | Parameters | Distilled | Accuracy | F1 | Error Reduction |
|-------|-----------|-----------|----------|-----|-----------------|
| BERT-tiny | 4M | No | 96.41% | 96.45% | — |
| **BERT-tiny** | **4M** | **Yes** | **97.24%** | **97.28%** | **23.1%** |
| DeBERTa | 184M | No | 99.17% | 99.17% | — |
| DeBERTa | 184M | Yes | 99.08% | 99.08% | — |

> Distillation helps weak models most: BERT-tiny gains +0.83% (23% fewer errors), while DeBERTa is already near-ceiling.

### Dataset Hardening Eliminated Source Shortcuts

| Baseline | Original Dataset | Hardened Dataset |
|----------|-----------------|-----------------|
| Source-only classifier | 99.75% | 54.24% (near random) |
| Bag-of-words classifier | 99.24% | 92.54% |

Contrastive augmentation — generating opposite-label questions for the same story contexts — broke the correlation between dataset source and label.

## How It Works

```
Input: (context, question)
         │
         ▼
   ┌─────────────┐
   │   Router     │  DeBERTa-v3-base classifier
   │  (Student)   │  P(requires_tom) → 0.0 to 1.0
   └──────┬──────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌────────┐ ┌────────┐
│  ToM   │ │Social  │
│ Expert │ │ Expert │  OLMo-3-7B with role-specific prompts
└────────┘ └────────┘
```

## Dataset

Built from 6 source datasets, merged into a unified schema:

| Dataset | Samples | Label | Description |
|---------|---------|-------|-------------|
| SimpleToM | 3,441 | ToM | False belief stories |
| theory_of_mind | 4,058 | ToM | Aggregated ToM benchmarks |
| tomi-nli | 17,982 | ToM | Sally-Anne style false belief tracking |
| KokoMind | 770 | Mixed | Social dialogues with ToM, emotion, norm questions |
| SocialIQA | 34,934 | Non-ToM | Social commonsense QA |
| CICERO | 22,731 | Non-ToM | Commonsense inference in dialogues |

**Final dataset:** 10,782 balanced samples (5,391 ToM / 5,391 Non-ToM) including 2,922 OLMo-3-generated contrastive pairs.

Processed datasets are included in `data/processed/`. Raw data can be re-downloaded by running the preparation scripts.

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA-capable GPU (tested on RTX 5090, 34GB VRAM).

### Run Full Pipeline

```bash
bash run_all.sh
```

Or run individual steps:

```bash
# 1. Download and prepare datasets
python scripts/prepare_simpletom.py
python scripts/prepare_kokomind.py
python scripts/prepare_theory_of_mind.py
python scripts/prepare_tomi_nli.py
python scripts/prepare_social_iqa.py
python scripts/prepare_cicero.py
python scripts/build_router_dataset.py

# 2. Generate teacher labels (OLMo-3, ~2 hours)
python scripts/generate_teacher_labels.py

# 3. Train student router (~3 minutes for DeBERTa)
python scripts/train_student_router.py

# 4. Evaluate
python scripts/eval_router.py
python scripts/eval_routed_system.py

# 5. Dataset hardening (optional, ~50 minutes)
python scripts/generate_contrastive_questions.py
python scripts/build_hardened_dataset.py

# 6. Ablation studies (optional)
python scripts/run_distillation_ablation.py
python scripts/run_hardened_ablation.py
```

## Project Structure

```
├── configs/                    # YAML configuration files
│   ├── data.yaml               # Dataset paths and split ratios
│   ├── teacher.yaml            # Teacher model settings
│   ├── router_student.yaml     # Student training hyperparameters
│   ├── experts.yaml            # Expert model settings
│   ├── eval_router.yaml        # Router evaluation settings
│   └── eval_routed.yaml        # Routed system evaluation settings
│
├── data/
│   ├── raw/                    # Downloaded source datasets (gitignored)
│   ├── interim/                # Per-source normalized datasets (gitignored)
│   └── processed/              # Final merged datasets (tracked)
│       ├── router_dataset.parquet          # Original dataset (7,966 samples)
│       └── router_dataset_hardened.parquet # Hardened dataset (10,782 samples)
│
├── scripts/
│   ├── prepare_*.py            # Dataset download and preprocessing (6 scripts)
│   ├── build_router_dataset.py # Merge, deduplicate, balance, split
│   ├── generate_teacher_labels.py      # OLMo-3 teacher labeling
│   ├── train_student_router.py         # Student distillation training
│   ├── train_experts.py                # Expert model configuration
│   ├── eval_router.py                  # Router evaluation with full metrics
│   ├── eval_routed_system.py           # End-to-end system comparison
│   ├── export_router.py               # Export for deployment
│   ├── generate_contrastive_questions.py  # Contrastive augmentation
│   ├── style_normalize.py             # Style transfer normalization
│   ├── build_hardened_dataset.py       # Build hardened dataset
│   ├── run_distillation_ablation.py    # Distillation ablation study
│   └── run_hardened_ablation.py        # Original vs hardened comparison
│
├── src/
│   ├── data/                   # Data schemas, cleaners, splitters
│   ├── models/                 # Router (teacher & student), experts, losses
│   ├── training/               # Distillation training loop
│   ├── inference/              # Router and routed QA pipelines
│   ├── eval/                   # Metrics, error analysis
│   └── utils/                  # Seeds, config loading, prompts
│
├── outputs/
│   ├── reports/                # Evaluation reports and ablation results
│   ├── teacher_labels/         # Teacher-labeled dataset
│   ├── contrastive/            # Generated contrastive question pairs
│   └── checkpoints/            # Model checkpoints (gitignored)
│
├── PROGRESS_LOG.md             # Detailed experiment log with all results
├── requirements.txt
├── pyproject.toml
├── run_all.sh                  # One-command full pipeline
└── README.md
```

## Models Used

| Role | Model | Parameters | Purpose |
|------|-------|-----------|---------|
| Teacher | `allenai/Olmo-3-7B-Instruct` | 7B (loaded in 4-bit) | Generate soft labels and contrastive questions |
| Student | `microsoft/deberta-v3-base` | 184M | Deployed router classifier |
| Weak student | `google/bert_uncased_L-2_H-128_A-2` | 4M | Ablation: test distillation on weak models |
| Experts | `allenai/Olmo-3-7B-Instruct` | 7B | ToM and Social expert answering (prompt-differentiated) |

## Knowledge Distillation

The student learns from two signals:
- **Hard labels** (70% weight): Ground-truth binary labels from the dataset
- **Soft labels** (30% weight): Teacher probability outputs (e.g., 0.85 instead of 1)

```
Loss = 0.7 × BCE(student_logits, hard_label) + 0.3 × BCE(σ(logits/T), teacher_prob)
```

Temperature `T = 1.5` softens the student's predictions during soft-label learning.

## Ablation Studies

See `PROGRESS_LOG.md` for full ablation results, including:

1. **Shortcut baselines** — Majority class, source-only, context-length-only, bag-of-words
2. **Distillation comparison** — Hard labels only vs. distilled, across 3 model sizes
3. **Dataset hardening** — Original vs. contrastive-augmented dataset
4. **Source shortcut analysis** — Proving contrastive augmentation eliminates dataset artifacts

## Citation

If you use this code or dataset, please cite the source datasets:

- **SimpleToM**: Gu et al., "SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs" (2024)
- **KokoMind**: CHATS Lab, KokoMind benchmark
- **SocialIQA**: Sap et al., "Social IQa: Commonsense Reasoning about Social Interactions" (2019)
- **CICERO**: Ghosal et al., "CICERO: A Dataset for Contextualized Commonsense Inference in Dialogues" (2022)
- **ToMi-NLI**: from tasksource
- **theory_of_mind**: aggregated from ToMBench and Hi-ToM

## License

MIT License. See `LICENSE` for details.

Note: The source datasets have their own licenses. Check each dataset's original repository for terms of use.
