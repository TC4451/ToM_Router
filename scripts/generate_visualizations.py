"""Generate all dataset and result visualizations."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIG_DIR = Path("outputs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

ORIG_PATH = Path("data/processed/router_dataset.parquet")
HARD_PATH = Path("data/processed/router_dataset_hardened.parquet")
TEACHER_PATH = Path("outputs/teacher_labels/router_dataset_with_teacher.parquet")
TEACHER_CACHE = Path("outputs/teacher_labels/teacher_cache.jsonl")

sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = {"ToM": "#E74C3C", "Non-ToM": "#3498DB"}
PALETTE = [COLORS["Non-ToM"], COLORS["ToM"]]


def fig1_dataset_composition():
    """Bar chart: samples per source, colored by label."""
    df = pd.read_parquet(ORIG_PATH)
    ct = pd.crosstab(df["source_dataset"], df["requires_tom"])
    ct.columns = ["Non-ToM", "ToM"]
    ct = ct.reindex(ct.sum(axis=1).sort_values(ascending=True).index)

    fig, ax = plt.subplots(figsize=(10, 5))
    ct.plot.barh(stacked=True, ax=ax, color=PALETTE, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("")
    ax.set_title("Original Dataset: Samples per Source Dataset")
    ax.legend(title="Label", loc="lower right")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_dataset_composition.png", dpi=150)
    plt.close()
    print("  fig1_dataset_composition.png")


def fig2_hardened_composition():
    """Side-by-side: original vs hardened source-label distribution."""
    orig = pd.read_parquet(ORIG_PATH)
    hard = pd.read_parquet(HARD_PATH)

    # Collapse contrastive sources to base name for comparison
    hard["base_source"] = hard["source_dataset"].str.replace("_contrastive", "")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (df, title) in zip(axes, [(orig, "Original Dataset"), (hard, "Hardened Dataset")]):
        src_col = "source_dataset" if "base_source" not in df.columns else "base_source"
        if src_col == "base_source":
            ct = pd.crosstab(df["base_source"], df["requires_tom"])
        else:
            ct = pd.crosstab(df["source_dataset"], df["requires_tom"])
        ct.columns = ["Non-ToM", "ToM"]
        ct = ct.reindex(ct.sum(axis=1).sort_values(ascending=True).index)
        ct.plot.barh(stacked=True, ax=ax, color=PALETTE, edgecolor="white", linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel("Number of Samples")
        ax.set_ylabel("")
        ax.legend(title="Label", loc="lower right")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    plt.suptitle("Source Shortcut Elimination: Both Labels Now Appear in Every Source", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_hardened_composition.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig2_hardened_composition.png")


def fig3_shortcut_comparison():
    """Bar chart: baseline accuracies on original vs hardened."""
    data = {
        "Baseline": ["Majority\nclass", "Source-only\nclassifier", "Bag-of-words\nclassifier", "BERT-tiny\n(4M params)", "DeBERTa\n(184M params)"],
        "Original": [50.00, 99.75, 99.24, 99.24, 99.75],
        "Hardened": [50.00, 54.24, 92.54, 96.41, 99.17],
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    w = 0.35
    bars1 = ax.bar(x - w/2, df["Original"], w, label="Original Dataset", color="#95A5A6", edgecolor="white")
    bars2 = ax.bar(x + w/2, df["Hardened"], w, label="Hardened Dataset", color="#2ECC71", edgecolor="white")

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Shortcut Baselines: Original vs. Hardened Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Baseline"])
    ax.legend()
    ax.set_ylim(40, 105)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random chance")

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_shortcut_comparison.png", dpi=150)
    plt.close()
    print("  fig3_shortcut_comparison.png")


def fig4_teacher_soft_labels():
    """Histogram of teacher soft label distribution."""
    with open(TEACHER_CACHE) as f:
        records = [json.loads(l) for l in f]
    probs = [r["teacher_prob_tom"] for r in records]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probs, bins=50, color="#8E44AD", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Teacher P(requires ToM)")
    ax.set_ylabel("Number of Samples")
    ax.set_title("OLMo-3 Teacher Soft Label Distribution")
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.7, label="Decision boundary")
    ax.legend()

    # Annotate regions
    ax.annotate(f"Confident\nNon-ToM\n({sum(1 for p in probs if p < 0.1):,})",
                xy=(0.05, ax.get_ylim()[1]*0.7), fontsize=10, color=COLORS["Non-ToM"], fontweight="bold")
    ax.annotate(f"Boundary\ncases\n({sum(1 for p in probs if 0.2 < p < 0.8):,})",
                xy=(0.4, ax.get_ylim()[1]*0.5), fontsize=10, color="gray", fontweight="bold")
    ax.annotate(f"Confident\nToM\n({sum(1 for p in probs if p > 0.9):,})",
                xy=(0.85, ax.get_ylim()[1]*0.7), fontsize=10, color=COLORS["ToM"], fontweight="bold")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_teacher_soft_labels.png", dpi=150)
    plt.close()
    print("  fig4_teacher_soft_labels.png")


def fig5_teacher_agreement():
    """Heatmap: teacher label vs hard label agreement."""
    df = pd.read_parquet(TEACHER_PATH)
    ct = pd.crosstab(
        df["requires_tom"].map({0: "Hard: Non-ToM", 1: "Hard: ToM"}),
        df["teacher_label"].map({0: "Teacher: Non-ToM", 1: "Teacher: ToM"}),
    )

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                linewidths=1, linecolor="white", cbar_kws={"label": "Count"})
    ax.set_title("Teacher vs. Ground-Truth Label Agreement")
    ax.set_ylabel("Ground-Truth Label")
    ax.set_xlabel("Teacher Label")

    total = len(df)
    agree = ((df["requires_tom"] == df["teacher_label"]).sum())
    ax.text(0.5, -0.15, f"Agreement: {agree}/{total} ({agree/total:.1%})",
            transform=ax.transAxes, ha="center", fontsize=11, fontstyle="italic")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_teacher_agreement.png", dpi=150)
    plt.close()
    print("  fig5_teacher_agreement.png")


def fig6_text_length_distributions():
    """Overlapping histograms: context and question length by label."""
    orig = pd.read_parquet(ORIG_PATH)
    hard = pd.read_parquet(HARD_PATH)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for col_idx, (df, title_prefix) in enumerate([(orig, "Original"), (hard, "Hardened")]):
        for row_idx, field in enumerate(["context", "question"]):
            ax = axes[row_idx][col_idx]
            for label, color, name in [(0, COLORS["Non-ToM"], "Non-ToM"), (1, COLORS["ToM"], "ToM")]:
                lengths = df[df["requires_tom"] == label][field].str.len()
                ax.hist(lengths, bins=40, alpha=0.6, color=color, label=name, edgecolor="white")
            ax.set_xlabel(f"{field.capitalize()} Length (characters)")
            ax.set_ylabel("Count")
            ax.set_title(f"{title_prefix}: {field.capitalize()} Length")
            ax.legend()
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

    plt.suptitle("Text Length Distributions by Label", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_text_length_distributions.png", dpi=150)
    plt.close()
    print("  fig6_text_length_distributions.png")


def fig7_subtype_distribution():
    """Pie/bar chart of question subtypes."""
    df = pd.read_parquet(ORIG_PATH)
    subtypes = df["subtype"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("Set2", len(subtypes))
    subtypes.plot.barh(ax=ax, color=colors, edgecolor="white")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("")
    ax.set_title("Question Subtype Distribution")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig7_subtype_distribution.png", dpi=150)
    plt.close()
    print("  fig7_subtype_distribution.png")


def fig8_distillation_gain():
    """Grouped bar: hard-only vs distilled across models and datasets."""
    data = {
        "config": [
            "BERT-tiny\nOriginal", "BERT-tiny\nHardened",
            "DeBERTa\nOriginal", "DeBERTa\nHardened"
        ],
        "Hard only": [99.24, 96.41, 99.75, 99.17],
        "Distilled": [99.37, 96.22, 99.62, 99.54],
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    w = 0.3
    bars1 = ax.bar(x - w/2, df["Hard only"], w, label="Hard labels only", color="#E67E22", edgecolor="white")
    bars2 = ax.bar(x + w/2, df["Distilled"], w, label="Distilled (hard + soft)", color="#9B59B6", edgecolor="white")

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Knowledge Distillation: Hard Labels Only vs. Distilled")
    ax.set_xticks(x)
    ax.set_xticklabels(df["config"])
    ax.legend(loc="lower left")
    ax.set_ylim(95, 100.5)

    # Add value labels and delta
    for i, (h, d) in enumerate(zip(df["Hard only"], df["Distilled"])):
        delta = d - h
        sign = "+" if delta >= 0 else ""
        color = "#27AE60" if delta > 0 else "#E74C3C"
        ax.text(i, max(h, d) + 0.15, f"{sign}{delta:.2f}%", ha="center", fontsize=10,
                fontweight="bold", color=color)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig8_distillation_gain.png", dpi=150)
    plt.close()
    print("  fig8_distillation_gain.png")


def fig9_error_reduction():
    """Visual showing error rate reduction from distillation on hardened dataset."""
    fig, ax = plt.subplots(figsize=(7, 4))

    models = ["BERT-tiny (4M)", "DeBERTa (184M)"]
    hard_errors = [100 - 96.41, 100 - 99.17]  # error rates
    dist_errors = [100 - 96.22, 100 - 99.54]

    x = np.arange(len(models))
    w = 0.3
    bars1 = ax.bar(x - w/2, hard_errors, w, label="Hard labels only", color="#E67E22", edgecolor="white")
    bars2 = ax.bar(x + w/2, dist_errors, w, label="Distilled", color="#9B59B6", edgecolor="white")

    ax.set_ylabel("Error Rate (%)")
    ax.set_title("Error Rate Reduction from Distillation (Hardened Dataset)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    # Add reduction percentages
    for i, (h, d) in enumerate(zip(hard_errors, dist_errors)):
        if h > 0:
            reduction = (h - d) / h * 100
            sign = "+" if reduction < 0 else ""
            ax.annotate(f"↓ {abs(reduction):.0f}% fewer\nerrors",
                       xy=(i + w/2, d), xytext=(i + w/2 + 0.25, d + 0.3),
                       fontsize=10, fontweight="bold", color="#27AE60" if reduction > 0 else "#E74C3C",
                       arrowprops=dict(arrowstyle="->", color="gray"))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig9_error_reduction.png", dpi=150)
    plt.close()
    print("  fig9_error_reduction.png")


def fig10_pipeline_overview():
    """Create a text-based pipeline diagram as a figure."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (0.5, 4.5, "6 Source\nDatasets", "#3498DB"),
        (2.5, 4.5, "Unified\nSchema\n(7,966)", "#2ECC71"),
        (4.5, 4.5, "Contrastive\nAugmentation\n(+2,922)", "#E74C3C"),
        (6.5, 4.5, "Hardened\nDataset\n(10,782)", "#9B59B6"),
        (2.5, 1.5, "OLMo-3\nTeacher\n(7B params)", "#E67E22"),
        (5.5, 1.5, "Soft Labels\nP(ToM)\n[0.0 – 1.0]", "#F39C12"),
        (8.0, 3.0, "DeBERTa\nStudent\n(184M params)", "#1ABC9C"),
    ]

    for x, y, text, color in boxes:
        rect = mpatches.FancyBboxPatch((x-0.7, y-0.6), 1.4, 1.2,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=9, fontweight="bold")

    # Arrows
    arrows = [
        (1.3, 4.5, 1.7, 4.5), (3.3, 4.5, 3.7, 4.5), (5.3, 4.5, 5.7, 4.5),
        (3.3, 1.5, 4.7, 1.5),  # teacher -> soft labels
        (7.2, 4.5, 7.5, 3.6),  # hardened -> student
        (6.3, 1.5, 7.5, 2.4),  # soft labels -> student
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=2))

    ax.set_title("System Pipeline Overview", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig10_pipeline_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig10_pipeline_overview.png")


def fig11_contrastive_explanation():
    """Visual showing contrastive pair concept."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Context box (shared)
    rect = mpatches.FancyBboxPatch((3, 3.2), 6, 1.2,
                                    boxstyle="round,pad=0.15",
                                    facecolor="#ECF0F1", edgecolor="#7F8C8D", linewidth=2)
    ax.add_patch(rect)
    ax.text(6, 3.8, "Same Story Context", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#2C3E50")
    ax.text(6, 3.35, '"Isla moved the cucumber. Chloe exited the porch."',
            ha="center", va="center", fontsize=9, fontstyle="italic", color="#7F8C8D")

    # ToM question
    rect_tom = mpatches.FancyBboxPatch((0.5, 0.5), 4.5, 1.8,
                                        boxstyle="round,pad=0.15",
                                        facecolor=COLORS["ToM"], alpha=0.15,
                                        edgecolor=COLORS["ToM"], linewidth=2)
    ax.add_patch(rect_tom)
    ax.text(2.75, 1.9, "ToM Question (label = 1)", ha="center", fontsize=10,
            fontweight="bold", color=COLORS["ToM"])
    ax.text(2.75, 1.2, '"Where does Chloe think\nthe cucumber is?"', ha="center",
            fontsize=9, fontstyle="italic")
    ax.text(2.75, 0.65, "Requires modeling Chloe's false belief",
            ha="center", fontsize=8, color="#7F8C8D")

    # Non-ToM question
    rect_nontom = mpatches.FancyBboxPatch((7.0, 0.5), 4.5, 1.8,
                                           boxstyle="round,pad=0.15",
                                           facecolor=COLORS["Non-ToM"], alpha=0.15,
                                           edgecolor=COLORS["Non-ToM"], linewidth=2)
    ax.add_patch(rect_nontom)
    ax.text(9.25, 1.9, "Non-ToM Question (label = 0)", ha="center", fontsize=10,
            fontweight="bold", color=COLORS["Non-ToM"])
    ax.text(9.25, 1.2, '"Where is the cucumber\nafter Isla moved it?"', ha="center",
            fontsize=9, fontstyle="italic")
    ax.text(9.25, 0.65, "Factual — directly stated in the story",
            ha="center", fontsize=8, color="#7F8C8D")

    # Arrows from context to questions
    ax.annotate("", xy=(2.75, 2.3), xytext=(4.5, 3.2),
                arrowprops=dict(arrowstyle="->", color="#7F8C8D", lw=1.5))
    ax.annotate("", xy=(9.25, 2.3), xytext=(7.5, 3.2),
                arrowprops=dict(arrowstyle="->", color="#7F8C8D", lw=1.5))

    ax.set_title("Contrastive Augmentation: Same Story, Opposite Labels",
                 fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig11_contrastive_explanation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig11_contrastive_explanation.png")


def main():
    print("Generating visualizations...")
    fig1_dataset_composition()
    fig2_hardened_composition()
    fig3_shortcut_comparison()
    fig4_teacher_soft_labels()
    fig5_teacher_agreement()
    fig6_text_length_distributions()
    fig7_subtype_distribution()
    fig8_distillation_gain()
    fig9_error_reduction()
    fig10_pipeline_overview()
    fig11_contrastive_explanation()
    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
