"""Train the student router via knowledge distillation.

Loads the teacher-labeled dataset, trains a DeBERTa-v3-base student
using combined hard + soft label distillation loss.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.seed import set_seed
from src.utils.config import load_config
from src.models.router_student import StudentRouter, get_tokenizer
from src.models.collators import RouterCollator
from src.training.trainer_distill import RouterDataset, DistillationTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/router_student.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    # Load dataset
    dataset_path = Path("outputs/teacher_labels/router_dataset_with_teacher.parquet")
    if not dataset_path.exists():
        # Fall back to base dataset (without teacher labels, use hard labels as soft)
        dataset_path = Path("data/processed/router_dataset.parquet")
        print(f"WARNING: No teacher labels found, using hard labels as soft targets")

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_parquet(dataset_path)

    # If no teacher labels, use hard labels as soft targets
    if "teacher_prob_tom" not in df.columns:
        df["teacher_prob_tom"] = df["requires_tom"].astype(float)

    # Split into train/val/test
    train_records = df[df["split"] == "train"].to_dict("records")
    val_records = df[df["split"] == "val"].to_dict("records")
    test_records = df[df["split"] == "test"].to_dict("records")

    print(f"  Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")

    train_dataset = RouterDataset(train_records)
    val_dataset = RouterDataset(val_records)

    # Model and tokenizer
    model_name = config.get("model_name", "microsoft/deberta-v3-base")
    print(f"Loading model: {model_name}...")
    model = StudentRouter(model_name=model_name)
    tokenizer = get_tokenizer(model_name)
    collator = RouterCollator(tokenizer, max_length=config.get("max_length", 512))

    # Train
    trainer = DistillationTrainer(
        model=model,
        collator=collator,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        output_dir=config.get("output_dir", "outputs/checkpoints/router_student"),
    )

    results = trainer.train()

    # Save training history
    output_dir = Path(config.get("output_dir", "outputs/checkpoints/router_student"))
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(results["history"], f, indent=2)

    # Save tokenizer alongside model for easy loading
    tokenizer.save_pretrained(output_dir / "best_f1")
    tokenizer.save_pretrained(output_dir / "best_auroc")

    print(f"\nTraining complete!")
    print(f"  Best F1: {results['best_f1']:.4f}")
    print(f"  Best AUROC: {results['best_auroc']:.4f}")
    print(f"  Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
