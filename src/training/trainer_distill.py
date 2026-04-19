"""Distillation trainer for the student router."""

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.metrics import f1_score, roc_auc_score

from src.models.losses import DistillationLoss


class RouterDataset(Dataset):
    """Simple dataset for router training."""

    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        return {
            "text": f"[CONTEXT] {rec['context']} [QUESTION] {rec['question']}",
            "requires_tom": float(rec["requires_tom"]),
            "teacher_prob_tom": float(rec["teacher_prob_tom"]),
            "sample_id": rec["sample_id"],
        }


class DistillationTrainer:
    """Train student router with hard + soft label distillation."""

    def __init__(
        self,
        model,
        collator,
        train_dataset: RouterDataset,
        val_dataset: RouterDataset,
        config: dict,
        output_dir: str = "outputs/checkpoints/router_student",
    ):
        self.model = model
        self.collator = collator
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = DistillationLoss(
            alpha_hard=config.get("alpha_hard", 0.7),
            beta_soft=config.get("beta_soft", 0.3),
            temperature=config.get("distill_temperature", 1.5),
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get("batch_size", 16),
            shuffle=True,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.get("batch_size", 16),
            shuffle=False,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True,
        )

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=float(config.get("lr", 2e-5)),
            weight_decay=float(config.get("weight_decay", 0.01)),
        )

        # Scheduler
        num_training_steps = len(self.train_loader) * config.get("epochs", 5)
        warmup_steps = int(num_training_steps * config.get("warmup_ratio", 0.1))
        self.scheduler = get_scheduler(
            config.get("scheduler", "cosine"),
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        # DeBERTa has fp16 weight issues with AMP — run in fp32
        self.use_amp = False
        # Convert model to fp32 to avoid mixed precision issues
        self.model.float()

        # Tracking
        self.best_f1 = 0.0
        self.best_auroc = 0.0
        self.patience_counter = 0
        self.history = []

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_hard = 0
        total_soft = 0
        n_batches = 0

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            hard_labels = batch["hard_labels"].to(self.device)
            teacher_probs = batch["teacher_probs"].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    losses = self.criterion(logits.float(), hard_labels, teacher_probs)
            else:
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                losses = self.criterion(logits, hard_labels, teacher_probs)

            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            self.scheduler.step()

            total_loss += losses["total_loss"].item()
            total_hard += losses["hard_loss"].item()
            total_soft += losses["soft_loss"].item()
            n_batches += 1

        return {
            "train_loss": total_loss / n_batches,
            "train_hard_loss": total_hard / n_batches,
            "train_soft_loss": total_soft / n_batches,
        }

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate on validation set."""
        self.model.eval()
        all_logits = []
        all_labels = []
        total_loss = 0
        n_batches = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            hard_labels = batch["hard_labels"].to(self.device)
            teacher_probs = batch["teacher_probs"].to(self.device)

            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    losses = self.criterion(logits.float(), hard_labels, teacher_probs)
            else:
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                losses = self.criterion(logits, hard_labels, teacher_probs)

            all_logits.append(logits.cpu())
            all_labels.append(hard_labels.cpu())
            total_loss += losses["total_loss"].item()
            n_batches += 1

        all_logits = torch.cat(all_logits).float().numpy()
        all_labels = torch.cat(all_labels).numpy().astype(int)
        all_probs = 1 / (1 + np.exp(-all_logits))  # sigmoid
        all_preds = (all_probs >= 0.5).astype(int)

        f1 = f1_score(all_labels, all_preds, average="macro")
        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auroc = 0.5

        return {
            "val_loss": total_loss / n_batches,
            "val_f1": f1,
            "val_auroc": auroc,
            "val_accuracy": (all_preds == all_labels).mean(),
        }

    def save_checkpoint(self, name: str, metrics: dict):
        """Save model checkpoint."""
        path = self.output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        # Save config for reproducibility
        import json
        with open(path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(path / "config.json", "w") as f:
            json.dump(dict(self.config) if hasattr(self.config, 'items') else self.config, f, indent=2)

    def train(self) -> dict:
        """Full training loop with early stopping."""
        epochs = self.config.get("epochs", 5)
        patience = self.config.get("patience", 2)

        print(f"Training for {epochs} epochs...")
        print(f"  Train: {len(self.train_dataset)} samples")
        print(f"  Val: {len(self.val_dataset)} samples")
        print(f"  Batch size: {self.config.get('batch_size', 16)}")
        print(f"  Alpha (hard): {self.config.get('alpha_hard', 0.7)}")
        print(f"  Beta (soft): {self.config.get('beta_soft', 0.3)}")
        print(f"  Temperature: {self.config.get('distill_temperature', 1.5)}")

        for epoch in range(epochs):
            start = time.time()

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()

            elapsed = time.time() - start
            metrics = {**train_metrics, **val_metrics, "epoch": epoch, "time": elapsed}
            self.history.append(metrics)

            print(
                f"  Epoch {epoch+1}/{epochs} ({elapsed:.0f}s): "
                f"loss={metrics['train_loss']:.4f}, "
                f"val_loss={metrics['val_loss']:.4f}, "
                f"val_f1={metrics['val_f1']:.4f}, "
                f"val_auroc={metrics['val_auroc']:.4f}, "
                f"val_acc={metrics['val_accuracy']:.4f}"
            )

            # Save best by F1
            if val_metrics["val_f1"] > self.best_f1:
                self.best_f1 = val_metrics["val_f1"]
                self.save_checkpoint("best_f1", metrics)
                print(f"    -> New best F1: {self.best_f1:.4f}")

            # Save best by AUROC
            if val_metrics["val_auroc"] > self.best_auroc:
                self.best_auroc = val_metrics["val_auroc"]
                self.save_checkpoint("best_auroc", metrics)
                print(f"    -> New best AUROC: {self.best_auroc:.4f}")

            # Early stopping on F1
            if val_metrics["val_f1"] < self.best_f1:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            else:
                self.patience_counter = 0

        # Save last checkpoint
        self.save_checkpoint("last", metrics)

        return {
            "best_f1": self.best_f1,
            "best_auroc": self.best_auroc,
            "history": self.history,
        }
