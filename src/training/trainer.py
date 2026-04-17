"""PyTorch Lightning module for jet classification with transfer learning."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy


class JetClassifier(pl.LightningModule):
    """Lightning module for training jet classifiers with any transfer strategy.

    Logs: train_loss, train_acc, val_loss, val_acc, val_auc, test_loss, test_acc, test_auc.
    Uses sklearn roc_auc_score for AUC (torchmetrics multiclass AUROC is unreliable
    when not all classes appear in every batch).
    """

    def __init__(self, model: torch.nn.Module, cfg) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.num_classes = cfg.model.num_classes

        # Accuracy via torchmetrics (works fine per-batch)
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

        # AUC computed from accumulated predictions (sklearn, epoch-level)
        self._val_probs: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []
        self._test_probs: list[torch.Tensor] = []
        self._test_labels: list[torch.Tensor] = []

        # Timing
        self._train_start_time: float | None = None

    def forward(self, batch: dict[str, torch.Tensor]):
        return self.model(batch["features"], batch["lorentz_vectors"], batch["mask"])

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        logits, _ = self(batch)
        loss = F.cross_entropy(logits, batch["label"])
        preds = logits.argmax(dim=-1)
        self.train_acc(preds, batch["label"])
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        logits, _ = self(batch)
        loss = F.cross_entropy(logits, batch["label"])
        preds = logits.argmax(dim=-1)
        self.val_acc(preds, batch["label"])
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self._val_probs.append(F.softmax(logits, dim=-1).detach().cpu())
        self._val_labels.append(batch["label"].detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_probs:
            return
        auc = self._compute_auc(self._val_probs, self._val_labels)
        self.log("val_auc", auc, prog_bar=True)
        self._val_probs.clear()
        self._val_labels.clear()

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        logits, _ = self(batch)
        loss = F.cross_entropy(logits, batch["label"])
        preds = logits.argmax(dim=-1)
        self.test_acc(preds, batch["label"])
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self._test_probs.append(F.softmax(logits, dim=-1).detach().cpu())
        self._test_labels.append(batch["label"].detach().cpu())

    def on_test_epoch_end(self) -> None:
        if not self._test_probs:
            return
        auc = self._compute_auc(self._test_probs, self._test_labels)
        self.log("test_auc", auc)
        self._test_probs.clear()
        self._test_labels.clear()

    @staticmethod
    def _compute_auc(probs_list: list[torch.Tensor], labels_list: list[torch.Tensor]) -> float:
        from sklearn.metrics import roc_auc_score
        probs = torch.cat(probs_list).numpy()
        labels = torch.cat(labels_list).numpy()
        try:
            return roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        except ValueError:
            return 0.0

    def on_train_start(self) -> None:
        self._train_start_time = time.time()

    def configure_optimizers(self) -> dict:
        """Configure optimizer with strategy-appropriate LR handling.

        - frozen / from_scratch: all trainable params at cfg lr
        - partial_ft: all trainable params at cfg lr
        - full_ft: backbone at transfer.backbone_lr, head at training.optimizer.lr
        """
        strategy = self.cfg.transfer.strategy
        lr = self.cfg.training.optimizer.lr
        wd = self.cfg.training.optimizer.weight_decay

        if strategy == "full_ft":
            backbone_lr = self.cfg.transfer.get("backbone_lr", lr * 0.1)
            param_groups = self.model.get_param_groups(backbone_lr, lr, wd)
            optimizer = torch.optim.AdamW(param_groups)
        else:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)

        warmup_epochs = self.cfg.training.scheduler.warmup_epochs
        max_epochs = self.cfg.training.max_epochs

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, max_epochs - warmup_epochs), eta_min=1e-6
        )
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup, cosine], milestones=[warmup_epochs]
            )
        else:
            scheduler = cosine

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def save_results(self, output_dir: str, val_metrics: dict | None = None) -> None:
        """Save results.json with all metrics and config."""
        total, trainable = type(self.model).count_params(self.model)
        elapsed = time.time() - self._train_start_time if self._train_start_time else 0

        # Test metrics from callback_metrics (available after trainer.test())
        m = self.trainer.callback_metrics

        def _get(key: str) -> float:
            # Check test metrics first, then val_metrics snapshot
            for source in (m, val_metrics or {}):
                if key in source:
                    v = source[key]
                    return v.item() if isinstance(v, torch.Tensor) else float(v)
            return float("nan")

        results = {
            "strategy": self.cfg.transfer.strategy,
            "train_size": self.cfg.data.train_size,
            "seed": self.cfg.project.seed,
            "num_classes": self.num_classes,
            "best_val_loss": _get("val_loss"),
            "best_val_acc": _get("val_acc"),
            "val_auc_macro": _get("val_auc"),
            "test_loss": _get("test_loss"),
            "test_acc": _get("test_acc"),
            "test_auc_macro": _get("test_auc"),
            "total_params": total,
            "trainable_params": trainable,
            "num_epochs": self.current_epoch + 1,
            "wall_clock_seconds": elapsed,
        }

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_dir}/results.json")
