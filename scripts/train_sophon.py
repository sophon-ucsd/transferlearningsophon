#!/usr/bin/env python3
"""Main training entry point.

Usage:
    python scripts/train.py transfer.strategy=frozen data.train_size=10000
    python scripts/train.py transfer.strategy=full_ft transfer.checkpoint=path/to/model.pt
"""
import os
import sys

import torch
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.sophon_wrapper import create_model, SophonTransferModel
from src.data.jetclass import JetClassDataModule
from src.training.trainer import JetClassifier
from src.utils.reproducibility import seed_everything


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.project.seed)

    # Data
    datamodule = JetClassDataModule(
        data_dir=cfg.data.get("data_dir", ""),
        train_dir=cfg.data.get("train_dir", None),
        val_dir=cfg.data.get("val_dir", None),
        test_dir=cfg.data.get("test_dir", None),
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
        test_size=cfg.data.test_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        seed=cfg.project.seed,
    )

    # Model
    model = create_model(
        strategy=cfg.transfer.strategy,
        checkpoint_path=cfg.transfer.checkpoint,
        num_classes=cfg.model.num_classes,
        head_type=cfg.transfer.get("head_type", "mlp"),
        frozen_layers=cfg.transfer.get("frozen_layers", 4),
    )
    total, trainable = SophonTransferModel.count_params(model)
    print(f"Model: {total:,} total params, {trainable:,} trainable")

    # Lightning module
    lightning_model = JetClassifier(model, cfg)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=cfg.training.early_stopping_patience, mode="min"),
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Compute log_every_n_steps safely
    if cfg.data.train_size:
        steps_per_epoch = max(1, cfg.data.train_size // cfg.data.batch_size)
        log_every = min(50, steps_per_epoch)
    else:
        log_every = 50

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accelerator="auto",
        devices=1,
        log_every_n_steps=log_every,
        enable_progress_bar=True,
        deterministic=cfg.project.get("deterministic", False),
    )

    # Train
    trainer.fit(lightning_model, datamodule=datamodule)

    # Capture val metrics before test overwrites them
    val_metrics = {k: v.item() if isinstance(v, torch.Tensor) else float(v)
                   for k, v in trainer.callback_metrics.items()}

    # Test on best checkpoint
    trainer.test(lightning_model, datamodule=datamodule, ckpt_path="best")

    # Save results
    output_dir = os.path.join(
        "results",
        cfg.get("sweep_name", "default"),
        f"{cfg.transfer.strategy}_{cfg.data.train_size}_{cfg.project.seed}",
    )
    lightning_model.save_results(output_dir, val_metrics=val_metrics)


if __name__ == "__main__":
    main()
