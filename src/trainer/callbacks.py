from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch


class LossHistoryCallback(pl.Callback):
    """Record per-epoch train/val losses and persist as txt + plot."""

    def __init__(self, output_dir: Path | str) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        value = _get_metric(trainer, "train_total_loss")
        if value is not None:
            self.train_losses.append(value)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        value = _get_metric(trainer, "val_total_loss")
        if value is not None:
            self.val_losses.append(value)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.train_losses and not self.val_losses:
            return
        txt_path = self.output_dir / "loss_history.txt"
        max_len = max(len(self.train_losses), len(self.val_losses))
        with txt_path.open("w", encoding="utf-8") as fh:
            fh.write("epoch\ttrain_total_loss\tval_total_loss\n")
            for epoch in range(max_len):
                train_loss = _format_loss(self.train_losses, epoch)
                val_loss = _format_loss(self.val_losses, epoch)
                fh.write(f"{epoch}\t{train_loss}\t{val_loss}\n")

        plot_path = self.output_dir / "loss_curve.png"
        epochs_train = range(len(self.train_losses))
        epochs_val = range(len(self.val_losses))
        plt.figure(figsize=(8, 5))
        if self.train_losses:
            plt.plot(epochs_train, self.train_losses, label="Train", color="tab:blue")
        if self.val_losses:
            plt.plot(epochs_val, self.val_losses, label="Validation", color="tab:orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Per-epoch Loss History")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()


def _get_metric(trainer: pl.Trainer, key: str) -> float | None:
    value = trainer.callback_metrics.get(key)
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _format_loss(losses: List[float], index: int) -> str:
    if index >= len(losses):
        return ""
    return f"{losses[index]:.6f}"
