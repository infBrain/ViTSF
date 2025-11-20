from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl

from src.trainer.data_module import ViTSFDataModule
from src.trainer.lightning_trainer import ViTSFLightningModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ViTSF with PyTorch Lightning")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/processed/ett/ETTh1/data_with_TR.npz"),
        help="Path to the preprocessed npz file that includes T/R components.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional directory to cache rendered ViT images.",
    )
    parser.add_argument("--seq-len", type=int, default=336, help="Input sequence length for the causal path.")
    parser.add_argument("--pred-len", type=int, default=96, help="Forecast horizon.")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--fusion-mode", choices=["add", "gate"], default="gate")
    parser.add_argument("--fast-dev-run", action="store_true", help="Enable Lightning fast_dev_run mode.")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(42, workers=True)

    data_module = ViTSFDataModule(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.stride,
        cache_dir=args.cache_dir,
    )
    data_module.setup("fit")

    model_config = {
        "pred_len": args.pred_len,
        "num_nodes": data_module.num_nodes,
        "fusion_mode": args.fusion_mode,
    }

    lightning_module = ViTSFLightningModule(
        model_kwargs=model_config,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(lightning_module, datamodule=data_module)


if __name__ == "__main__":
    main()
