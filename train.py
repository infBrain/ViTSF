from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

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
        default=Path("cache/images"),
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
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine", "plateau"],
        default="cosine",
        help="Learning-rate annealing strategy.",
    )
    parser.add_argument(
        "--lr-t-max",
        type=int,
        default=None,
        help="T_max for cosine scheduler (defaults to max_epochs).",
    )
    parser.add_argument("--lr-min", type=float, default=1e-6, help="Minimum learning rate for schedulers.")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="Factor for ReduceLROnPlateau scheduler.")
    parser.add_argument("--lr-patience", type=int, default=5, help="Patience for ReduceLROnPlateau scheduler.")
    parser.add_argument("--fusion-mode", choices=["add", "gate"], default="gate")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="Enable EarlyStopping if >0 and set its patience (in validation checks).",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum change in monitored metric to qualify as improvement.",
    )
    parser.add_argument(
        "--monitor-metric",
        type=str,
        default="val_total_loss",
        help="Metric name to monitor for callbacks.",
    )
    parser.add_argument(
        "--monitor-mode",
        choices=["min", "max"],
        default="min",
        help="Optimization direction for monitored metric.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to store model checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-save-top-k",
        type=int,
        default=1,
        help="Number of best checkpoints (by monitored metric) to keep.",
    )
    parser.add_argument(
        "--checkpoint-save-last",
        action="store_true",
        help="Also persist the last checkpoint in addition to the top-k best.",
    )
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

    scheduler_config = None
    if args.lr_scheduler != "none":
        scheduler_config = {"type": args.lr_scheduler}
        if args.lr_scheduler == "cosine":
            scheduler_config["t_max"] = args.lr_t_max or args.max_epochs
            scheduler_config["eta_min"] = args.lr_min
        elif args.lr_scheduler == "plateau":
            scheduler_config.update(
                {
                    "patience": args.lr_patience,
                    "factor": args.lr_factor,
                    "min_lr": args.lr_min,
                    "monitor": args.monitor_metric,
                    "mode": args.monitor_mode,
                }
            )

    lightning_module = ViTSFLightningModule(
        model_kwargs=model_config,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_config=scheduler_config,
    )

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="vitsf-{epoch:02d}",
            monitor=args.monitor_metric,
            mode=args.monitor_mode,
            save_top_k=args.checkpoint_save_top_k,
            save_last=args.checkpoint_save_last,
            auto_insert_metric_name=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if args.early_stop_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=args.monitor_metric,
                mode=args.monitor_mode,
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
            )
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        fast_dev_run=args.fast_dev_run,
        callbacks=callbacks,
    )

    trainer.fit(lightning_module, datamodule=data_module)


if __name__ == "__main__":
    main()
