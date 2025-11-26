from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trainer.data_module import ViTSFDataModule
from src.trainer.lightning_trainer import ViTSFLightningModule


class MetricTracker:
    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self.sq_error = 0.0
        self.abs_error = 0.0
        self.abs_perc_error = 0.0
        self.smape_error = 0.0
        self.sum_y = 0.0
        self.sum_y_sq = 0.0
        self.count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        diff = pred - target
        self.sq_error += torch.sum(diff * diff).item()
        self.abs_error += torch.sum(diff.abs()).item()
        self.abs_perc_error += torch.sum(diff.abs() / (target.abs().clamp(min=self.eps))).item()
        denom = (pred.abs() + target.abs()).clamp(min=self.eps)
        self.smape_error += torch.sum(2.0 * diff.abs() / denom).item()
        self.sum_y += torch.sum(target).item()
        self.sum_y_sq += torch.sum(target * target).item()
        self.count += target.numel()

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "mape": 0.0, "smape": 0.0, "r2": 0.0}
        mse = self.sq_error / self.count
        mae = self.abs_error / self.count
        rmse = mse ** 0.5
        mape = (self.abs_perc_error / self.count) * 100.0
        smape = (self.smape_error / self.count) * 100.0
        mean_y = self.sum_y / self.count
        sst = self.sum_y_sq - self.count * (mean_y ** 2)
        if sst <= self.eps:
            r2 = 0.0
        else:
            r2 = 1.0 - (self.sq_error / max(sst, self.eps))
        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
            "smape": smape,
            "r2": r2,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ViTSF checkpoints on a dataset split.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the .ckpt file to evaluate.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/processed/ett/ETTh1/data_with_TR.npz"),
        help="Path to the preprocessed npz file with T/R components.",
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="Cache directory for ViT images (optional).")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("eval_outputs"))
    parser.add_argument("--plot-node", type=int, default=0, help="Which node index to visualize in the plots.")
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="Optional cap on number of batches to evaluate (useful for smoke tests).",
    )
    return parser.parse_args()


def select_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def prepare_dataloader(args: argparse.Namespace) -> ViTSFDataModule:
    data_module = ViTSFDataModule(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.stride,
        image_size=args.image_size,
        cache_dir=args.cache_dir,
        return_history_components=True,
    )
    stage = "test" if args.split == "test" else "fit"
    data_module.setup(stage)
    return data_module


def plot_predictions(
    output_dir: Path,
    sample: Dict[str, torch.Tensor],
    node_idx: int,
    split: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    num_nodes = sample["target"].shape[-1]
    node_idx = max(0, min(node_idx, num_nodes - 1))
    history_total = sample.get("history")
    trend_history = sample.get("trend_history")
    residual_history = sample.get("residual_history")
    history = history_total[..., node_idx].cpu().numpy() if history_total is not None else None
    target = sample["target"][..., node_idx].cpu().numpy()
    final_pred = sample["final_pred"][..., node_idx].cpu().numpy()
    trend_true = sample["trend_true"][..., node_idx].cpu().numpy()
    trend_pred = sample["trend_pred"][..., node_idx].cpu().numpy()
    res_true = sample["res_true"][..., node_idx].cpu().numpy()
    res_pred = sample["res_pred"][..., node_idx].cpu().numpy()
    history_len = len(history) if history is not None else 0
    forecast_len = len(target)
    history_x = list(range(-history_len, 0)) if history_len > 0 else []
    forecast_x = list(range(0, forecast_len))
    truth_x = history_x + forecast_x if history_len > 0 else forecast_x
    truth_y = (history.tolist() if history is not None else []) + target.tolist()
    separator_x = -0.5 if history_len > 0 else None

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(truth_x, truth_y, label="Ground Truth (history+future)", color="black")
    axes[0].plot(forecast_x, final_pred, label="Final Prediction", color="tab:purple")
    if separator_x is not None:
        axes[0].axvline(separator_x, color="tab:gray", linestyle="--", linewidth=1)
    axes[0].set_title(f"Final vs Ground Truth (node {node_idx}, split={split})")
    axes[0].legend()

    if trend_history is not None:
        trend_hist = trend_history[..., node_idx].cpu().numpy()
        axes[1].plot(history_x + forecast_x, trend_hist.tolist() + trend_true.tolist(), label="Trend Truth", color="black")
    else:
        axes[1].plot(forecast_x, trend_true, label="Trend Truth", color="black")
    axes[1].plot(forecast_x, trend_pred, label="Trend Pred", color="tab:blue")
    if separator_x is not None:
        axes[1].axvline(separator_x, color="tab:gray", linestyle="--", linewidth=1)
    axes[1].set_title("Trend Path")
    axes[1].legend()

    if residual_history is not None:
        res_hist = residual_history[..., node_idx].cpu().numpy()
        axes[2].plot(history_x + forecast_x, res_hist.tolist() + res_true.tolist(), label="Residual Truth", color="black")
    else:
        axes[2].plot(forecast_x, res_true, label="Residual Truth", color="black")
    axes[2].plot(forecast_x, res_pred, label="Residual Pred", color="tab:green")
    if separator_x is not None:
        axes[2].axvline(separator_x, color="tab:gray", linestyle="--", linewidth=1)
    axes[2].set_title("Residual Path")
    axes[2].legend()
    axes[2].set_xlabel("Timesteps (history<0, forecast>=0)")

    fig.tight_layout()
    out_path = output_dir / f"prediction_overview_node{node_idx}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def evaluate(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    data_module = prepare_dataloader(args)
    if args.split == "train":
        dataloader = data_module.train_dataloader()
    elif args.split == "val":
        dataloader = data_module.val_dataloader()
    else:
        dataloader = data_module.test_dataloader()

    model = ViTSFLightningModule.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to(device)

    final_tracker = MetricTracker()
    trend_tracker = MetricTracker()
    residual_tracker = MetricTracker()
    sample_to_plot: Optional[Dict[str, torch.Tensor]] = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if args.limit_batches is not None and batch_idx >= args.limit_batches:
                break
            if len(batch) == 6:
                x_img, x_ts, y_trend, y_residual, history_trend, history_residual = batch
            else:
                x_img, x_ts, y_trend, y_residual = batch
                history_trend = history_residual = None
            x_img = x_img.to(device)
            x_ts = x_ts.to(device)
            y_trend = y_trend.to(device)
            y_residual = y_residual.to(device)
            y_target = y_trend + y_residual

            final_pred, trend_pred, residual_pred, _ = model(x_img, x_ts)

            final_tracker.update(final_pred, y_target)
            trend_tracker.update(trend_pred, y_trend)
            residual_tracker.update(residual_pred, y_residual)

            if sample_to_plot is None:
                sample_to_plot = {
                    "history": x_ts[0].detach().cpu(),
                    "target": y_target[0].detach().cpu(),
                    "final_pred": final_pred[0].detach().cpu(),
                    "trend_true": y_trend[0].detach().cpu(),
                    "trend_pred": trend_pred[0].detach().cpu(),
                    "res_true": y_residual[0].detach().cpu(),
                    "res_pred": residual_pred[0].detach().cpu(),
                }
                if history_trend is not None:
                    sample_to_plot["trend_history"] = history_trend[0].detach().cpu()
                if history_residual is not None:
                    sample_to_plot["residual_history"] = history_residual[0].detach().cpu()

    metrics = {
        "final": final_tracker.compute(),
        "trend_path": trend_tracker.compute(),
        "residual_path": residual_tracker.compute(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    if sample_to_plot is not None:
        plot_path = plot_predictions(args.output_dir, sample_to_plot, args.plot_node, args.split)
    else:
        plot_path = None

    print("\n===== Evaluation Summary =====")
    for name, stats in metrics.items():
        print(f"[{name}]")
        for metric_name, value in stats.items():
            print(f"  {metric_name}: {value:.6f}")
    print(f"Metrics saved to: {metrics_path}")
    if plot_path:
        print(f"Prediction plots saved to: {plot_path}")


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
