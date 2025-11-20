from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trainer.data_module import ViTSFDataModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ViTSF training inputs.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/processed/ett/ETTh1/data_with_TR.npz"),
        help="Path to the preprocessed npz file that includes T/R components.",
    )
    parser.add_argument("--cache-dir", type=Path, default=None, help="Directory containing cached ViT images.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--index", type=int, default=0, help="Sample index within the chosen split.")
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", type=Path, default=Path("vis_input.png"))
    parser.add_argument("--show", action="store_true", help="Display the matplotlib window in addition to saving.")
    return parser.parse_args()


def get_dataset(args: argparse.Namespace):
    data_module = ViTSFDataModule(
        dataset_path=args.dataset_path,
        batch_size=1,
        num_workers=0,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.stride,
        image_size=args.image_size,
        cache_dir=args.cache_dir,
    )
    data_module.setup(None)
    if args.split == "train":
        dataset = data_module.train_dataset
    elif args.split == "val":
        dataset = data_module.val_dataset
    else:
        dataset = data_module.test_dataset
    if dataset is None:
        raise RuntimeError(f"Dataset for split '{args.split}' is not initialized.")
    return dataset


def main() -> None:
    args = parse_args()
    dataset = get_dataset(args)
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"Index {args.index} out of range for split '{args.split}' (size {len(dataset)}).")

    image, series, trend, residual = dataset[args.index]

    img = image.detach().cpu()
    img_np = img.numpy()
    img_min = img_np.min()
    img_max = img_np.max()
    norm_img = (img_np - img_min) / (img_max - img_min + 1e-6)

    time_axis = range(series.shape[0])
    mean_series = series.mean(dim=1).cpu().numpy()
    first_node = series[:, 0].cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    axes[0].imshow(norm_img.transpose(1, 2, 0), aspect="auto")
    axes[0].set_title(f"ViT Input Image — split={args.split} index={args.index}")
    axes[0].axis("off")

    axes[1].plot(time_axis, mean_series, label="Mean across nodes")
    axes[1].plot(time_axis, first_node, label="Node 0")
    axes[1].set_title("Original time-series window")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Normalized value")
    axes[1].legend()

    fig.tight_layout()
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"✅ Visualization saved to {output_path}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
