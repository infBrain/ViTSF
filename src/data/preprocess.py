
# python preprocess.py ett --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
METADATA_PATH = ROOT / "data" / "metadata.json"

SUPPORTED_DATASETS = {"ett": {"files": ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv"], "datetime": "date"},
                      "weather": {"files": ["weather.csv"], "datetime": "date"}
                      }


@dataclass
class SplitConfig:
    train_ratio: float = 0.6
    val_ratio: float = 0.1
    test_ratio: float = 0.3

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}.")
        for name, value in ("train", self.train_ratio), ("val", self.val_ratio), ("test", self.test_ratio):
            if value <= 0:
                raise ValueError(f"Split ratio for {name} must be positive, got {value}.")


@dataclass
class DatasetArtifacts:
    features_path: Path
    stats_path: Path


def load_metadata() -> Dict[str, Dict]:
    with METADATA_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _infer_datetime_column(df: pd.DataFrame) -> str:
    for candidate in ("date", "Date", "timestamp", "time", "Time"):
        if candidate in df.columns:
            try:
                pd.to_datetime(df[candidate])
                return candidate
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Failed to parse datetime column '{candidate}': {exc}") from exc
    raise ValueError("Unable to infer datetime column. Provide --datetime-column explicitly.")


def preprocess_csv(
    dataset_key: str,
    file_name: str,
    split_cfg: SplitConfig,
    datetime_column: str | None = None,
    freq: str | None = None,
    output_dir: Path | None = None,
) -> DatasetArtifacts:
    raw_path = RAW_DIR / dataset_key / file_name
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = pd.read_csv(raw_path)
    dt_col = datetime_column or _infer_datetime_column(df)
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).reset_index(drop=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError(f"No numeric columns found in file {file_name}.")

    df = df.set_index(dt_col)
    if freq:
        df = df.resample(freq).mean()

    df[numeric_cols] = df[numeric_cols].interpolate(method="time").fillna(method="bfill").fillna(method="ffill")
    df = df.reset_index()

    # values = df[numeric_cols].to_numpy(dtype=np.float32)
    # mean = values.mean(axis=0, keepdims=True)
    # std = values.std(axis=0, keepdims=True) + 1e-6
    # norm_values = (values - mean) / std

    # 计算切分点
    split_cfg.validate()
    n_samples = len(df)
    train_end = int(n_samples * split_cfg.train_ratio)
    val_end = train_end + int(n_samples * split_cfg.val_ratio)


    # 标准化：只用 train 进行拟合
    values = df[numeric_cols].to_numpy(dtype=np.float32)
    train_values = values[:train_end]
    mean = train_values.mean(axis=0, keepdims=True)
    std  = train_values.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)  # 防止除以零
    norm_values = (values - mean) / std   # 标准化整个数据集

    # 切分数据集
    splits = {
        "train": norm_values[:train_end],
        "val": norm_values[train_end:val_end],
        "test": norm_values[val_end:],
    }

    timestamps = df[dt_col].to_numpy()
    meta = {
        "dataset": dataset_key,
        "file": file_name,
        "datetime_column": dt_col,
        "numeric_columns": numeric_cols,
        "split": {
            "train": [0, train_end],
            "val": [train_end, val_end],
            "test": [val_end, n_samples],
        },
        "original_length": n_samples,
        "frequency": freq,
    }

    out_dir = output_dir or (PROCESSED_DIR / dataset_key / Path(file_name).stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_path = out_dir / "data.npz"
    stats_path = out_dir / "stats.json"

    np.savez_compressed(
        features_path,
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
        timestamps=timestamps.astype("datetime64[ns]"),
    )

    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump({"mean": mean.squeeze().tolist(), "std": std.squeeze().tolist(), **meta}, fh, indent=2)

    return DatasetArtifacts(features_path=features_path, stats_path=stats_path)


def preprocess_dataset(dataset_key: str, split_cfg: SplitConfig, freq: str | None = None) -> Dict[str, DatasetArtifacts]:
    artifacts: Dict[str, DatasetArtifacts] = {}
    if dataset_key not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{dataset_key}'. Supported: {list(SUPPORTED_DATASETS)}")
    files = SUPPORTED_DATASETS[dataset_key]["files"]
    for file_name in tqdm(files, desc=f"Preprocessing {dataset_key.upper()}"):
        artifacts[file_name] = preprocess_csv(
            dataset_key,
            file_name,
            split_cfg,
            datetime_column=SUPPORTED_DATASETS[dataset_key].get("datetime"),
            freq=freq,
        )
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw time series datasets for forecasting tasks.")
    parser.add_argument("dataset", choices=SUPPORTED_DATASETS.keys(), help="Dataset key, e.g. 'ett'.")
    parser.add_argument("--freq", default=None, help="Optional resample frequency string (e.g. 'H', '15T').")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_cfg = SplitConfig(train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    artifacts = preprocess_dataset(args.dataset, split_cfg, freq=args.freq)
    for file_name, artifact in artifacts.items():
        print(f"Prepared {file_name}: data → {artifact.features_path}, stats → {artifact.stats_path}")


if __name__ == "__main__":
    main()