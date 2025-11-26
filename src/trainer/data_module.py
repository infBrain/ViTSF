from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

Split = str


@dataclass(slots=True)
class WindowConfig:
    seq_len: int
    pred_len: int
    stride: int = 1


class SeriesImageRenderer:
    """Convert a multivariate window into a 3xHxW pseudo-image for ViT."""

    def __init__(self, image_size: int = 224) -> None:
        self.image_size = image_size

    def __call__(self, window: np.ndarray) -> torch.Tensor:
        # window: (seq_len, num_nodes)
        if window.ndim != 2:
            raise ValueError(f"Expected 2D window, got shape {window.shape}")
        vec_mean = window.mean(axis=1)
        vec_std = window.std(axis=1)
        vec_first = window[:, 0]
        channels = [vec_mean, vec_std, vec_first]
        rendered: List[torch.Tensor] = []
        for vec in channels:
            torch_vec = torch.from_numpy(vec.astype(np.float32))
            torch_vec = (torch_vec - torch_vec.mean()) / (torch_vec.std() + 1e-6)
            resized = F.interpolate(
                torch_vec.view(1, 1, -1), size=self.image_size, mode="linear", align_corners=False
            ).view(-1)
            channel = resized.unsqueeze(0).repeat(self.image_size, 1)
            rendered.append(channel)
        image = torch.stack(rendered, dim=0)
        return image


class ViTSFDataset(Dataset[Tuple[torch.Tensor, ...]]):
    """Dataset returning (image, series, trend_target, residual_target) tuples."""

    def __init__(
        self,
        split_name: Split,
        arrays: Dict[str, np.ndarray],
        window_cfg: WindowConfig,
        renderer: SeriesImageRenderer,
        context: Dict[str, np.ndarray] | None = None,
        cache_dir: Path | None = None,
        return_history_components: bool = False,
    ) -> None:
        self.split = split_name
        split_series = arrays[split_name]
        split_trend = arrays[f"{split_name}_T"]
        split_residual = arrays[f"{split_name}_R"]
        if (
            split_series.shape != split_trend.shape
            or split_series.shape != split_residual.shape
        ):
            raise ValueError(f"Series, trend, residual shapes must match for split {split_name}.")

        context_series = context.get("series") if context else None
        context_trend = context.get("trend") if context else None
        context_residual = context.get("residual") if context else None
        self.context_len = 0 if context_series is None else context_series.shape[0]
        self.split_len = split_series.shape[0]

        def _concat(main: np.ndarray, prefix: np.ndarray | None) -> np.ndarray:
            if prefix is None or prefix.size == 0:
                return main.astype(np.float32)
            if prefix.shape[1] != main.shape[1]:
                raise ValueError("Context and split must have the same feature dimension.")
            return np.concatenate([prefix.astype(np.float32), main.astype(np.float32)], axis=0)

        self.series = _concat(split_series, context_series)
        self.trend = _concat(split_trend, context_trend)
        self.residual = _concat(split_residual, context_residual)

        self.window_cfg = window_cfg
        self.renderer = renderer
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.return_history_components = return_history_components
        self.num_nodes = self.series.shape[1]
        total = self.series.shape[0]
        start_lower = max(0, self.context_len - self.window_cfg.seq_len)
        start_upper = self.context_len + self.split_len - (self.window_cfg.seq_len + self.window_cfg.pred_len)
        if start_upper < start_lower:
            raise ValueError("Window lengths exceed available timesteps.")
        self.indices: List[Tuple[int, int]] = []
        for start in range(start_lower, start_upper + 1, window_cfg.stride):
            end = start + window_cfg.seq_len
            target_end = end + window_cfg.pred_len
            self.indices.append((start, target_end))
        if not self.indices:
            raise ValueError("No samples generated; adjust seq_len/pred_len/stride.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        start, target_end = self.indices[idx]
        end = start + self.window_cfg.seq_len
        seq_window = self.series[start:end]
        trend_window = self.trend[end:target_end]
        residual_window = self.residual[end:target_end]

        image: torch.Tensor
        if self.cache_dir is not None:
            cache_file = self.cache_dir / f"{self.split}_{start:08d}.pt"
            if cache_file.exists():
                image = torch.load(cache_file, map_location="cpu")
            else:
                image = self.renderer(seq_window)
                tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
                torch.save(image, tmp_file)
                tmp_file.replace(cache_file)
        else:
            image = self.renderer(seq_window)
        x_series = torch.from_numpy(seq_window.astype(np.float32))
        y_trend = torch.from_numpy(trend_window.astype(np.float32))
        y_residual = torch.from_numpy(residual_window.astype(np.float32))
        if not self.return_history_components:
            return image, x_series, y_trend, y_residual

        history_trend = torch.from_numpy(self.trend[start:end].astype(np.float32))
        history_residual = torch.from_numpy(self.residual[start:end].astype(np.float32))
        return image, x_series, y_trend, y_residual, history_trend, history_residual


class ViTSFDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: str | Path,
        batch_size: int = 8,
        num_workers: int = 4,
        seq_len: int = 336,
        pred_len: int = 96,
        stride: int = 1,
        image_size: int = 224,
        cache_dir: str | Path | None = None,
        return_history_components: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_cfg = WindowConfig(seq_len=seq_len, pred_len=pred_len, stride=stride)
        self.renderer = SeriesImageRenderer(image_size=image_size)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.return_history_components = return_history_components
        self.arrays: Dict[str, np.ndarray] | None = None
        self._num_nodes: int | None = None
        self.train_dataset: ViTSFDataset | None = None
        self.val_dataset: ViTSFDataset | None = None
        self.test_dataset: ViTSFDataset | None = None

    @property
    def num_nodes(self) -> int:
        if self._num_nodes is None:
            raise RuntimeError("Call setup() before accessing num_nodes")
        return self._num_nodes

    def setup(self, stage: str | None = None) -> None:
        if self.arrays is None:
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
            payload = np.load(self.dataset_path, allow_pickle=False)
            required_keys = {
                "train",
                "val",
                "test",
                "train_T",
                "val_T",
                "test_T",
                "train_R",
                "val_R",
                "test_R",
            }
            missing = required_keys - set(payload.keys())
            if missing:
                raise KeyError(f"Dataset missing required keys: {missing}")
            self.arrays = {key: payload[key] for key in required_keys}
            self._num_nodes = int(payload["train"].shape[1])

        def _tail(arr: np.ndarray, length: int) -> np.ndarray:
            if length <= 0:
                return arr[:0]
            if arr.shape[0] >= length:
                return arr[-length:]
            return arr

        if stage in ("fit", None):
            train_context = None
            train_cache = self.cache_dir / "train" if self.cache_dir is not None else None
            val_cache = self.cache_dir / "val" if self.cache_dir is not None else None
            context_for_val = {
                "series": _tail(self.arrays["train"], self.window_cfg.seq_len),
                "trend": _tail(self.arrays["train_T"], self.window_cfg.seq_len),
                "residual": _tail(self.arrays["train_R"], self.window_cfg.seq_len),
            }
            self.train_dataset = ViTSFDataset(
                "train",
                self.arrays,
                self.window_cfg,
                self.renderer,
                context=train_context,
                cache_dir=train_cache,
                return_history_components=self.return_history_components,
            )
            self.val_dataset = ViTSFDataset(
                "val",
                self.arrays,
                self.window_cfg,
                self.renderer,
                context=context_for_val,
                cache_dir=val_cache,
                return_history_components=self.return_history_components,
            )
        if stage in ("test", None):
            train_val_series = np.concatenate([self.arrays["train"], self.arrays["val"]], axis=0)
            train_val_trend = np.concatenate([self.arrays["train_T"], self.arrays["val_T"]], axis=0)
            train_val_residual = np.concatenate([self.arrays["train_R"], self.arrays["val_R"]], axis=0)
            context_for_test = {
                "series": _tail(train_val_series, self.window_cfg.seq_len),
                "trend": _tail(train_val_trend, self.window_cfg.seq_len),
                "residual": _tail(train_val_residual, self.window_cfg.seq_len),
            }
            test_cache = self.cache_dir / "test" if self.cache_dir is not None else None
            self.test_dataset = ViTSFDataset(
                "test",
                self.arrays,
                self.window_cfg,
                self.renderer,
                context=context_for_test,
                cache_dir=test_cache,
                return_history_components=self.return_history_components,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting train_dataloader")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting val_dataloader")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before requesting test_dataloader")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
