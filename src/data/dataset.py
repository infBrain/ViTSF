from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

Split = Literal["train", "val", "test"]


@dataclass
class SlidingWindowConfig:
    input_length: int
    forecast_horizon: int
    stride: int = 1
    target_columns: tuple[int, ...] | None = None


class SlidingWindowDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Create supervised samples from preprocessed arrays stored in .npz files."""

    def __init__(
        self,
        npz_path: Path | str,
        split: Split,
        window_cfg: SlidingWindowConfig,
    ) -> None:
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.npz_path}")
        payload = np.load(self.npz_path, allow_pickle=False)
        if split not in payload:
            raise KeyError(f"Split '{split}' not found in {self.npz_path}. Available: {list(payload.keys())}")
        data = payload[split]
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array for split '{split}', got shape {data.shape}")

        self.data = data.astype(np.float32)
        self.window_cfg = window_cfg
        self.samples: list[tuple[int, int]] = []

        total_len, _ = self.data.shape
        max_start = total_len - (window_cfg.input_length + window_cfg.forecast_horizon)
        for start in range(0, max_start + 1, window_cfg.stride):
            end = start + window_cfg.input_length
            target_end = end + window_cfg.forecast_horizon
            self.samples.append((start, target_end))

        if not self.samples:
            raise ValueError(
                "No samples generated. Check input_length, forecast_horizon, and stride relative to data length."
            )

        if window_cfg.target_columns is not None:
            self.target_columns = np.array(window_cfg.target_columns, dtype=np.int64)
        else:
            self.target_columns = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start, target_end = self.samples[idx]
        end = start + self.window_cfg.input_length
        input_window = self.data[start:end]
        target_window = self.data[end:target_end]

        if self.target_columns is not None:
            target_window = target_window[:, self.target_columns]

        x = torch.from_numpy(input_window)
        y = torch.from_numpy(target_window)
        return x, y
