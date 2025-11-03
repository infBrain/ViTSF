# src/make_tr_labels.py
# -*- coding: utf-8 -*-
"""
基于命令行参数的 趋势-残差 打标脚本（无泄漏 & 无边界阶跃）：
- 支持 {train,val,test} 或 {X_all, train_end, val_end} 两种 npz 格式
- 先拼接整段 → 因果趋势(EWMA/MA) → 切回三段
示例：
python -m src.make_tr_labels \
  --npz data/processed/ett/ETTh1/data.npz \
  --trend ewma --window 24
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
from typing import Dict, Tuple
import numpy as np


def causal_moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """因果(只看过去)简单移动平均。"""
    T, D = arr.shape
    out = np.empty_like(arr, dtype=np.float32)
    csum = np.cumsum(arr, axis=0)
    for t in range(T):
        s = 0 if t < window - 1 else t - window + 1
        seg = csum[t] if s == 0 else csum[t] - csum[s - 1]
        out[t] = seg / (t - s + 1)
    return out


def causal_ewma(arr: np.ndarray, alpha: float) -> np.ndarray:
    """因果 EWMA：out[t] = a*x[t] + (1-a)*out[t-1]。起点 out[0]=x[0]。"""
    T, D = arr.shape
    out = np.empty_like(arr, dtype=np.float32)
    out[0] = arr[0]
    a = float(alpha)
    b = 1.0 - a
    for t in range(1, T):
        out[t] = a * arr[t] + b * out[t - 1]
    return out


def make_TR_full_series(
    X_full: np.ndarray, trend_kind: str, window: int, alpha: float | None
) -> Tuple[np.ndarray, np.ndarray]:
    """整段上计算因果趋势，再得残差。"""
    if trend_kind == "ewma":
        if alpha is None:
            alpha = 2.0 / (window + 1.0)  # 常用经验：与 MA 窗口对应
        T_full = causal_ewma(X_full, alpha=float(alpha))
    elif trend_kind == "ma":
        T_full = causal_moving_average(X_full, window=window)
    else:
        raise ValueError("trend_kind must be 'ewma' or 'ma'")
    R_full = X_full - T_full
    return T_full.astype(np.float32), R_full.astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Make trend/residual labels (causal, no leakage)")
    p.add_argument("--npz", required=True, type=str, help="preprocess 输出的 npz 路径")
    p.add_argument("--trend", choices=["ewma", "ma"], default="ewma", help="趋势器类型")
    p.add_argument("--window", type=int, default=24, help="MA 窗口或用于换算 ewma 的窗口")
    p.add_argument("--alpha", type=float, default=None, help="ewma 的 alpha；缺省按 2/(W+1) 计算")
    p.add_argument("--out-suffix", type=str, default="_with_TR.npz", help="输出文件后缀")
    return p.parse_args()


def main():
    args = parse_args()
    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    z = np.load(npz_path, allow_pickle=False)
    has_splits = all(k in z for k in ["train", "val", "test"])
    has_all = "X_all" in z and "train_end" in z and "val_end" in z
    if not (has_splits or has_all):
        raise KeyError("npz must contain either {train,val,test} or {X_all, train_end, val_end}.")

    if has_splits:
        X_train = z["train"].astype(np.float32)
        X_val = z["val"].astype(np.float32)
        X_test = z["test"].astype(np.float32)
        X_full = np.concatenate([X_train, X_val, X_test], axis=0)
        train_end = len(X_train)
        val_end = train_end + len(X_val)
    else:
        X_full = z["X_all"].astype(np.float32)
        train_end = int(z["train_end"][0])
        val_end = int(z["val_end"][0])

    # 整段做因果趋势
    T_full, R_full = make_TR_full_series(
        X_full, trend_kind=args.trend, window=args.window, alpha=args.alpha
    )

    # 切回三段
    T_train, T_val, T_test = T_full[:train_end], T_full[train_end:val_end], T_full[val_end:]
    R_train, R_val, R_test = R_full[:train_end], R_full[train_end:val_end], R_full[val_end:]

    # 组织输出
    out_dict: Dict[str, np.ndarray] = {
        "train": X_full[:train_end],
        "val":   X_full[train_end:val_end],
        "test":  X_full[val_end:],
        "train_T": T_train, "val_T": T_val, "test_T": T_test,
        "train_R": R_train, "val_R": R_val, "test_R": R_test,
    }
    for k in ["timestamps", "columns", "train_end", "val_end", "scaler_mean", "scaler_std", "freq"]:
        if k in z:
            out_dict[k] = z[k]

    out_path = npz_path.with_name(npz_path.stem + args.out_suffix)
    np.savez_compressed(out_path, **out_dict)
    print(f"[OK] Saved: {out_path}")

    meta = {
        "trend_kind": args.trend,
        "ma_window": args.window,
        "ewma_alpha": (2.0 / (args.window + 1.0) if args.trend == "ewma" and args.alpha is None else args.alpha),
        "causal": True,
        "labels_space": "standardized",
        "note": "Trend computed causally on full timeline, then sliced back to splits."
    }
    meta_path = npz_path.parent / "tr_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved meta: {meta_path}")


if __name__ == "__main__":
    main()
