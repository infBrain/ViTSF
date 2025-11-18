# scripts/inspect_npz.py
# -*- coding: utf-8 -*-
"""
一个用于快速检查 .npz 文件内容的通用脚本。

它会打印出文件中每个数组的：
- 键名 (Key)
- 形状 (Shape)
- 数据类型 (Dtype)
- 少量样本数据

用法:
python -m scripts.inspect_npz [path_to_your_npz_file]

示例:
python -m scripts.inspect_npz data/processed/ett/ETTh1/data_with_TR.npz
"""
import numpy as np
import argparse
from pathlib import Path

def inspect_npz(file_path: Path):
    """加载 .npz 文件并打印其内容的详细信息。"""
    if not file_path.exists():
        print(f"❌ 错误：文件不存在 -> {file_path}")
        return

    print(f"🔍 正在检查: {file_path.name}")
    print("=" * 40)

    try:
        with np.load(file_path, allow_pickle=True) as data:
            keys = list(data.keys())
            print(f"包含的键 (Keys): {keys}\n")

            for key in keys:
                array = data[key]
                print(f"--- 键: '{key}' ---")
                print(f"  • 形状 (Shape): {array.shape}")
                print(f"  • 数据类型 (Dtype): {array.dtype}")

                # 根据维度打印不同格式的样本
                if array.ndim == 0:  # 标量
                    print(f"  • 值 (Value): {array}")
                elif array.ndim == 1:
                    sample = array[:5]
                    print(f"  • 样本 (前5个): {sample}")
                else:
                    sample = array[:3, :5] # 最多看3行5列
                    print(f"  • 样本 (前3行, 前5列):\n{sample}")
                print("-" * 20)

    except Exception as e:
        print(f"❌ 加载或读取文件时出错: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="检查 .npz 文件内容的工具。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "npz_file",
        type=str,
        help="要检查的 .npz 文件的路径。"
    )
    args = parser.parse_args()
    
    inspect_npz(Path(args.npz_file))

if __name__ == "__main__":
    main()
