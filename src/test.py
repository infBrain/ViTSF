import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

data = np.load(PROCESSED_DIR / "weather" / "weather" / "data.npz")

print(data.files)  # 查看包含的数组名称

print(data["train"][:10])
print(data["val"][:10])
print(data["test"][:10])
print(data["timestamps"][:10])

data.close()  # 关闭文件句柄