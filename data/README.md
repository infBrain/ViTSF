# 开源时序数据集集合

本目录收录了时序预测领域常用的开源数据集，并提供自动化下载脚本与元数据说明，方便快速复现实验。所有数据均保留在各自的原始来源，脚本仅用于便捷下载，请在使用前自行确认许可条款。

## 数据集列表

| 数据集 | 领域 | 频率 | 原始来源 | 许可 |
| --- | --- | --- | --- | --- |
| ETT (ETTh1/ETTh2/ETTm1/ETTm2) | 能源负载 | 小时/15分钟 | [Tsinghua ETT](https://github.com/zhouhaoyi/ETDataset) | 原始仓库许可 (MIT) |
| Electricity | 电力消耗 | 小时 | [UCI ML Repository](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) | 公开使用，需遵循 UCI 条款 |
| Traffic | 公路交通流量 | 小时 | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/PEMS-SF) | 公开使用，需遵循 UCI 条款 |
| Exchange Rate | 金融汇率 | 日 | [Lai et al.](https://github.com/laiguokun/multivariate-time-series-data) | 原始仓库许可 (MIT) |
| Weather | 气象观测 | 10分钟 | [NREL](https://www.nrel.gov/grid/solar-power-data.html) | 公开使用，需遵循 NREL 条款 |
| ILI | 流感指数 | 周 | [CDC](https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html) | 公开使用，需注明来源 |
| M4 | 商业+宏观混合 | 多频率 | [M4 Competition](https://www.mcompetitions.unic.ac.cy/the-dataset/) | 研究使用许可 |
| Solar | 太阳能发电 | 10分钟 | [NREL](https://www.nrel.gov/grid/solar-power-data.html) | 公开使用，需遵循 NREL 条款 |

## 使用方式

1. 安装依赖（仅需标准库，无额外包）。
2. 运行 `scripts/download_datasets.py` 中的命令以下载所需数据集：
   - 列出数据集：`python scripts/download_datasets.py --list`
   - 下载指定数据集：`python scripts/download_datasets.py --dataset ett`
   - 下载全部数据集：`python scripts/download_datasets.py --all`

脚本将自动在 `data/raw/<dataset_name>` 下创建目录并保存下载的压缩包或原始文件。

## 示例数据

`sample/` 目录提供了裁剪后的时间序列样本，便于快速测试数据管道。当前包含：
- `ETTh1_sample.csv`：`ETTh1.csv` 的前 200 条记录（含表头）。
- `ETTm1_sample.csv`：`ETTm1.csv` 的前 200 条记录（含表头）。
- `electricity_sample.txt`：`LD2011_2014.txt` 的前 200 条记录（保留原始分隔符）。

## 目录结构

```text
data/
├── README.md
├── metadata.json          # 数据集元信息
├── sample/
│   ├── ETTh1_sample.csv   # 小型 ETT 子集
│   ├── ETTm1_sample.csv   # 小型 ETT 子集
│   └── electricity_sample.txt
└── raw/                   # 下载脚本存放原始数据的位置
```

## 法律与许可声明

- 所有数据版权归原始提供方所有。
- 在使用任何数据集之前，请阅读并遵守对应的许可协议与引用要求。
- 部分数据集可能包含个人隐私或敏感信息，使用时请遵循数据治理与伦理规范。
