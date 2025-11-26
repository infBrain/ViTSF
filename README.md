# ViTSF: A Two-Path Framework for Time Series Forecasting

本仓库实现了一个创新的**双路径**时间序列预测框架（ViTSF），其核心思想是**将时间序列的“形状”预测与“值”预测分离**，通过两条并行的路径分别建模，最后融合信息以提升预测精度。

## 算法思路

传统模型通常试图在一个单一模型中同时学习时间序列的复杂模式（形状）和数值大小（值），这可能导致顾此失彼。本项目将该问题分解为两个更专注的子任务，并通过双路径架构解决。

### 1. 双路径架构：形状与值的分离

我们的框架包含两条核心路径：

- **ViT 路径 (Shape Modeling)**：专注于学习和预测时间序列曲线的**形状**或**模式**。
- **因果路径 (Value Modeling)**：专注于预测目标变量的**绝对值**或**量级**。

通过这种方式，模型可以更精准地捕捉不同层面的信息，避免单一模型在复杂任务下的学习混淆。

### 2. Path 1: 使用 Vision Transformer (ViT) 进行形状建模

此路径的目标是理解序列的动态模式，而不受其绝对数值大小的干扰。

#### 引入频域信息：从一维序列到二维图像（推荐策略）

为了更精准地捕捉“形状”，一个非常有效的策略是引入**频域信息**。
- **动机**：时间序列的“形状”很大程度上由其内在的**周期性**决定。频域分析（如傅里叶变换）能将这些周期性成分**显式地**提取出来，为模型提供极强的特征信号。
- **实现**：通过**短时傅里叶变换 (STFT)** 或**小波变换 (CWT)**，我们可以将一维的时间序列窗口转换成一张二维的**时频图（Spectrogram / Scalogram）**。这张图展示了信号的频率如何随时间演变，本身就是一张信息丰富的“图像”。
- **优势**：将这张二维时频图作为 ViT 的输入，可以让 ViT **像处理真实图像一样**去捕捉复杂的时频模式，这比直接处理一维序列更能发挥其强大的模式识别能力。

#### ViT 模型
- **输入**：二维时频图（推荐）或经过标准化的原始一维序列。
- **分块 (Patching)**：将输入（无论是二维图还是一维序列）切分成若干个“块”(Patch)。
- **特征嵌入**：将每个块线性映射为特征向量 (Token)，并加入位置编码。
- **Transformer 编码器**：利用自注意力机制捕捉序列的长期依赖和全局模式。
- **输出**：此路径输出的是关于序列未来“形状”的深度表征。

### 3. Path 2: 使用因果模型进行值建模

此路径的目标是精准预测目标变量的**数值**。

- **思路**：采用**多变量预测单变量**的策略。它利用所有可用的相关变量（多变量输入）来预测目标变量的未来值。
- **因果性**：模型设计确保在预测 `t` 时刻的值时，只使用 `t` 时刻及之前的历史信息，保证无未来数据泄漏。
- **输出**：此路径输出对目标变量未来**绝对值**的直接预测。

### 4. 信息融合

最后，我们将两条路径的输出进行有效融合，以产生最终的预测结果。

- **融合层**：通过一个简单的**线性层**或其他可学习的模块，将“形状”路径的模式表征与“值”路径的数值预测相结合。
- **最终预测**：融合后的信息能够同时兼顾序列的宏观模式和精确的数值量级，从而实现比单一路径更准确的预测。

## 数据处理流程

为了支撑上述双路径模型，数据处理流程至关重要：

1.  **原始数据预处理 (`preprocess.py`)**:
    - 读取原始 CSV 文件，处理缺失值。
    - **标准化**：**仅使用训练集**的均值和标准差来标准化整个数据集，为“形状”建模提供纯净的模式输入。
    - 将处理好的数据保存为 `.npz` 格式。

2.  **（可选）趋势-残差标签生成 (`make_tr_labels.py`)**:
    - 此脚本可将标准化后的序列进一步分解为**趋势 (T)** 和**残差 (R)** 分量。
    - 这些分解后的分量可以作为“形状”路径的更优输入，让 ViT 更专注于学习特定模式。
    - 分解过程同样遵循**因果**原则，通过在完整拼接的序列上计算，避免了边界效应。

3.  **构建 PyTorch 数据集 (`dataset.py`)**:
    - `SlidingWindowDataset` 类负责从处理好的 `.npz` 文件中，根据指定的输入长度和预测范围，生成滑动窗口样本，供双路径模型训练。

## 如何运行

1.  **下载数据集**
    ```bash
    python scripts/download_datasets.py
    ```

2.  **预处理数据**
    ```bash
    # 以 ETT 数据集为例
    python src/data/preprocess.py ett
    ```

3.  **（可选）生成趋势-残差标签**
    ```bash
    python src/data/make_tr_labels.py --npz data/processed/ett/ETTh1/data.npz
    ```

4.  **训练模型**
    ```bash
    # 基础示例：使用缓存图像，并在训练结束后立即跑一次测试
    python train.py \
        --dataset-path data/processed/ett/ETTh1/data_with_TR.npz \
        --cache-dir cache/images \
        --batch-size 8 \
        --max-epochs 5 \
        --checkpoint-dir checkpoints/full_run \
        --test-after-train --test-ckpt-path best
    ```

## 训练脚本参数说明 (`train.py`)

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--dataset-path` | 是 | `data/processed/ett/ETTh1/data_with_TR.npz` | 预处理后的 `.npz` 数据（需包含 `train/val/test` 及 `T/R` 键）。 |
| `--cache-dir` | 否 | `None` | 若提供，将渲染好的 ViT 输入缓存为 `.pt`（按 split+起点命名），训练和可视化可复用。 |
| `--seq-len` / `--pred-len` / `--stride` | 否 | `336 / 96 / 1` | 滑动窗口相关超参。 |
| `--batch-size` / `--num-workers` | 否 | `8 / 4` | DataLoader 设置；显存不够可调低 batch。 |
| `--max-epochs` | 否 | `5` | 训练轮数。 |
| `--learning-rate` / `--weight-decay` | 否 | `1e-4 / 1e-5` | AdamW 优化器参数。 |
| `--lr-scheduler` | 否 | `cosine` | 支持 `none/cosine/plateau`。结合 `--lr-t-max/--lr-min/--lr-factor/--lr-patience` 进一步控制。 |
| `--fusion-mode` | 否 | `gate` | 双路径融合方式，`add` 或 `gate`。 |
| `--early-stop-patience` / `--early-stop-min-delta` | 否 | `10 / 1e-4` | >0 时启用 EarlyStopping（监控 `--monitor-metric`）。 |
| `--monitor-metric` / `--monitor-mode` | 否 | `val_total_loss / min` | ModelCheckpoint & EarlyStopping 共同引用的指标。 |
| `--checkpoint-dir` | 否 | `checkpoints` | 保存最优/最后模型的目录。`--checkpoint-save-top-k`、`--checkpoint-save-last` 控制保留策略。 |
| `--fast-dev-run` | 否 | `False` | Lightning 的调试模式，只跑一两个 batch，不会写 checkpoint。 |
| `--accelerator` / `--devices` | 否 | `auto / auto` | Lightning 硬件配置，可指定 `cpu`、`gpu`、`devices=4` 等。 |
| `--test-after-train` | 否 | `False` | 训练结束后立即运行 `trainer.test()`。 |
| `--test-only` | 否 | `False` | 跳过训练，直接测试（需搭配 `--test-ckpt-path` 给出 `.ckpt`）。 |
| `--test-ckpt-path` | 否 | `best` | `trainer.test()` 使用的 checkpoint。可填 `best`、`last` 或具体路径。 |

> 常见组合：
> - **标准训练 + 自动测试**：`--cache-dir cache/images --checkpoint-dir checkpoints/full_run --test-after-train`
> - **仅测试已有模型**：`--test-only --test-ckpt-path checkpoints/full_run/vitsf-epoch=03.ckpt`

## 评估脚本 (`scripts/evaluate_model.py`)

用于离线评测任意 checkpoint（支持 train/val/test split），并输出完整指标与路径分解图。

### 基础命令

```bash
python scripts/evaluate_model.py \
    --checkpoint checkpoints/full_run/vitsf-epoch=03.ckpt \
    --dataset-path data/processed/ett/ETTh1/data_with_TR.npz \
    --cache-dir cache/images \
    --split test \
    --batch-size 8 --num-workers 4 \
    --output-dir eval_outputs/full_test
```

### 参数列表

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--checkpoint` | 是 | 无 | 要评测的 `.ckpt` 文件路径。 |
| `--dataset-path` | 否 | `data/processed/ett/ETTh1/data_with_TR.npz` | 与训练阶段一致的数据文件。 |
| `--cache-dir` | 否 | `None` | 若训练时开启缓存，应传同一路径以复用。 |
| `--split` | 否 | `test` | 可选 `train/val/test`。 |
| `--batch-size` / `--num-workers` | 否 | `8 / 4` | 推理批大小及 DataLoader worker。 |
| `--seq-len` / `--pred-len` / `--stride` / `--image-size` | 否 | 训练默认值 | 应与训练时匹配。 |
| `--device` | 否 | `auto` | `cpu`、`cuda` 或自动。 |
| `--plot-node` | 否 | `0` | 生成图像时关注的节点索引。 |
| `--limit-batches` | 否 | `None` | 只评测若干 batch 以快速 sanity check。 |
| `--output-dir` | 否 | `eval_outputs` | 指标与图像的输出目录。 |

### 产物

- `metrics.json`：包含 **Final**（融合输出）、**Trend Path**（ViT 通道）、**Residual Path**（因果通道）的 MAE/MSE/RMSE/MAPE/SMAPE/R²。
- `prediction_overview_node{k}.png`：三行子图展示真值与各路径预测，可直观比较路径贡献。

> 若已在 `train.py` 中用 `--test-after-train`，Lightning 会直接调用 `trainer.test()` 并在日志中输出相同指标；`evaluate_model.py` 适合更灵活的离线分析（自定义节点、limit-batches、不同输出目录等）。