# src/model/causal_path.py
# -*- coding: utf-8 -*-
"""
ViTSF 模型的因果路径 (Causal Path)。
使用 TCN 和 Graph Attention Network 捕捉时序的残差信息。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import TemporalConvNet, GraphAttentionLayer

class CausalPath(nn.Module):
    """
    因果路径模型。
    - 包含一个可学习的邻接矩阵。
    - 使用 TCN 提取时间特征。
    - 使用 GAT 提取变量间关系。
    """
    def __init__(self, 
                 num_nodes: int, 
                 in_dim: int, 
                 out_dim: int,
                 pred_len: int,
                 tcn_channels: list[int],
                 tcn_kernel_size: int = 2,
                 tcn_dropout: float = 0.2,
                 gat_dropout: float = 0.2,
                 gat_alpha: float = 0.2):
        """
        Args:
            num_nodes (int): 节点数量，即变量数量。
            in_dim (int): 输入特征维度 (通常为历史窗口长度 L)。
            out_dim (int): TCN 输出的特征维度。
            pred_len (int): 预测长度 H。
            tcn_channels (list[int]): TCN 每层的通道数。
            tcn_kernel_size (int): TCN 卷积核大小。
            tcn_dropout (float): TCN 中的 dropout 率。
            gat_dropout (float): GAT 中的 dropout 率。
            gat_alpha (float): GAT LeakyReLU 的负斜率。
        """
        super(CausalPath, self).__init__()
        self.num_nodes = num_nodes
        self.pred_len = pred_len

        # 可学习的邻接矩阵
        self.adj = nn.Parameter(torch.randn(num_nodes, num_nodes))

        # TCN 用于时间特征提取
        self.tcn = TemporalConvNet(in_dim, tcn_channels, kernel_size=tcn_kernel_size, dropout=tcn_dropout)
        
        # GAT 用于变量关系提取
        # 注意：这里的 GAT 实现是一个简化版本，实际应用中可能需要更复杂的结构
        # 这里的 in_features 是 TCN 输出的最后一个 channel 数
        self.gat = GraphAttentionLayer(tcn_channels[-1], out_dim, dropout=gat_dropout, alpha=gat_alpha, concat=True)

        # 最终的预测头
        self.prediction_head = nn.Linear(out_dim, pred_len)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): 输入的时间序列，shape: (N, L, D)，其中 D 是变量数 (num_nodes)
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: 
                - 预测的残差，shape: (N, H, D)
                - 归一化后的邻接矩阵，shape: (D, D)
        """
        # x shape: (N, L, D)
        N, L, D = x.shape
        
        # 1. TCN 处理时间维度
        # TCN 需要 (N, C, L) 的输入，这里 C 是变量数 D，L 是历史长度
        x_tcn_in = x.permute(0, 2, 1) # -> (N, D, L)
        
        # 为了让每个变量有自己的 TCN，我们将 N 和 D 合并
        x_tcn_in = x_tcn_in.reshape(N * D, 1, L) # -> (N*D, 1, L)
        
        # 经过 TCN 并在时间维度上进行池化，得到每个节点的紧凑表示
        tcn_out = self.tcn(x_tcn_in) # -> (N*D, tcn_channels[-1], L)
        tcn_out = tcn_out.mean(dim=-1) # -> (N*D, tcn_channels[-1])
        
        # 恢复形状
        tcn_out = tcn_out.view(N, D, -1) # -> (N, D, tcn_channels[-1])

        # 2. GAT 处理变量维度
        # GAT 需要 (num_nodes, in_features) 的输入
        # 我们对 batch 中的每个样本都进行 GAT 计算
        
        # 归一化邻接矩阵
        adj_normalized = F.softmax(F.relu(self.adj), dim=1)
        
        gat_outputs = []
        for i in range(N):
            # h shape: (D, tcn_channels[-1])
            h = tcn_out[i]
            # adj_normalized shape: (D, D)
            gat_out = self.gat(h, adj_normalized) # -> (D, out_dim)
            gat_outputs.append(gat_out)
        
        gat_out_batch = torch.stack(gat_outputs) # -> (N, D, out_dim)

        # 3. 预测头
        residual_pred = self.prediction_head(gat_out_batch) # -> (N, D, H)
        
        # 调整为 (N, H, D)
        residual_pred = residual_pred.permute(0, 2, 1)
        
        return residual_pred, adj_normalized

if __name__ == '__main__':
    # 测试 CausalPath 模型
    N, L, D = 16, 96, 7  # Batch, History Len, Num Vars
    H = 24 # Pred Len
    
    dummy_series = torch.randn(N, L, D)
    
    model = CausalPath(
        num_nodes=D,
        in_dim=1, # TCN 输入通道为1
        out_dim=64, # GAT 输出维度
        pred_len=H,
        tcn_channels=[32, 64], # TCN 通道数
    )
    
    print("模型结构:")
    # print(model) # 打印 GAT 会很长
    
    output, adj = model(dummy_series)
    
    print(f"\n输入 shape: {dummy_series.shape}")
    print(f"输出 shape: {output.shape}")
    print(f"邻接矩阵 shape: {adj.shape}")
    
    assert output.shape == (N, H, D)
    assert adj.shape == (D, D)
    print("✅ CausalPath 测试通过!")
