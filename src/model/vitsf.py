# src/model/vitsf.py
# -*- coding: utf-8 -*-
"""
ViTSF (Vision Transformer for Time Series Forecasting) 主模型。
融合了视觉路径 (ViTPath) 和因果路径 (CausalPath)。
"""
import torch
import torch.nn as nn
from .vit_path import ViTPath
from .causal_path import CausalPath

class ViTSF(nn.Module):
    """
    ViTSF 主模型。
    """
    def __init__(self,
                 # ViT Path Args
                 vit_model_name: str = 'vit_base_patch16_224',
                 vit_pretrained: bool = True,
                 vit_in_chans: int = 3,
                 
                 # Causal Path Args
                 num_nodes: int = 7,
                 causal_in_dim: int = 1,
                 causal_out_dim: int = 64,
                 tcn_channels: list[int] = [32, 64],
                 
                 # Common Args
                 pred_len: int = 24,
                 d_model: int = 768, # ViT 特征维度
                 
                 # Fusion
                 fusion_mode: str = 'add'):
        """
        Args:
            vit_model_name (str): ViT 模型名称。
            vit_pretrained (bool): ViT 是否预训练。
            vit_in_chans (int): 图像通道数。
            num_nodes (int): 变量/节点数量。
            causal_in_dim (int): 因果路径 TCN 的输入维度。
            causal_out_dim (int): 因果路径 GAT 的输出维度。
            tcn_channels (list[int]): TCN 通道列表。
            pred_len (int): 预测长度。
            d_model (int): ViT 的特征维度。
            fusion_mode (str): 融合方式, 'add' 或 'gate'。
        """
        super(ViTSF, self).__init__()
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.fusion_mode = fusion_mode

        # 1. 视觉路径
        self.vit_path = ViTPath(
            model_name=vit_model_name,
            pretrained=vit_pretrained,
            in_chans=vit_in_chans,
            pred_len=pred_len,
            d_model=d_model
        )

        # 2. 因果路径
        self.causal_path = CausalPath(
            num_nodes=num_nodes,
            in_dim=causal_in_dim,
            out_dim=causal_out_dim,
            pred_len=pred_len,
            tcn_channels=tcn_channels
        )
        
        # 3. 融合层 (如果需要)
        if self.fusion_mode == 'gate':
            # 门控融合: gamma = sigmoid(Linear(features))
            # 使用 ViT 的 CLS token 作为上下文特征
            self.fusion_gate = nn.Sequential(
                nn.Linear(d_model, pred_len * num_nodes),
                nn.Sigmoid()
            )

    def forward(self, 
                x_image: torch.Tensor, 
                x_series: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_image (torch.Tensor): 视觉路径的输入图像, shape (N, C, H, W)。
            x_series (torch.Tensor): 因果路径的输入序列, shape (N, L, D)。

        Returns:
            tuple: (最终预测, 趋势预测, 残差预测, 邻接矩阵)
        """
        # 1. 视觉路径预测趋势
        # 注意：ViTPath 的输出是单变量的，需要扩展到多变量
        trend_pred_single = self.vit_path(x_image) # (N, H, 1)
        trend_pred = trend_pred_single.repeat(1, 1, self.num_nodes) # (N, H, D)

        # 2. 因果路径预测残差
        residual_pred, adj = self.causal_path(x_series) # (N, H, D), (D, D)

        # 3. 融合
        if self.fusion_mode == 'add':
            final_pred = trend_pred + residual_pred
        elif self.fusion_mode == 'gate':
            # 从 ViT 获取 CLS token 用于门控
            # 这部分代码依赖于 ViTPath 的内部实现，需要确保一致
            with torch.no_grad(): # 门控的特征提取不参与梯度计算
                features = self.vit_path.vit.forward_features(x_image)
                if features.ndim == 3:
                    cls_token = features[:, 0]
                else:
                    cls_token = features
            
            gamma = self.fusion_gate(cls_token).view(-1, self.pred_len, self.num_nodes) # (N, H, D)
            final_pred = trend_pred + gamma * residual_pred
        else:
            raise ValueError(f"未知的融合模式: {self.fusion_mode}")

        return final_pred, trend_pred, residual_pred, adj

if __name__ == '__main__':
    # 测试 ViTSF 主模型
    N, L, D = 16, 96, 7
    H = 24
    C, IMG_H, IMG_W = 3, 224, 224

    dummy_image = torch.randn(N, C, IMG_H, IMG_W)
    dummy_series = torch.randn(N, L, D)

    # 测试加性融合
    print("--- 测试加性融合 (add) ---")
    model_add = ViTSF(num_nodes=D, pred_len=H, fusion_mode='add')
    final_pred, trend, res, adj = model_add(dummy_image, dummy_series)
    
    print(f"输入图像 shape: {dummy_image.shape}")
    print(f"输入序列 shape: {dummy_series.shape}")
    print(f"最终预测 shape: {final_pred.shape}")
    print(f"趋势预测 shape: {trend.shape}")
    print(f"残差预测 shape: {res.shape}")
    print(f"邻接矩阵 shape: {adj.shape}")
    
    assert final_pred.shape == (N, H, D)
    print("✅ 加性融合测试通过!")

    # 测试门控融合
    print("\n--- 测试门控融合 (gate) ---")
    model_gate = ViTSF(num_nodes=D, pred_len=H, fusion_mode='gate')
    final_pred_g, _, _, _ = model_gate(dummy_image, dummy_series)
    assert final_pred_g.shape == (N, H, D)
    print("✅ 门控融合测试通过!")

