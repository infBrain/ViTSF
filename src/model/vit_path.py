# src/model/vit_path.py
# -*- coding: utf-8 -*-
"""
ViTSF 模型的视觉路径 (Visual Path)。
使用预训练的 Vision Transformer (ViT) 提取图像特征，并预测时间序列的趋势。
"""
import torch
import torch.nn as nn
import timm

class ViTPath(nn.Module):
    """
    视觉路径模型。
    - 加载预训练的 ViT 模型。
    - 替换掉 ViT 的头部，以适应趋势预测任务。
    """
    def __init__(self, 
                 model_name: str = 'vit_base_patch16_224', 
                 pretrained: bool = True, 
                 num_classes: int = 0, # 移除原始分类头
                 in_chans: int = 3, # 输入图像通道数
                 pred_len: int = 24,
                 d_model: int = 768):
        """
        Args:
            model_name (str): timm 库中的 ViT 模型名称。
            pretrained (bool): 是否加载预训练权重。
            num_classes (int): 原始分类头的类别数，设为0以移除它。
            in_chans (int): 输入图像的通道数。
            pred_len (int): 预测长度。
            d_model (int): ViT 模型的特征维度。
        """
        super(ViTPath, self).__init__()
        self.pred_len = pred_len
        self.d_model = d_model

        # 加载预训练的 ViT 模型
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes, # 移除分类头
            in_chans=in_chans
        )
        
        # 冻结 ViT 的大部分层 (可选)
        # for name, param in self.vit.named_parameters():
        #     if 'head' not in name:
        #         param.requires_grad = False

        # 新的预测头
        self.prediction_head = nn.Linear(self.d_model, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 输入的图像张量，shape: (N, C, H, W)
        
        Returns:
            torch.Tensor: 预测的趋势，shape: (N, pred_len, 1)
        """
        # x shape: (N, C, H, W)
        # 从 ViT 获取特征 (CLS token)
        # timm < 0.9.0, features = self.vit.forward_features(x)
        # timm >= 0.9.0, features = self.vit.forward_features(x)[:, 0]
        features = self.vit.forward_features(x)
        
        # 不同的 timm 版本，forward_features 的输出可能不同
        if features.ndim == 3: # (N, num_patches + 1, d_model)
            cls_token = features[:, 0] # 取 [CLS] token
        else: # (N, d_model)
            cls_token = features

        # cls_token shape: (N, d_model)
        
        # 通过预测头得到预测结果
        trend_pred = self.prediction_head(cls_token) # shape: (N, pred_len)
        
        return trend_pred.unsqueeze(-1) # shape: (N, pred_len, 1)

if __name__ == '__main__':
    # 测试 ViTPath 模型
    dummy_image = torch.randn(16, 3, 224, 224) # (N, C, H, W)
    
    # 假设预测长度为 96
    pred_len = 96
    
    model = ViTPath(pred_len=pred_len)
    
    print("模型结构:")
    print(model)
    
    output = model(dummy_image)
    
    print(f"\n输入 shape: {dummy_image.shape}")
    print(f"输出 shape: {output.shape}")
    assert output.shape == (16, pred_len, 1)
    print("✅ ViTPath 测试通过!")
