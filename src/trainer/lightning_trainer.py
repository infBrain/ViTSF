from __future__ import annotations

from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.model.vitsf import ViTSF

class ViTSFLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_kwargs,
        lr=1e-4,
        weight_decay=1e-5,
        loss_weights={'trend': 1.0, 'residual': 1.0},
        scheduler_config: Dict[str, Any] | None = None,
    ):
        """
        PyTorch Lightning Module for the ViTSF model.

        Args:
            model_kwargs (dict): Arguments for the ViTSF model.
            lr (float): Learning rate.
            weight_decay (float): Weight decay for the optimizer.
            loss_weights (dict): Weights for the trend and residual losses.
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.model = ViTSF(**model_kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Define loss functions
        self.trend_loss_fn = nn.MSELoss()
        self.residual_loss_fn = nn.HuberLoss() # Huber loss is robust to outliers
        
        self.loss_weights = loss_weights
        self.scheduler_config = scheduler_config

    def forward(self, x_img, x_ts):
        return self.model(x_img, x_ts)

    def _calculate_loss(self, batch, stage='train'):
        """
        Generic step to calculate loss for training, validation, or testing.
        """
        # Assuming batch is a tuple (x_img, x_ts, y_trend, y_residual)
        # You will need to adapt this based on your Dataset implementation
        x_img, x_ts, y_trend, y_residual = batch
        
        # Forward pass
        final_pred, trend_pred, residual_pred, adj = self.forward(x_img, x_ts)
        
        # Calculate losses
        trend_loss = self.trend_loss_fn(trend_pred, y_trend)
        residual_loss = self.residual_loss_fn(residual_pred, y_residual)
        
        # Weighted total loss
        total_loss = (self.loss_weights['trend'] * trend_loss + 
                      self.loss_weights['residual'] * residual_loss)
        
        # Logging
        self.log(f'{stage}_total_loss', total_loss, prog_bar=True)
        self.log(f'{stage}_trend_loss', trend_loss)
        self.log(f'{stage}_residual_loss', residual_loss)
        
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, stage='train')

    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch, stage='val')

    def test_step(self, batch, batch_idx):
        return self._calculate_loss(batch, stage='test')

    def configure_optimizers(self):
        """Configure optimizer and the requested LR scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if not self.scheduler_config or self.scheduler_config.get("type", "none") == "none":
            return optimizer

        cfg = self.scheduler_config
        sched_type = cfg.get("type", "none").lower()

        if sched_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.get("t_max", 10),
                eta_min=cfg.get("eta_min", 0.0),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }
        if sched_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=cfg.get("mode", "min"),
                patience=cfg.get("patience", 5),
                factor=cfg.get("factor", 0.5),
                min_lr=cfg.get("min_lr", 0.0),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": cfg.get("monitor", "val_total_loss"),
                },
            }

        raise ValueError(f"Unsupported scheduler type: {sched_type}")

if __name__ == '__main__':
    # To run this test, execute from the root directory:
    # python -m src.trainer.lightning_trainer

    # This is a simple test to check if the Lightning module can be instantiated.
    # A full test would require a dummy DataLoader.
    
    # Basic data dimensions for the dummy test
    seq_len = 336
    pred_len = 96
    num_nodes = 7 # Example for electricity dataset

    # Model configuration that matches ViTSF.__init__ signature
    model_config = {
        'pred_len': pred_len,
        'num_nodes': num_nodes,
        'vit_model_name': 'vit_base_patch16_224',
        'vit_pretrained': True,
        'vit_in_chans': 3,
        'causal_in_dim': 1,
        'causal_out_dim': 64,
        'tcn_channels': [32, 64],
        'd_model': 768,
        'fusion_mode': 'gate'
    }

    # Instantiate the Lightning Module
    lightning_module = ViTSFLightningModule(model_kwargs=model_config)
    
    print("ViTSFLightningModule instantiated successfully!")
    print("Model architecture:")
    print(lightning_module.model)

    # You can also try a dummy forward pass if you create dummy data
    # Dummy data
    batch_size = 16
    x_img_dummy = torch.randn(batch_size, 3, 224, 224)
    x_ts_dummy = torch.randn(batch_size, seq_len, num_nodes)
    y_trend_dummy = torch.randn(batch_size, pred_len, num_nodes)
    y_residual_dummy = torch.randn(batch_size, pred_len, num_nodes)
    
    dummy_batch = (x_img_dummy, x_ts_dummy, y_trend_dummy, y_residual_dummy)
    
    print("\nTesting a training step...")
    loss = lightning_module.training_step(dummy_batch, 0)
    print(f"Calculated loss: {loss.item()}")
    print("✅ Lightning module test passed!")

