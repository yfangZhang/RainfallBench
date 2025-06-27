import torch
import torch.nn as nn

class MixedLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9], alpha=0.5):
        super().__init__()
        self.quantiles = quantiles
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        
    def quantile_loss(self, preds, target):
        """
        计算多时间点的分位数损失
        输入:
            preds: [batch_size, num_timesteps, num_quantiles]
            target: [batch_size, num_timesteps, 1]
        """
        # 扩展目标维度以匹配预测
        target = target.expand(-1, -1, len(self.quantiles))  # [B, T, Q]
        errors = target - preds
        
        losses = []
        for i, q in enumerate(self.quantiles):
            loss = torch.max((q-1)*errors[..., i], q*errors[..., i])
            losses.append(loss)
            
        # 按时间步和分位数求和后取平均
        return torch.mean(torch.stack(losses, dim=-1).sum(dim=-1))
    
    def forward(self, preds, target):
        """
        输入:
            preds: [batch_size, num_timesteps, num_outputs]
                   num_outputs = len(quantiles) + 1
            target: [batch_size, num_timesteps, 1]
        """
        # 分离分位数预测和MSE预测
        quant_preds = preds[..., :-1]  # [B, T, Q]
        mse_preds = preds[..., -1]     # [B, T]
        
        # 计算分位数损失
        quant_loss = self.quantile_loss(quant_preds, target)
        
        # 计算MSE损失（所有时间点）
        mse_loss = self.mse_loss(mse_preds, target.squeeze(-1))
        
        return self.alpha * mse_loss + (1 - self.alpha) * quant_loss