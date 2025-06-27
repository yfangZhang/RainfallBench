import torch
import torch.nn as nn

def decompose(seq_x, zeromask):
    non_zero_indices = torch.where(~zeromask)[0]
    if len(non_zero_indices) == 0:
        return torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.float32)
    
    vals = seq_x[non_zero_indices].to(torch.float32)
    gaps = []
    prev_idx = -1
    for idx in non_zero_indices:
        if prev_idx == -1:
            gap = idx
        else:
            gap = idx - prev_idx - 1
        gaps.append(gap)
        prev_idx = idx
    gaps = torch.tensor(gaps, dtype=torch.float32)
    return gaps, vals
class DualLSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1):
        super().__init__()
        # Gap prediction LSTM
        self.gap_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.gap_fc = nn.Linear(hidden_dim, 1)
        
        # Value prediction LSTM
        self.val_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.val_fc = nn.Linear(hidden_dim, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        gap_input, val_input = decompose(x_enc,mask)
        # Gap branch
        gap_out, _ = self.gap_lstm(gap_input)
        gap_pred = self.gap_fc(gap_out[:, -self.pred_len:, :])  # (batch_size, 1)
        
        # Value branch
        val_out, _ = self.val_lstm(val_input)
        val_pred = self.val_fc(val_out[:, -self.pred_len:, :])   # (batch_size, 1)
        
        return val_pred.squeeze()/gap_pred.squeeze()