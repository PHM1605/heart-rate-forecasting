import torch.nn as nn

class LSTMForecaster(nn.Module):
  def __init__(self, feat_dim, pred_len, hidden=128, layers=2, dropout=0.1, n_quantiles=3):
    super().__init__()
    self.pred_len = pred_len
    self.nq = n_quantiles 
    self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=hidden, num_layers=layers, batch_first=True, dropout=dropout)
    self.head = nn.Linear(hidden, pred_len*n_quantiles)
  
  def forward(self, x):
    out, _ = self.lstm(x) # out: [batch, seq_len, hidden]
    out = self.head(out[:,-1,:]) # sub-out: [batch, hidden] => [batch, pred_len*n_quantiles]
    batch = out.shape[0]
    return out.view(batch, self.pred_len, self.nq)