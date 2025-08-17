import torch.nn as nn 

class MLP(nn.Module):
  def __init__(self, seq_len, feat_dim, pred_len, hidden=256, dropout=0.1):
    super().__init__()
    self.net = nn.Sequential(
      nn.Flatten(), # [time*dim]
      nn.Linear(seq_len*feat_dim, hidden),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hidden, hidden),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hidden, pred_len) # [pred_len]
    )
  
  def forward(self, x):
    return self.net(x)
    