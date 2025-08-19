import torch.nn as nn 

class MLP(nn.Module):
  def __init__(self, seq_len, feat_dim, pred_len, hidden=256, dropout=0.1, n_quantiles=3):
    super().__init__()
    self.pred_len = pred_len 
    self.nq = n_quantiles
    self.net = nn.Sequential(
      nn.Flatten(), # [time*dim]
      nn.Linear(seq_len*feat_dim, hidden),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hidden, hidden),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hidden, pred_len*n_quantiles) # [pred_len*n_quantiles]
    )
  
  def forward(self, x):
    out = self.net(x)
    batch = out.shape[0]
    return out.view(batch, self.pred_len, self.nq)
    