import math, torch
import torch.nn as nn 

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=2048):
    super().__init__()
    # (2048,model_dim)
    pe = torch.zeros(max_len, d_model) 
    # (2048,)=[0,1,..,2047] => (2048,1)
    pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    # exp( [0,2,4...] * (-ln(10000)/model_dim) ) => (model_dim/2,)
    div = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
    # broadcast => (2048,model_dim/2)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    self.register_buffer('pe', pe.unsqueeze(0))

  def forward(self, x):
    L = x.size(1) # time
    return x + self.pe[:,:L,:]

class TransformerForecaster(nn.Module):
  def __init__(self, feat_dim, pred_len, d_model=128, nhead=4, num_layers=3, dim_ff=256, dropout=0.1, n_quantiles=3):
    super().__init__()
    self.pred_len = pred_len 
    self.nq = n_quantiles
    self.in_proj = nn.Linear(feat_dim, d_model)
    self.pos = PositionalEncoding(d_model)
    enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
    self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
    self.head = nn.Linear(d_model, pred_len*n_quantiles)

  def forward(self, x):
    # x: [batch,time,feat_dim]
    z = self.in_proj(x) # z: [batch,time,d_model]
    z = self.pos(z) # z: [batch,time,d_model]
    z = self.encoder(z) # z: [batch, time, d_model]
    out = self.head(z[:,-1,:]) # [batch, d_model] => [batch, pred_len*n_quantiles]
    batch = out.shape[0]
    return out.view(batch, self.pred_len, self.nq)