import torch.nn as nn 

class Chomp1d(nn.Module):
  def __init__(self, chomp_size):
    super().__init__()
    self.chomp_size = chomp_size 
  
  def forward(self, x):
    return x[:,:,:-self.chomp_size].contiguous()

class TCNBlock(nn.Module):
  def __init__(self, in_ch, out_ch, k=3, dilation=1, dropout=0.1):
    super().__init__()
    pad = (k-1)*dilation # padding=2
    self.net = nn.Sequential(
      nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation),
      Chomp1d(pad),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation),
      Chomp1d(pad),
      nn.ReLU(),
      nn.Dropout(dropout)
    )
    self.down = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
  
  def forward(self, x):
    return self.net(x) + self.down(x)

class TCN(nn.Module):
  def __init__(self, feat_dim, pred_len, channels=(64,64,64), k=3, dropout=0.1):
    super().__init__()
    layers = []
    in_ch = feat_dim 
    for i, ch in enumerate(channels):
      layers.append(TCNBlock(in_ch, ch, k=k, dilation=2**i, dropout=dropout))
      in_ch = ch 
    self.tcn = nn.Sequential(*layers)
    self.head = nn.Linear(in_ch, pred_len)
  
  def forward(self, x):
    x = x.transpose(1,2) # [batch,time,feat]=>[batch,feat,time]
    y = self.tcn(x) # [batch,channels,time]
    last = y[:,:,-1] # [batch,channels]
    return self.head(last) # [batch,pred_len]