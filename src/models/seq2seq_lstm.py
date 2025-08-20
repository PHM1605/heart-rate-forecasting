import torch 
import torch.nn as nn

class Encoder(nn.Module):
  def __init__(self, feat_dim, hidden=128, layers=2, dropout=0.1):
    super().__init__() 
    self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=hidden, num_layers=layers, batch_first=True, dropout=dropout)
  
  def forward(self, x):
    outputs, (hidden, cell) = self.lstm(x)
    return hidden, cell # (num_layers,batch,hidden) for both 

class Decoder(nn.Module):
  def __init__(self, hidden=128, layers=2, dropout=0.1, n_quantiles=3):
    super().__init__()
    self.inp = nn.Linear(1, hidden)
    self.lstm = nn.LSTM(input_size=hidden, hidden_size=hidden, num_layers=layers, batch_first=True, dropout=dropout)
    self.out = nn.Linear(hidden, n_quantiles)
    self.nq = n_quantiles
  
  def forward(self, y_prev, hidden, cell):
    # y_prev: previous heart-rate; (batch,)
    dec_in = self.inp(y_prev.unsqueeze(1)) # (batch,1)=>(batch,hidden)
    dec_in = dec_in.unsqueeze(1) # (batch,1,hidden)
    # hidden, cell: (num_layers,batch,hidden) for both 
    # out: (batch,1,hidden)
    out, (hidden, cell) = self.lstm(dec_in, (hidden, cell))
    y_q = self.out(out.squeeze(1)) # (batch, nq)
    return y_q, hidden, cell 


class Seq2SeqLSTM(nn.Module):
  def __init__(self, feat_dim, pred_len, hidden=128, layers=2, dropout=0.1, teacher_forcing_ratio=0.5, n_quantiles=3):
    super().__init__()
    self.encoder = Encoder(feat_dim, hidden, layers, dropout)
    self.decoder = Decoder(hidden, layers, dropout)
    self.pred_len = pred_len 
    self.teacher_forcing_ratio = teacher_forcing_ratio
    self.nq = n_quantiles
    self.mid_idx = 1 if n_quantiles==3 else (n_quantiles//2)
  
  # x: (batch,time,dim)
  # y_future, if provided: teacher-forcing used during training
  def forward(self, x, y_future=None):
    device = x.device 
    batch = x.size(0)
    hidden, cell = self.encoder(x) # (num_layers,batch,hidden) for both 
    # start_token: use last observed HR from the input features
    y_prev = x[:,-1,0] # feature 0 of x is heart rate, shape (batch,)
    preds = []
    for t in range(self.pred_len):
      # y_t: (batch,n_quantiles)
      y_q, hidden, cell = self.decoder(y_prev, hidden, cell)
      if self.nq > 1:
        y_q, _ = torch.sort(y_q, dim=-1) # ascending: p10<p50<p90
      preds.append(y_q.unsqueeze(1)) # append (batch,1,n_quantiles)
      if (y_future is not None) and (torch.rand(1, device=device) < self.teacher_forcing_ratio):
        y_prev = y_future[:,t]
      else:
        # Autoregressive: feedback the median (p50) prediction
        y_prev = y_q[:, self.mid_idx].detach()
    return torch.cat(preds, dim=1) # (batch,pred_len, n_quantiles)

