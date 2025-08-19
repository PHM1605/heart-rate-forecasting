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
  def __init__(self, hidden=128, layers=2, dropout=0.1):
    super().__init__()
    self.inp = nn.Linear(1, hidden)
    self.lstm = nn.LSTM(input_size=hidden, hidden_size=hidden, num_layers=layers, batch_first=True, dropout=dropout)
    self.out = nn.Linear(hidden, 1)
  
  def forward(self, y_prev, hidden, cell):
    # y_prev: previous heart-rate; (batch,)
    dec_in = self.inp(y_prev.unsqueeze(1)) # (batch,1)=>(batch,hidden)
    dec_in = dec_in.unsqueeze(1) # (batch,1,hidden)
    # hidden, cell: (num_layers,batch,hidden) for both 
    # out: (batch,1,hidden)
    out, (hidden, cell) = self.lstm(dec_in, (hidden, cell))
    y = self.out(out.squeeze(1)).squeeze(-1)
    return y, hidden, cell 


class Seq2SeqLSTM(nn.Module):
  def __init__(self, feat_dim, pred_len, hidden=128, layers=2, dropout=0.1, teacher_forcing_ratio=0.5):
    super().__init__()
    self.encoder = Encoder(feat_dim, hidden, layers, dropout)
    self.decoder = Decoder(hidden, layers, dropout)
    self.pred_len = pred_len 
    self.teacher_forcing_ratio = teacher_forcing_ratio
  
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
      # y_t: (batch,)
      y_t, hidden, cell = self.decoder(y_prev, hidden, cell)
      preds.append(y_t.unsqueeze(-1)) # append (batch,1)
      if (y_future is not None) and (torch.rand(1, device=device) < self.teacher_forcing_ratio):
        y_prev = y_future[:,t]
      else:
        y_prev = y_t.detach()
    return torch.cat(preds, dim=1) # (batch,pred_len)

