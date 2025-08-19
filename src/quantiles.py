import torch 
import torch.nn as nn

class PinballLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.quantiles = [0.1, 0.5, 0.9]
    self.qs = torch.tensor(self.quantiles).view(1,1,-1) # (1,1,3)
  
  # yhat: (batch,pred_len,3)
  # h: (batch,pred_len)
  def forward(self, yhat, y):
    qs = self.qs.to(yhat.device)
    y = y.unsqueeze(-1) # (batch,pred_len,1)
    diff = y-yhat # broadcast => (batch,pred_len,Q)
    # qs:[0.1,0.5,0.9] => (1-qs):[0.9,0.5,0.1]
    # if y>yhat (diff>0) we choose <qs*diff>
    # if y<yhat (diff<0) we choose <(1-qs)*-diff>
    loss = torch.maximum(qs*diff, (1-qs)*(-diff))
    return loss.mean()

# (batch,pred_len,3) => pick middle channel (batch,pred_len)
def median_channel(yhat_3q):
  return yhat_3q[..., 1]