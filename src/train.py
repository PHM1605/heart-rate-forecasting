import numpy as np 
import torch 
from torch import nn 
from src.quantiles import PinballLoss

def metrics_dict(y_true, y_pred):
  eps = 1e-8 
  mae = np.mean(np.abs(y_true - y_pred))
  rmse = np.sqrt(np.mean((y_true-y_pred)**2))
  mask = np.abs(y_true) > 1e-6 # to avoid divided by zero
  mape = np.mean( abs((y_true[mask]-y_pred[mask]) / (y_true[mask]+eps)) )*100.0 if mask.any() else np.nan
  # R^2
  ss_res = np.sum((y_true-y_pred)**2)
  ss_tot = np.sum((y_true - np.mean(y_true))**2) + eps 
  r2 = 1.0 - ss_res/ss_tot 
  return dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2)
  
def run_epoch(model, loader, device, optimizer=None):
  # train=> optimizer has a value
  is_train = optimizer is not None
  is_seq2seq = hasattr(model, "encoder") and hasattr(model, "decoder")
  model.train(is_train)
  # list of losses of each batch
  losses, y_all, yhat50_all = [], [], [] 

  qloss = PinballLoss()
  with torch.set_grad_enabled(is_train):
    print("DEVICE IS ", device)
    for x, y in loader:
      x = x.to(device)
      y = y.to(device)
      if is_train and is_seq2seq:
        yhat_3q = model(x, y_future=y) # (batch,pred_len,3) with teacher forcing
      else:
        yhat_3q = model(x) # (batch,pred_len,3)
      loss = qloss(yhat_3q, y)

      if is_train:
        optimizer.zero_grad()
        loss.backward() 
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step() 
      losses.append(loss.item())
      # for eval metrics: using p50 as point forecast
      if not is_train:
        y_all.append(y.detach().cpu().numpy())
        yhat50_all.append(median_channel(yhat_3q).detach().cpu().numpy())

  # ONE very long time series of ALL samples 
  avg_loss = float(np.mean(losses)) if losses else float('nan')
  if training:
    return avg_loss, None, None 
  else:
    y_all = np.concatenate(y_all, axis=0) if y_all else None
    yhat50_all = np.concatenate(yhat50_all, axis=0) if yhat50_all else None
  return avg_loss, y_all, yhat50_all 