import numpy as np 
import torch 
from torch import nn 

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
  model.train(is_train)
  # list of losses of each batch
  losses, y_all, yhat_all = [], [], [] 

  crit = nn.MSELoss() 
  for x, y in loader:
    x = x.to(device)
    y = y.to(device)
    if is_train:
      optimizer.zero_grad()
    yhat = model(x)
    loss = crit(yhat, y)
    if is_train:
      loss.backward() 
      nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step() 
    losses.append(loss.item())
    y_all.append(y.detach().cpu().numpy())
    yhat_all.append(yhat.detach().cpu().numpy())
  # ONE very long time series of ALL samples 
  y_all = np.concatenate(y_all, axis=0) if y_all else np.zeros((0,1))
  yhat_all = np.concatenate(yhat_all, axis=0) if yhat_all else np.zeros((0,1))
  return float(np.mean(losses)) if losses else float('nan'), y_all, yhat_all 