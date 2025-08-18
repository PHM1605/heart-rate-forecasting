import matplotlib, random, torch
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import torch 

def set_seed(seed=1337):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True 
  torch.backends.cudnn.benchmark = False 

def save_example_plot(frame, input_cols, norm, model, seq_len, pred_len, out_path):
  # check last 60s of ground-truth 
  # each <pred_len> is e.g. 10s => we must forecast 6 times (S=6)
  duration_to_check = 60
  if len(frame) >= duration_to_check:
    S = duration_to_check // pred_len
  else:
    S = (len(frame) - seq_len) // pred_len
 
  if S <= 0:
    print("[plot] Not enough data to make one horizon.")
    return

  # Use the *training* feature columns and order (from norm stats)
  if hasattr(norm['mean'], 'index'):
    train_cols = list(norm['mean'].index)
  else:
    train_cols = list(input_cols)  # fallback
  first_start = max(0, len(frame) - S*pred_len - seq_len) 

  ys, yhats = [], []
  device = next(model.parameters()).device 
  model.eval() 
  with torch.no_grad():
    for st in range(first_start, first_start+S*pred_len, pred_len):
      x = frame.iloc[st:st+seq_len][input_cols]
      x = (x-norm['mean']) / norm['std']
      x_t = torch.tensor(x.values, dtype=torch.float32, device=device).unsqueeze(0)
      yhat = model(x_t).squeeze(0).cpu().numpy()
      y_true = frame.iloc[st+seq_len : st+seq_len+pred_len]['hr'].values
      yhats.extend(list(yhat))
      ys.extend(list(y_true))
  # time at 0s, 10s, 20s ....
  ts = frame.iloc[first_start+seq_len : first_start+seq_len+len(ys)]['timestamp'].values 
  plt.figure()
  # plot BACK-TO-BACK. at 0th-second we plot 10 samples, then at 10th-second we plot next 10 samples...
  plt.plot(ts, ys, label="HR true")
  plt.plot(ts, yhats, label="HR pred", alpha=0.8)
  plt.xlabel("Time")
  plt.ylabel("HR (bpm)")
  plt.title("Test segment: ground truth vs prediction")
  plt.legend() 
  plt.tight_layout()
  plt.savefig(out_path, dpi=180)
