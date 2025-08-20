from torch.utils.data import Dataset 
import torch 
import pandas as pd 

class WindowDataset(Dataset):
  def __init__(self, frames, input_cols, target_col, seq_len=60, pred_len=10, stride=1, norm_stats=None, fit_stats=False):
    self.samples = [] 
    self.frames = frames
    self.input_cols = input_cols 
    self.target_col = target_col
    self.seq_len = seq_len 
    self.pred_len = pred_len 
    self.stride = stride
    # fit_stats True always goes with norm_stats=None (i.e. we'll calculate norm_stats) and vice versa
    if fit_stats:
      concat = pd.concat([f[input_cols] for f in frames], axis=0)
      self.mean = concat.mean()
      self.std = concat.std().replace(0, 1.0)
    else:
      self.mean = norm_stats['mean']
      self.std = norm_stats['std']
    
    self.target_mean = float(self.mean[self.target_col])
    self.target_std = float(self.std[self.target_col] if self.std[self.target_col] != 0 else 1.0)

    for frame_id, frame in enumerate(frames):
      n = len(frame)
      max_start = n-(seq_len+pred_len)
      for start in range(0, max_start+1, stride):
        self.samples.append((frame_id, start))
  
  def __len__(self):
    return len(self.samples)
  
  def __getitem__(self, idx):
    frame_id, start = self.samples[idx]
    frame = self.frames[frame_id]
    x = frame.iloc[start:start+self.seq_len][self.input_cols]
    y = frame.iloc[start+self.seq_len : start+self.seq_len+self.pred_len][self.target_col]
    x = (x-self.mean) / self.std 
    y = (y-self.target_mean) / self.target_std
    x = torch.tensor(x.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32)
    return x, y
