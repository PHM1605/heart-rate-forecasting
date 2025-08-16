import matplotlib, random, torch
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt 

def set_seed(seed=1337):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True 
  torch.backends.cudnn.benchmark = False 

def save_example_plot(frame, input_cols, norm, model, seq_len, pred_len, out_path):
  L = min(len(frame), 600)
