import argparse
import torch 
import numpy as np 
import pandas as pd
from pathlib import Path 
from src import preprocess, utils 
from torch.utils.data import DataLoader

def main():
  p = argparse.ArgumentParser()
  p.add_argument('--data_dir', type=str, required=True)
  p.add_argument('--seq_len', type=int, default=60)
  p.add_argument('--pred_len', type=int, default=10)
  p.add_argument('--stride', type=int, default=1)
  p.add_argument('--batch_size', type=int, default=64)
  p.add_argument('--epochs', type=int, default=20)
  p.add_argument('--seed', type=int, default=1337)
  p.add_argument('--lr', type=float, default=1e-3)
  p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
  p.add_argument('--out_dir', type=str, default=str(Path(__file__).resolve().parents[1] / 'outputs'))
  args = p.parse_args()

  utils.set_seed(args.seed)
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  data_dir = Path(args.data_dir)
  pairs = preprocess.find_pairs(data_dir)
  if not pairs:
    raise SystemExit("No HR/XeThru pairs found. Check folder structure and filenames.")
  sessions = []
  for hr_path, xe_path, activity, user in pairs:
    try:
      df = preprocess.build_session_dataframe(hr_path, xe_path, activity, user)
      break
    except:
      pass

if __name__ == "__main__":
  main()