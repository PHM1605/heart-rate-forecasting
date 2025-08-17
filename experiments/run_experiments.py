import argparse
import torch 
import numpy as np 
import pandas as pd
from pathlib import Path 
from src import preprocess, utils 
from src.dataset import WindowDataset
from src.models import LSTMForecaster, MLP
from src.train import run_epoch 
from torch.utils.data import DataLoader

def make_loaders(train_frames, val_frames, test_frames, input_cols, target_col,
  seq_len, pred_len, batch_size, stride):
  train_ds = WindowDataset(train_frames, input_cols, target_col, seq_len, pred_len, stride, norm_stats=None, fit_stats=True)
  norm = {"mean": train_ds.mean, "std": train_ds.std}
  val_ds = WindowDataset(val_frames, input_cols, target_col, seq_len, pred_len, stride, norm_stats=norm, fit_stats=False)
  test_ds = WindowDataset(test_frames, input_cols, target_col, seq_len, pred_len, stride, norm_stats=norm, fit_stats=False)
  loaders = {
    'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True), # drop_last mean what to do if #lastSamples != batch_size
    'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False),
    'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
  }
  return loaders, norm 

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
  # (path-sensor-1, path-sensor-2, activity-type, user-x)
  pairs = preprocess.find_pairs(data_dir)
  if not pairs:
    raise SystemExit("No HR/XeThru pairs found. Check folder structure and filenames.")
  sessions = []
  for hr_path, xe_path, activity, user in pairs:
    try:
      # df: <user1> <resting> <hr-data> <xe-data>
      df = preprocess.build_session_dataframe(hr_path, xe_path, activity, user)
      if len(df) >= (args.seq_len + args.pred_len + 1):
        sessions.append(df)
    except Exception as e:
      print(f"[WARN] Skipping pair {hr_path.name} / {xe_path.name}: {e}")
  # *[(), ()] => [...]
  all_cols = sorted(set().union(*[set(df.columns) for df in sessions]))
  xe_cols = [c for c in all_cols if c.startswith('xe')]
  input_cols = ['hr'] + xe_cols 
  target_col = 'hr'

  # user independent split
  users = sorted({df['user'].iloc[0] for df in sessions}) # ["user11", "user12", "user13", "user14"]
  n_users = len(users)
  rng = np.random.RandomState(args.seed)
  rng.shuffle(users)
  
  train_ratio, val_ratio = 0.7, 0.15
  n_train = max(1, int(round(train_ratio * n_users)))
  n_val = max(1, int(round(val_ratio * n_users)))
  
  train_users = set(users[:n_train])
  val_users = set(users[n_train : n_train+n_val])
  test_users = set(users[n_train+n_val:])

  def by_users(frames, chosen_users):
    return [f for f in frames if f['user'].iloc[0] in chosen_users]
  
  train_frames = by_users(sessions, train_users)
  val_frames = by_users(sessions, val_users)
  test_frames = by_users(sessions, test_users)

  # norm = {'mean': <x_mean>, 'std': <x_std>}
  loaders, norm = make_loaders(
    train_frames, val_frames, test_frames, input_cols, target_col,
    args.seq_len, args.pred_len, args.batch_size, args.stride)
  device = torch.device(args.device)
  feat_dim = len(input_cols)
  models = {
    'mlp': MLP(seq_len=args.seq_len, feat_dim=feat_dim, pred_len=args.pred_len, hidden=256, dropout=0.1),
    'lstm': LSTMForecaster(feat_dim=feat_dim, pred_len=args.pred_len, hidden=128, layers=2, dropout=0.1)
  }

  results = []
  best_plot_saved = False 
  for name, model in models.items():
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")
    best_state = None 
    patience = 5 
    bad = 0 

    for epoch in range(1, args.epochs+1):
      tr_loss, _, _ = run_epoch(model, loaders['train'], device, optimizer=opt)
      val_loss, _, _ = run_epoch(model, loaders['val'], device, optimizer=None)
      print(f"[{name}] epoch {epoch:02d} train MSE={tr_loss:.4f} val MSE={val_loss:.4f}")
if __name__ == "__main__":
  main()