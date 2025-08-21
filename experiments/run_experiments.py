import argparse
import torch 
import numpy as np 
import pandas as pd
from collections import defaultdict 
from pathlib import Path 
from src import preprocess, utils 
from src.dataset import WindowDataset
from src.models import LSTMForecaster, MLP, TransformerForecaster, TCN, Seq2SeqLSTM
from src.train import run_epoch, metrics_dict
from torch.utils.data import DataLoader

def make_loaders(train_frames, val_frames, test_frames, input_cols, target_col,
  seq_len, pred_len, batch_size, stride):
  train_ds = WindowDataset(train_frames, input_cols, target_col, seq_len, pred_len, stride, norm_stats=None, fit_stats=True)
  norm = {"mean": train_ds.mean, "std": train_ds.std}
  val_ds = WindowDataset(val_frames, input_cols, target_col, seq_len, pred_len, stride, norm_stats=norm, fit_stats=False)
  test_ds = WindowDataset(test_frames, input_cols, target_col, seq_len, pred_len, stride, norm_stats=norm, fit_stats=False)
  loaders = {
    'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True), # drop_last mean what to do if #lastSamples != batch_size
    'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True),
    'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)
  }
  target_mean, target_std = float(train_ds.target_mean), float(train_ds.target_std)

  return loaders, norm, target_mean, target_std

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
  print("DEVICE IS: ", args.device)
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
  loaders, norm, target_mean, target_std = make_loaders(
    train_frames, val_frames, test_frames, input_cols, target_col,
    args.seq_len, args.pred_len, args.batch_size, args.stride)
  device = torch.device(args.device)
  feat_dim = len(input_cols)
  models = {
    'mlp': MLP(seq_len=args.seq_len, feat_dim=feat_dim, pred_len=args.pred_len, hidden=256, dropout=0.1),
    'lstm': LSTMForecaster(feat_dim=feat_dim, pred_len=args.pred_len, hidden=128, layers=2, dropout=0.1),
    'tcn': TCN(feat_dim=feat_dim, pred_len=args.pred_len, channels=(64,64,64), k=3, dropout=0.1),
    'transformer': TransformerForecaster(feat_dim, args.pred_len, d_model=128, nhead=4, num_layers=3, dim_ff=256, dropout=0.1),
    "seq2seq_lstm": Seq2SeqLSTM(feat_dim=feat_dim, pred_len=args.pred_len, hidden=128, layers=2, teacher_forcing_ratio=0.5)
  }

  results = []
  best_plot_saved = False 
  for name, model in models.items():
    model = model.to(device)
    print("MODEL: ", next(model.parameters()).device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")
    best_state = None 
    patience = 5 
    bad = 0 

    for epoch in range(1, args.epochs+1):
      tr_loss, _, _ = run_epoch(model, loaders['train'], device, optimizer=opt, target_mean=target_mean, target_std=target_std)
      val_loss, _, _ = run_epoch(model, loaders['val'], device, optimizer=None, target_mean=target_mean, target_std=target_std)
      print(f"[{name}] epoch {epoch:02d} train Pinball(z)={tr_loss:.4f} val Pinball(z)={val_loss:.4f}")
      
      if val_loss < best_val - 1e-6:
        best_val = val_loss 
        best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        bad = 0 
      else:
        bad += 1 
        if bad >= patience:
          break
    
    if best_state is not None:
      model.load_state_dict(best_state)
    # test 
    _, y_test, yhat50_test = run_epoch(model, loaders['test'], device, optimizer=None, target_mean=target_mean, target_std=target_std)
    md = metrics_dict(y_test, yhat50_test)
    md.update(model=name, split='test_overall')
    results.append(md)

    # per-activity predictions => invert model outputs
    per_activity = defaultdict(lambda: {'y': [], 'yhat': []})
    test_ds = loaders['test'].dataset 
    model.eval()
    with torch.no_grad():
      for (frame_id, start) in test_ds.samples:
        frame = test_ds.frames[frame_id]
        activity = frame["activity"].iloc[0]
        x = frame.iloc[start:start+test_ds.seq_len][test_ds.input_cols]
        y = frame.iloc[start+test_ds.seq_len : start+test_ds.seq_len+test_ds.pred_len][test_ds.target_col]
        x = (x - test_ds.mean) / test_ds.std 
        # [batch, time, dim]
        x = torch.tensor(x.values, dtype=torch.float32, device=device).unsqueeze(0)
        yhat3_z = model(x).squeeze(0).cpu().numpy() # (batch,3)
        y50_z = yhat3_z[:,1] # p50
        y50 = y50_z * target_std + target_mean
        per_activity[activity]['y'].append(y.values)
        per_activity[activity]['yhat'].append(y50)
    
    # d: {'y': [[], [],...], 'yhat': [[], [], [], ..]}
    for activity, d in per_activity.items():
      y = np.stack(d['y'], axis=0)
      yhat = np.stack(d['yhat'], axis=0)
      md = metrics_dict(y, yhat)
      md.update(model=name, split=f"test_{activity}")
      results.append(md)
    
    # Save one example plot
    if not best_plot_saved and len(test_frames) > 0:
      frame = test_frames[0]
      plot_path = Path(out_dir) / "example_test_segment.png"
      utils.save_example_plot(frame, input_cols, norm, model, args.seq_len, args.pred_len, plot_path)
      best_plot_saved = True 
  
  # save result 
  res_df = pd.DataFrame(results)
  res_path = Path(out_dir) / "results_summary.csv"
  res_df.to_csv(res_path, index=False)
  print(f"\nSaved metrics table -> {res_path}")
  if best_plot_saved:
    print(f"Saved example plot -> {Path(out_dir) / 'example_test_segment.png'}")

if __name__ == "__main__":
  main()