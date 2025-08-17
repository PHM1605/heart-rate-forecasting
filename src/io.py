import csv, re 
from datetime import timedelta 
from pathlib import Path 
import pandas as pd 
from io import StringIO

def parse_hr_csv(path:Path):
  rows = []
  # newline='' means show EXACTLY \r\n or \n, as written in file, regardless of system
  with open(path, newline='') as f:
    reader = csv.reader(f)
    for r in reader:
      rows.append(r)
  if len(rows) < 4:
    raise ValueError(f"Unexpected HR csv format: {path}")
  
  # find date time of the file
  header_meta = rows[0]
  values_meta = rows[1]
  meta = dict(zip(header_meta, values_meta)) # dict with key=1st line and value=2nd line 
  date_str = meta.get("Date", "")
  time_str = meta.get("Start time", "")
  start_dt = pd.to_datetime(f"{date_str} {time_str}", dayfirst=True, errors="coerce")
  if pd.isna(start_dt):
    raise ValueError(f"Cannot parse Date/Start time in {path}")
  # find row starting with "Sample rate" header cell, whom "Time" and "HR (bpm)" would follow
  idx = None # row that contains "sample rate" header
  for i, r in enumerate(rows):
    if len(r) > 0 and r[0].strip().lower() == 'sample rate':
      idx = i 
      break 
  if idx is None or idx + 1 >= len(rows):
    raise ValueError(f"'Sample rate' section not found in {path}")
  data_header = rows[idx]
  name_to_ix = {name.strip(): j for j, name in enumerate(data_header)}
  if "Time" not in name_to_ix or "HR (bpm)" not in name_to_ix:
    raise ValueError(f"'Time' or 'HR (bpm)' not found in header for {path}")
  # picking "Time" and "Header" out
  time_ix = name_to_ix["Time"]
  hr_ix = name_to_ix["HR (bpm)"]
  ts_list, hr_list = [], []
  for r in rows[idx+1:]:
    if len(r) <= max(time_ix, hr_ix):
      continue 
    t_str = r[time_ix].strip()
    h_str = r[hr_ix].strip()
    if t_str == '' or h_str == '':
      continue 
    # parse 00:00:11
    h, m, s = t_str.split(":")
    delta = timedelta(hours=int(h), minutes=int(m), seconds=int(s))
    try:
      hr_val = float(h_str)
    except Exception:
      continue 
    ts_list.append(start_dt + delta)
    hr_list.append(hr_val)
  df = pd.DataFrame({"timestamp": pd.to_datetime(ts_list), "hr": hr_list})
  df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
  return df 

# group(1): yyyy-mm-dd HH:MM:SS.<fraction-seconds>
TS_REGEX = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)')
# Parse files like user1_rest_xethru.csv to ['timestamp', 'xe0', 'xe1',...]
def parse_xethru_csv(path:Path):
  clean_lines = []
  with open(path, 'r') as f:
    for line in f:
      line = line.strip()
      if not line: continue 
      m = TS_REGEX.match(line)
      if not m: continue 
      ts = m.group(1)
      rest = line[len(ts):]
      if rest[0] in ['-', '+'] or rest[0].isdigit():
        clean_lines.append(f"{ts},{rest}")
      else:
        continue 

  # convert ["x,x,x", "y,y,y", ...] => "x,x,x\ny,y,y\n" => make it to a file for pd.read_csv()
  buf = StringIO('\n'.join(clean_lines))
  df = pd.read_csv(buf, header=None)
  df.rename(columns={0:'timestamp'}, inplace=True) # new df: 1st-col=timestamp, 2nd-col=1, 3rd=2, ...
  df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
  df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
  df = df.resample('1s').mean().reset_index() # make one row = one second gap
  # rename column '1'=>'xe0', '2'=>'xe1',...
  for j in range(1, df.shape[1]):
    df.rename(columns={j: f'xe{j-1}'}, inplace=True)

  return df 

