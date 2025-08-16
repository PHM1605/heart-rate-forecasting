import re
import pandas as pd 
from pathlib import Path 
from .io import parse_hr_csv, parse_xethru_csv 

# return list of (path-sensor-1, path-sensor-2, activity-type, user-x)
def find_pairs(data_dir: Path):
  pairs = []
  for activity in ['rest', 'normal', 'abnormal']:
    folder = (data_dir / activity)
    if not folder.exists():
      continue 
    for hr_path in folder.glob('*_hr.csv'):
      base = hr_path.name.replace('_hr.csv', '')
      xe_path = hr_path.with_name(base+'_xethru.csv')
      if not xe_path.exists():
        continue 
      # base: "user1_rest"
      # ^(.*): means "beginning with any chars"
      # $: means "ends with" 
      # => pattern: ^(.*)_rest$; comparing with <base>  => m.group(1)="user1"
      m = re.match(r'^(.*)_' + activity + r'$', base)
      user = m.group(1) if m else "unknown"
      pairs.append((hr_path, xe_path, activity, user))
  return pairs

def build_session_dataframe(hr_path, xe_path, activity, user):
  hr = parse_hr_csv(hr_path)
  # xe = parse_xethru_csv(xe_path)
  print(hr)