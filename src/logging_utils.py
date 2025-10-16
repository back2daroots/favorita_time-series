import json
import pandas as pd 
from datetime import datetime
import subprocess

def get_git_hash(short=True) -> str | None:
	try:
		args = ['git', 'rev-parse', '--short', 'HEAD'] if short else ['git', 'rev-parse', 'HEAD']
		return subprocess.check_output(args, text=True).strip()
	except Exception:
		return None

def append_log(row: dict, path: str = 'experiments_log.csv') -> None:
	'''Append a single experiment record to CSV (creates file if missing).'''
	r = row.copy()
	r['timestamp'] = datetime.now().isoformat(timespec='seconds')

	for k, v in list(r.items()):
		if isinstance(v, dict):
			r[k] = json.dumps(v, ensure_ascii=False)

	try:
		old = pd.read_csv(path)
		out = pd.concat([old, pd.DataFrame([r])], ignore_index=True)
	except FileNotFoundError:
		out = pd.DataFrame([r])

	out.to_csv(path, index=False)