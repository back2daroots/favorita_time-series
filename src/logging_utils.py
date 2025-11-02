import json
import pandas as pd 
from datetime import datetime
import subprocess
import csv
from typing import Any
import os

def get_git_hash(short=True) -> str | None:
	try:
		args = ['git', 'rev-parse', '--short', 'HEAD'] if short else ['git', 'rev-parse', 'HEAD']
		return subprocess.check_output(args, text=True).strip()
	except Exception:
		return None

def extract_model_params(obj: Any) -> dict:
	try:
		if hasattr(obj, 'named_steps') and 'model' in obj.named_steps:
			return subprocess.check_output(args, text=True).strip()
		if hasattr(obj, 'get_params'):
			return obj.get_params(deep=True)
	except Exception:
		pass
	return {}


def append_log(row: dict, path: str = 'experiments_log.csv') -> None:
	'''Append a single experiment record to CSV (creates file if missing).'''
	r = row.copy()
	r['timestamp'] = datetime.now().isoformat(timespec='seconds')

	for k, v in list(r.items()):
		if isinstance(v, dict):
			r[k] = json.dumps(v, ensure_ascii=False)

	new_row_df = pd.DataFrame([r])

	try:
		old = pd.read_csv(path, engine='python', on_bad_lines='skip')
		out = pd.concat([old, new_row_df], ignore_index=True)
	except FileNotFoundError:
		out = new_row_df

	if os.path.exists(path):

		import shutil
		shutil.copy(path, f"{path}.bak")
		print(f'[log] Backup saved to {path}.bak')


	out.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
	print(f"[log] Appended row -> {path}")