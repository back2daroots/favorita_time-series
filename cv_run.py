import argparse, yaml, json, os
import pandas as pd 
import joblib
from src.data import load_favorita
from src.features import make_features
from src.models import make_model
from src.metrics import rmse, mae, smape
from src.validation import rolling_splits
from src.logging_utils import get_git_hash


def build_feats_on_full(
	df: pd.DataFrame,
	frames: dict,
	cfg: dict
) -> pd.DataFrame:
	''' Building features on a sent frame (any dates selection), no leakage (shift(1) inside).'''
	return make_features(
		df,
		cfg['data']['date_col'],
		cfg['data']['id_cols'],
		cfg['data']['target_col'],
		cfg['features']['lags'],
		cfg['features']['rolling_windows'],
		add_cal=cfg['features']['add_calendar'],
		calendar_extras=cfg['features'].get('calendar_extras', False),
		rolling_stats=tuple(cfg['features'].get('rolling_stats', ['mean'])),
		group_specs=cfg['features'].get('group_aggregates', []),
		oil=frames['oil'] if cfg['features'].get('use_oil', False) else None,
		hol=frames['hol'] if cfg['features'].get('use_holidays', False) else None,
		trans=frames['trans'] if cfg['features'].get('use_transactions', False) else None
	)

def main(cfg_path: str):
	with open(cfg_path, 'r') as f:
		cfg = yaml.safe_load(f)

		date_col = cfg['data']['date_col']
		target_col = cfg['data']['target_col']
		id_cols = cfg['data']['id_cols']

		frames = load_favorita(cfg['data'], date_col)
		train = frames['train'].copy()

		splits = cfg['cv']['splits']
		horizon = cfg['cv']['horizon']
		step = cfg['cv']['step']

		rows = []
		fold_idx = 1

		for train_end, val_end in rolling_splits(train, date_col, splits, horizon, step):
			tr = train[train[date_col] <= train_end].copy()
			va = train[(train[date_col] > train_end) & (train[date_col] <= val_end)].copy()

			full = pd.concat([tr, va], ignore_index=True)
			full_f = build_feats_on_full(full, frames, cfg)

			trf = full_f[full_f[date_col] <= train_end].reset_index(drop=True)
			vaf = full_f[(full_f[date_col] > train_end) & (full_f[date_col] <= val_end)].reset_index(drop=True)

			X_tr, y_tr = trf.drop(columns=[target_col, date_col]), trf[target_col]
			X_va, y_va = vaf.drop(columns=[target_col, date_col]), vaf[target_col]


			obj_cols = X_tr.select_dtypes(include='object').columns.tolist()
			if obj_cols:
				for c in obj_cols:
					X_tr[c] = X_tr[c].astype('category')
				for c in obj_cols:
					X_va[c] = pd.Categorical(X_va[c], categories=X_tr[c].cat.categories)

			model = make_model(cfg['model']['kind'], cfg['model']['params'])
			model.fit(X_tr, y_tr)
			y_pred = model.predict(X_va)

			fold_metrics = {
			'fold': fold_idx,
			'train_end': str(train_end.date()),
			'val_end': str(val_end.date()),
			'n_train': int(len(X_tr)),
			'n_val': int(len(X_va)),
			'rmse': rmse(y_va, y_pred),
			'mae': mae(y_va, y_pred),
			'smape': smape(y_va, y_pred),
			}
			print(f'[fold {fold_idx}] {fold_metrics}')
			rows.append(fold_metrics)
			fold_idx += 1

		cv_df = pd.DataFrame(rows)
		os.makedirs('plots', exist_ok=True)
		cv_path = 'cv_log.csv'
		cv_df.to_csv(cv_path, index=False)


		agg = cv_df[['rmse', 'mae', 'smape']].mean().to_dict()
		agg = {k: float(v) for k,v in agg.items()}

		print('[CV mean]', agg)

		exp_row = {
		'project': 'Favorita-StoreSales',
		'dataset': 'Favorita (Kaggle)',
		'target': target_col,
		'model': cfg['model']['kind'],
		'cv_splits': splits,
		'cv_step': step,
		'cv_horizon': horizon,
		'rmse': agg['rmse'],
		'mae': agg['mae'],
		'smape': agg['smape'],
		'git': get_git_hash() or '',
		'notes': 'rolling CV run (mean over folds)',
		'cv_log_path': cv_path,
		'model_params': cfg['model']['params']
		}

		if os.path.exists(cfg['log']['path']):
			base_log = pd.read_csv(cfg['log']['path'])
			base_log = pd.concat([base_log, pd.DataFrame([exp_row])], ignore_index=True)
		else:
			base_log = pd.DataFrame([exp_row])
			print(f"[log] appended CV mean into {cfg['log']['path']}")


if __name__ == '__main__':
	p = argparse.ArgumentParser()
	p.add_argument('--config', '-c', default='configs/config.yaml')
	args = p.parse_args()
	main(args.config)






