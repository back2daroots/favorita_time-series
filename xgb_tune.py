import argparse, json, os, yaml
import optuna
import pandas as pd 

from src.data import load_favorita
from src.features import make_features
from src.validation import rolling_splits
from src.metrics import smape

def build_feats(df, frames, cfg):
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

def objective(trial, cfg, frames, train):
	params = {
	'n_estimators': trial.suggest_int('n_estimators', 400, 1200, step=100),
	'max_depth': trial.suggest_int('max_depth', 3, 10),
	'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
	'subsample': trial.suggest_float('subsample', 0.6, 1.0),
	'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
	'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, step=0.01),
	'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.1),
	'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0),
	'n_jobs': -1,
	}

	splits = cfg['cv']['splits']; horizon = cfg['cv']['horizon']; step = cfg['cv']['step']
	scores = []

	from xgboost import XGBRegressor
	date_col = cfg['data']['date_col']; target_col = cfg['data']['target_col']

	for train_end, val_end in rolling_splits(train, date_col, splits, horizon, step):
		tr = train[train[date_col] <= train_end].copy()
		va = train[(train[date_col] > train_end) & (train[date_col] <= val_end)].copy()

		full = pd.concat([tr, va], ignore_index=True)
		full_f = build_feats(full, frames, cfg)

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

		model = XGBRegressor(
			objective='reg:squarederror',
			tree_method='hist',
			enable_categorical=True,
			**params
		)

		model.fit(X_tr, y_tr)
		y_pred = model.predict(X_va)
		scores.append(smape(y_va, y_pred))

	return float(sum(scores) / len(scores))

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--config', '-c', default='configs/config.yaml')
	ap.add_argument('--n-trials', type=int, default=30)
	ap.add_argument('--storage', default='sqlite:///optuna_xgb.db', help='for continuation / parallelization')
	ap.add_argument('--study-name', default='xgb_smape')
	args = ap.parse_args()


	with open(args.config, 'r') as f:
		cfg = yaml.safe_load(f)
	frames = load_favorita(cfg['data'], cfg['data']['date_col'])
	train = frames['train'].copy()

	sampler = optuna.samplers.TPESampler(seed=42)
	pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

	study = optuna.create_study(direction='minimize',
		study_name=args.study_name,
		storage=args.storage,
		load_if_exists=True,
		sampler=sampler,
		pruner=pruner)

	study.optimize(lambda t: objective(t, cfg, frames, train),
		n_trials=args.n_trials,
		gc_after_trial=True)

	print('Best sMAPE:', study.best_value)
	print('Best params:', study.best_params)

	os.makedirs('configs', exist_ok=True)
	best_json = 'configs/best_xgb.json'
	with open(best_json, 'w') as f:
		json.dump(study.best_params, f, indent=2)
	print(f'[save] best params -> {best_json}')

	cfg['model']['params'].update(study.best_params)
	with open(args.config, 'w') as f:
		yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
	print(f'[update] {args.config} model.params updated')

if __name__ == '__main__':
	main()



