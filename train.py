import argparse, yaml, joblib
import pandas as pd
from src.data import load_favorita, train_test_split_time
from src.features import make_features
from src.models import make_model
from src.metrics import rmse, mae, smape
from src.logging_utils import append_log, get_git_hash

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    date_col   = cfg["data"]["date_col"]
    target_col = cfg["data"]["target_col"]
    id_cols    = cfg["data"]["id_cols"]

    # load
    frames = load_favorita(cfg["data"], date_col)
    train, test = frames["train"], frames["test"]

    # train/holdout split (внутри train.csv)
    tr, ho = train_test_split_time(train, date_col, cfg["data"]["holdout_days"])

    # features for train/holdout
    trf = make_features(
        tr, date_col, id_cols, target_col,
        cfg["features"]["lags"], cfg["features"]["rolling_mean_windows"],
        add_cal=cfg["features"]["add_calendar"],
        oil=frames["oil"] if cfg["features"]["use_oil"] else None,
        hol=frames["hol"] if cfg["features"]["use_holidays"] else None,
        trans=frames["trans"] if cfg["features"]["use_transactions"] else None
    )

    # для holdout нужно построить фичи, включая хвост train для лагов
    tail_len = max(cfg["features"]["lags"] + cfg["features"]["rolling_mean_windows"])
    seed_df = pd.concat([tr.tail(tail_len), ho], axis=0)
    hof = make_features(
        seed_df, date_col, id_cols, target_col,
        cfg["features"]["lags"], cfg["features"]["rolling_mean_windows"],
        add_cal=cfg["features"]["add_calendar"],
        oil=frames["oil"] if cfg["features"]["use_oil"] else None,
        hol=frames["hol"] if cfg["features"]["use_holidays"] else None,
        trans=frames["trans"] if cfg["features"]["use_transactions"] else None
    ).iloc[-len(ho):]

    # split X/y
    X_tr, y_tr = trf.drop(columns=[target_col, date_col]), trf[target_col]
    X_ho, y_ho = hof.drop(columns=[target_col, date_col]), hof[target_col]


    obj_cols = X_tr.select_dtypes(include='object').columns.tolist()

    for c in obj_cols:
    	X_tr[c] = X_tr[c].astype('category')

    for c in obj_cols:
    	X_ho[c] = pd.Categorical(X_ho[c], categories=X_tr[c].cat.categories)

    # model
    model = make_model(cfg["model"]["kind"], cfg["model"]["params"])
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_ho)

    metrics = {
        "rmse": rmse(y_ho, y_pred),
        "mae":  mae(y_ho, y_pred),
        "smape": smape(y_ho, y_pred),
    }
    print("Holdout metrics:", metrics)

    # save model
    joblib.dump(model, f"models/{cfg['model']['kind']}_baseline.joblib")

    row = {
    'project': 'Favorita-StoreSales',
    'dataset': 'Favorita (Kaggle)',
    'target': cfg['data']['target_col'],
    'model': cfg['model']['kind'],
    'model_params': cfg['model']['params'],
    'cv_splits': cfg['cv']['splits'],
    'cv_step': cfg['cv']['step'],
    'cv_horizon': cfg['cv']['horizon'],
    'rmse': float(metrics['rmse']),
    'mae': float(metrics['mae']),
    'smape': float(metrics['smape']),
    'git': get_git_hash() or '',
    'notes': 'baseline run'
    }

    append_log(row, path=cfg['log']['path'])
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="configs/config.yaml")
    args = p.parse_args()
    main(args.config)