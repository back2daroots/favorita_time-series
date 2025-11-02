#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import traceback
import json

import yaml
import joblib
import pandas as pd
import csv 

from src.data import load_favorita, train_test_split_time
from src.features import make_features
from src.models import make_model
from src.metrics import rmse, mae, smape
from src.logging_utils import append_log, get_git_hash, extract_model_params
from src.utils import encode_categoricals_for_gbm



def main(cfg_path: str) -> None:
    # === 1) load config ===
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    date_col   = cfg["data"]["date_col"]
    target_col = cfg["data"]["target_col"]
    id_cols    = cfg["data"]["id_cols"]

    # === 2) load raw tables ===
    frames = load_favorita(cfg["data"], date_col)
    train = frames["train"]
    test  = frames["test"]     
    print(f"[load] train={train.shape}, test={test.shape}")

    # === 3) split inside train.csv to build holdout ===
    tr, ho = train_test_split_time(train, date_col, cfg["data"]["holdout_days"])
    print(f"[split] tr={tr.shape}, ho={ho.shape}")

    # === 4) build features on FULL (train ∪ holdout), then split by date ===
    cutoff = tr[date_col].max()
    full_df = pd.concat([tr, ho], axis=0, ignore_index=True)

    try:
        full_f = make_features(
            full_df, date_col, id_cols, target_col,
            cfg["features"]["lags"],
            cfg["features"]["rolling_windows"],
            add_cal=cfg["features"]["add_calendar"],
            calendar_extras=cfg["features"].get("calendar_extras", False),
            rolling_stats=tuple(cfg["features"].get("rolling_stats", ["mean"])),
            group_specs=cfg["features"].get("group_aggregates", []),
            oil=frames["oil"] if cfg["features"].get("use_oil", False) else None,
            hol=frames["hol"] if cfg["features"].get("use_holidays", False) else None,
            trans=frames["trans"] if cfg["features"].get("use_transactions", False) else None,
            use_onpromotion=cfg['features'].get('use_onpromotion', False),
            prepost_offsets=cfg['features'].get('prepost_holiday', []),
            add_interactions_flag=cfg['features'].get('add_interactions', False),
        )
    except Exception as e:
        print("[features] failed while building features on full_df")
        traceback.print_exc()
        raise e

    # split by date (not by iloc), so we keep correct group histories
    trf = full_f[full_f[date_col] <= cutoff].reset_index(drop=True)
    hof = full_f[full_f[date_col] >  cutoff].reset_index(drop=True)
    print(f"[features] trf={trf.shape}, hof={hof.shape}")

    # === 5) split X/y ===
    X_tr, y_tr = trf.drop(columns=[target_col, date_col]), trf[target_col]
    X_ho, y_ho = hof.drop(columns=[target_col, date_col]), hof[target_col]
    print(f"[split XY] X_tr={X_tr.shape}, y_tr={y_tr.shape}, X_ho={X_ho.shape}, y_ho={y_ho.shape}")

    # === 6) categorical sync for XGB (object -> category; align categories) ===
    obj_cols = X_tr.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        for c in obj_cols:
            X_tr[c] = X_tr[c].astype("category")
        for c in obj_cols:
            X_ho[c] = pd.Categorical(X_ho[c], categories=X_tr[c].cat.categories)

    # === 7) build & fit model ===
   # kind   = cfg["model"]["kind"]
   # params = cfg["model"]["params"]
    model = make_model(cfg["model"]["kind"], cfg["model"].get("params", {}), cfg)

    print(f"[fit] model={cfg['model']['kind']} params={json.dumps(cfg['model'].get('params', {}))}")
    kind = cfg['model']['kind'].lower()
    cat_cols = X_tr.select_dtypes(include=['category', 'object']).columns.tolist()
            

    if kind == 'cat':
        cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols]
        model.fit(X_tr, y_tr, cat_features=cat_idx)
            
    elif kind in ['lgbm', 'xgb']:
        X_tr, X_ho = encode_categoricals_for_gbm(X_tr, X_ho)
        assert not any (X_tr.dtypes.astype(str).str.contains('category|object')), 'X_tr still has categorical dtypes' 
        assert not any (X_ho.dtypes.astype(str).str.contains('category|object')), 'X_ho still has categorical dtypes' 
        model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, y_tr)

    # === 8) evaluate on holdout ===
    y_pred = model.predict(X_ho)
    metrics = {
        "rmse": rmse(y_ho, y_pred),
        "mae":  mae(y_ho, y_pred),
        "smape": smape(y_ho, y_pred),
    }
    print("[metrics] holdout:", metrics)

    # === 9) save model ===
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{cfg['model']['kind']}_baseline.joblib"
    joblib.dump(model, model_path)
    print(f"[save] model -> {model_path}")

    # === 10) log experiment ===
    params = extract_model_params(model)
    row = {
        "project": "Favorita-StoreSales",
        "dataset": "Favorita (Kaggle)",
        "target":  target_col,
        "model":   cfg['model']['kind'],
        "model_params": cfg['model'].get('params', {}),               # will be JSON-encoded in append_log
        "holdout_days": cfg["data"]["holdout_days"],
        "cv_splits": cfg["cv"]["splits"],
        "cv_step":   cfg["cv"]["step"],
        "cv_horizon": cfg["cv"]["horizon"],
        "rmse": float(metrics["rmse"]),
        "mae":  float(metrics["mae"]),
        "smape": float(metrics["smape"]),
        "git": get_git_hash() or "",
        "notes": "baseline with group_promo_holiday feats (holdout)",
    }
    log_path = cfg["log"]["path"]
    append_log(row, path=log_path)
#    print(f"[log] appended to {log_path}")
#    log_path = cfg['log']['path']
#    row_df = pd.DataFrame([row])

#serialising parameteres into a JSON line
#    if 'model_params' in row_df.columns:
#        row_df['model_params'] = row_df['model_params'].apply(
#            lambda x: json.dumps(x, ensure_ascii=False)
#            if not isinstance(x, str) else x
#            )
#    if os.path.exists(log_path):
#        try:
#            base_log = pd.read_csv(log_path, engine='python')
#        except Exception:
##            base_log - pd.read_csv(log_path, engine='python', on_bad_lines='skip')
 #       base_log = pd.concat([base_log, row_df], ignore_index=True)
 #   else:
 #       base_log = row_df

#    base_log.to_csv(log_path, index=False, quoting=csv.QUOTE_MINIMAL)
#    print(f'[log] appended CV mean into {log_path}')
    out_path = 'models/lgbm_retrained.joblib'
    joblib.dump(model, out_path)
    print(f'[save] model -> {out_path}')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)

