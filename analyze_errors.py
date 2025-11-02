#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import joblib
import pandas as pd

from src.data import load_favorita
from src.features import make_features

CFG_PATH = "configs/config.yaml"

def load_cfg(path: str) -> dict:
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    
    if "data" in cfg and isinstance(cfg["data"], dict):
        data_cfg = cfg["data"]
        feat_cfg = cfg.get("features", {})
        cv_cfg   = cfg.get("cv", {})
        log_cfg  = cfg.get("log", {})
        model_cfg= cfg.get("model", {})
    else:
        
        data_keys = ["date_col","target_col","id_cols","holdout_days","train_path","test_path"]
        data_cfg = {k: cfg[k] for k in data_keys if k in cfg}
        feat_cfg = cfg.get("features", {})
        cv_cfg   = cfg.get("cv", {})
        log_cfg  = cfg.get("log", {})
        model_cfg= cfg.get("model", {})
    
    for k in ["date_col","target_col","id_cols","holdout_days"]:
        if k not in data_cfg:
            sys.exit(f"[ERROR] Missing `{k}` in config under `data` (or top-level).")
    return {"data": data_cfg, "features": feat_cfg, "cv": cv_cfg, "log": log_cfg, "model": model_cfg}

def main():
    cfg_all = load_cfg(CFG_PATH)
    cfg_d   = cfg_all["data"]
    cfg_f   = cfg_all["features"]

    date_col    = cfg_d["date_col"]
    target_col  = cfg_d["target_col"]
    id_cols     = cfg_d["id_cols"]
    holdout_days= int(cfg_d["holdout_days"])

    
    frames = load_favorita(cfg_d, date_col)
    train  = frames["train"].copy()
    print(f"[load] train={train.shape}")

    
    max_date = pd.to_datetime(train[date_col]).max()
    cutoff   = max_date - pd.Timedelta(days=holdout_days)
    tr = train[train[date_col] <= cutoff].copy()
    ho = train[train[date_col] >  cutoff].copy()
    print(f"[split] tr={tr.shape}, ho={ho.shape} (cutoff={cutoff.date()})")

    
    full = pd.concat([tr, ho], ignore_index=True)

    full_f = make_features(
        full,
        date_col,
        id_cols,
        target_col,
        cfg_f["lags"],
        cfg_f["rolling_windows"],
        add_cal=cfg_f.get("add_calendar", True),
        calendar_extras=cfg_f.get("calendar_extras", False),
        rolling_stats=tuple(cfg_f.get("rolling_stats", ["mean"])),
        group_specs=cfg_f.get("group_aggregates", []),
        oil=frames.get("oil") if cfg_f.get("use_oil", False) else None,
        hol=frames.get("hol") if cfg_f.get("use_holidays", False) else None,
        trans=frames.get("trans") if cfg_f.get("use_transactions", False) else None,
        use_onpromotion=cfg_f.get('use_onpromotion', False),
        prepost_offsets=cfg_f.get('prepost_holiday', []),
        add_interactions_flag=cfg_f.get('add_interactions', False),
    )

    trf = full_f[full_f[date_col] <= cutoff].reset_index(drop=True)
    hof = full_f[full_f[date_col] >  cutoff].reset_index(drop=True)
    print(f"[features] trf={trf.shape}, hof={hof.shape}")

   
    model_path = "models/xgb_baseline.joblib" 
    if not os.path.exists(model_path):
        sys.exit(f"[ERROR] Model file not found: {model_path}")
    model = joblib.load(model_path)
    print(f"[model] loaded {model_path}")

    
    X_tr = trf.drop(columns=[target_col, date_col])
    X_ho = hof.drop(columns=[target_col, date_col])
    y_ho = hof[target_col]

    
    obj_cols = X_tr.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        for c in obj_cols:
            X_tr[c] = X_tr[c].astype("category")
        for c in obj_cols:
            X_ho[c] = pd.Categorical(X_ho[c], categories=X_tr[c].cat.categories)

    y_pred = model.predict(X_ho)

    hof = hof.copy()
    hof["pred"] = y_pred
    hof["resid"] = hof[target_col] - hof["pred"]

    
    def abs_mae(s: pd.Series) -> float:
        return float(s.abs().mean())

    by_family = hof.groupby("family")["resid"].apply(abs_mae).sort_values(ascending=False)
    by_store  = hof.groupby("store_nbr")["resid"].apply(abs_mae).sort_values(ascending=False)
    by_dow    = hof.groupby("dow")["resid"].apply(abs_mae).sort_values(ascending=False)

    print("\n🔸 Top-10 categories (family) by |resid| (MAE):")
    print(by_family.head(10))
    print("\n🔸 Top-10 shops (store_nbr) by |resid| (MAE):")
    print(by_store.head(10))
    print("\n🔸 Error by days of the week (MAE |resid|):")
    print(by_dow)

    
    os.makedirs("results", exist_ok=True)
    by_family.to_csv("results/errors_by_family.csv", header=["abs_mae_resid"])
    by_store.to_csv("results/errors_by_store.csv",   header=["abs_mae_resid"])
    by_dow.to_csv("results/errors_by_dow.csv",       header=["abs_mae_resid"])

    cols_keep = [date_col, "store_nbr", "family", target_col, "pred", "resid"]
    hof[cols_keep].to_csv("results/holdout_residuals.csv", index=False)
    print("\n[save] results -> results/errors_by_*.csv, results/holdout_residuals.csv")

if __name__ == "__main__":
    main()