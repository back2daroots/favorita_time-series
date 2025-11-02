import json, joblib, numpy as np, pandas as pd
from src.utils import load_cfg, load_feature_schema, align_frame_to_schema
from src.data import load_favorita
from src.features import make_features
from src.utils import encode_categoricals_for_gbm

def smape(y, yhat):
    return 100 * np.mean(2*np.abs(yhat - y) / (np.abs(yhat) + np.abs(y) + 1e-8))

def main(cfg_path="configs/config.yaml"):
    cfg = load_cfg(cfg_path)
    d, f = cfg["data"], cfg["features"]

    frames = load_favorita(d, d["date_col"])
    train  = frames["train"].copy()

    # holdout split
    maxd = pd.to_datetime(train[d["date_col"]]).max()
    cutoff = maxd - pd.Timedelta(days=d["holdout_days"])
    tr = train[train[d["date_col"]] <= cutoff].copy()
    ho = train[train[d["date_col"]] >  cutoff].copy()
    full = pd.concat([tr, ho], ignore_index=True)

    full_f = make_features(
        full, d["date_col"], d["id_cols"], d["target_col"],
        f["lags"], f["rolling_windows"],
        add_cal=f.get("add_calendar", True),
        calendar_extras=f.get("calendar_extras", False),
        rolling_stats=tuple(f.get("rolling_stats", ["mean"])),
        group_specs=f.get("group_aggregates", []),
        oil=frames.get("oil") if f.get("use_oil", False) else None,
        hol=frames.get("hol") if f.get("use_holidays", False) else None,
        trans=frames.get("trans") if f.get("use_transactions", False) else None,
        use_onpromotion=f.get("use_onpromotion", False),
        prepost_offsets=f.get("prepost_holiday", []),
        add_interactions_flag=f.get("add_interactions", False),
    )

    hof = full_f[full_f[d["date_col"]] >  cutoff].reset_index(drop=True)
    trf = full_f[full_f[d["date_col"]] <= cutoff].reset_index(drop=True)
    y_ho = hof[d["target_col"]].values

    X_ho = hof.drop(columns=[d["target_col"], d["date_col"]], errors="ignore")
    y_ho = hof[d["target_col"]].values
    X_tr_ref = trf.drop(columns=[d["target_col"], d["date_col"]], errors="ignore")

    X_tr_ref_enc, X_ho_enc = encode_categoricals_for_gbm(X_tr_ref, X_ho)


    assert not any(X_ho_enc.dtypes.astype(str).str.contains('category|object')), X_ho_enc.dtypes


    try:
        schema = load_feature_schema("models/xgb_feature_schema.json")
    except Exception:
        schema = None

    if schema is not None:
        X_ho_final = align_frame_to_schema(X_ho_enc, schema)
    else:
        X_ho_final = X_ho_enc


    try:
        schema = load_feature_schema("models/xgb_feature_schema.json")
        X_ho_aligned = align_frame_to_schema(X_ho, schema)
    except Exception:
        X_ho_aligned = X_ho


    m_lgb = joblib.load("models/lgbm_retrained.joblib")
    m_xgb = joblib.load("models/xgb_baseline.joblib") 
    # m_cat = joblib.load("models/cat_tuned.joblib")

    p_lgb = m_lgb.predict(X_ho_final)
    p_xgb = m_xgb.predict(X_ho_final)
    # p_cat = m_cat.predict(X_ho_final)


    grid = np.linspace(0, 1, 21)
    best = (1e9, None)
    for w in grid:
        yb = w * p_lgb + (1 - w) * p_xgb
        s = smape(y_ho, yb)
        if s < best[0]:
            best = (s, w)

    print(f"[blend LGBM/XGB] best sMAPE={best[0]:.4f} at w_lgb={best[1]:.2f}, w_xgb={1-best[1]:.2f}")

    # if adding CatBoost:
    # best3 = (1e9, None)
    # for w1 in grid:
    #   for w2 in grid:
    #       if w1 + w2 <= 1:
    #           w3 = 1 - w1 - w2
    #           yb = w1*p_lgb + w2*p_xgb + w3*p_cat
    #           s = smape(y_ho, yb)
    #           if s < best3[0]:
    #               best3 = (s, (w1, w2, w3))
    # print(f"[blend LGBM/XGB/CAT] best sMAPE={best3[0]:.4f} weights={best3[1]}")

if __name__ == "__main__":
    main()