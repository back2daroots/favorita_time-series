import os
import joblib
import yaml
import pandas as pd
import numpy as np 
from src.features import make_features
from src.utils import load_feature_schema, align_frame_to_schema, encode_categoricals_for_gbm, load_cfg
from src.data import load_favorita

def main(cfg_path="configs/config.yaml"):

    cfg = load_cfg(cfg_path)
    d, f = cfg["data"], cfg["features"]
    blend = cfg.get("blend", {})

    date_col   = d["date_col"]
    target_col = d["target_col"]
    id_cols    = d["id_cols"]


    frames = load_favorita(d, date_col)
    train  = frames["train"].copy()
    test   = frames["test"].copy()


    test[target_col] = np.nan


    full = pd.concat([train, test], ignore_index=True)


    full_f = make_features(
        full, date_col, id_cols, target_col,
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
        stage='inference',
    )


    test_key = test[[date_col, *id_cols]].drop_duplicates()
    full_te_f = full_f.merge(test_key, on=[date_col, *id_cols], how="inner")
    full_tr_f = full_f.merge(test_key, on=[date_col, *id_cols], how="left", indicator=True)
    full_tr_f = full_tr_f[full_tr_f["_merge"] == "left_only"].drop(columns="_merge")


    print("[debug] full_f:", full_f.shape,
        "| test_key:", test_key.shape,
        "| full_te_f:", full_te_f.shape,
        "| full_tr_f:", full_tr_f.shape)

    X_tr_ref = full_tr_f.drop(columns=[target_col, date_col], errors="ignore")
    X_test   = full_te_f.drop(columns=[target_col, date_col], errors="ignore")

    X_tr_ref_enc, X_test_enc = encode_categoricals_for_gbm(X_tr_ref, X_test)

    schema = None
    for schema_path in ["models/lgbm_feature_schema.json",
                        "models/xgb_feature_schema.json"]:
        try:
            schema = load_feature_schema(schema_path)
            if schema is not None:
                break
        except Exception:
            pass

    if schema is not None:
        X_test_final = align_frame_to_schema(X_test_enc, schema)
    else:
        X_test_final = X_test_enc


    bad = X_test_final.dtypes.astype(str).str.contains("category|object")
    assert not any(bad), X_test_final.dtypes[bad]


    members = blend.get("members", ["lgbm", "xgb"])
    weights = np.array(blend.get("weights", [1.0]*len(members)), dtype=float)
    weights = weights / weights.sum()
    paths   = blend.get("model_paths", {
        "lgbm": "models/lgbm_retrained.joblib",
        "xgb" : "models/xgb_baseline.joblib",
    })

    preds = []
    for m in members:
        model = joblib.load(paths[m])

        p = model.predict(X_test_final.to_numpy())
        preds.append(p.reshape(-1, 1))

    P = np.hstack(preds)  # [n, k]
    y_blend = (P * weights.reshape(1, -1)).sum(axis=1).clip(min=0)


    sub = pd.DataFrame({
        "id": test["id"].values,
        "sales": y_blend
    })

    os.makedirs("submissions", exist_ok=True)
    out_path = "submissions/submission.csv"
    sub.to_csv(out_path, index=False)
    print(f"[save] submission -> {out_path} (shape={sub.shape})")

    assert not sub["sales"].isna().any(), "NaNs in submission!"
    assert (sub["sales"] >= 0).all(), "Negative sales detected!"
    print("✅ Submission sanity check passed:", sub.shape)

if __name__ == "__main__":
    main()