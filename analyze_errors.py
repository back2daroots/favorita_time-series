# analyze_errors.py
# -*- coding: utf-8 -*-
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from src.utils import load_cfg, encode_categoricals_for_gbm, load_feature_schema, align_frame_to_schema
from src.data import load_favorita
from src.features import make_features


PLOTS_DIR = "plots"
CFG_PATH = "configs/config.yaml"



def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def smape(y, yhat):
    y = np.asarray(y).reshape(-1)
    yhat = np.asarray(yhat).reshape(-1)
    return 100 * np.mean(2 * np.abs(yhat - y) / (np.abs(yhat) + np.abs(y) + 1e-8))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(yhat) - np.asarray(y)) ** 2)))


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(yhat) - np.asarray(y))))


def plot_residuals_scatter(y, yhat, title, outpath):
    resid = y - yhat
    plt.figure()
    plt.scatter(yhat, resid, alpha=0.4, s=8)
    plt.axhline(0, ls="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_residuals_hist(y, yhat, title, outpath):
    resid = y - yhat
    plt.figure()
    plt.hist(resid, bins=40)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def bar_top(series: pd.Series, title: str, outpath: str, top: int = 10, ascending=False):
    s = series.sort_values(ascending=ascending)

    s = s.sort_values(ascending=False).head(top)
    plt.figure()
    s.plot(kind="bar")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def try_load(path: str):
    try:
        return joblib.load(path)
    except Exception:
        return None


def infer_feature_schema_hint(model, fallback_schema_path=None):

    names = getattr(model, "feature_name_", None)
    if names:
        return {"columns": list(names)}
    if fallback_schema_path and os.path.exists(fallback_schema_path):
        try:
            with open(fallback_schema_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None



def main():
    ensure_dir(PLOTS_DIR)


    cfg_all = load_cfg(CFG_PATH)
    cfg_d = cfg_all["data"]
    cfg_f = cfg_all["features"]
    cfg_blend = cfg_all.get("blend", {})

    date_col = cfg_d["date_col"]
    target_col = cfg_d["target_col"]
    id_cols = cfg_d["id_cols"]
    holdout_days = int(cfg_d["holdout_days"])


    frames = load_favorita(cfg_d, date_col)
    train = frames["train"].copy()

    full_f = make_features(
        train,
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
        use_onpromotion=cfg_f.get("use_onpromotion", False),
        prepost_offsets=cfg_f.get("prepost_holiday", []),
        add_interactions_flag=cfg_f.get("add_interactions", False),
        stage="inference",  
    )

    if len(full_f) == 0:
        raise RuntimeError("[analyze] make_features returned empty frame for train — check FE logic")


    cutoff = pd.to_datetime(full_f[date_col].max()) - pd.Timedelta(days=holdout_days)
    hof = full_f[full_f[date_col] > cutoff].reset_index(drop=True)
    trf = full_f[full_f[date_col] <= cutoff].reset_index(drop=True)

    if len(hof) == 0:
        raise RuntimeError("[analyze] Holdout is empty — check holdout_days or FE filters")

    X_ho = hof.drop(columns=[target_col, date_col], errors="ignore")
    y_ho = hof[target_col].values
    X_tr_ref = trf.drop(columns=[target_col, date_col], errors="ignore")


    X_tr_ref_enc, X_ho_enc = encode_categoricals_for_gbm(X_tr_ref, X_ho)


    schema = None
    for schema_path in [
        "models/lgbm_feature_schema.json",
        "models/xgb_feature_schema.json",
    ]:
        try:
            tmp = load_feature_schema(schema_path)
            if tmp:
                schema = tmp
                break
        except Exception:
            pass

    if schema:
        X_ho_final = align_frame_to_schema(X_ho_enc, schema)
        X_tr_ref_final = align_frame_to_schema(X_tr_ref_enc, schema)
    else:
        X_ho_final = X_ho_enc
        X_tr_ref_final = X_tr_ref_enc


    bad = X_ho_final.dtypes.astype(str).str.contains("category|object")
    assert not any(bad), f"[analyze] X_ho_final still has non-numeric dtypes: {X_ho_final.dtypes[bad]}"


    members = cfg_blend.get("members", [])
    paths = cfg_blend.get("model_paths", {})
    weights = np.array(cfg_blend.get("weights", []), dtype=float) if members else None


    m_lgb = try_load(paths.get("lgbm", "models/lgbm_retrained.joblib"))
    m_xgb = try_load(paths.get("xgb", "models/xgb_baseline.joblib"))
    m_cat = try_load(paths.get("cat", "models/cat_baseline.joblib"))

    preds = {}
    if m_lgb is not None:

        preds["lgbm"] = m_lgb.predict(X_ho_final.to_numpy())
    if m_xgb is not None:
        preds["xgb"] = m_xgb.predict(X_ho_final.to_numpy())
    if m_cat is not None:
 
        preds["cat"] = m_cat.predict(X_ho_final)


    pred_best = None
    if members and weights is not None and len(members) == len(weights):
        weights = weights / (weights.sum() if weights.sum() != 0 else 1.0)
        stack = []
        for m in members:
            if m not in preds:
                raise RuntimeError(f"[analyze] Blend member '{m}' has no predictions (model missing?)")
            stack.append(preds[m].reshape(-1, 1))
        P = np.hstack(stack)
        pred_best = (P * weights.reshape(1, -1)).sum(axis=1)
        best_name = f"blend({', '.join(f'{m}:{w:.2f}' for m, w in zip(members, weights))})"
    else:
        # иначе: приоритет LGBM → XGB → Cat
        if "lgbm" in preds:
            pred_best = preds["lgbm"]
            best_name = "lgbm"
        elif "xgb" in preds:
            pred_best = preds["xgb"]
            best_name = "xgb"
        elif "cat" in preds:
            pred_best = preds["cat"]
            best_name = "cat"
        else:
            raise RuntimeError("[analyze] No models available to predict")


    metrics_best = {
        "rmse": rmse(y_ho, pred_best),
        "mae": mae(y_ho, pred_best),
        "smape": smape(y_ho, pred_best),
    }
    print(f"[analyze] BEST={best_name} metrics:", metrics_best)


    hof_plot = hof.copy()
    hof_plot["pred_best"] = pred_best
    hof_plot["resid_best"] = hof_plot[target_col] - hof_plot["pred_best"]


    ensure_dir(PLOTS_DIR)
    plot_residuals_scatter(y_ho, pred_best, f"Residuals vs Predicted — {best_name}", os.path.join(PLOTS_DIR, "residuals_scatter_best.png"))
    plot_residuals_hist(y_ho, pred_best, f"Residuals Histogram — {best_name}", os.path.join(PLOTS_DIR, "residuals_hist_best.png"))


    bar_top(
        hof_plot.groupby("family")["resid_best"].mean(),
        "MAE by family (top-10)",
        os.path.join(PLOTS_DIR, "mae_by_family.png"),
        top=10,
    )
    bar_top(
        hof_plot.groupby("store_nbr")["resid_best"].mean(),
        "MAE by store (top-10)",
        os.path.join(PLOTS_DIR, "mae_by_store.png"),
        top=10,
    )


    if "dow" not in hof_plot.columns and "date" in hof_plot.columns:
        hof_plot["dow"] = pd.to_datetime(hof_plot[date_col]).dt.dayofweek
    s_dow = hof_plot.groupby("dow").apply(lambda g: smape(g[target_col].values, g["pred_best"].values))
    plt.figure()
    s_dow.plot(kind="bar")
    plt.title("sMAPE by day-of-week (best)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "smape_by_dow.png"), dpi=150)
    plt.close()


    best_model = None
    if best_name.startswith("blend"):
        # возьмём LGBM или XGB для FI
        best_model = m_lgb or m_xgb
    else:
        best_model = {"lgbm": m_lgb, "xgb": m_xgb, "cat": m_cat}.get(best_name)

    if best_model is not None and hasattr(best_model, "feature_importances_"):

        schema_hint = infer_feature_schema_hint(best_model) or {"columns": list(X_tr_ref_final.columns)}
        feat_names = schema_hint["columns"]
        fi = pd.Series(best_model.feature_importances_, index=feat_names).sort_values(ascending=False).head(20)
        plt.figure()
        fi.sort_values().plot(kind="barh")
        plt.title(f"Top-20 Feature Importance ({best_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"fi_{best_name}_top20.png"), dpi=150)
        plt.close()

    # (опционально) сохраним hof с предсказаниями для быстрых последующих графиков
    hof_out = "data/holdout_with_pred.parquet"
    os.makedirs("data", exist_ok=True)
    hof_plot.to_parquet(hof_out)
    print(f"[analyze] saved plots -> {PLOTS_DIR}/, and hof -> {hof_out}")

    # Если доступны отдельные предсказания моделей — выведем их метрики тоже
    for name in ("lgbm", "xgb", "cat"):
        if name in preds:
            print(f"[analyze] {name}: rmse={rmse(y_ho, preds[name]):.4f}, mae={mae(y_ho, preds[name]):.4f}, smape={smape(y_ho, preds[name]):.4f}")


if __name__ == "__main__":
    main()