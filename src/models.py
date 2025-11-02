# src/models.py
from __future__ import annotations

from xgboost import XGBRegressor

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None  

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None


def make_model(kind: str, params: dict, cfg: dict | None = None):
    """
    Factory: 'xgb'|'xgboost' | 'lgbm'|'lightgbm' | 'cat'|'catboost'
    """
    k = (kind or "").lower()

    if k in ("xgb", "xgboost"):
        base = {"objective": "reg:squarederror", "tree_method": "hist", "n_jobs": -1}
        base.update(params or {})
        return XGBRegressor(**base)

    if k in ("lgbm", "lightgbm"):
        if LGBMRegressor is None:
            raise ImportError("LightGBM is not installed. Try: conda install -c conda-forge lightgbm")
        
        if (not params) and cfg:
            params = cfg.get("model", {}).get("params_lgbm", {})
        base = {"n_estimators": 800, "learning_rate": 0.05, "num_leaves": 63,
                "subsample": 0.9, "colsample_bytree": 0.9, "n_jobs": -1}
        base.update(params or {})
        return LGBMRegressor(**base)

    if k in ("cat", "catboost"):
        if CatBoostRegressor is None:
            raise ImportError("CatBoost is not installed. Try: conda install -c conda-forge catboost")
        if (not params) and cfg:
            params = cfg.get("model", {}).get("params_cat", {})
        base = {"iterations": 1000, "learning_rate": 0.05, "depth": 6,
                "loss_function": "RMSE", "random_seed": 42, "verbose": False}
        base.update(params or {})
        return CatBoostRegressor(**base)

    raise ValueError(f"Unknown model kind: {kind}")