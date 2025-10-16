from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def make_model(kind: str, params: dict):
    if kind == "xgb":
        return XGBRegressor(objective="reg:squarederror", 
        		tree_method='hist',
        		enable_categorical=True,
        		**params)
    if kind == "lgbm":
        return LGBMRegressor(objective="rmse", **params)
    raise ValueError(f"Unknown model kind: {kind}")