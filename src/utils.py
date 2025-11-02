import yaml
import json
import pandas as pd 

def load_cfg(cfg_path: str) -> dict:
    '''Load YAML configuration file.'''
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)

def load_feature_schema(path: str) -> dict | None:
    '''Load saved feature schema (JSON) if exists.'''
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def align_frame_to_schema(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    '''Ensure DataFrame has same columns as schema )add missing cols as 0).'''
    if schema is None:
        return df
    cols = schema.get('columns', list(df.columns))
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0
            return out[cols]


def encode_categoricals_for_gbm(X_tr: pd.DataFrame, X_va: pd.DataFrame):
    X_tr = X_tr.copy()
    X_va = X_va.copy()
    cat_cols = X_tr.select_dtypes(include=['category', 'object']).columns.tolist()
    for c in cat_cols:
        X_tr[c] = X_tr[c].astype('category')
        cats = X_tr[c].cat.categories
        X_va[c] = pd.Categorical(X_va[c], categories=cats)
        X_tr[c] = X_tr[c].cat.codes.astype("int32")
        X_va[c] = X_va[c].cat.codes.astype("int32")
    return X_tr, X_va

#X_tr, X_val = encode_categoricals_for_gbm(X_tr, X_val, cat_cols)   # в cv_run
#X_tr, X_ho  = encode_categoricals_for_gbm(X_tr, X_ho,  cat_cols)   # в train