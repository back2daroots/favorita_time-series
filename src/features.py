import pandas as pd
import numpy as np

def add_calendar(df, date_col):
    out = df.copy()
    dt = out[date_col]
    out["dow"] = dt.dt.dayofweek
    out["dom"] = dt.dt.day
    out["week"] = dt.dt.isocalendar().week.astype(int)
    out["month"] = dt.dt.month
    out["year"] = dt.dt.year
    return out

def merge_externals(df, oil=None, hol=None, trans=None, date_col="date"):
    out = df.copy()
    if oil is not None:
        oil = oil.sort_values(date_col).copy()
        oil["dcoilwtico"] = oil["dcoilwtico"].interpolate().bfill()
        out = out.merge(oil, on=date_col, how="left")
    if trans is not None and "store_nbr" in out.columns:
        out = out.merge(trans, on=[date_col, "store_nbr"], how="left")
        out["transactions"] = out["transactions"].fillna(0)
    if hol is not None:
        # simple holiday flag
        hol_flag = hol[[date_col]].drop_duplicates().assign(is_holiday=1)
        out = out.merge(hol_flag, on=date_col, how="left")
        out["is_holiday"] = out["is_holiday"].fillna(0)
    return out

def add_lags(df, group_cols, target_col, lags):
    out = df.copy()
    g = out.groupby(group_cols, observed=True)
    for L in lags:
        out[f"lag_{L}"] = g[target_col].shift(L)
    return out

def add_rollmeans(df, group_cols, target_col, windows):
    out = df.copy()
    g = out.groupby(group_cols, observed=True)
    for w in windows:
        out[f"rmean_{w}"] = g[target_col].shift(1).rolling(w).mean()
    return out

def make_features(df, date_col, id_cols, target_col, lags, rolling_windows,
                  add_cal=True, oil=None, hol=None, trans=None):
    out = df.copy()
    if add_cal:
        out = add_calendar(out, date_col)
    out = merge_externals(out, oil=oil, hol=hol, trans=trans, date_col=date_col)
    out = add_lags(out, id_cols, target_col, lags)
    out = add_rollmeans(out, id_cols, target_col, rolling_windows)
    out = out.dropna().reset_index(drop=True)
    return out