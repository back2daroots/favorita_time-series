import pandas as pd
import numpy as np

def add_calendar(df, date_col, extras: bool = False):
    out = df.copy()
    dt = out[date_col]
    out["dow"]   = dt.dt.dayofweek
    out["dom"]   = dt.dt.day
    out["week"]  = dt.dt.isocalendar().week.astype(int)
    out["month"] = dt.dt.month
    out["year"]  = dt.dt.year
    if extras:
        out["is_weekend"]   = (out["dow"] >= 5).astype(int)
        out["is_month_start"] = dt.dt.is_month_start.astype(int)
        out["is_month_end"]   = dt.dt.is_month_end.astype(int)
    return out

def merge_externals(df, oil=None, hol=None, trans=None, date_col="date"):
    out = df.copy()
    if oil is not None:
        oil = oil.sort_values(date_col).copy()
        if "dcoilwtico" in oil.columns:
            oil["dcoilwtico"] = oil["dcoilwtico"].interpolate().bfill()
        out = out.merge(oil[[date_col, "dcoilwtico"]], on=date_col, how="left")

    if trans is not None and "store_nbr" in out.columns:
        out = out.merge(trans[[date_col, "store_nbr", "transactions"]],
                        on=[date_col, "store_nbr"], how="left")
        out["transactions"] = out["transactions"].fillna(0)

    if hol is not None:
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

def add_rolling_stats(df, group_cols, target_col, windows, stats=("mean",)):
    """
    stats: tuple/list из ('mean','std','median','min','max')
    все окна считаем со смещением 1, чтобы не было утечек.
    """
    out = df.copy()
    g = out.groupby(group_cols, observed=True)[target_col]
    s = g.shift(1)
    for w in windows:
        roll = s.rolling(w)
        if "mean" in stats:
            out[f"rmean_{w}"] = roll.mean()
        if "std" in stats:
            out[f"rstd_{w}"] = roll.std()
        if "median" in stats:
            out[f"rmed_{w}"] = roll.median()
        if "min" in stats:
            out[f"rmin_{w}"] = roll.min()
        if "max" in stats:
            out[f"rmax_{w}"] = roll.max()
    return out

def add_group_aggregates(df, date_col, target_col, specs: list[dict]):
    """
    specs: список словарей вида:
      { group_by: [store_nbr], windows: [7,28], stats: [mean, median] }
    Считаем агрегаты по target в группах *без* лагов (это сглаженные уровни).
    """
    out = df.copy()
    for spec in specs:
        group_by = spec["group_by"]
        windows  = spec.get("windows", [])
        stats    = spec.get("stats", ["mean"])
        # группируем и считаем rolling по датам внутри группы
        g = out.sort_values(date_col).groupby(group_by, observed=True)[target_col]
        s = g.shift(1)  # чтобы не смотреть в будущее
        for w in windows:
            roll = s.rolling(w)
            prefix = f"{'_'.join(group_by)}_{w}"
            if "mean" in stats:
                out[f"{prefix}_mean"] = roll.mean().values
            if "median" in stats:
                out[f"{prefix}_med"] = roll.median().values
            if "std" in stats:
                out[f"{prefix}_std"] = roll.std().values
            if "min" in stats:
                out[f"{prefix}_min"] = roll.min().values
            if "max" in stats:
                out[f"{prefix}_max"] = roll.max().values
    return out

def make_features(df, date_col, id_cols, target_col,
                  lags, rolling_windows, add_cal=True,
                  calendar_extras=False,
                  rolling_stats=("mean",),
                  group_specs=None,
                  oil=None, hol=None, trans=None):
    out = df.copy()
    if add_cal:
        out = add_calendar(out, date_col, extras=calendar_extras)
    out = merge_externals(out, oil=oil, hol=hol, trans=trans, date_col=date_col)
    out = add_lags(out, id_cols, target_col, lags)
    out = add_rolling_stats(out, id_cols, target_col, rolling_windows, stats=rolling_stats)
    if group_specs:
        out = add_group_aggregates(out, date_col, target_col, group_specs)
    out = out.dropna().reset_index(drop=True)
    return out