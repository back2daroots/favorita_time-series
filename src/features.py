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
    stats: tuple/list from ('mean','std','median','min','max')
    all windows are calculated with shift 1 to avoid leakage
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
    specs: dicts as:
      { group_by: [store_nbr], windows: [7,28], stats: [mean, median] }
    Counting aggregates by target in groups without lags.
    """
    if df is None:
        raise ValueError("add_group_aggregates got df=None — previous step returned None (likely inplace=True misuse).")
    out = df.copy()

    for spec in specs:
        group_by = spec["group_by"]
        windows  = spec.get("windows", [])
        stats    = spec.get("stats", ["mean"])
        # grouping and calculating rolling via date by groups
        g = out.sort_values(date_col).groupby(group_by, observed=True)[target_col]
        s = g.shift(1)  # to avoid leakage
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

def add_onpromotion_feats(df: pd.DataFrame, id_cols, onpromo_col = 'onpromotion') -> pd.DataFrame:
    '''current onpromotion + lags + rollings (with shift 1).'''
    if df is None:
        raise ValueError("add_onpromotion_feats got df=None — previous step returned None (likely inplace=True misuse).")
    out = df.copy()
    if onpromo_col not in out.columns:
        return out

        #make sure it's numerical
    out[onpromo_col] = out[onpromo_col].fillna(0).astype(int)

    g = out.groupby(id_cols, observed=True)[onpromo_col]
    out[f'{onpromo_col}_lag1'] = g.shift(1)
    out[f'{onpromo_col}_lag7'] = g.shift(7)

    s = g.shift(1)
    out[f'{onpromo_col}_rsum7'] = s.rolling(7, min_periods=1).sum()
    out[f'{onpromo_col}_rmean28'] = s.rolling(28, min_periods=1).mean()

    return out

def add_prepost_holiday(df: pd.DataFrame, hol: pd.DataFrame | None,
                        date_col="date", offsets=(1,)) -> pd.DataFrame:
    if df is None:
        raise ValueError("add_prepost_holiday got df=None — previous step returned None (likely inplace=True misuse).")
    '''binaries of days before/after holdidays'''
    out = df.copy()
    if hol is None or date_col not in hol.columns:
        # creating columns of nulls
        for k in offsets:
            out[f"is_preholiday_{k}"]  = 0
            out[f"is_postholiday_{k}"] = 0
        return out

    hol = hol[[date_col]].drop_duplicates().copy()
    hol[date_col] = pd.to_datetime(hol[date_col])

    for k in offsets:
        pre  = hol.copy();  pre[date_col]  = pre[date_col]  + pd.Timedelta(days=k)
        post = hol.copy();  post[date_col] = post[date_col] - pd.Timedelta(days=k)

        out = out.merge(pre.assign(**{f"is_preholiday_{k}": 1}),
                        on=date_col, how="left")
        out = out.merge(post.assign(**{f"is_postholiday_{k}": 1}),
                        on=date_col, how="left")

        out[f"is_preholiday_{k}"]  = out[f"is_preholiday_{k}"].fillna(0).astype(int)
        out[f"is_postholiday_{k}"] = out[f"is_postholiday_{k}"].fillna(0).astype(int)

    return out

#aggregates for store and store x family+ z-score deviation
def add_store_group_aggregates(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    ''' making store_7_mean, (store, family)_7_mean/28_mean/7_std and z-score on store x family
    assuming that add_rolling_stats/add_group_aggregates already made rmean/std
    if not - calculating minimum here'''
    if df is None:
        raise ValueError("add_store_group_aggregates got df=None — previous step returned None (likely inplace=True misuse).")

    out = df.copy().sort_values(date_col)

    #store 7-day mean(no shift(1))

    if 'store_nbr' in out.columns:
        g_store = out.groupby(['store_nbr'], observed=True)[target_col]
        out['store_7_mean'] = g_store.shift(1).rolling(7, min_periods=1).mean()

    #store x family 7/28 mean and 7 std
    if {'store_nbr', 'family'}.issubset(out.columns):
        g_sf = out.groupby(['store_nbr', 'family'], observed=True)[target_col]
        s = g_sf.shift(1)
        out['store_family_7_mean'] = s.rolling(7, min_periods=1).mean()
        out['store_family_28_mean'] = s.rolling(28, min_periods=1).mean()
        out['store_family_7_std'] = s.rolling(7, min_periods=1).std()

        #z-score deviation (only if both columns exist)
        eps = 1e-6
        mu = out['store_family_7_mean']
        sd = out['store_family_7_std'].fillna(0)
        out['sf_zscore'] = (out[target_col] - mu) / (sd + eps)

    return out

#interactions
def add_interactions(df: pd.DataFrame, date_col: str,
                    target_col: str,onpromo_col='onpromotion') -> pd.DataFrame:
    '''
    -rmean_sales_7 * onpromotion
    -Friday & Month_end (binary)
    '''
    if df is None:
        raise ValueError("add_interactions got df=None — previous step returned None (likely inplace=True misuse).")
    out = df.copy()

    #if rmean_7 (moving average by target) already exists - using it
    rmean7_col = None
    for cand in ['rmean_7', 'rmed_7', 'store_family_7_mean']:
        if cand in out.columns:
            rmean7_col = cand
            break

    if rmean7_col and onpromo_col in out.columns:
        out['int_rmean7_onpromo'] = out[rmean7_col].fillna(0) * out[onpromo_col].fillna(0)

    #Friday and Month_end
    if 'dow' in out.columns and 'is_month_end' in out.columns:
        out['is_friday_month_end'] = ((out['dow'] == 4) & (out['is_month_end'] == 1)).astype(int)

    return out      

def make_features(df, date_col, id_cols, target_col,
                  lags, rolling_windows, add_cal=True,
                  calendar_extras=False,
                  rolling_stats=("mean",),
                  group_specs=None,
                  oil=None, hol=None, trans=None,
                  use_onpromotion=False,
                  prepost_offsets=(),
                  add_interactions_flag=False,
                  stage: str = 'train'):
    out = df.copy()
    if add_cal:
        out = add_calendar(out, date_col, extras=calendar_extras)
    out = merge_externals(out, oil=oil, hol=hol, trans=trans, date_col=date_col)

    out = add_lags(out, id_cols, target_col, lags)

    out = add_rolling_stats(out, id_cols, target_col, rolling_windows, stats=rolling_stats)
    if group_specs:
        out = add_group_aggregates(out, date_col, target_col, group_specs)

    out = add_store_group_aggregates(out, date_col, target_col)

    if use_onpromotion:
        out = add_onpromotion_feats(out, id_cols, onpromo_col='onpromotion')
    if prepost_offsets:
        out = add_prepost_holiday(out, hol=hol, date_col=date_col, offsets=tuple(prepost_offsets))

    if add_interactions_flag:
        out = add_interactions(out,date_col, target_col, onpromo_col='onpromotion')

    lag_cols = [f'lag_{L}' for L in lags]
    #out = out.dropna(subset=lag_cols).reset_index(drop=True)
    if stage == 'train':
        out = out.dropna().reset_index(drop=True)
    else:
        out = out.fillna(0).reset_index(drop=True)
    return out