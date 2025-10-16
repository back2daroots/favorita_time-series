import pandas as pd
def load_favorita(paths: dict, date_col: str):
    train = pd.read_csv(paths["train_path"])
    test  = pd.read_csv(paths["test_path"])
    oil   = pd.read_csv(paths["oil_path"])
    hol   = pd.read_csv(paths["hol_path"])
    trans = pd.read_csv(paths["trans_path"])

    # parse dates
    for df in (train, test, oil, hol, trans):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])

    # basic sorting
    train = train.sort_values([date_col, "store_nbr", "family"]).reset_index(drop=True)
    test  = test.sort_values([date_col, "store_nbr", "family"]).reset_index(drop=True)

    return {"train": train, "test": test, "oil": oil, "hol": hol, "trans": trans}

def train_test_split_time(df: pd.DataFrame, date_col: str, holdout_days: int):
    cutoff = df[date_col].max() - pd.Timedelta(days=holdout_days)
    train = df[df[date_col] <= cutoff].copy()
    test  = df[df[date_col]  > cutoff].copy()
    return train, test