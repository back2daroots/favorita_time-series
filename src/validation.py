# src/validation.py
from __future__ import annotations
import pandas as pd
from typing import Iterator, Tuple

def rolling_splits(
    df: pd.DataFrame,
    date_col: str,
    splits: int,
    horizon_days: int,
    step_days: int
) -> Iterator[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Генерируем (train_end, val_end) даты для rolling backtest.
    На каждом фолде:
      train: <= train_end
      val:   (train_end, val_end]
    """
    dates = pd.to_datetime(df[date_col]).sort_values().unique()
    last = dates.max()
    # начинаем с конца, откатываясь назад step_days на каждом фолде
    for i in range(splits, 0, -1):
        val_end   = last - pd.Timedelta(days=step_days * (i - 1))
        train_end = val_end - pd.Timedelta(days=horizon_days)
        yield train_end, val_end