"""Découpage train/validation/test."""
from __future__ import annotations

import pandas as pd


def time_split(df: pd.DataFrame, date_col: str, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)
    cut = int(n * (1 - test_size))
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()
    # no leakage
    assert train[date_col].max() < test[date_col].min()
    return train, test