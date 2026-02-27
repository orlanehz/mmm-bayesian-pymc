"""Validation des données en entrée."""
from __future__ import annotations

import pandas as pd


class DataValidationError(ValueError):
    pass


def assert_time_index(df: pd.DataFrame, date_col: str) -> None:
    if date_col not in df.columns:
        raise DataValidationError(f"Missing date column: {date_col}")

    if df[date_col].isna().any():
        raise DataValidationError(f"Null dates found in {date_col}")

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise DataValidationError(f"{date_col} must be datetime dtype")

    if not df[date_col].is_monotonic_increasing:
        raise DataValidationError(f"{date_col} must be sorted ascending")

    if not df[date_col].is_unique:
        raise DataValidationError(f"{date_col} must be unique (one row per period)")


def assert_non_negative(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")

    for c in cols:
        if df[c].isna().any():
            raise DataValidationError(f"Null values found in {c}")
        if (df[c] < 0).any():
            raise DataValidationError(f"Negative values found in {c}")


def validate_processed_weekly(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    channel_cols: list[str],
    control_cols: list[str],
) -> None:
    assert_time_index(df, date_col=date_col)

    if target_col not in df.columns:
        raise DataValidationError(f"Missing target column: {target_col}")
    if df[target_col].isna().any():
        raise DataValidationError(f"Null values found in target {target_col}")

    # sales/conversions typically non-negative
    if (df[target_col] < 0).any():
        raise DataValidationError(f"Negative values found in target {target_col}")

    assert_non_negative(df, cols=channel_cols)

    # controls can be nullable, but if present, must be numeric
    for c in control_cols:
        if c not in df.columns:
            raise DataValidationError(f"Missing control column: {c}")
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise DataValidationError(f"Control column must be numeric: {c}")