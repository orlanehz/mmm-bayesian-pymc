"""Chargement/sauvegarde des données."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mmm.validation import validate_processed_weekly


@dataclass(frozen=True)
class DataArtifacts:
    weekly_source_csv: Path
    daily_derived_parquet: Path
    weekly_model_parquet: Path


def download_weekly_source(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    return df


def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="raise")
    return out


def _detect_holiday_controls(columns: list[str]) -> list[str]:
    return [col for col in columns if "hldy_" in col]


def derive_daily_from_weekly_uniform(
    weekly_df: pd.DataFrame,
    week_start_col: str,
    value_cols: list[str],
    keep_cols: list[str],
) -> pd.DataFrame:
    """
    Create 7 daily rows per weekly row by uniform split for value_cols.
    week_start_col: start date for the week (e.g., Monday).
    value_cols: numeric columns to split across days (sales + spends).
    keep_cols: columns to repeat as-is (controls, etc.).
    """
    df = weekly_df.copy()

    # Build 7 rows per week
    repeated = df.loc[df.index.repeat(7)].reset_index(drop=True)
    day_offsets = np.tile(np.arange(7), len(df))
    repeated["date"] = repeated[week_start_col] + pd.to_timedelta(day_offsets, unit="D")

    # Uniform split
    for c in value_cols:
        repeated[c] = repeated[c].astype(float) / 7.0

    # Keep only what we need + new daily date
    cols = ["date"] + keep_cols + value_cols
    return repeated[cols].sort_values("date").reset_index(drop=True)


def aggregate_daily_to_weekly(
    daily_df: pd.DataFrame,
    date_col: str,
    sum_cols: list[str],
    keep_last_cols: list[str],
    week_start: str = "MON",
) -> pd.DataFrame:
    """
    Aggregate daily to weekly.
    - sum_cols are summed over the week (sales, spends)
    - keep_last_cols are taken as last observation in the week (controls)
    Week anchored on Monday if week_start="MON".
    """
    df = daily_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="raise")

    # Pandas weekly period anchored:
    # W-MON means weekly frequency ending on Monday.
    # To get "week start Monday", easiest is to compute week_start_date:
    # week_start_date = date - weekday_offset (Mon=0)
    weekday = df[date_col].dt.weekday  # Mon=0
    df["week_start_date"] = df[date_col] - pd.to_timedelta(weekday, unit="D")

    agg = {c: "sum" for c in sum_cols}
    for c in keep_last_cols:
        agg[c] = "last"

    out = (
        df.sort_values(date_col)
        .groupby("week_start_date", as_index=False)
        .agg(agg)
        .rename(columns={"week_start_date": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    return out


def run_data_pipeline(
    *,
    config: dict,
    project_root: str | Path,
) -> DataArtifacts:
    root = Path(project_root)
    paths = config["paths"]
    raw_dir = root / paths["raw_data"]
    processed_dir = root / paths["processed_data"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config["data"]
    url = data_cfg["source_url"]

    src_date = data_cfg["source_date_col"]
    src_target = data_cfg["source_target_col"]
    src_channels = list(data_cfg["source_channel_cols"])
    src_controls = list(data_cfg.get("source_control_cols", []))

    processed_date = data_cfg["processed_date_col"]
    processed_target = data_cfg["processed_target_col"]
    processed_channels = list(data_cfg["processed_channel_cols"])
    processed_controls = list(data_cfg.get("processed_control_cols", []))

    # 1) Download weekly source
    weekly = download_weekly_source(url)
    weekly = _ensure_datetime(weekly, src_date)
    weekly = weekly.sort_values(src_date).reset_index(drop=True)

    # Optional auto-detection for holiday controls.
    if not src_controls:
        src_controls = _detect_holiday_controls(list(weekly.columns))
    if not processed_controls:
        processed_controls = list(src_controls)

    weekly_source_csv = raw_dir / "weekly_source.csv"
    weekly.to_csv(weekly_source_csv, index=False)

    # 2) Derive daily (uniform split)
    value_cols = [src_target] + src_channels
    keep_cols = src_controls  # controls replicated daily
    daily = derive_daily_from_weekly_uniform(
        weekly_df=weekly,
        week_start_col=src_date,
        value_cols=value_cols,
        keep_cols=keep_cols,
    )

    daily_derived_parquet = raw_dir / "daily_derived.parquet"
    daily.to_parquet(daily_derived_parquet, index=False)

    # 3) Aggregate back to weekly (model table)
    weekly_model = aggregate_daily_to_weekly(
        daily_df=daily,
        date_col="date",
        sum_cols=value_cols,
        keep_last_cols=keep_cols,
        week_start=data_cfg["aggregate_to_weekly"]["week_start"],
    )

    # 4) Rename columns to processed schema if needed
    rename_map = {src_target: processed_target}
    rename_map.update({c: c for c in src_channels})
    rename_map.update({c: c for c in src_controls})

    weekly_model = weekly_model.rename(columns=rename_map)
    # Ensure processed_date_col name
    if processed_date != "date":
        weekly_model = weekly_model.rename(columns={"date": processed_date})

    # 5) Validate processed weekly
    weekly_model = weekly_model.sort_values(processed_date).reset_index(drop=True)
    validate_processed_weekly(
        weekly_model,
        date_col=processed_date,
        target_col=processed_target,
        channel_cols=processed_channels,
        control_cols=processed_controls,
    )

    weekly_model_parquet = processed_dir / "weekly_model.parquet"
    weekly_model.to_parquet(weekly_model_parquet, index=False)

    return DataArtifacts(
        weekly_source_csv=weekly_source_csv,
        daily_derived_parquet=daily_derived_parquet,
        weekly_model_parquet=weekly_model_parquet,
    )
