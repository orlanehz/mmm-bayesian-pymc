from __future__ import annotations

from pathlib import Path

import pandas as pd

from mmm.data_io import run_data_pipeline


def test_daily_weekly_invariants(tmp_path: Path):
    # --- Build a tiny "public dataset" locally (weekly) ---
    weekly = pd.DataFrame(
        {
            "wk_strt_dt": pd.to_datetime(["2024-01-01", "2024-01-08"]),  # Mondays
            "sales": [700.0, 1400.0],
            "mdsp_tv": [70.0, 140.0],
            "mdsp_radio": [35.0, 70.0],
            "mdsp_online": [14.0, 28.0],
            "mdsp_social": [7.0, 14.0],
            "hldy_1": [0, 1],
            "hldy_2": [0, 0],
        }
    )

    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    local_csv = raw_dir / "local_source.csv"
    weekly.to_csv(local_csv, index=False)

    # --- Minimal config (mirrors your base.yaml keys) ---
    cfg = {
        "paths": {"raw_data": "data/raw", "processed_data": "data/processed", "artifacts": "artifacts"},
        "data": {
            "source_url": local_csv.as_posix(),  # local path works with pd.read_csv
            "source_date_col": "wk_strt_dt",
            "source_target_col": "sales",
            "source_channel_cols": ["mdsp_tv", "mdsp_radio", "mdsp_online", "mdsp_social"],
            "source_control_cols": ["hldy_1", "hldy_2"],
            "derive_daily": {"enabled": True, "distribution": "uniform"},
            "aggregate_to_weekly": {"enabled": True, "week_start": "MON"},
            "processed_date_col": "date",
            "processed_target_col": "sales",
            "processed_channel_cols": ["mdsp_tv", "mdsp_radio", "mdsp_online", "mdsp_social"],
            "processed_control_cols": ["hldy_1", "hldy_2"],
        },
    }

    artifacts = run_data_pipeline(config=cfg, project_root=tmp_path)

    daily = pd.read_parquet(artifacts.daily_derived_parquet)
    weekly_model = pd.read_parquet(artifacts.weekly_model_parquet)

    # 1) 7 daily rows per weekly row
    assert len(daily) == len(weekly) * 7

    # 2) Totals preserved after daily split + weekly re-aggregation
    # (sales and spends should match original weekly totals exactly)
    assert weekly_model["sales"].sum() == weekly["sales"].sum()
    for c in ["mdsp_tv", "mdsp_radio", "mdsp_online", "mdsp_social"]:
        assert weekly_model[c].sum() == weekly[c].sum()

    # 3) Weekly model has one row per week and sorted unique dates
    assert weekly_model["date"].is_monotonic_increasing
    assert weekly_model["date"].is_unique
    assert len(weekly_model) == len(weekly)