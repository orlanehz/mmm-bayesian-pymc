import pandas as pd
import pytest

from mmm.validation import DataValidationError, validate_processed_weekly


def test_validate_processed_weekly_ok():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="W-MON"),
            "sales": [10.0, 11.0, 9.0],
            "mdsp_tv": [100.0, 110.0, 90.0],
            "mdsp_social": [20.0, 22.0, 18.0],
            "hldy_1": [0, 1, 0],
            "hldy_2": [0, 0, 0],
        }
    )

    validate_processed_weekly(
        df,
        date_col="date",
        target_col="sales",
        channel_cols=["mdsp_tv", "mdsp_social"],
        control_cols=["hldy_1", "hldy_2"],
    )


def test_validate_processed_weekly_fails_on_unsorted_dates():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-08", "2024-01-01"]),
            "sales": [10.0, 11.0],
            "mdsp_tv": [100.0, 110.0],
            "hldy_1": [0, 1],
            "hldy_2": [0, 0],
        }
    )

    with pytest.raises(DataValidationError):
        validate_processed_weekly(
            df,
            date_col="date",
            target_col="sales",
            channel_cols=["mdsp_tv"],
            control_cols=["hldy_1", "hldy_2"],
        )


def test_validate_processed_weekly_fails_on_negative_spend():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2, freq="W-MON"),
            "sales": [10.0, 11.0],
            "mdsp_tv": [100.0, -110.0],
            "hldy_1": [0, 1],
            "hldy_2": [0, 0],
        }
    )

    with pytest.raises(DataValidationError):
        validate_processed_weekly(
            df,
            date_col="date",
            target_col="sales",
            channel_cols=["mdsp_tv"],
            control_cols=["hldy_1", "hldy_2"],
        )