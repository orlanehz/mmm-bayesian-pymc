import pandas as pd

from mmm.tuning import tune_adstock_decays


def test_tuning_runs_and_returns_decays():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=60, freq="W-MON"),
            "sales": [200 + i for i in range(60)],
            "mdsp_tv": [100.0] * 60,
            "mdsp_social": [20.0] * 60,
            "hldy_1": [0, 0, 1, 0, 0] * 12,
            "hldy_2": [0] * 60,
        }
    )

    res = tune_adstock_decays(
        df,
        date_col="date",
        target_col="sales",
        channel_cols=["mdsp_tv", "mdsp_social"],
        control_cols=["hldy_1", "hldy_2"],
        decay_candidates=[0.0, 0.3],
        metric="rmse",
        test_size=0.2,
        seasonal_order=0,  # keep fast
        target_transform="log1p",
    )

    assert set(res.best_decays.keys()) == {"mdsp_tv", "mdsp_social"}
    assert len(res.results_df) == 2 ** 2
    assert res.best_score >= 0.0