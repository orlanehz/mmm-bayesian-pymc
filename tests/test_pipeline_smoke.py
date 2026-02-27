import pandas as pd

from mmm.features import build_design_matrix
from mmm.model import fit_mmm
from mmm.evaluation import evaluate_fit


def test_full_pipeline_smoke():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=30, freq="W-MON"),
            "sales": [100 + i for i in range(30)],
            "mdsp_tv": [50.0] * 30,
            "mdsp_radio": [20.0] * 30,
            "mdsp_online": [10.0] * 30,
            "mdsp_social": [5.0] * 30,
            "hldy_1": [0, 0, 1, 0, 0] * 6,
            "hldy_2": [0] * 30,
        }
    )

    dm = build_design_matrix(
        df,
        date_col="date",
        target_col="sales",
        channel_cols=["mdsp_tv", "mdsp_radio", "mdsp_online", "mdsp_social"],
        control_cols=["hldy_1", "hldy_2"],
        adstock_decay={
            "mdsp_tv": 0.5,
            "mdsp_radio": 0.3,
            "mdsp_online": 0.2,
            "mdsp_social": 0.2,
        },
        seasonal_order=0,  # keep test fast
        target_transform="log1p",
    )

    res = fit_mmm(dm, draws=50, tune=50, chains=2, target_accept=0.9)
    metrics = evaluate_fit(dm, res.idata, target_transform="log1p")

    assert "rmse" in metrics
    assert metrics["rmse"] >= 0.0