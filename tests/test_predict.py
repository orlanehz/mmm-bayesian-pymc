import numpy as np
import pandas as pd

from mmm.features import build_design_matrix
from mmm.model import fit_mmm
from mmm.predict import posterior_predictive_normal_linear


def test_pp_test_shape():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=30, freq="W-MON"),
            "sales": [100 + i for i in range(30)],
            "mdsp_tv": [50.0] * 30,
            "mdsp_social": [5.0] * 30,
            "hldy_1": [0] * 30,
            "hldy_2": [0] * 30,
        }
    )
    dm = build_design_matrix(
        df,
        date_col="date",
        target_col="sales",
        channel_cols=["mdsp_tv", "mdsp_social"],
        control_cols=["hldy_1", "hldy_2"],
        adstock_decay={"mdsp_tv": 0.3, "mdsp_social": 0.2},
        seasonal_order=0,
        target_transform="log1p",
    )
    res = fit_mmm(dm, draws=30, tune=30, chains=2, target_accept=0.9)
    pp = posterior_predictive_normal_linear(res.idata, dm.X)
    assert pp.ndim == 2
    assert pp.shape[1] == dm.X.shape[0]