from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mmm.features import build_design_matrix
from mmm.split import time_split
from mmm.evaluation import rmse, mape, invert_target_transform


@dataclass(frozen=True)
class TuningResult:
    best_decays: dict[str, float]
    best_score: float
    metric: str
    results_df: pd.DataFrame


def _make_decay_grid(
    channel_cols: list[str],
    candidates: list[float],
) -> Iterable[dict[str, float]]:
    for combo in product(candidates, repeat=len(channel_cols)):
        yield dict(zip(channel_cols, combo))


def tune_adstock_decays(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    channel_cols: list[str],
    control_cols: list[str],
    decay_candidates: list[float],
    metric: str = "rmse",
    test_size: float = 0.2,
    ridge_alpha: float = 1.0,
    seasonal_order: int = 3,
    target_transform: str = "log1p",
) -> TuningResult:
    """
    Grid search over adstock decay values using a fast frequentist model (Ridge).
    Returns best decays and a full results table.
    """
    train_df, test_df = time_split(df, date_col=date_col, test_size=test_size)

    rows = []
    best_decays = None
    best_score = float("inf")

    for decays in _make_decay_grid(channel_cols, decay_candidates):
        dm_train = build_design_matrix(
            train_df,
            date_col=date_col,
            target_col=target_col,
            channel_cols=channel_cols,
            control_cols=control_cols,
            adstock_decay=decays,
            seasonal_order=seasonal_order,
            target_transform=target_transform,
        )
        dm_test = build_design_matrix(
            test_df,
            date_col=date_col,
            target_col=target_col,
            channel_cols=channel_cols,
            control_cols=control_cols,
            adstock_decay=decays,
            seasonal_order=seasonal_order,
            target_transform=target_transform,
        )

        model = Ridge(alpha=ridge_alpha, fit_intercept=True)
        model.fit(dm_train.X, dm_train.y)
        pred_t = model.predict(dm_test.X)

        # Score on original scale (more interpretable)
        y_true = invert_target_transform(dm_test.y, target_transform)
        y_pred = invert_target_transform(pred_t, target_transform)

        if metric == "rmse":
            score = rmse(y_true, y_pred)
        elif metric == "mape":
            score = mape(y_true, y_pred)
        else:
            raise ValueError("metric must be 'rmse' or 'mape'")

        rows.append(
            {
                **{f"decay_{k}": v for k, v in decays.items()},
                "score": score,
            }
        )

        if score < best_score:
            best_score = score
            best_decays = decays

    results_df = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    assert best_decays is not None
    return TuningResult(
        best_decays=best_decays,
        best_score=best_score,
        metric=metric,
        results_df=results_df,
    )