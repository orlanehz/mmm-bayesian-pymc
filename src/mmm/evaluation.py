"""Evaluation des performances."""
from __future__ import annotations

import numpy as np
import arviz as az

from mmm.features import DesignMatrix


def posterior_predict_mean(idata: az.InferenceData) -> np.ndarray:
    # y_obs is in posterior_predictive: dims (chain, draw, obs)
    y_pp = idata.posterior_predictive["y_obs"].values
    return y_pp.mean(axis=(0, 1))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-9)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def invert_target_transform(y_transformed: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log1p":
        return np.expm1(y_transformed)
    if transform == "none":
        return y_transformed
    raise ValueError(f"Unknown transform: {transform}")


def evaluate_fit(
    dm: DesignMatrix,
    idata: az.InferenceData,
    *,
    target_transform: str,
) -> dict[str, float]:
    y_pred_t = posterior_predict_mean(idata)
    y_true_t = dm.y

    # metrics in transformed space
    metrics = {
        "rmse_transformed": rmse(y_true_t, y_pred_t),
        "mape_transformed": mape(y_true_t, y_pred_t),
    }

    # metrics in original space
    y_true = invert_target_transform(y_true_t, target_transform)
    y_pred = invert_target_transform(y_pred_t, target_transform)

    metrics.update(
        {
            "rmse": rmse(y_true, y_pred),
            "mape": mape(y_true, y_pred),
        }
    )
    return metrics