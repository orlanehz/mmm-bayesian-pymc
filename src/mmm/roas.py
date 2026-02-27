from __future__ import annotations

import numpy as np
import pandas as pd
import arviz as az

from mmm.predict import _stack_posterior


def compute_roas(
    idata: az.InferenceData,
    X: np.ndarray,
    raw_spend: pd.DataFrame,
    feature_names: list[str],
    channel_cols: list[str],
    *,
    hdi_prob: float = 0.9,
) -> pd.DataFrame:
    """
    Compute Bayesian ROAS distribution per channel.
    Returns summary dataframe.
    """

    beta = _stack_posterior(idata, "beta")  # (S, K)
    roas_rows = []

    for channel in channel_cols:
        feat_name = f"media_{channel}"
        j = feature_names.index(feat_name)

        # Contribution samples
        contrib = beta[:, j][:, None] * X[:, j][None, :]
        total_contrib = contrib.sum(axis=1)

        total_spend = raw_spend[channel].sum()

        roas_samples = total_contrib / total_spend

        roas_rows.append(
            {
                "channel": channel,
                "mean": roas_samples.mean(),
                "median": np.median(roas_samples),
                "hdi_low": np.quantile(roas_samples, (1 - hdi_prob) / 2),
                "hdi_high": np.quantile(roas_samples, 1 - (1 - hdi_prob) / 2),
            }
        )

    return pd.DataFrame(roas_rows)