from __future__ import annotations

import numpy as np
import pandas as pd
import arviz as az

from mmm.predict import _stack_posterior


def media_feature_indices(feature_names: list[str]) -> dict[str, int]:
    """
    Return indices for media features (those starting with 'media_').
    """
    out = {}
    for i, name in enumerate(feature_names):
        if name.startswith("media_"):
            out[name] = i
    return out


def compute_contributions(
    idata: az.InferenceData,
    X: np.ndarray,
    feature_names: list[str],
    *,
    hdi_prob: float = 0.9,
) -> pd.DataFrame:
    """
    Returns a dataframe with per-feature contribution summary across time:
    mean, hdi_low, hdi_high (on the model scale).
    """
    beta = _stack_posterior(idata, "beta")                 # (S, K)
    media_idx = media_feature_indices(feature_names)

    rows = []
    for feat, j in media_idx.items():
        # contribution per sample per time: beta_sj * X_tj
        contrib = beta[:, j][:, None] * X[:, j][None, :]   # (S, N)
        mean = contrib.mean(axis=0)
        lo = np.quantile(contrib, (1 - hdi_prob) / 2, axis=0)
        hi = np.quantile(contrib, 1 - (1 - hdi_prob) / 2, axis=0)

        rows.append(
            pd.DataFrame(
                {
                    "feature": feat,
                    "t": np.arange(X.shape[0]),
                    "mean": mean,
                    "hdi_low": lo,
                    "hdi_high": hi,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)