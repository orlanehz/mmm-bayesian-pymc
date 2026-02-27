from __future__ import annotations

import numpy as np
import pandas as pd
import arviz as az

from mmm.predict import _stack_posterior


def _hdi(arr: np.ndarray, hdi_prob: float) -> tuple[np.ndarray, np.ndarray]:
    lo = np.quantile(arr, (1 - hdi_prob) / 2, axis=0)
    hi = np.quantile(arr, 1 - (1 - hdi_prob) / 2, axis=0)
    return lo, hi


def compute_baseline_vs_media(
    idata: az.InferenceData,
    X: np.ndarray,
    feature_names: list[str],
    dates: pd.Series,
    *,
    hdi_prob: float = 0.9,
) -> pd.DataFrame:
    """
    Decompose mu_t into baseline_t + media_t using posterior draws.
    Returns long dataframe with component in {"baseline", "media", "total"}.
    Values are on the model scale (e.g., log1p(sales) if you used that).
    """
    beta = _stack_posterior(idata, "beta")          # (S, K)
    intercept = _stack_posterior(idata, "intercept").reshape(-1)  # (S,)

    # indices
    media_idx = [i for i, n in enumerate(feature_names) if n.startswith("media_")]
    base_idx = [i for i in range(len(feature_names)) if i not in media_idx]

    # Baseline: intercept + X_base @ beta_base
    X_base = X[:, base_idx] if base_idx else np.zeros((X.shape[0], 0))
    beta_base = beta[:, base_idx] if base_idx else np.zeros((beta.shape[0], 0))
    baseline = intercept[:, None] + (beta_base @ X_base.T)  # (S, N)

    # Media: X_media @ beta_media
    X_media = X[:, media_idx] if media_idx else np.zeros((X.shape[0], 0))
    beta_media = beta[:, media_idx] if media_idx else np.zeros((beta.shape[0], 0))
    media = beta_media @ X_media.T  # (S, N)

    total = baseline + media

    def pack(component: str, arr: np.ndarray) -> pd.DataFrame:
        mean = arr.mean(axis=0)
        lo, hi = _hdi(arr, hdi_prob=hdi_prob)
        return pd.DataFrame(
            {
                "date": pd.to_datetime(dates).values,
                "component": component,
                "mean": mean,
                "hdi_low": lo,
                "hdi_high": hi,
            }
        )

    out = pd.concat(
        [pack("baseline", baseline), pack("media", media), pack("total", total)],
        ignore_index=True,
    ).sort_values(["date", "component"]).reset_index(drop=True)

    return out