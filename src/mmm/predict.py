from __future__ import annotations

import numpy as np
import arviz as az


def _stack_posterior(idata: az.InferenceData, var: str) -> np.ndarray:
    """Return posterior samples stacked as (n_samples, ...)"""
    arr = idata.posterior[var].values  # (chain, draw, ...)
    return arr.reshape(-1, *arr.shape[2:])


def posterior_predictive_normal_linear(
    idata: az.InferenceData,
    X: np.ndarray,
    *,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Generate posterior predictive samples for y ~ Normal(intercept + X @ beta, sigma).
    Returns array with shape (n_samples, n_obs).
    """
    rng = np.random.default_rng(random_seed)

    intercept = _stack_posterior(idata, "intercept").reshape(-1)          # (S,)
    beta = _stack_posterior(idata, "beta")                                # (S, K)
    sigma = _stack_posterior(idata, "sigma").reshape(-1)                  # (S,)

    mu = intercept[:, None] + beta @ X.T                                  # (S, N)
    y = rng.normal(loc=mu, scale=sigma[:, None])
    return y


def pp_mean(pp: np.ndarray) -> np.ndarray:
    return pp.mean(axis=0)


def pp_hdi(pp: np.ndarray, hdi_prob: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    lo = np.quantile(pp, (1 - hdi_prob) / 2, axis=0)
    hi = np.quantile(pp, 1 - (1 - hdi_prob) / 2, axis=0)
    return lo, hi