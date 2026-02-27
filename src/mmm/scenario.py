from __future__ import annotations

import numpy as np
import pandas as pd
import arviz as az

from mmm.features import build_design_matrix
from mmm.predict import posterior_predictive_normal_linear


def apply_budget_scenario(
    df: pd.DataFrame,
    channel_cols: list[str],
    multipliers: dict[str, float],
) -> pd.DataFrame:
    """
    Apply multiplicative budget change per channel.
    Example: {"mdsp_tv": 1.2, "mdsp_social": 0.8}
    """
    df_new = df.copy()

    for c in channel_cols:
        if c in multipliers:
            df_new[c] = df_new[c] * multipliers[c]

    return df_new


def simulate_scenario(
    idata: az.InferenceData,
    df_base: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    channel_cols: list[str],
    control_cols: list[str],
    adstock_decay: dict[str, float],
    seasonal_order: int,
    target_transform: str,
    multipliers: dict[str, float],
    random_seed: int = 42,
):
    """
    Returns delta distribution between scenario and baseline.
    """

    # Baseline design
    dm_base = build_design_matrix(
        df_base,
        date_col=date_col,
        target_col=target_col,
        channel_cols=channel_cols,
        control_cols=control_cols,
        adstock_decay=adstock_decay,
        seasonal_order=seasonal_order,
        target_transform=target_transform,
    )

    # Scenario df
    df_scenario = apply_budget_scenario(df_base, channel_cols, multipliers)

    dm_scenario = build_design_matrix(
        df_scenario,
        date_col=date_col,
        target_col=target_col,
        channel_cols=channel_cols,
        control_cols=control_cols,
        adstock_decay=adstock_decay,
        seasonal_order=seasonal_order,
        target_transform=target_transform,
    )

    # Posterior predictive
    pp_base = posterior_predictive_normal_linear(idata, dm_base.X, random_seed=random_seed)
    pp_scen = posterior_predictive_normal_linear(idata, dm_scenario.X, random_seed=random_seed)

    delta = pp_scen.sum(axis=1) - pp_base.sum(axis=1)

    return delta

def summarize_delta(delta: np.ndarray, hdi_prob: float = 0.9) -> dict:
    return {
        "mean": delta.mean(),
        "median": np.median(delta),
        "hdi_low": np.quantile(delta, (1 - hdi_prob) / 2),
        "hdi_high": np.quantile(delta, 1 - (1 - hdi_prob) / 2),
    }