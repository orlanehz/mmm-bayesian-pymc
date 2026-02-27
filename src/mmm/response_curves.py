from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import arviz as az

from mmm.transforms import geometric_adstock, log_saturation
from mmm.predict import posterior_predictive_normal_linear
from mmm.evaluation import invert_target_transform


@dataclass(frozen=True)
class MediaScaler:
    mean: float
    std: float


def _fourier_series(t: np.ndarray, period: float, order: int) -> tuple[np.ndarray, list[str]]:
    feats = []
    names = []
    for k in range(1, order + 1):
        feats.append(np.sin(2 * np.pi * k * t / period))
        names.append(f"sin_{int(period)}_{k}")
        feats.append(np.cos(2 * np.pi * k * t / period))
        names.append(f"cos_{int(period)}_{k}")
    return np.column_stack(feats), names


def _build_X_with_fixed_media_scalers(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    channel_cols: list[str],
    control_cols: list[str],
    adstock_decay: dict[str, float],
    add_trend: bool,
    seasonal_period: float,
    seasonal_order: int,
    target_transform: str,
    fixed_media_scalers: dict[str, MediaScaler],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = df.sort_values(date_col).reset_index(drop=True)

    # y
    y_raw = df[target_col].astype(float).to_numpy()
    if target_transform == "log1p":
        y = np.log1p(np.maximum(y_raw, 0.0))
    elif target_transform == "none":
        y = y_raw
    else:
        raise ValueError(f"Unknown target_transform: {target_transform}")

    X_parts: list[np.ndarray] = []
    names: list[str] = []

    # trend
    if add_trend:
        t = np.arange(len(df), dtype=float)
        t = (t - t.mean()) / (t.std() + 1e-12)
        X_parts.append(t.reshape(-1, 1))
        names.append("trend")
        t_for_season = np.arange(len(df), dtype=float)
    else:
        t_for_season = np.arange(len(df), dtype=float)

    # seasonality
    if seasonal_order > 0:
        seas, seas_names = _fourier_series(t=t_for_season, period=seasonal_period, order=seasonal_order)
        X_parts.append(seas)
        names.extend(seas_names)

    # controls (standardized per run; controls unchanged across scenarios usually)
    for c in control_cols:
        x = df[c].astype(float).to_numpy()
        x = (x - x.mean()) / (x.std() + 1e-12)
        X_parts.append(x.reshape(-1, 1))
        names.append(f"ctrl_{c}")

    # media (fixed scalers from baseline)
    for c in channel_cols:
        spend = df[c].astype(float).to_numpy()
        decay = adstock_decay[c]

        ad = geometric_adstock(spend, decay=decay)
        sat = log_saturation(ad)

        sc = fixed_media_scalers[c]
        sat_std = (sat - sc.mean) / (sc.std + 1e-12)

        X_parts.append(sat_std.reshape(-1, 1))
        names.append(f"media_{c}")

    X = np.column_stack(X_parts) if X_parts else np.empty((len(df), 0))
    return X, y, names


def _compute_baseline_media_scalers(
    df: pd.DataFrame,
    *,
    date_col: str,
    channel_cols: list[str],
    adstock_decay: dict[str, float],
) -> dict[str, MediaScaler]:
    df = df.sort_values(date_col).reset_index(drop=True)
    scalers: dict[str, MediaScaler] = {}
    for c in channel_cols:
        spend = df[c].astype(float).to_numpy()
        ad = geometric_adstock(spend, decay=adstock_decay[c])
        sat = log_saturation(ad)
        scalers[c] = MediaScaler(mean=float(sat.mean()), std=float(sat.std()))
    return scalers


def _summarize(samples: np.ndarray, hdi_prob: float = 0.9) -> dict[str, float]:
    return {
        "mean": float(samples.mean()),
        "median": float(np.median(samples)),
        "hdi_low": float(np.quantile(samples, (1 - hdi_prob) / 2)),
        "hdi_high": float(np.quantile(samples, 1 - (1 - hdi_prob) / 2)),
    }


def compute_response_curve(
    idata: az.InferenceData,
    df_base: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    channel_cols: list[str],
    control_cols: list[str],
    adstock_decay: dict[str, float],
    channel: str,
    grid: np.ndarray,
    grid_type: Literal["multiplier", "absolute"] = "multiplier",
    seasonal_order: int = 3,
    seasonal_period: float = 52.0,
    add_trend: bool = True,
    target_transform: str = "log1p",
    hdi_prob: float = 0.9,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Build a response curve for one channel.
    - grid_type="multiplier": grid values multiply the baseline spend series
    - grid_type="absolute": grid values replace spend with a constant per period

    Returns a df with spend_total, uplift_total_mean/median/HDI on original scale.
    """
    if channel not in channel_cols:
        raise ValueError(f"channel must be in channel_cols. Got {channel}")

    df_base = df_base.sort_values(date_col).reset_index(drop=True)

    # baseline scalers (fixed for media standardization)
    fixed_scalers = _compute_baseline_media_scalers(
        df_base, date_col=date_col, channel_cols=channel_cols, adstock_decay=adstock_decay
    )

    # baseline X/y
    X0, y0_t, feat_names0 = _build_X_with_fixed_media_scalers(
        df_base,
        date_col=date_col,
        target_col=target_col,
        channel_cols=channel_cols,
        control_cols=control_cols,
        adstock_decay=adstock_decay,
        add_trend=add_trend,
        seasonal_period=seasonal_period,
        seasonal_order=seasonal_order,
        target_transform=target_transform,
        fixed_media_scalers=fixed_scalers,
    )

    pp0 = posterior_predictive_normal_linear(idata, X0, random_seed=random_seed)
    y0 = invert_target_transform(pp0, target_transform)  # (S, N) original scale
    base_total = y0.sum(axis=1)  # (S,)

    rows = []
    base_spend_total = float(df_base[channel].sum())

    for g in grid:
        df_s = df_base.copy()
        if grid_type == "multiplier":
            df_s[channel] = df_s[channel] * float(g)
        else:
            df_s[channel] = float(g)

        Xs, _, feat_names_s = _build_X_with_fixed_media_scalers(
            df_s,
            date_col=date_col,
            target_col=target_col,
            channel_cols=channel_cols,
            control_cols=control_cols,
            adstock_decay=adstock_decay,
            add_trend=add_trend,
            seasonal_period=seasonal_period,
            seasonal_order=seasonal_order,
            target_transform=target_transform,
            fixed_media_scalers=fixed_scalers,
        )

        # sanity: same feature order
        if feat_names_s != feat_names0:
            raise RuntimeError("Feature names mismatch between baseline and scenario design matrices.")

        pps = posterior_predictive_normal_linear(idata, Xs, random_seed=random_seed)
        ys = invert_target_transform(pps, target_transform)  # (S, N)
        scen_total = ys.sum(axis=1)  # (S,)

        uplift = scen_total - base_total  # (S,)

        spend_total = float(df_s[channel].sum())
        summary = _summarize(uplift, hdi_prob=hdi_prob)
        rows.append(
            {
                "channel": channel,
                "grid_value": float(g),
                "grid_type": grid_type,
                "spend_total": spend_total,
                "base_spend_total": base_spend_total,
                **{f"uplift_total_{k}": v for k, v in summary.items()},
            }
        )

    return pd.DataFrame(rows).sort_values("spend_total").reset_index(drop=True)