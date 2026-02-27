"""Construction des features."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from mmm.transforms import geometric_adstock, log_saturation


@dataclass(frozen=True)
class DesignMatrix:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]


def fourier_series(t: np.ndarray, period: float, order: int) -> tuple[np.ndarray, list[str]]:
    """Return Fourier series features for seasonality."""
    feats = []
    names = []
    for k in range(1, order + 1):
        feats.append(np.sin(2 * np.pi * k * t / period))
        names.append(f"sin_{int(period)}_{k}")
        feats.append(np.cos(2 * np.pi * k * t / period))
        names.append(f"cos_{int(period)}_{k}")
    return np.column_stack(feats), names


def build_design_matrix(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    channel_cols: list[str],
    control_cols: list[str],
    adstock_decay: dict[str, float],
    add_trend: bool = True,
    seasonal_period: float = 52.0,
    seasonal_order: int = 3,
    target_transform: str = "log1p",
) -> DesignMatrix:
    df = df.copy()
    df = df.sort_values(date_col).reset_index(drop=True)

    # target
    y_raw = df[target_col].astype(float).to_numpy()
    if target_transform == "log1p":
        y = np.log1p(np.maximum(y_raw, 0.0))
    elif target_transform == "none":
        y = y_raw
    else:
        raise ValueError(f"Unknown target_transform: {target_transform}")

    X_parts = []
    names = []

    # trend as time index
    if add_trend:
        t = np.arange(len(df), dtype=float)
        t = (t - t.mean()) / (t.std() + 1e-12)
        X_parts.append(t.reshape(-1, 1))
        names.append("trend")
    else:
        t = np.arange(len(df), dtype=float)

    # seasonality
    if seasonal_order > 0:
        seas, seas_names = fourier_series(t=np.arange(len(df), dtype=float), period=seasonal_period, order=seasonal_order)
        X_parts.append(seas)
        names.extend(seas_names)

    # controls
    for c in control_cols:
        x = df[c].astype(float).to_numpy()
        # standardize controls (optional but usually helpful)
        x = (x - x.mean()) / (x.std() + 1e-12)
        X_parts.append(x.reshape(-1, 1))
        names.append(f"ctrl_{c}")

    # channels: adstock + saturation
    for c in channel_cols:
        spend = df[c].astype(float).to_numpy()
        decay = adstock_decay.get(c)
        if decay is None:
            raise ValueError(f"Missing adstock decay for channel: {c}")

        ad = geometric_adstock(spend, decay=decay)
        sat = log_saturation(ad)

        # standardize so priors are comparable
        sat = (sat - sat.mean()) / (sat.std() + 1e-12)

        X_parts.append(sat.reshape(-1, 1))
        names.append(f"media_{c}")

    X = np.column_stack(X_parts) if X_parts else np.empty((len(df), 0))
    return DesignMatrix(X=X, y=y, feature_names=names)