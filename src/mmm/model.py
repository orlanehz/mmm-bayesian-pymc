"""Définition du modèle PyMC."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pymc as pm
import arviz as az

from mmm.features import DesignMatrix


@dataclass(frozen=True)
class FitResult:
    idata: az.InferenceData
    feature_names: list[str]


def fit_mmm(
    dm: DesignMatrix,
    *,
    draws: int = 500,
    tune: int = 500,
    chains: int = 2,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> FitResult:
    X = dm.X
    y = dm.y
    n_features = X.shape[1]

    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=n_features)
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        mu = intercept + pm.math.dot(X, beta)

        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=False,
        )

        # Posterior predictive for evaluation
        idata = pm.sample_posterior_predictive(
            idata,
            var_names=["y_obs"],
            random_seed=random_seed,
            progressbar=False,
        )

    return FitResult(idata=idata, feature_names=dm.feature_names)


def save_fit_result(res: FitResult, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save idata as netcdf
    idata_path = out / "idata.nc"
    res.idata.to_netcdf(idata_path)

    # Save metadata
    meta = {"feature_names": res.feature_names}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_fit_result(out_dir: str | Path) -> FitResult:
    out = Path(out_dir)
    idata = az.from_netcdf(out / "idata.nc")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    return FitResult(idata=idata, feature_names=meta["feature_names"])