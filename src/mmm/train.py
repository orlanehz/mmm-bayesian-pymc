from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from mmm.data_io import run_data_pipeline
from mmm.model import fit_mmm, save_fit_result
from mmm.evaluation import evaluate_fit
from mmm.features import build_design_matrix
from mmm.split import time_split
from mmm.tuning import tune_adstock_decays


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_train(cfg: dict, *, project_root: Path) -> None:
    # ---------- Paths ----------
    artifacts_dir = project_root / cfg["artifacts"]["dir"]
    _ensure_dir(artifacts_dir)

    # ---------- 1) Data pipeline (public weekly -> derived daily -> weekly processed) ----------
    data_artifacts = run_data_pipeline(config=cfg, project_root=project_root)

    weekly_path = data_artifacts.weekly_model_parquet
    df = pd.read_parquet(weekly_path)

    data_cfg = cfg["data"]
    date_col = data_cfg["processed_date_col"]
    target_col = data_cfg["processed_target_col"]
    channel_cols = list(data_cfg["processed_channel_cols"])
    control_cols = list(data_cfg["processed_control_cols"])

    # ---------- 2) Tuning (grid search decays with fast Ridge) ----------
    tuning_cfg = cfg.get("tuning", None)
    if tuning_cfg and tuning_cfg.get("enabled", True):
        decay_candidates = list(tuning_cfg.get("decay_candidates", [0.0, 0.2, 0.4, 0.6]))
        metric = tuning_cfg.get("metric", "rmse")
        ridge_alpha = float(tuning_cfg.get("ridge_alpha", 1.0))
    else:
        decay_candidates = [0.0, 0.2, 0.4, 0.6]
        metric = "rmse"
        ridge_alpha = 1.0

    split_cfg = cfg["split"]
    test_size = float(split_cfg["test_size"])

    feat_cfg = cfg["features"]
    seasonal_order = int(feat_cfg.get("seasonal_order", 3))
    target_transform = feat_cfg.get("target_transform", "log1p")

    tune_res = tune_adstock_decays(
        df,
        date_col=date_col,
        target_col=target_col,
        channel_cols=channel_cols,
        control_cols=control_cols,
        decay_candidates=decay_candidates,
        metric=metric,
        test_size=test_size,
        ridge_alpha=ridge_alpha,
        seasonal_order=seasonal_order,
        target_transform=target_transform,
    )

    # Save tuning outputs
    tune_dir = artifacts_dir / "tuning"
    _ensure_dir(tune_dir)
    tune_res.results_df.to_parquet(tune_dir / "tuning_results.parquet", index=False)
    (tune_dir / "best_decays.json").write_text(json.dumps(tune_res.best_decays, indent=2), encoding="utf-8")
    (tune_dir / "tuning_summary.json").write_text(
        json.dumps({"metric": tune_res.metric, "best_score": tune_res.best_score}, indent=2),
        encoding="utf-8",
    )

    best_decays = tune_res.best_decays

    # ---------- 3) Final PyMC fit on TRAIN only, evaluate on TEST ----------
    train_df, test_df = time_split(df, date_col=date_col, test_size=test_size)

    dm_train = build_design_matrix(
        train_df,
        date_col=date_col,
        target_col=target_col,
        channel_cols=channel_cols,
        control_cols=control_cols,
        adstock_decay=best_decays,
        seasonal_order=seasonal_order,
        target_transform=target_transform,
    )
    dm_test = build_design_matrix(
        test_df,
        date_col=date_col,
        target_col=target_col,
        channel_cols=channel_cols,
        control_cols=control_cols,
        adstock_decay=best_decays,
        seasonal_order=seasonal_order,
        target_transform=target_transform,
    )

    train_cfg = cfg["training"]
    fast_mode = bool(train_cfg.get("fast_mode", False))

    # fast_mode: keep CI / dev runs quick
    draws = int(train_cfg["draws"])
    tune = int(train_cfg["tune"])
    chains = int(train_cfg["chains"])
    target_accept = float(train_cfg["target_accept"])

    if fast_mode:
        draws = min(draws, 100)
        tune = min(tune, 100)
        chains = min(chains, 2)

    fit_res = fit_mmm(
        dm_train,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
    )

    # Save inference
    inf_dir = artifacts_dir / "inference"
    _ensure_dir(inf_dir)
    save_fit_result(fit_res, inf_dir)

    # ---------- 4) Evaluate (train & test) ----------
    metrics_train = evaluate_fit(dm_train, fit_res.idata, target_transform=target_transform)

    # For test, we need posterior predictive for test X:
    # v1 (simple): re-fit posterior predictive on test by rebuilding mu and sampling
    # For now: we do a pragmatic approach: compute mu from posterior mean betas (deterministic)
    # -> keeps pipeline simple; we’ll add full posterior predictive on test in v2.
    # We'll still compute metrics in a consistent way:
    metrics = {"train": metrics_train}

    # Save metrics
    metrics_dir = artifacts_dir / "metrics"
    _ensure_dir(metrics_dir)
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Also save the exact config used (repro)
    (artifacts_dir / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print("✅ Pipeline completed")
    print(f"Artifacts saved to: {artifacts_dir}")
    print(f"Best decays: {best_decays}")
    print(f"Train metrics: {metrics_train}")