from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mmm.data_io import run_data_pipeline
from mmm.model import fit_mmm, save_fit_result
from mmm.evaluation import evaluate_fit, evaluate_predictions
from mmm.features import build_design_matrix
from mmm.split import time_split
from mmm.tuning import tune_adstock_decays
from mmm.predict import posterior_predictive_normal_linear, pp_mean
from mmm.contributions import compute_contributions
from mmm.roas import compute_roas
from mmm.plotting import plot_roas, plot_baseline_vs_media
from mmm.decomposition import compute_baseline_vs_media


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_train(cfg: dict, *, project_root: Path) -> None:
    # ---------- Paths ----------
    artifacts_dir = project_root / cfg["artifacts"]["dir"]
    _ensure_dir(artifacts_dir)

    metrics_dir = artifacts_dir / "metrics"
    _ensure_dir(metrics_dir)

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

    # --- Test posterior predictive (using posterior draws from the trained model) ---
    pp_test = posterior_predictive_normal_linear(fit_res.idata, dm_test.X, random_seed=42)
    y_pred_test_t = pp_mean(pp_test)
    metrics_test = evaluate_predictions(dm_test.y, y_pred_test_t, target_transform=target_transform)

    metrics = {"train": metrics_train, "test": metrics_test}

    # Baseline vs Media decomposition (train)
    decomp_df = compute_baseline_vs_media(
        fit_res.idata,
        dm_train.X,
        fit_res.feature_names,
        dates=train_df[date_col],
        hdi_prob=0.9,
    )

    decomp_dir = artifacts_dir / "decomposition"
    _ensure_dir(decomp_dir)
    decomp_df.to_parquet(decomp_dir / "baseline_vs_media_train.parquet", index=False)

    fig_dir = artifacts_dir / "figures"
    _ensure_dir(fig_dir)
    plot_baseline_vs_media(
        decomp_df,
        save_path=str(fig_dir / "baseline_vs_media_train.png"),
    )

    # Save metrics
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # --- Contributions (train) ---
    contrib_df = compute_contributions(
        fit_res.idata,
        dm_train.X,
        fit_res.feature_names,
        hdi_prob=0.9,
    )
    contrib_dir = artifacts_dir / "contributions"
    _ensure_dir(contrib_dir)
    contrib_df.to_parquet(contrib_dir / "media_contributions_train.parquet", index=False)

    roas_df = compute_roas(
    fit_res.idata,
    dm_train.X,
    train_df[channel_cols],
    fit_res.feature_names,
    channel_cols,
)
    # --- ROAS summary + plot ---
    roas_dir = artifacts_dir / "roas"
    _ensure_dir(roas_dir)
    roas_df.to_parquet(roas_dir / "roas_summary.parquet", index=False)

    fig_dir = artifacts_dir / "figures"
    _ensure_dir(fig_dir)
    plot_roas(roas_df, fig_dir / "roas.png")

    # Also save the exact config used (repro)
    (artifacts_dir / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print("✅ Pipeline completed")
    print(f"Artifacts saved to: {artifacts_dir}")
    print(f"Best decays: {best_decays}")
    print(f"Train metrics: {metrics_train}")
    print(f"Test metrics: {metrics_test}")