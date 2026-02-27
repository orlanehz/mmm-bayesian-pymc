"""Point d'entrée CLI."""
from __future__ import annotations

import argparse
from pathlib import Path

from mmm.config import load_config
from mmm.train import run_train


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mmm", description="MMM Bayesian PyMC pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run full pipeline: data -> tuning -> PyMC fit -> artifacts")
    train.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML config")
    train.add_argument("--project-root", type=str, default=".", help="Project root (where data/, artifacts/ live)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    project_root = Path(args.project_root)

    if args.command == "train":
        run_train(cfg, project_root=project_root)
    else:
        raise SystemExit(f"Unknown command: {args.command}")