#!/usr/bin/env python3
"""
MS&E 342 Project — Master Pipeline Runner

Usage:
  python run_project.py <stage> [options]

Stages:
  data              Download data, create train/val/test splits
  leakage-audit     Run leakage audit (must pass before final report)
  train-score       Train VP-SDE score model on 2014-2020 data
  sample            Generate base scenario set
  ot                Gaussian OT calibration (+ optional Sinkhorn subset)
  ot-augmented      Train OT-augmented score model (optional)
  finetune          Fine-tune with KL-penalized portfolio reward
  eta-sweep         Eta sweep on validation data (selects eta)
  backtest-fixed    Fixed-scenario backtest on test period
  backtest-rolling  Rolling-diffusion backtest on test period
  compare           Run both backtests and produce final metrics (requires eta-sweep)
  report-assets     Generate final report tables/figures (requires audit pass)
  all               Run full pipeline in order

Options:
  --fast            Use reduced epochs/samples for smoke testing
  --epochs N        Override score model training epochs
  --n_samples N     Override scenario count
  --ot-augmented    Enable OT-augmented training in ot stage
  --reward MODE     Reward mode for finetune: portfolio_validation_reward | theory_quadratic
  --eta ETA         Fixed eta for finetune (skip sweep)

Examples:
  python run_project.py data
  python run_project.py leakage-audit
  python run_project.py train-score --epochs 2000
  python run_project.py all --fast
  python run_project.py all
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Always run from project root
PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
EXPERIMENTS = PROJECT_ROOT / "experiments"
PYTHON = sys.executable


def run(script, extra_args=None, env=None):
    """Run an experiment script, passing through fast/extra flags."""
    cmd  = [PYTHON, str(EXPERIMENTS / script)] + (extra_args or [])
    # Pass fast flag through if present
    if "--fast" in sys.argv and "--fast" not in cmd:
        cmd.append("--fast")
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    t0  = time.time()
    ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    dt  = time.time() - t0
    if ret.returncode != 0:
        print(f"\nERROR: {script} failed with code {ret.returncode}")
        sys.exit(ret.returncode)
    print(f"[done in {dt:.1f}s]")


def is_fast():
    return "--fast" in sys.argv


def parse_extra(key):
    """Return [key, value] if key appears in sys.argv, else []."""
    if key in sys.argv:
        idx = sys.argv.index(key)
        if idx + 1 < len(sys.argv):
            return [key, sys.argv[idx + 1]]
    return []


def stage_data():
    run("01_data.py")


def stage_leakage_audit():
    run("00_leakage_audit.py")


def stage_train_score():
    extra = parse_extra("--epochs")
    if not extra and is_fast():
        extra = ["--fast"]
    run("02_score_model.py", extra)


def stage_sample():
    extra = parse_extra("--n_samples")
    run("03_sample_and_stylized.py", extra)


def stage_ot():
    extra = []
    if "--ot-augmented" in sys.argv:
        extra.append("--ot-augmented")
    run("07_ot_calibration.py", extra)


def stage_ot_augmented():
    run("07_ot_calibration.py", ["--ot-augmented"])


def stage_finetune():
    extra = parse_extra("--reward") + parse_extra("--eta") + parse_extra("--epochs")
    run("05_finetune.py", extra)


def stage_eta_sweep():
    run("08_eta_sweep.py")


def stage_backtest_fixed():
    run("04_portfolio_baseline.py")


def stage_compare():
    if not (PROJECT_ROOT / "results" / "eta_selected.csv").exists():
        print("ERROR: results/eta_selected.csv not found.")
        print("Run eta-sweep first: python run_project.py eta-sweep")
        sys.exit(1)
    run("06_compare.py")


def stage_report_assets():
    run("09_make_final_report_assets.py")


STAGES = {
    "data":              stage_data,
    "leakage-audit":     stage_leakage_audit,
    "train-score":       stage_train_score,
    "sample":            stage_sample,
    "ot":                stage_ot,
    "ot-augmented":      stage_ot_augmented,
    "finetune":          stage_finetune,
    "eta-sweep":         stage_eta_sweep,
    "backtest-fixed":    stage_backtest_fixed,
    "compare":           stage_compare,
    "report-assets":     stage_report_assets,
    "all": None,  # handled below
}

PIPELINE_ORDER = [
    "data",
    "train-score",
    "sample",
    "ot",
    "eta-sweep",
    "compare",
    "leakage-audit",
    "report-assets",
]


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    stage = sys.argv[1]

    if stage == "all":
        print("Running full pipeline...")
        if is_fast():
            print("FAST MODE: reduced epochs and samples.")
        for s in PIPELINE_ORDER:
            STAGES[s]()
        print("\nPipeline complete.")
        return

    if stage not in STAGES:
        print(f"Unknown stage: {stage}")
        print(f"Valid stages: {', '.join(STAGES.keys())}")
        sys.exit(1)

    STAGES[stage]()


if __name__ == "__main__":
    main()
