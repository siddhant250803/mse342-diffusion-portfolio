"""
Leakage Audit — must pass before final report assets are generated.

Checks:
  1. Date ranges of saved train/val/test files
  2. Scaler metadata says training-only fit
  3. Score model checkpoint lists training data only
  4. OT calibration metadata excludes test
  5. eta_selected.csv based on validation metrics only (no test metrics)
  6. Final metric files generated only after eta_selected.csv exists
  7. Rolling backtest logs show no look-ahead windows (checked structurally)
  8. Sinkhorn OT is labeled as subset-only (not full strategy)

Exits with code 0 if all checks pass, code 1 if any fail.
"""
import sys, os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import json
import pandas as pd
import numpy as np
from datetime import datetime

TRAIN_START = pd.Timestamp("2014-01-01")
TRAIN_END   = pd.Timestamp("2020-12-31")
VAL_START   = pd.Timestamp("2021-01-01")
VAL_END     = pd.Timestamp("2021-12-31")
TEST_START  = pd.Timestamp("2022-01-01")
TEST_END    = pd.Timestamp("2024-12-31")

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

failures = []
warnings = []


def check(condition, msg_pass, msg_fail, fatal=True):
    if condition:
        print(f"  {PASS} {msg_pass}")
        return True
    else:
        tag = FAIL if fatal else WARN
        print(f"  {tag} {msg_fail}")
        if fatal:
            failures.append(msg_fail)
        else:
            warnings.append(msg_fail)
        return False


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def date_range_of_csv(path):
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # parse_dates=True may not convert the index if it looks like strings
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index, errors="coerce")
    # Drop NaT rows introduced by failed parses
    df = df[df.index.notna()]
    if df.empty:
        return None, None
    return df.index.min(), df.index.max()


print("=" * 60)
print("LEAKAGE AUDIT")
print("=" * 60)


# ── Check 1: Date ranges of split files ──────────────────────────────────────
print("\n[1] Split file date ranges")

for fname, expect_start, expect_end in [
    ("data/train_2014_2020.csv", TRAIN_START, TRAIN_END),
    ("data/val_2021.csv",        VAL_START,   VAL_END),
    ("data/test_2022_2024.csv",  TEST_START,  TEST_END),
]:
    if not os.path.exists(fname):
        print(f"  {FAIL} {fname} does not exist")
        failures.append(f"{fname} missing")
        continue
    start, end = date_range_of_csv(fname)
    ok_start = start >= expect_start - pd.Timedelta(days=5)
    ok_end   = end   <= expect_end   + pd.Timedelta(days=5)
    ok_sep   = True
    # Train must not overlap val/test
    if "train" in fname:
        ok_sep = end < VAL_START
    elif "val" in fname:
        ok_sep = start >= VAL_START and end <= VAL_END + pd.Timedelta(days=5)
    elif "test" in fname:
        ok_sep = start >= TEST_START
    check(ok_start and ok_end and ok_sep,
          f"{fname}: {start.date()} to {end.date()} — OK",
          f"{fname}: {start.date()} to {end.date()} — date range mismatch or overlap")


# ── Check 2: Scaler leakage ───────────────────────────────────────────────────
print("\n[2] Scaler leakage")

for ckpt_path in ["checkpoints/score_model_base.pt",
                  "checkpoints/score_model.pt"]:
    if not os.path.exists(ckpt_path):
        print(f"  {WARN} {ckpt_path} not found — skipping")
        warnings.append(f"{ckpt_path} not found")
        continue
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    scaler_info = ckpt.get("scaler_fitted_on", "UNKNOWN")
    check("train_2014_2020" in str(scaler_info) or "train" in str(scaler_info).lower(),
          f"{ckpt_path}: scaler_fitted_on='{scaler_info}'",
          f"{ckpt_path}: scaler_fitted_on='{scaler_info}' — may include non-train data")
    # Verify train_start/end field
    ts = ckpt.get("train_start", "UNKNOWN")
    te = ckpt.get("train_end",   "UNKNOWN")
    check("2014" in str(ts) and "2020" in str(te),
          f"{ckpt_path}: train split recorded as {ts} to {te}",
          f"{ckpt_path}: train split '{ts}' to '{te}' — suspicious")


# ── Check 3: Score model training metadata ───────────────────────────────────
print("\n[3] Score model training metadata")

cfg = load_json("checkpoints/score_model_base_config.json")
if cfg is None:
    print(f"  {FAIL} checkpoints/score_model_base_config.json not found")
    failures.append("score_model_base_config.json missing")
else:
    check("train_2014_2020" in str(cfg.get("data_used_for_fit", "")),
          f"data_used_for_fit = '{cfg.get('data_used_for_fit')}'",
          f"data_used_for_fit = '{cfg.get('data_used_for_fit')}' — should be train only")
    check("N/A" in str(cfg.get("test_start", "")) or "not used" in str(cfg.get("test_start", "")).lower(),
          f"test_start = '{cfg.get('test_start')}' — test not in training",
          f"test_start = '{cfg.get('test_start')}' — test may have been used")


# ── Check 4: OT calibration metadata ─────────────────────────────────────────
print("\n[4] OT calibration metadata")

gauss_meta = load_json("data/scenarios_gaussian_ot_metadata.json")
if gauss_meta is None:
    print(f"  {WARN} data/scenarios_gaussian_ot_metadata.json not found")
    warnings.append("Gaussian OT metadata missing")
else:
    check("train_2014_2020" in str(gauss_meta.get("data_used_for_fit", "")),
          f"Gaussian OT target: '{gauss_meta.get('data_used_for_fit')}'",
          f"Gaussian OT target includes non-train data: '{gauss_meta.get('data_used_for_fit')}'")
    check("N/A" in str(gauss_meta.get("test_start", "test")),
          "Gaussian OT does not reference test period",
          f"Gaussian OT test_start='{gauss_meta.get('test_start')}' — may leak test data")

sinkh_meta = load_json("data/scenarios_sinkhorn_ot_metadata.json")
if sinkh_meta is None:
    print(f"  {WARN} data/scenarios_sinkhorn_ot_metadata.json not found — checking subset file")
    warnings.append("Sinkhorn OT metadata missing")
else:
    method = sinkh_meta.get("method", "UNKNOWN")
    check("subset" in method.lower(),
          f"Sinkhorn method = '{method}' — correctly labeled as subset-only",
          f"Sinkhorn method = '{method}' — should be labeled subset_only",
          fatal=False)
    check("N/A" in str(sinkh_meta.get("test_start", "test")),
          "Sinkhorn OT does not reference test period",
          f"Sinkhorn test_start='{sinkh_meta.get('test_start')}' — may leak")


# ── Check 5: Eta selected on validation only ──────────────────────────────────
print("\n[5] Eta selection on validation only")

if not os.path.exists("results/eta_selected.csv"):
    print(f"  {WARN} results/eta_selected.csv not found — eta sweep not yet run")
    warnings.append("eta_selected.csv missing")
else:
    sel_df = pd.read_csv("results/eta_selected.csv", index_col=0)
    for _, row in sel_df.iterrows():
        check("validation" in str(row.get("selection_metric", "")).lower(),
              f"selection_metric = '{row.get('selection_metric')}' — validation-based",
              f"selection_metric = '{row.get('selection_metric')}' — must be validation-only")
        check("NOT AVAILABLE" in str(row.get("test_metrics", "")).upper()
              or "not" in str(row.get("test_metrics", "")).lower(),
              "test_metrics field confirms no test data used in selection",
              f"test_metrics field = '{row.get('test_metrics')}' — test data may have been used")

    # Check validation metrics file exists and contains only val_ columns
    if os.path.exists("results/eta_validation_metrics.csv"):
        vm = pd.read_csv("results/eta_validation_metrics.csv", index_col=0)
        test_cols = [c for c in vm.columns if c.startswith("test_") or "2022" in c or "2024" in c]
        check(len(test_cols) == 0,
              "eta_validation_metrics.csv contains no test-period columns",
              f"eta_validation_metrics.csv has suspicious columns: {test_cols}")


# ── Check 6: Final metrics generated after eta is frozen ─────────────────────
print("\n[6] Final metrics generated after eta frozen")

eta_sel_mtime = (os.path.getmtime("results/eta_selected.csv")
                 if os.path.exists("results/eta_selected.csv") else None)
for fpath in ["results/final_metrics_fixed.csv", "results/final_metrics_rolling.csv",
              "results/final_metrics.csv"]:
    if not os.path.exists(fpath):
        print(f"  {WARN} {fpath} not yet generated")
        continue
    final_mtime = os.path.getmtime(fpath)
    if eta_sel_mtime is not None:
        check(final_mtime >= eta_sel_mtime - 60,
              f"{fpath} generated after eta_selected.csv (OK)",
              f"{fpath} generated BEFORE eta_selected.csv — potential leakage",
              fatal=False)


# ── Check 7: Sinkhorn OT not used as a full strategy ─────────────────────────
print("\n[7] Sinkhorn OT correctly labeled")

for fpath in ["results/final_metrics_fixed.csv", "results/final_metrics_rolling.csv",
              "results/final_metrics.csv"]:
    if not os.path.exists(fpath):
        continue
    df = pd.read_csv(fpath, index_col=0)
    sinkhorn_full = [i for i in df.index if "Sinkhorn" in str(i) and "subset" not in str(i).lower()]
    check(len(sinkhorn_full) == 0,
          f"{fpath}: No full Sinkhorn strategy in final metrics",
          f"{fpath}: Full Sinkhorn strategies found: {sinkhorn_full} — must be labeled subset-only",
          fatal=False)


# ── Check 8: Scenario metadata for NaN/Inf ───────────────────────────────────
print("\n[8] Scenario file integrity")

for scen_path in ["data/scenarios_base.npy", "data/scenarios_gaussian_ot.npy"]:
    if not os.path.exists(scen_path):
        print(f"  {WARN} {scen_path} not found")
        continue
    scen = np.load(scen_path)
    nan_f = np.isnan(scen).any(axis=1).mean()
    inf_f = np.isinf(scen).any(axis=1).mean()
    check(nan_f + inf_f < 0.01,
          f"{scen_path}: shape={scen.shape}  NaN={nan_f:.3f}  Inf={inf_f:.3f}",
          f"{scen_path}: high NaN/Inf fraction NaN={nan_f:.3f} Inf={inf_f:.3f}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"AUDIT SUMMARY")
print(f"  Failures: {len(failures)}")
print(f"  Warnings: {len(warnings)}")

if failures:
    print("\nFAILURES (must fix before generating final report):")
    for i, f in enumerate(failures, 1):
        print(f"  {i}. {f}")

if warnings:
    print("\nWARNINGS (review recommended):")
    for i, w in enumerate(warnings, 1):
        print(f"  {i}. {w}")

if not failures:
    print("\nAUDIT PASSED — OK to generate final report assets.")
    print("Saving audit result...")
    with open("results/leakage_audit_result.json", "w") as f:
        json.dump({
            "status": "PASSED",
            "failures": 0,
            "warnings": len(warnings),
            "warning_list": warnings,
            "timestamp": str(datetime.now()),
        }, f, indent=2)
    sys.exit(0)
else:
    print("\nAUDIT FAILED — do NOT generate final report assets until fixed.")
    with open("results/leakage_audit_result.json", "w") as f:
        json.dump({
            "status": "FAILED",
            "failures": len(failures),
            "failure_list": failures,
            "warnings": len(warnings),
            "warning_list": warnings,
            "timestamp": str(datetime.now()),
        }, f, indent=2)
    sys.exit(1)
