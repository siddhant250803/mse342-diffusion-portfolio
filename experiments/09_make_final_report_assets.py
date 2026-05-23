"""
Generate final report assets (tables, figures, summary markdown).

REQUIRES: leakage audit must pass before this script runs.
Checks results/leakage_audit_result.json for PASSED status.

Produces:
  results/final_summary.md    — concise text summary
  results/final_table_fixed.csv, final_table_rolling.csv — formatted tables
  figures/ — consolidated final figures
"""
import sys, os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)


def require_audit_pass():
    audit_path = "results/leakage_audit_result.json"
    if not os.path.exists(audit_path):
        print("ERROR: Leakage audit result not found.")
        print("Run: python experiments/00_leakage_audit.py")
        print("Audit must PASS before generating report assets.")
        sys.exit(1)
    with open(audit_path) as f:
        result = json.load(f)
    if result.get("status") != "PASSED":
        print("ERROR: Leakage audit FAILED.")
        print("Failures:")
        for failure in result.get("failure_list", []):
            print(f"  - {failure}")
        print("Fix all failures and re-run audit before generating report assets.")
        sys.exit(1)
    print(f"Leakage audit passed at {result.get('timestamp', 'unknown time')}.")
    if result.get("warnings"):
        print(f"Warnings ({len(result['warnings'])}): see leakage_audit_result.json")


def load_final_metrics():
    metrics = {}
    for mode in ["fixed", "rolling"]:
        path = f"results/final_metrics_{mode}.csv"
        if os.path.exists(path):
            metrics[mode] = pd.read_csv(path, index_col=0)
    if not metrics and os.path.exists("results/final_metrics.csv"):
        metrics["fixed"] = pd.read_csv("results/final_metrics.csv", index_col=0)
    return metrics


def format_table(df):
    """Format metrics DataFrame for report display."""
    display_cols = {
        "ann_ret": "Ann. Return",
        "ann_vol": "Ann. Volatility",
        "sharpe": "Sharpe",
        "sharpe_ci_lo": "Sharpe CI Lo",
        "sharpe_ci_hi": "Sharpe CI Hi",
        "cvar95": "CVaR(95%)",
        "max_dd": "Max Drawdown",
        "turnover": "Avg Turnover",
        "avg_hhi": "Avg HHI",
    }
    cols = [c for c in display_cols if c in df.columns]
    out  = df[cols].rename(columns=display_cols).copy()
    for col in ["Ann. Return", "Ann. Volatility", "Max Drawdown"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
    for col in ["Sharpe", "Sharpe CI Lo", "Sharpe CI Hi", "CVaR(95%)", "Avg Turnover", "Avg HHI"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
    return out


def make_final_summary(metrics_dict, eta_selected, audit_result):
    """Generate final_summary.md."""
    lines = []
    lines.append("# MS&E 342 Project: Final Summary")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Leakage audit status: **{audit_result.get('status', 'UNKNOWN')}**")
    if audit_result.get("warnings"):
        lines.append(f"Audit warnings: {len(audit_result['warnings'])}")
        for w in audit_result["warnings"]:
            lines.append(f"  - {w}")

    lines.append("\n## Dataset Split\n")
    lines.append("| Split | Start | End |")
    lines.append("|-------|-------|-----|")
    lines.append("| Training | 2014-01-01 | 2020-12-31 |")
    lines.append("| Validation | 2021-01-01 | 2021-12-31 |")
    lines.append("| Test | 2022-01-01 | 2024-12-31 |")
    lines.append("\nAssets: XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE (10 S&P 500 sector ETFs)")

    lines.append("\n## Leakage Controls\n")
    lines.append("- Scaler (mean, std) fitted on training data only (2014-2020).")
    lines.append("- Score model trained on 2014-2020 data only.")
    lines.append("- OT calibration target moments computed from training data only.")
    lines.append("- Eta selected on 2021 validation Sharpe. Test data not used for selection.")
    lines.append("- Test data evaluated once for final metrics.")
    lines.append("- Sinkhorn OT reported as subset-only diagnostic, not a full strategy.")

    lines.append("\n## Methods Compared\n")
    lines.append("- **Equal-Weight**: uniform 1/d weights, monthly rebalanced.")
    lines.append("- **Markowitz**: plug-in MVO on rolling 252-day window.")
    lines.append("- **Wasserstein-Robust**: robust MVO with W2 ball (Blanchet et al. 2022).")
    lines.append("- **Diffusion-Base**: MVO using VP-SDE generated scenarios (no calibration).")
    lines.append("- **Diffusion+GaussOT**: Gaussian OT calibration applied to generated scenarios.")
    lines.append("  Corrects first two moments under Gaussian approximation only.")
    lines.append("- **Diffusion+FT(eta*)**: fine-tuned with KL-penalized portfolio reward.")
    lines.append("  Eta selected on 2021 validation Sharpe.")

    if eta_selected:
        lines.append("\n## Selected Eta\n")
        for model, info in eta_selected.items():
            lines.append(f"- Model `{model}`: eta* = {info.get('selected_eta')} "
                         f"(val Sharpe = {float(info.get('validation_score', float('nan'))):.3f})")

    for mode, df in metrics_dict.items():
        lines.append(f"\n## Final Metrics ({mode.title()} Backtest, 2022-2024)\n")
        lines.append("Sharpe CI uses simple i.i.d. bootstrap (2000 draws). "
                     "With a ~3-year test window, CI width is approximately 0.6-0.8 Sharpe units. "
                     "Results should be interpreted as exploratory, not statistically decisive.")
        lines.append("")
        ft = format_table(df)
        try:
            lines.append(ft.to_markdown())
        except ImportError:
            # tabulate not installed — fall back to CSV-style
            lines.append("```")
            lines.append(ft.to_string())
            lines.append("```")

    lines.append("\n## Key Diagnostics\n")
    if os.path.exists("results/scenario_diagnostics_base.csv"):
        diag = pd.read_csv("results/scenario_diagnostics_base.csv")
        if "bw_distance" in diag.columns:
            bw = diag["bw_distance"].iloc[0]
            lines.append(f"- Bures-Wasserstein distance (generated vs. train): {bw:.4f}")
        if "cov_error" in diag.columns:
            lines.append(f"- Covariance Frobenius error: {diag['cov_error'].iloc[0]:.4f}")

    if os.path.exists("results/ot_gaussian_diagnostics.csv"):
        ot_diag = pd.read_csv("results/ot_gaussian_diagnostics.csv")
        if "w2_raw" in ot_diag.columns:
            lines.append(f"- W2 before Gaussian OT: {ot_diag['w2_raw'].iloc[0]:.4f}")
            lines.append(f"- W2 after  Gaussian OT: {ot_diag['w2_gauss_ot'].iloc[0]:.4f}")

    lines.append("\n## Limitations\n")
    lines.append("- Score model is a simple MLP; more expressive architectures may improve realism.")
    lines.append("- Gaussian OT corrects first two moments only. Non-Gaussian tail behavior "
                 "is not corrected by this calibration.")
    lines.append("- Sinkhorn OT applied to a subset only. Full-set extension was not implemented.")
    lines.append("- The portfolio reward uses a differentiable soft-Markowitz approximation, "
                 "not a hard constrained optimizer. Weights may differ from exact MVO.")
    lines.append("- Bootstrap Sharpe CIs use simple i.i.d. bootstrap. With autocorrelated "
                 "returns, block bootstrap would give more reliable intervals.")
    lines.append("- A 3-year test window (~756 trading days) provides limited statistical power. "
                 "Pairwise Sharpe differences are not individually statistically significant at "
                 "conventional levels.")
    lines.append("- Rolling backtests use Gaussian OT recalibration, not full model retraining.")

    lines.append("\n## Reproducibility\n")
    lines.append("Full pipeline: `python run_project.py all`")
    lines.append("Fast smoke test: `python run_project.py all --fast`")
    lines.append("Config: `configs/default.yaml`")
    lines.append("Seeds: see config `seeds` section.")

    return "\n".join(lines)


def make_comparison_figure(metrics_dict):
    """Consolidated figure of Sharpe ratios across modes."""
    if not metrics_dict:
        return
    fig, axes = plt.subplots(1, len(metrics_dict), figsize=(8 * len(metrics_dict), 5))
    if len(metrics_dict) == 1:
        axes = [axes]
    for ax, (mode, df) in zip(axes, metrics_dict.items()):
        if "sharpe" not in df.columns:
            continue
        strategies = list(df.index)
        sharpes    = df["sharpe"].tolist()
        ci_lo      = df.get("sharpe_ci_lo", pd.Series([s for s in sharpes])).tolist()
        ci_hi      = df.get("sharpe_ci_hi", pd.Series([s for s in sharpes])).tolist()
        lo_err = [s - l for s, l in zip(sharpes, ci_lo)]
        hi_err = [h - s for s, h in zip(sharpes, ci_hi)]
        ax.barh(strategies, sharpes, alpha=0.8, color="#4CAF50",
                xerr=[lo_err, hi_err],
                error_kw=dict(ecolor="black", capsize=4, linewidth=1.2))
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(f"Sharpe Ratio — {mode.title()} Backtest (2022-2024)\n95% bootstrap CI")
        ax.set_xlabel("Sharpe ratio")
        ax.text(0.02, 0.02, "Simple i.i.d. bootstrap; wide CIs expected (~3yr window)",
                transform=ax.transAxes, fontsize=7, color="gray")
    plt.tight_layout()
    plt.savefig("figures/final_sharpe_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved figures/final_sharpe_comparison.png")


if __name__ == "__main__":
    print("=" * 60)
    print("FINAL REPORT ASSET GENERATION")
    print("=" * 60)

    # Require leakage audit pass
    require_audit_pass()

    with open("results/leakage_audit_result.json") as f:
        audit_result = json.load(f)

    # Load eta selection result
    eta_selected = {}
    if os.path.exists("results/eta_selected.json"):
        with open("results/eta_selected.json") as f:
            eta_selected = json.load(f)

    # Load final metrics
    metrics_dict = load_final_metrics()
    if not metrics_dict:
        print("WARNING: No final metrics found. Run 06_compare.py first.")

    # Save formatted tables
    for mode, df in metrics_dict.items():
        ft = format_table(df)
        ft.to_csv(f"results/final_table_{mode}.csv")
        print(f"Saved results/final_table_{mode}.csv")

    # Make consolidated comparison figure
    make_comparison_figure(metrics_dict)

    # Write final summary markdown
    summary = make_final_summary(metrics_dict, eta_selected, audit_result)
    with open("results/final_summary.md", "w") as f:
        f.write(summary)
    print("Saved results/final_summary.md")

    print("\nFinal report assets generated successfully.")
