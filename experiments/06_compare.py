"""
Final comparison of all strategies on 2022-2024 test period.

LEAKAGE CONTROLS:
  - Uses eta_selected.csv to load the validation-selected eta.
  - Test period (2022-2024) is evaluated exactly once here.
  - No hyperparameter tuning or model selection occurs in this script.
  - Both fixed-scenario and rolling-diffusion backtests are produced.

Must be run AFTER 08_eta_sweep.py has produced results/eta_selected.csv.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from portfolio_baseline import (
    mvo, compute_metrics, bootstrap_sharpe_ci, backtest,
    LAMBDA, robust_mvo, diffusion_mvo,
)
from csv_utils import load_returns_csv

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)


def load_selected_scenarios():
    """
    Load scenario files for the validation-selected eta.
    Returns dict: {label: path}.
    """
    scenario_sets = {}

    # Base generated scenarios
    if os.path.exists("data/scenarios_base.npy"):
        scenario_sets["Diffusion-Base"] = "data/scenarios_base.npy"

    # Gaussian OT calibrated
    if os.path.exists("data/scenarios_gaussian_ot.npy"):
        scenario_sets["Diffusion+GaussOT"] = "data/scenarios_gaussian_ot.npy"

    # Validation-selected fine-tuned scenarios
    if not os.path.exists("results/eta_selected.csv"):
        print("WARNING: results/eta_selected.csv not found — fine-tuned strategies skipped.")
        return scenario_sets

    sel_df = pd.read_csv("results/eta_selected.csv", index_col=0)
    for _, row in sel_df.iterrows():
        path  = str(row.get("scenarios_path", ""))
        label = f"Diffusion+FT(eta={row['selected_eta']},{row.get('model','base')})"
        if os.path.exists(path):
            scenario_sets[label] = path
        else:
            print(f"WARNING: {path} not found — skipping {label}")

    return scenario_sets


def run_all_strategies(all_df, scenario_sets, mode="fixed",
                       test_start="2022-01-01"):
    """
    Run all strategies and collect daily returns, metrics, and bootstrap CIs.
    """
    all_results = {}
    all_turnovers = {}
    all_portfolios = {}

    # Run baseline strategies once (they don't depend on scenarios)
    ref_gen   = (np.load("data/scenarios_base.npy")
                 if os.path.exists("data/scenarios_base.npy")
                 else np.ones((100, all_df.shape[1])) / all_df.shape[1])
    ref_res, ref_to, ref_ports = backtest(all_df, ref_gen, mode=mode,
                                          test_start=test_start)
    for name in ("Equal-Weight", "Markowitz", "Wasserstein-Robust"):
        if name in ref_res:
            all_results[name]    = ref_res[name]
            all_turnovers[name]  = ref_to.get(name, 0.0)
            all_portfolios[name] = ref_ports[name]["weights"]

    # Run diffusion strategies
    for label, path in scenario_sets.items():
        print(f"Backtesting {label} ({mode}) ...")
        gen = np.load(path)
        res, to, ports = backtest(all_df, gen, mode=mode, test_start=test_start)
        if "Diffusion-MVO" in res:
            all_results[label]    = res["Diffusion-MVO"]
            all_turnovers[label]  = to.get("Diffusion-MVO", 0.0)
            all_portfolios[label] = ports["Diffusion-MVO"]["weights"]

    # Compute metrics
    metrics = {}
    for name, r in all_results.items():
        m  = compute_metrics(r, test_start=test_start,
                             turnover=all_turnovers.get(name, 0.0),
                             weights_list=all_portfolios.get(name))
        ci = bootstrap_sharpe_ci(r, test_start=test_start)
        m["sharpe_ci_lo"] = ci[0]; m["sharpe_ci_hi"] = ci[1]
        metrics[name] = m

    return all_results, metrics


def print_metrics(metrics):
    print(f"\n{'Strategy':<35} {'Ann.Ret':>9} {'Sharpe':>8} {'95% CI':>14}"
          f" {'CVaR':>8} {'MaxDD':>8} {'HHI':>8} {'TO':>8}")
    print("-" * 105)
    for name, m in metrics.items():
        ci_str = f"[{m.get('sharpe_ci_lo', float('nan')):+.2f},{m.get('sharpe_ci_hi', float('nan')):+.2f}]"
        print(f"{name:<35} {m.get('ann_ret', float('nan')):>9.2%}"
              f" {m.get('sharpe', float('nan')):>8.3f}"
              f" {ci_str:>14}"
              f" {m.get('cvar95', float('nan')):>8.4f}"
              f" {m.get('max_dd', float('nan')):>8.2%}"
              f" {m.get('avg_hhi', float('nan')):>8.4f}"
              f" {m.get('turnover', float('nan')):>8.4f}")


def plot_comparison(all_results, metrics, mode, test_start="2022-01-01"):
    colors = {
        "Equal-Weight":          "#9E9E9E",
        "Markowitz":             "#2196F3",
        "Wasserstein-Robust":    "#FF9800",
        "Diffusion-Base":        "#9C27B0",
        "Diffusion+GaussOT":     "#4CAF50",
    }
    for name in all_results:
        if name not in colors:
            colors[name] = "#E91E63"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    mode_label = "Fixed-Scenario" if mode == "fixed" else "Rolling-Diffusion"
    fig.suptitle(f"Reward-Finetuned Diffusion vs. Baselines — Test 2022-2024 ({mode_label})",
                 fontsize=11)

    ax = axes[0]
    for name, r in all_results.items():
        r_test = r[r.index >= test_start]
        wealth = np.exp(r_test.cumsum())
        lw     = 2.5 if ("FT" in name or "OT" in name) else 1.4
        ax.plot(wealth.index, wealth, label=name,
                color=colors.get(name, "#888"), linewidth=lw)
    ax.set_title("Cumulative Wealth"); ax.legend(fontsize=7)
    ax.set_ylabel("Normalised wealth"); ax.axhline(1, color="k", lw=0.5, ls="--")

    ax = axes[1]
    names  = list(metrics.keys())
    sharps = [metrics[n].get("sharpe", 0) for n in names]
    lo_err = [metrics[n].get("sharpe", 0) - metrics[n].get("sharpe_ci_lo", metrics[n].get("sharpe", 0))
              for n in names]
    hi_err = [metrics[n].get("sharpe_ci_hi", metrics[n].get("sharpe", 0)) - metrics[n].get("sharpe", 0)
              for n in names]
    bar_cs = [colors.get(n, "#888") for n in names]
    ax.barh(names, sharps, color=bar_cs, alpha=0.80,
            xerr=[lo_err, hi_err],
            error_kw=dict(ecolor="black", capsize=4, linewidth=1.2))
    ax.set_title("Sharpe + 95% Bootstrap CI\n(simple i.i.d. bootstrap; ~3yr window)")
    ax.axvline(0, color="k", lw=0.5); ax.set_xlabel("Sharpe ratio")
    ax.text(0.02, 0.02, "Wide CIs expected: ~3yr test window",
            transform=ax.transAxes, fontsize=7, color="gray")

    ax = axes[2]
    for name, r in all_results.items():
        r_test = r[r.index >= test_start]
        wealth = np.exp(r_test.cumsum())
        dd     = wealth / wealth.cummax() - 1
        lw     = 2.5 if ("FT" in name or "OT" in name) else 1.4
        ax.plot(dd.index, dd, label=name,
                color=colors.get(name, "#888"), linewidth=lw)
    ax.set_title("Drawdown"); ax.legend(fontsize=7)
    ax.set_ylabel("Drawdown"); ax.axhline(0, color="k", lw=0.5, ls="--")

    plt.tight_layout()
    out = f"figures/final_comparison_{mode}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    if mode == "fixed":
        plt.savefig("figures/final_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


def plot_concentration_turnover(metrics, mode):
    names  = list(metrics.keys())
    hhis   = [metrics[n].get("avg_hhi", 0) for n in names]
    tos    = [metrics[n].get("turnover", 0) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Concentration and Turnover — {mode.title()} Backtest", fontsize=11)
    axes[0].barh(names, hhis, color="#4CAF50", alpha=0.8)
    axes[0].set_title("Average HHI (concentration)")
    axes[0].set_xlabel("HHI (higher = more concentrated)")
    axes[1].barh(names, tos, color="#2196F3", alpha=0.8)
    axes[1].set_title("Average Turnover per Rebalance")
    axes[1].set_xlabel("L1 turnover")
    plt.tight_layout()
    out = f"figures/portfolio_concentration_{mode}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    if mode == "fixed":
        plt.savefig("figures/portfolio_concentration.png", dpi=150, bbox_inches="tight")
        plt.savefig("figures/turnover_comparison.png",    dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


def plot_drawdown_comparison(all_results, mode, test_start="2022-01-01"):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title(f"Drawdown Comparison — {mode.title()} Backtest (2022-2024)")
    colors = {"Equal-Weight": "#9E9E9E", "Markowitz": "#2196F3",
              "Wasserstein-Robust": "#FF9800", "Diffusion-Base": "#9C27B0",
              "Diffusion+GaussOT": "#4CAF50"}
    for name, r in all_results.items():
        r_test = r[r.index >= test_start]
        wealth = np.exp(r_test.cumsum())
        dd     = wealth / wealth.cummax() - 1
        ax.plot(dd.index, dd, label=name,
                color=colors.get(name, "#E91E63"),
                lw=2.5 if ("FT" in name or "OT" in name) else 1.4)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Drawdown"); ax.legend(fontsize=7)
    plt.tight_layout()
    out = f"figures/drawdown_comparison_{mode}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    if mode == "fixed":
        plt.savefig("figures/drawdown_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    # Guard: check that eta is frozen
    if not os.path.exists("results/eta_selected.csv"):
        print("ERROR: results/eta_selected.csv not found.")
        print("Run 08_eta_sweep.py first to select eta on validation data.")
        sys.exit(1)

    train_df = load_returns_csv("data/train_2014_2020.csv")
    val_df   = load_returns_csv("data/val_2021.csv")
    test_df  = load_returns_csv("data/test_2022_2024.csv")
    all_df   = pd.concat([train_df, val_df, test_df]).sort_index()

    print(f"Test period: {test_df.index[0].date()} to {test_df.index[-1].date()}")
    print("NOTE: test data used for evaluation only — no model selection occurs here.")

    scenario_sets = load_selected_scenarios()
    print(f"\nScenario sets to evaluate: {list(scenario_sets.keys())}")

    all_final_metrics = {}

    for mode in ["fixed", "rolling"]:
        print(f"\n{'='*60}")
        print(f"Backtest mode: {mode}")
        all_results, metrics = run_all_strategies(all_df, scenario_sets, mode=mode)
        print_metrics(metrics)

        out_csv = f"results/final_metrics_{mode}.csv"
        pd.DataFrame(metrics).T.round(4).to_csv(out_csv)
        print(f"Saved {out_csv}")
        if mode == "fixed":
            pd.DataFrame(metrics).T.round(4).to_csv("results/final_metrics.csv")

        all_final_metrics[mode] = metrics
        plot_comparison(all_results, metrics, mode)
        plot_concentration_turnover(metrics, mode)
        plot_drawdown_comparison(all_results, mode)

    print("\nFinal comparison complete.")
    print("Test period evaluated once. Do not revise model selection based on these results.")
