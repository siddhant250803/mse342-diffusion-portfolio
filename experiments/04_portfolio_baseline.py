"""
Portfolio construction, backtesting, and metrics.

Strategies:
  1. Equal-weight (EW)
  2. Rolling Markowitz plug-in MVO (rolling 252-day historical window)
  3. Wasserstein-robust MVO (Blanchet et al. 2022)
  4. Diffusion-MVO (any scenario set, fixed or rolling-calibrated)

Two backtest modes:
  Mode A: fixed-scenario — uses one scenario set for the full test period.
  Mode B: rolling diffusion — at each rebalance, recalibrates generated
          scenarios to the rolling historical window using Gaussian OT,
          using only data up to the rebalance date (no future data).

Metrics include: annualized return, volatility, Sharpe, CVaR(95%),
max drawdown, turnover, average max weight, Herfindahl index,
and 95% bootstrap confidence interval for Sharpe.

LEAKAGE CONTROLS:
  - Rolling baselines use only data up to the rebalance date.
  - Rolling Gaussian OT uses only past data for target moments.
  - Bootstrap CI uses test-period returns only for CIs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import cvxpy as cp
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

np.random.seed(42)
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

LAMBDA = 5.0
WINDOW = 252
DELTA  = 0.02


# ── MVO helpers ───────────────────────────────────────────────────────────────

def mvo(mu, Sigma, lam=LAMBDA, long_only=True):
    d = len(mu)
    w = cp.Variable(d)
    obj = mu @ w - lam * cp.quad_form(w, Sigma)
    cons = [cp.sum(w) == 1]
    if long_only:
        cons.append(w >= 0)
    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver=cp.CLARABEL, warm_start=True)
    if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        return np.ones(d) / d
    wv = np.clip(w.value, 0, None)
    return wv / (wv.sum() + 1e-12)


def robust_mvo(r_samples, lam=LAMBDA, delta=DELTA):
    """Wasserstein-robust MVO (Blanchet et al. 2022). Tractable approx: Sigma_robust = Sigma + (delta/lam)*I."""
    mu_hat    = r_samples.mean(0)
    Sigma_hat = np.cov(r_samples.T) + (delta / lam) * np.eye(r_samples.shape[1])
    return mvo(mu_hat, Sigma_hat, lam=lam)


def diffusion_mvo(generated_scenarios, lam=LAMBDA):
    """MVO using generated diffusion scenarios as the distribution."""
    finite_mask = np.isfinite(generated_scenarios).all(axis=1)
    if finite_mask.sum() < 10:
        return np.ones(generated_scenarios.shape[1]) / generated_scenarios.shape[1]
    sc     = generated_scenarios[finite_mask]
    mu     = sc.mean(0)
    Sigma  = np.cov(sc.T)
    Sigma  = (Sigma + Sigma.T) / 2 + 1e-5 * np.eye(Sigma.shape[0])
    return mvo(mu, Sigma, lam=lam)


# ── Gaussian OT map (for rolling calibration) ─────────────────────────────────

def gaussian_ot_apply(X_gen, mu_target, Sigma_target):
    """
    Apply Gaussian OT map that matches X_gen to (mu_target, Sigma_target).
    Used for rolling recalibration of scenarios at each rebalance date.
    Target moments must be computed from data up to the rebalance date only.
    """
    d     = X_gen.shape[1]
    mu_s  = X_gen.mean(0)
    Sig_s = np.cov(X_gen.T) + 1e-6 * np.eye(d)
    Sig_t = Sigma_target + 1e-6 * np.eye(d)
    Sig_s_sqrt     = np.real(sqrtm(Sig_s))
    Sig_s_sqrt_inv = np.linalg.inv(Sig_s_sqrt)
    M      = Sig_s_sqrt @ Sig_t @ Sig_s_sqrt
    M_sqrt = np.real(sqrtm(M))
    A      = Sig_s_sqrt_inv @ M_sqrt @ Sig_s_sqrt_inv
    return mu_target + (X_gen - mu_s) @ A.T


# ── Backtest engine ───────────────────────────────────────────────────────────

def backtest(returns_df, generated_scenarios, window=WINDOW, mode="fixed",
             test_start="2022-01-01"):
    """
    Monthly rebalancing backtest.

    mode: "fixed" — use generated_scenarios as-is for Diffusion-MVO.
          "rolling" — recalibrate generated_scenarios at each rebalance date
                      using rolling Gaussian OT on past data only.
    """
    dates   = returns_df.index
    ret_mat = returns_df.values
    d       = ret_mat.shape[1]
    monthly_ends = returns_df.resample("ME").last().index

    portfolios = {
        "Equal-Weight":       {"weights": [], "dates": []},
        "Markowitz":          {"weights": [], "dates": []},
        "Wasserstein-Robust": {"weights": [], "dates": []},
        "Diffusion-MVO":      {"weights": [], "dates": []},
    }

    prev_weights = {name: np.ones(d) / d for name in portfolios}

    for rebal_date in monthly_ends:
        mask_train = dates <= rebal_date
        idx        = np.where(mask_train)[0]
        if len(idx) < window:
            continue

        r_train = ret_mat[idx[-window:]]

        w_ew  = np.ones(d) / d
        w_mvo = mvo(r_train.mean(0), np.cov(r_train.T) + 1e-6 * np.eye(d))
        w_rob = robust_mvo(r_train)

        if mode == "rolling":
            # Recalibrate generated scenarios to rolling historical moments
            # using only data up to and including rebal_date
            mu_roll    = r_train.mean(0)
            Sigma_roll = np.cov(r_train.T)
            scenarios_cal = gaussian_ot_apply(generated_scenarios, mu_roll, Sigma_roll)
            w_diff = diffusion_mvo(scenarios_cal)
        else:
            w_diff = diffusion_mvo(generated_scenarios)

        for name, w in zip(portfolios.keys(), [w_ew, w_mvo, w_rob, w_diff]):
            portfolios[name]["weights"].append(w)
            portfolios[name]["dates"].append(rebal_date)

    # Compute daily portfolio returns and track turnover
    results  = {}
    turnovers = {}

    for name, data in portfolios.items():
        if not data["dates"]:
            continue
        weights_list = data["weights"]
        dates_list   = data["dates"]

        daily_rets  = []
        to_list     = []

        for i, (rebal_date, w) in enumerate(zip(dates_list, weights_list)):
            next_date = dates_list[i + 1] if i + 1 < len(dates_list) else dates[-1]
            mask      = (dates > rebal_date) & (dates <= next_date)
            period_ret = ret_mat[mask] @ w
            daily_rets.extend(zip(dates[mask], period_ret))

            # Turnover = L1 change in weights at rebalance
            if i > 0:
                to = np.sum(np.abs(w - weights_list[i - 1]))
                to_list.append(to)

        if daily_rets:
            idx_r, vals = zip(*daily_rets)
            results[name]   = pd.Series(vals, index=idx_r)
            turnovers[name] = float(np.mean(to_list)) if to_list else 0.0

    return results, turnovers, portfolios


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(daily_rets, test_start="2022-01-01", turnover=0.0,
                    weights_list=None):
    """Compute full metric suite for a strategy."""
    r = daily_rets[daily_rets.index >= test_start]
    if len(r) == 0:
        return {}
    ann_ret  = float(np.exp(r.mean() * 252) - 1)
    ann_vol  = float(r.std() * np.sqrt(252))
    sharpe   = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
    cvar95   = float(-r[r <= r.quantile(0.05)].mean())
    wealth   = np.exp(r.cumsum())
    max_dd   = float((wealth / wealth.cummax() - 1).min())

    # Concentration
    avg_max_w = float("nan")
    avg_hhi   = float("nan")
    if weights_list is not None:
        max_ws = [w.max() for w in weights_list]
        hhis   = [float(np.sum(w ** 2)) for w in weights_list]
        avg_max_w = float(np.mean(max_ws))
        avg_hhi   = float(np.mean(hhis))

    return dict(
        ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
        cvar95=cvar95, max_dd=max_dd, turnover=turnover,
        avg_max_weight=avg_max_w, avg_hhi=avg_hhi,
    )


def bootstrap_sharpe_ci(daily_rets, test_start="2022-01-01",
                        n_boot=2000, alpha=0.05, seed=42):
    """
    95% bootstrap CI for annualised Sharpe ratio.
    Uses simple i.i.d. bootstrap — note that daily returns are autocorrelated,
    so CIs may be slightly narrow. Report as indicative, not exact.
    """
    r = daily_rets[daily_rets.index >= test_start].values
    if len(r) < 10:
        return (float("nan"), float("nan"))
    rng     = np.random.default_rng(seed)
    sharpes = []
    for _ in range(n_boot):
        rb = rng.choice(r, size=len(r), replace=True)
        sr = rb.mean() * np.sqrt(252) / (rb.std() + 1e-10)
        sharpes.append(sr)
    return (float(np.percentile(sharpes, alpha / 2 * 100)),
            float(np.percentile(sharpes, (1 - alpha / 2) * 100)))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_df = pd.read_csv("data/train_2014_2020.csv", index_col=0, parse_dates=True)
    val_df   = pd.read_csv("data/val_2021.csv",         index_col=0, parse_dates=True)
    test_df  = pd.read_csv("data/test_2022_2024.csv",   index_col=0, parse_dates=True)
    all_df   = pd.concat([train_df, val_df, test_df]).sort_index()

    generated = np.load("data/scenarios_base.npy")
    print(f"Generated scenarios shape: {generated.shape}")

    for mode in ["fixed", "rolling"]:
        print(f"\nRunning {mode} backtest ...")
        results, turnovers, portfolios = backtest(all_df, generated, mode=mode)

        metrics = {}
        print(f"\n{'Strategy':<22} {'Ann.Ret':>8} {'Sharpe':>8} {'CVaR95':>8} "
              f"{'MaxDD':>8} {'Turnover':>10} {'HHI':>8}")
        print("-" * 80)
        for name, r in results.items():
            w_list = portfolios[name]["weights"]
            m = compute_metrics(r, test_start="2022-01-01",
                                turnover=turnovers.get(name, 0.0),
                                weights_list=w_list)
            ci = bootstrap_sharpe_ci(r)
            m["sharpe_ci_lo"] = ci[0]; m["sharpe_ci_hi"] = ci[1]
            metrics[name] = m
            print(f"{name:<22} {m['ann_ret']:>8.2%} {m['sharpe']:>8.3f}"
                  f" {m['cvar95']:>8.4f} {m['max_dd']:>8.2%}"
                  f" {m['turnover']:>10.4f} {m['avg_hhi']:>8.4f}")

        out_csv = f"results/baseline_metrics_{mode}.csv"
        pd.DataFrame(metrics).T.round(4).to_csv(out_csv)
        print(f"Saved {out_csv}")
