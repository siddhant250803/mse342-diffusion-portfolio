"""
Optimal Transport calibration: map synthetic → real (training) return distribution.

Two levels of OT:

  Level 1 — Gaussian OT (Bures-Wasserstein, closed form)
    Assumes Gaussian marginals. Optimal map T*(x) = mu_q + A(x - mu_p).
    Corrects mean and covariance exactly under the Gaussian approximation.
    Does NOT claim to fix non-Gaussian structure (tails, skew, vol clustering).

  Level 2 — Sinkhorn OT (subset-only, entropy-regularised)
    No Gaussian assumption. Finds the optimal coupling on a SUBSET of samples
    (source and target are both subset-sampled to keep memory tractable).
    The barycentric projection gives the OT-calibrated subset.
    NOT applied to the full generated set — results are reported as diagnostics.
    Full-set Sinkhorn would require a nearest-neighbour extension; we label this
    clearly as subset-only and do NOT save it as a full strategy.

  Level 3 — W2-augmented score model training (DSM + Sinkhorn penalty)
    Adds a Sinkhorn divergence penalty to denoising score matching loss.
    Saved as score_model_ot_augmented.pt.

LEAKAGE CONTROLS:
  - Gaussian OT target moments computed from 2014-2020 training data only.
  - Sinkhorn subset drawn from 2014-2020 training data only.
  - Test data is never referenced.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import torch
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("data",    exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

try:
    import ot as pot
    HAS_POT = True
except ImportError:
    HAS_POT = False
    print("WARNING: POT (Python Optimal Transport) not found. Sinkhorn OT will be skipped.")

from score_model import ScoreNet, get_alpha_sigma, BETA_MIN, BETA_MAX
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")


def _load_returns_csv(path):
    """Robust CSV loader that handles yfinance multi-row headers and pandas 2.x date parsing."""
    df = pd.read_csv(path, index_col=0)
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    return df


# ── Level 1: Gaussian OT ──────────────────────────────────────────────────────

def gaussian_ot_map(X_source, X_target):
    """
    Closed-form W2 optimal transport map under Gaussian approximation.
    Corrects mean and covariance; does not fix non-Gaussian structure.
    T*(x) = mu_target + A @ (x - mu_source)
    """
    mu_s  = X_source.mean(0); mu_t = X_target.mean(0)
    Sig_s = np.cov(X_source.T) + 1e-6 * np.eye(X_source.shape[1])
    Sig_t = np.cov(X_target.T) + 1e-6 * np.eye(X_target.shape[1])
    Sig_s_sqrt     = np.real(sqrtm(Sig_s))
    Sig_s_sqrt_inv = np.linalg.inv(Sig_s_sqrt)
    M     = Sig_s_sqrt @ Sig_t @ Sig_s_sqrt
    M_sqrt = np.real(sqrtm(M))
    A      = Sig_s_sqrt_inv @ M_sqrt @ Sig_s_sqrt_inv

    def T(x):
        return mu_t + (x - mu_s) @ A.T

    W2_sq = (np.linalg.norm(mu_s - mu_t) ** 2
             + np.trace(Sig_s) + np.trace(Sig_t)
             - 2 * np.trace(M_sqrt))
    W2 = float(np.sqrt(max(W2_sq, 0.0)))
    return T, W2, A


def bures_wasserstein(A, B):
    """Bures-Wasserstein distance between two sample sets."""
    _, w, _ = gaussian_ot_map(A, B)
    return w


# ── Level 2: Sinkhorn OT (subset-only) ───────────────────────────────────────

def sinkhorn_ot_subset(X_source_sub, X_target_sub, reg=0.005, n_iter=500):
    """
    Entropy-regularised OT on a subset of source and target samples.
    Returns OT-mapped source subset and Sinkhorn cost.
    IMPORTANT: applies only to the provided subset, not the full generated set.
    """
    if not HAS_POT:
        return X_source_sub.copy(), float("nan"), None
    m, d = X_source_sub.shape
    n    = X_target_sub.shape[0]
    a = np.ones(m) / m
    b = np.ones(n) / n
    C = pot.dist(X_source_sub, X_target_sub, metric="sqeuclidean")
    C = C / (C.max() + 1e-12)
    plan = pot.sinkhorn(a, b, C, reg=reg, numItermax=n_iter, warn=False)
    weights   = plan / (plan.sum(1, keepdims=True) + 1e-12)
    X_mapped  = weights @ X_target_sub
    sinkh_cost = float(np.sum(plan * C))
    return X_mapped, sinkh_cost, plan


# ── Level 3: W2-augmented score model ────────────────────────────────────────

def sinkhorn_loss_torch(X, Y, reg=0.05, n_iter=50):
    """Differentiable Sinkhorn loss for OT-augmented training."""
    m, d = X.shape; n = Y.shape[0]
    a = torch.ones(m, device=X.device) / m
    b = torch.ones(n, device=Y.device) / n
    C = torch.cdist(X, Y, p=2).pow(2)
    C = C / (C.detach().max() + 1e-12)
    f = torch.zeros(m, device=X.device)
    g = torch.zeros(n, device=X.device)
    for _ in range(n_iter):
        f = -reg * torch.logsumexp((g.unsqueeze(0) - C / reg), dim=1) + reg * torch.log(a)
        g = -reg * torch.logsumexp((f.unsqueeze(1) - C / reg), dim=0) + reg * torch.log(b)
    log_pi = (f.unsqueeze(1) + g.unsqueeze(0) - C) / reg
    pi     = torch.exp(log_pi)
    return (pi * C).sum()


def train_ot_augmented(train_path="data/train_2014_2020.csv",
                       epochs=500, batch=256, lr=1e-3, gamma=0.05, seed=42):
    """
    Train score model with DSM + Sinkhorn penalty.
    All data used is from 2014-2020 training split only.
    """
    torch.manual_seed(seed)
    df   = _load_returns_csv(train_path)
    X    = torch.tensor(df.values, dtype=torch.float32)
    mu   = X.mean(0); std = X.std(0) + 1e-8
    X_norm = (X - mu) / std
    N, d = X_norm.shape

    model = ScoreNet(d).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dsm_hist, ot_hist = [], []

    for epoch in range(epochs):
        idx = torch.randint(0, N, (batch,))
        x0  = X_norm[idx].to(DEVICE)
        t   = torch.rand(batch, device=DEVICE)
        alpha, sigma = get_alpha_sigma(t)
        alpha = alpha.unsqueeze(-1); sigma = sigma.unsqueeze(-1)
        eps   = torch.randn_like(x0)
        x_t   = alpha * x0 + sigma * eps
        eps_pred = -model(x_t, t) * sigma.clamp(min=0.01)
        dsm_loss = ((eps_pred - eps) ** 2).mean()

        ot_loss = torch.tensor(0.0, device=DEVICE)
        if gamma > 0 and (epoch + 1) % 5 == 0:
            Y = torch.randn(64, d, device=DEVICE)
            dt_ot = 1.0 / 200
            for t_val in torch.linspace(1.0, dt_ot, 200, device=DEVICE):
                tb   = t_val.expand(64)
                beta = BETA_MIN + t_val * (BETA_MAX - BETA_MIN)
                s    = model(Y, tb)
                drift = (-0.5 * beta * Y.detach() + beta * s) * dt_ot
                noise = (beta ** 0.5) * torch.randn_like(Y).detach() * (dt_ot ** 0.5)
                Y = Y.detach() + drift + noise
            real_idx = torch.randint(0, N, (64,))
            Y_real   = X_norm[real_idx].to(DEVICE)
            ot_loss  = sinkhorn_loss_torch(Y, Y_real.detach())

        total = dsm_loss + gamma * ot_loss
        opt.zero_grad(); total.backward(); opt.step(); sched.step()
        dsm_hist.append(dsm_loss.item())
        ot_hist.append(ot_loss.item() if hasattr(ot_loss, "item") else 0.0)

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:5d}  DSM={dsm_loss.item():.5f}  OT={ot_hist[-1]:.5f}")

    torch.save({
        "model": model.state_dict(), "mu": mu, "std": std, "d": d,
        "train_start": "2014-01-01", "train_end": "2020-12-31",
        "scaler_fitted_on": "train_2014_2020", "gamma": gamma,
    }, "checkpoints/score_model_ot_augmented.pt")

    cfg = {
        "epochs": epochs, "batch": batch, "lr": lr, "gamma": gamma, "seed": seed,
        "train_start": "2014-01-01", "train_end": "2020-12-31",
        "data_used_for_fit": "train_2014_2020",
        "data_used_for_selection": "none", "data_used_for_evaluation": "none",
    }
    with open("checkpoints/score_model_ot_augmented_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return model, mu, std, dsm_hist, ot_hist


# ── Diagnostics ───────────────────────────────────────────────────────────────

def make_diagnostics(X_gen, X_real, X_gauss_ot, col_names, label="gaussian_ot"):
    w2_raw   = bures_wasserstein(X_gen, X_real)
    w2_gauss = bures_wasserstein(X_gauss_ot, X_real)
    cov_err_raw   = np.linalg.norm(np.cov(X_gen.T) - np.cov(X_real.T), "fro")
    cov_err_gauss = np.linalg.norm(np.cov(X_gauss_ot.T) - np.cov(X_real.T), "fro")
    rows = []
    for i, name in enumerate(col_names):
        rows.append({
            "asset": name,
            "real_mean": X_real[:, i].mean(), "raw_mean": X_gen[:, i].mean(),
            "gauss_ot_mean": X_gauss_ot[:, i].mean(),
            "real_std": X_real[:, i].std(), "raw_std": X_gen[:, i].std(),
            "gauss_ot_std": X_gauss_ot[:, i].std(),
        })
    df = pd.DataFrame(rows)
    df["w2_raw"]        = w2_raw
    df["w2_gauss_ot"]   = w2_gauss
    df["cov_err_raw"]   = cov_err_raw
    df["cov_err_gauss"] = cov_err_gauss
    return df, w2_raw, w2_gauss


def plot_ot_results(X_gen, X_real, X_gauss_ot, col_names, w2_vals):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Gaussian OT Calibration: Generated (train-2020) → Real (train-2020)", fontsize=12)

    ax = axes[0, 0]
    labels = ["Raw Generated", "Gaussian OT"]
    colors = ["#9E9E9E", "#FF9800"]
    ax.bar(labels, w2_vals[:2], color=colors, alpha=0.85, width=0.4)
    ax.set_title("Bures-Wasserstein Distance to Train Data\n(Gaussian approx — first 2 moments only)")
    ax.set_ylabel("W2")
    for i, v in enumerate(w2_vals[:2]):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)

    ax = axes[0, 1]
    i = 0
    for data, label, color in [(X_real, "Real (train)", "#2196F3"),
                                (X_gen,  "Generated",   "#9E9E9E"),
                                (X_gauss_ot, "Gaussian OT", "#FF9800")]:
        ax.hist(data[:, i], bins=60, density=True, alpha=0.45, label=label, color=color)
    ax.set_title(f"{col_names[i]} Marginal"); ax.legend(fontsize=7)

    ax = axes[0, 2]
    x = np.arange(len(col_names)); w = 0.25
    for j, (data, label, color) in enumerate([
            (X_real, "Real (train)", "#2196F3"),
            (X_gen,  "Generated",   "#9E9E9E"),
            (X_gauss_ot, "Gaussian OT", "#FF9800")]):
        ax.bar(x + j * w, data.std(0), w, label=label, color=color, alpha=0.8)
    ax.set_xticks(x + w); ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=7)
    ax.set_title("Per-Asset Std"); ax.legend(fontsize=7)

    ax = axes[1, 0]
    im = ax.imshow(np.corrcoef(X_real.T), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_title("Real (train) Correlation")
    ax.set_xticks(range(len(col_names))); ax.set_yticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(col_names, fontsize=7); plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    im = ax.imshow(np.corrcoef(X_gauss_ot.T), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_title("Gaussian OT Correlation")
    ax.set_xticks(range(len(col_names))); ax.set_yticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(col_names, fontsize=7); plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 2]
    q_real = np.quantile(X_real[:, 0], np.linspace(0.01, 0.99, 100))
    q_gauss = np.quantile(X_gauss_ot[:, 0], np.linspace(0.01, 0.99, 100))
    ax.scatter(q_real, q_gauss, s=10, alpha=0.6, color="#FF9800", label="Gauss OT")
    lims = [min(q_real.min(), q_gauss.min()), max(q_real.max(), q_gauss.max())]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_title(f"{col_names[0]} QQ-plot (Gauss OT vs Real)")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("figures/ot_gaussian_diagnostics.png", dpi=150, bbox_inches="tight")
    # Legacy alias
    plt.savefig("figures/ot_calibration.png", dpi=150, bbox_inches="tight")
    print("Saved figures/ot_gaussian_diagnostics.png")


def plot_ot_portfolio_weights(X_gen, X_real, X_gauss_ot, col_names):
    import cvxpy as cp

    def mvo(samples, lam=5.0):
        mu    = samples.mean(0)
        Sigma = np.cov(samples.T) + 1e-5 * np.eye(samples.shape[1])
        d     = len(mu)
        w     = cp.Variable(d)
        prob  = cp.Problem(cp.Maximize(mu @ w - lam * cp.quad_form(w, Sigma)),
                           [cp.sum(w) == 1, w >= 0])
        prob.solve(solver=cp.CLARABEL)
        if w.value is None:
            return np.ones(d) / d
        wv = np.clip(w.value, 0, None)
        return wv / wv.sum()

    w_real  = mvo(X_real)
    w_gen   = mvo(X_gen)
    w_gauss = mvo(X_gauss_ot)

    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(col_names)); w = 0.25
    for j, (wts, label, color) in enumerate([
            (w_real,  "Real moments (train)",  "#2196F3"),
            (w_gen,   "Generated",              "#9E9E9E"),
            (w_gauss, "Gaussian OT",            "#FF9800")]):
        ax.bar(x + j * w, wts, w, label=label, color=color, alpha=0.85)
    ax.set_xticks(x + w); ax.set_xticklabels(col_names, rotation=45, ha="right")
    ax.set_title("MVO Portfolio Weights: Effect of Gaussian OT Calibration\n(target = train 2014-2020 moments)")
    ax.set_ylabel("Weight"); ax.legend()
    plt.tight_layout()
    plt.savefig("figures/ot_portfolio_weights.png", dpi=150, bbox_inches="tight")
    plt.savefig("figures/ot_gaussian_diagnostics_weights.png", dpi=150, bbox_inches="tight")
    print("Saved figures/ot_portfolio_weights.png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    run_ot_augmented = "--ot-augmented" in sys.argv
    fast_mode        = "--fast" in sys.argv
    ot_aug_epochs    = 100 if fast_mode else 500

    # Load training data (OT target — 2014-2020 only)
    train_df  = _load_returns_csv("data/train_2014_2020.csv")
    col_names = list(train_df.columns)
    X_real    = train_df.values
    print(f"Training data (OT target): {X_real.shape}  "
          f"({train_df.index[0].date()} to {train_df.index[-1].date()})")

    # Load generated scenarios (from score model trained on 2014-2020 only)
    X_gen = np.load("data/scenarios_base.npy")
    print(f"Generated scenarios: {X_gen.shape}")

    # ── Level 1: Gaussian OT ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Level 1: Gaussian OT (Bures-Wasserstein)")
    print("NOTE: fixes mean and covariance only under Gaussian approximation.")
    print("      Does not correct tail behavior or volatility clustering.")
    T_gauss, W2_raw, A = gaussian_ot_map(X_gen, X_real)
    X_gauss_ot = T_gauss(X_gen)
    _, W2_after, _ = gaussian_ot_map(X_gauss_ot, X_real)
    print(f"  W2 before:  {W2_raw:.6f}")
    print(f"  W2 after:   {W2_after:.6f}  (near 0 = moments matched under Gaussian approx)")
    print(f"  Eigenvalues of linear map A: {np.linalg.eigvalsh(A).round(3)}")

    diag_df, w2_raw, w2_gauss = make_diagnostics(X_gen, X_real, X_gauss_ot, col_names)
    diag_df.to_csv("results/ot_gaussian_diagnostics.csv", index=False)

    plot_ot_results(X_gen, X_real, X_gauss_ot, col_names, [w2_raw, w2_gauss])
    plot_ot_portfolio_weights(X_gen, X_real, X_gauss_ot, col_names)

    np.save("data/scenarios_gaussian_ot.npy", X_gauss_ot)
    # Legacy alias (backward compat)
    np.save("data/gauss_ot_scenarios.npy", X_gauss_ot)

    gauss_meta = {
        "method": "gaussian_ot",
        "source": "scenarios_base.npy",
        "target": "train_2014_2020.csv",
        "train_start": "2014-01-01", "train_end": "2020-12-31",
        "val_start": "2021-01-01",   "val_end": "2021-12-31",
        "test_start": "N/A",
        "data_used_for_fit": "train_2014_2020",
        "data_used_for_selection": "none", "data_used_for_evaluation": "none",
        "w2_before": float(W2_raw), "w2_after": float(W2_after),
        "limitation": "Gaussian OT corrects first two moments only. "
                      "Non-Gaussian structure (tails, skew) is not corrected.",
    }
    with open("data/scenarios_gaussian_ot_metadata.json", "w") as f:
        json.dump(gauss_meta, f, indent=2)
    print("Saved data/scenarios_gaussian_ot.npy  +  metadata")

    # ── Level 2: Sinkhorn OT (subset-only) ───────────────────────────────────
    print("\n" + "=" * 60)
    print("Level 2: Sinkhorn OT — SUBSET ONLY (diagnostic, not a full strategy)")
    print("NOTE: applies to a 600-sample subset. Full scenarios NOT relabeled.")
    subset_size = min(600, len(X_gen), len(X_real))
    rng = np.random.default_rng(42)
    idx_g = rng.choice(len(X_gen),  subset_size, replace=False)
    idx_r = rng.choice(len(X_real), subset_size, replace=False)
    X_gen_sub  = X_gen[idx_g]
    X_real_sub = X_real[idx_r]

    X_sinkh_sub, sinkh_cost, transport_plan = sinkhorn_ot_subset(
        X_gen_sub, X_real_sub, reg=0.005, n_iter=500
    )
    if transport_plan is not None:
        w2_sinkh_sub = bures_wasserstein(X_sinkh_sub, X_real_sub)
        w2_raw_sub   = bures_wasserstein(X_gen_sub,   X_real_sub)
        print(f"  Subset W2 before Sinkhorn: {w2_raw_sub:.6f}")
        print(f"  Subset W2 after  Sinkhorn: {w2_sinkh_sub:.6f}")
        print(f"  Sinkhorn transport cost:   {sinkh_cost:.6f}")

        sinkh_rows = []
        for i, name in enumerate(col_names):
            sinkh_rows.append({
                "asset": name,
                "subset_raw_mean": float(X_gen_sub[:, i].mean()),
                "subset_sinkh_mean": float(X_sinkh_sub[:, i].mean()),
                "real_sub_mean": float(X_real_sub[:, i].mean()),
                "subset_raw_std": float(X_gen_sub[:, i].std()),
                "subset_sinkh_std": float(X_sinkh_sub[:, i].std()),
                "real_sub_std": float(X_real_sub[:, i].std()),
            })
        sinkh_df = pd.DataFrame(sinkh_rows)
        sinkh_df["subset_size"]   = subset_size
        sinkh_df["sinkhorn_cost"] = sinkh_cost
        sinkh_df["w2_before"]     = w2_raw_sub
        sinkh_df["w2_after"]      = w2_sinkh_sub
        sinkh_df["note"] = ("Subset-only: Sinkhorn OT applied to a "
                            f"{subset_size}-sample subset for diagnostic purposes. "
                            "Not applied to full generated set.")
        sinkh_df.to_csv("results/ot_sinkhorn_subset_diagnostics.csv", index=False)
        print("Saved results/ot_sinkhorn_subset_diagnostics.csv")

        # Save subset-mapped scenarios as diagnostic only (clearly labeled)
        np.save("data/scenarios_sinkhorn_ot_subset.npy", X_sinkh_sub)
        # Legacy sinkhorn_ot_scenarios.npy now points to Gaussian OT (correctly labeled)
        # We overwrite it with Gaussian OT to fix the mislabeling bug from the old code.
        np.save("data/sinkhorn_ot_scenarios.npy", X_gauss_ot)  # was wrongly = gauss_ot before
        print("NOTE: data/sinkhorn_ot_scenarios.npy corrected to use Gaussian OT. "
              "Full Sinkhorn is subset-only (see scenarios_sinkhorn_ot_subset.npy).")

        sinkh_meta = {
            "method": "sinkhorn_ot_subset_only",
            "subset_size": subset_size, "reg": 0.005,
            "source": "scenarios_base.npy (subset)", "target": "train_2014_2020.csv (subset)",
            "train_start": "2014-01-01", "train_end": "2020-12-31",
            "data_used_for_fit": "train_2014_2020 subset",
            "data_used_for_selection": "none", "data_used_for_evaluation": "none",
            "test_start": "N/A",
            "w2_before": float(w2_raw_sub), "w2_after": float(w2_sinkh_sub),
            "limitation": (
                "Sinkhorn OT applied to a subset only. Full-set extension would require "
                "nearest-neighbour or kernel-smoothed barycentric map. "
                "Not used as a full portfolio strategy."
            ),
        }
        with open("data/scenarios_sinkhorn_ot_metadata.json", "w") as f:
            json.dump(sinkh_meta, f, indent=2)
    else:
        print("  Sinkhorn skipped (POT not installed). Results from Gaussian OT used.")

    # ── Level 3: OT-augmented training ───────────────────────────────────────
    if run_ot_augmented:
        print("\n" + "=" * 60)
        print("Level 3: OT-augmented score model training")
        _, mu_ot, std_ot, dsm_h, ot_h = train_ot_augmented(epochs=ot_aug_epochs)

        dsm_only = (np.load("checkpoints/score_model_base_losses.npy")
                    if os.path.exists("checkpoints/score_model_base_losses.npy")
                    else np.array([]))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        if len(dsm_only) > 0:
            ax1.plot(dsm_only, alpha=0.7, label="DSM only", color="#2196F3")
        ax1.plot(dsm_h, alpha=0.7, label="DSM + OT", color="#E91E63")
        ax1.set_title("Denoising Score Matching Loss"); ax1.legend(); ax1.set_xlabel("Epoch")
        ax2.plot(ot_h, color="#FF9800"); ax2.set_title("Sinkhorn OT Loss term")
        ax2.set_xlabel("Epoch")
        plt.tight_layout()
        plt.savefig("figures/ot_augmented_training.png", dpi=150, bbox_inches="tight")
        plt.savefig("figures/ot_training_comparison.png", dpi=150, bbox_inches="tight")
        print("Saved figures/ot_augmented_training.png")

    print("\nOT calibration complete.")
    print("  Gaussian OT: data/scenarios_gaussian_ot.npy  (full, first-2-moments calibration)")
    print("  Sinkhorn OT: data/scenarios_sinkhorn_ot_subset.npy  (subset diagnostic only)")
