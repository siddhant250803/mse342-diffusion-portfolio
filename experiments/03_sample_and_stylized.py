"""
Generate scenarios via the reverse SDE and check stylized facts.

Reverse SDE (Euler-Maruyama):
  dY_t = [-0.5*beta(T-t)*Y_t + beta(T-t)*s_theta(Y_t, T-t)] dt
         + sqrt(beta(T-t)) dW_t

LEAKAGE CONTROLS:
  - Uses score_model_base.pt trained on 2014-2020 only.
  - Scaler (mu, std) loaded from checkpoint (fitted on 2014-2020 only).
  - Scenarios compared against training data only for diagnostics.
  - Validation and test data are not used here.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import sqrtm
from score_model import ScoreNet, get_alpha_sigma, BETA_MIN, BETA_MAX
from csv_utils import load_returns_csv

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("data",    exist_ok=True)


def load_model(ckpt_path="checkpoints/score_model_base.pt"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    d    = ckpt["d"]
    model = ScoreNet(d)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    return model, ckpt["mu"], ckpt["std"], ckpt


@torch.no_grad()
def reverse_sde(model, n_samples=10000, n_steps=500, seed=42, d=None,
                ckpt_path="checkpoints/score_model_base.pt"):
    """Euler-Maruyama reverse SDE sampler."""
    torch.manual_seed(seed)
    if d is None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        d = ckpt["d"]
    Y  = torch.randn(n_samples, d, device=DEVICE)
    dt = 1.0 / n_steps
    ts = torch.linspace(1.0, dt, n_steps, device=DEVICE)
    for t_val in ts:
        t_batch = t_val.expand(n_samples)
        beta    = BETA_MIN + t_val * (BETA_MAX - BETA_MIN)
        score   = model(Y, t_batch)
        drift   = -0.5 * beta * Y + beta * score
        diff    = (beta ** 0.5) * torch.randn_like(Y)
        Y = Y + drift * dt + diff * (dt ** 0.5)
    return Y.cpu().numpy()


def bures_wasserstein(A, B):
    """Bures-Wasserstein distance between two empirical samples (Gaussian approx)."""
    mu_A = A.mean(0); mu_B = B.mean(0)
    S_A  = np.cov(A.T) + 1e-6 * np.eye(A.shape[1])
    S_B  = np.cov(B.T) + 1e-6 * np.eye(B.shape[1])
    SA_sqrt = np.real(sqrtm(S_A))
    M       = np.real(sqrtm(SA_sqrt @ S_B @ SA_sqrt))
    W2_sq   = np.linalg.norm(mu_A - mu_B) ** 2 + np.trace(S_A) + np.trace(S_B) - 2 * np.trace(M)
    return float(np.sqrt(max(W2_sq, 0.0)))


def herfindahl(weights):
    """Herfindahl-Hirschman Index (portfolio concentration)."""
    return float(np.sum(weights ** 2))


def compute_diagnostics(samples, train_data, col_names):
    """Full diagnostic suite comparing generated to training data."""
    d = samples.shape[1]

    # Per-asset statistics
    rows = []
    for i, name in enumerate(col_names):
        rows.append({
            "asset": name,
            "gen_mean":  float(samples[:, i].mean()),
            "train_mean": float(train_data[:, i].mean()),
            "gen_std":   float(samples[:, i].std()),
            "train_std": float(train_data[:, i].std()),
            "gen_kurtosis":   float(stats.kurtosis(samples[:, i])),
            "train_kurtosis": float(stats.kurtosis(train_data[:, i])),
        })

    # Global measures
    bw  = bures_wasserstein(samples, train_data)
    cov_err  = np.linalg.norm(np.cov(samples.T) - np.cov(train_data.T), "fro")
    corr_err = np.linalg.norm(np.corrcoef(samples.T) - np.corrcoef(train_data.T), "fro")

    return rows, {"bures_wasserstein": bw, "cov_frobenius_err": cov_err, "corr_frobenius_err": corr_err}


def mvo_weights(samples, lam=5.0):
    import cvxpy as cp
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


def plot_stylized_facts(samples, train_data, col_names, out="figures/stylized_facts_base.png"):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Stylized Facts: Generated vs. Historical (2014-2020)", fontsize=13)

    real_kurt = [stats.kurtosis(train_data[:, i]) for i in range(train_data.shape[1])]
    gen_kurt  = [stats.kurtosis(samples[:, i])    for i in range(samples.shape[1])]

    def abs_autocorr(x, max_lag=10):
        x = np.abs(x - x.mean())
        return [np.corrcoef(x[:-lag], x[lag:])[0, 1] for lag in range(1, max_lag + 1)]

    real_ac = np.mean([abs_autocorr(train_data[:, i]) for i in range(train_data.shape[1])], 0)
    gen_ac  = np.mean([abs_autocorr(samples[:, i])    for i in range(samples.shape[1])],  0)

    ax = axes[0, 0]
    x  = np.arange(len(col_names))
    ax.bar(x - 0.2, real_kurt, 0.4, label="Historical (train)", color="#2196F3", alpha=0.8)
    ax.bar(x + 0.2, gen_kurt,  0.4, label="Generated",          color="#FF5722", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=8)
    ax.set_title("Excess Kurtosis"); ax.legend(); ax.axhline(0, color="k", lw=0.5)

    ax = axes[0, 1]
    lags = np.arange(1, 11)
    ax.plot(lags, real_ac, "o-", label="Historical", color="#2196F3")
    ax.plot(lags, gen_ac,  "s-", label="Generated",  color="#FF5722")
    ax.set_title("Autocorr of |Returns| (vol clustering)"); ax.legend()
    ax.axhline(0, color="k", lw=0.5)

    ax = axes[0, 2]
    im = ax.imshow(np.corrcoef(train_data.T), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(col_names))); ax.set_yticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(col_names, fontsize=7)
    ax.set_title("Historical Correlation"); plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    im = ax.imshow(np.corrcoef(samples.T), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(col_names))); ax.set_yticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(col_names, fontsize=7)
    ax.set_title("Generated Correlation"); plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    ax.hist(train_data[:, 0], bins=80, density=True, alpha=0.6,
            label="Historical", color="#2196F3")
    ax.hist(samples[:, 0],    bins=80, density=True, alpha=0.6,
            label="Generated",  color="#FF5722")
    ax.set_title(f"{col_names[0]} Return Distribution"); ax.legend()

    ax = axes[1, 2]
    q_real = np.quantile(train_data[:, 0], np.linspace(0.01, 0.99, 100))
    q_gen  = np.quantile(samples[:, 0],    np.linspace(0.01, 0.99, 100))
    ax.scatter(q_real, q_gen, s=10, alpha=0.6, color="#4CAF50")
    lims = [min(q_real.min(), q_gen.min()), max(q_real.max(), q_gen.max())]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("Historical quantiles"); ax.set_ylabel("Generated quantiles")
    ax.set_title(f"{col_names[0]} QQ-plot")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    import sys
    n_samples = 10000
    n_steps   = 500
    if "--fast" in sys.argv:
        n_samples = 500; n_steps = 100
    if "--n_samples" in sys.argv:
        n_samples = int(sys.argv[sys.argv.index("--n_samples") + 1])

    model, mu, std, ckpt_meta = load_model("checkpoints/score_model_base.pt")
    train_df  = load_returns_csv("data/train_2014_2020.csv")
    col_names = list(train_df.columns)
    train_data = train_df.values

    print(f"Generating {n_samples} scenarios via reverse SDE ({n_steps} steps)...")
    samples_std = reverse_sde(model, n_samples=n_samples, n_steps=n_steps, seed=42)
    samples     = samples_std * std.numpy() + mu.numpy()

    # Sanity checks
    nan_frac = np.isnan(samples).any(axis=1).mean()
    inf_frac = np.isinf(samples).any(axis=1).mean()
    print(f"NaN fraction: {nan_frac:.4f}  Inf fraction: {inf_frac:.4f}")
    if nan_frac + inf_frac > 0.01:
        print("WARNING: high NaN/Inf rate in generated scenarios")

    rows, global_diag = compute_diagnostics(samples, train_data, col_names)
    print(f"\nBures-Wasserstein distance (gen vs train): {global_diag['bures_wasserstein']:.6f}")

    w_gen  = mvo_weights(samples)
    w_real = mvo_weights(train_data)
    hhi_gen  = herfindahl(w_gen)
    hhi_real = herfindahl(w_real)
    print(f"HHI (generated): {hhi_gen:.4f}  HHI (historical): {hhi_real:.4f}")

    diag_df = pd.DataFrame(rows)
    diag_df["bw_distance"]      = global_diag["bures_wasserstein"]
    diag_df["cov_error"]        = global_diag["cov_frobenius_err"]
    diag_df["corr_error"]       = global_diag["corr_frobenius_err"]
    diag_df["hhi_gen"]          = hhi_gen
    diag_df["hhi_real"]         = hhi_real
    diag_df.to_csv("results/scenario_diagnostics_base.csv", index=False)

    plot_stylized_facts(samples, train_data, col_names)
    # Also save under legacy name
    plot_stylized_facts(samples, train_data, col_names, out="figures/stylized_facts.png")

    np.save("data/scenarios_base.npy", samples)
    # Legacy alias
    np.save("data/generated_scenarios.npy", samples)

    meta = {
        "checkpoint": "checkpoints/score_model_base.pt",
        "scaler_fitted_on": "train_2014_2020",
        "train_start": "2014-01-01", "train_end": "2020-12-31",
        "val_start": "2021-01-01",   "val_end": "2021-12-31",
        "test_start": "N/A — not used", "test_end": "N/A",
        "data_used_for_fit": "train_2014_2020",
        "data_used_for_selection": "none",
        "data_used_for_evaluation": "none",
        "n_samples": n_samples, "n_steps": n_steps, "seed": 42,
        "nan_fraction": float(nan_frac), "inf_fraction": float(inf_frac),
        "bures_wasserstein": global_diag["bures_wasserstein"],
        "hhi_gen": hhi_gen,
    }
    with open("data/scenarios_base_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved: data/scenarios_base.npy  shape={samples.shape}")
    print(f"Saved: data/scenarios_base_metadata.json")
    print(f"Saved: results/scenario_diagnostics_base.csv")
