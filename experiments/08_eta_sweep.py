"""
Eta sweep for KL-penalized fine-tuning.
Eta selection uses 2021 VALIDATION data only — never test data.

Protocol:
  1. For each candidate eta: fine-tune ControlNet using training data
     and portfolio_validation_reward on 2021 validation returns.
  2. Evaluate selection metric (validation Sharpe) on 2021 validation.
  3. Select eta* = argmax_{eta} val_sharpe(eta).
  4. Save eta_selected.csv with NO test metrics.
  5. Test evaluation happens separately in 06_compare.py.

LEAKAGE CONTROLS:
  - Score model (base) trained on 2014-2020 only.
  - Fine-tuning uses only training data and validation reward.
  - Eta selection criterion is validation Sharpe (2021 only).
  - 2022-2024 test returns are NEVER used during this script.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from score_model import ScoreNet, get_alpha_sigma, BETA_MIN, BETA_MAX
from csv_utils import load_returns_csv

torch.manual_seed(0)
DEVICE   = "mps" if torch.backends.mps.is_available() else "cpu"
LAMBDA   = 5.0
N_STEPS  = 100
BATCH    = 128
LR_FT    = 3e-4
EPOCHS   = 200
CTRL_SCALE = 1.5
RIDGE    = 1e-4
GAMMA_CVaR = 0.5
TAU_CONC   = 0.1

ETA_VALUES = [0.05, 0.1, 0.5, 1.0, 5.0]

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("figures",     exist_ok=True)
os.makedirs("results",     exist_ok=True)
os.makedirs("data",        exist_ok=True)


# ── ControlNet ────────────────────────────────────────────────────────────────
class ControlNet(nn.Module):
    def __init__(self, d, hidden=128, t_emb_dim=32):
        super().__init__()
        self.t_embed = nn.Sequential(
            nn.Linear(1, t_emb_dim), nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(d + t_emb_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d),
        )
        for p in self.net[-1].parameters():
            nn.init.zeros_(p)

    def forward(self, y, t):
        t_emb = self.t_embed(t.unsqueeze(-1))
        return self.net(torch.cat([y, t_emb], dim=-1))


# ── Reward ─────────────────────────────────────────────────────────────────────

def soft_cvar(port_rets, alpha=0.05):
    k = max(1, int(port_rets.shape[0] * alpha))
    sorted_rets, _ = torch.sort(port_rets)
    return -sorted_rets[:k].mean()


def differentiable_mvo_weights(mu_S, Sigma_S, ridge=RIDGE):
    d   = mu_S.shape[0]
    Sig = Sigma_S + ridge * torch.eye(d, device=mu_S.device)
    v   = torch.linalg.solve(Sig, mu_S)
    return torch.softmax(v, dim=0)


def portfolio_validation_reward(Y0_ret, r_val_tensor,
                                 lam=LAMBDA, gamma_cvar=GAMMA_CVaR, tau_conc=TAU_CONC):
    mu_S    = Y0_ret.mean(0)
    Sigma_S = (Y0_ret - mu_S).T @ (Y0_ret - mu_S) / (Y0_ret.shape[0] - 1)
    pi      = differentiable_mvo_weights(mu_S, Sigma_S)
    port    = r_val_tensor.to(DEVICE) @ pi
    ann_mean = port.mean() * 252
    ann_var  = port.var()  * 252
    cvar     = soft_cvar(port)
    hhi      = (pi ** 2).sum()
    return ann_mean - (lam / 2) * ann_var - gamma_cvar * cvar - tau_conc * hhi


# ── Rollout ───────────────────────────────────────────────────────────────────

def rollout(base_model, ctrl_net, n_samples, d, n_steps=N_STEPS):
    Y       = torch.randn(n_samples, d, device=DEVICE)
    dt      = 1.0 / n_steps
    ts      = torch.linspace(1.0, dt, n_steps, device=DEVICE)
    kl_cost = torch.zeros(n_samples, device=DEVICE)
    for t_val in ts:
        t_batch = t_val.expand(n_samples)
        beta    = BETA_MIN + t_val * (BETA_MAX - BETA_MIN)
        with torch.no_grad():
            score = base_model(Y, t_batch)
        u     = torch.tanh(ctrl_net(Y, t_batch)) * CTRL_SCALE
        drift = -0.5 * beta * Y + beta * score + u
        noise = (beta ** 0.5) * torch.randn_like(Y)
        Y     = Y + drift * dt + noise * (dt ** 0.5)
        kl_cost += (u ** 2).sum(-1) / (2 * beta + 1e-8) * dt
    return Y, kl_cost


# ── Training ──────────────────────────────────────────────────────────────────

def train_one(base_model, d, mu, std, r_val_tensor, eta, tag, epochs=EPOCHS):
    ctrl_net = ControlNet(d).to(DEVICE)
    opt = torch.optim.Adam(ctrl_net.parameters(), lr=LR_FT)
    for epoch in range(epochs):
        ctrl_net.train()
        Y0, kl_cost = rollout(base_model, ctrl_net, BATCH, d)
        Y0_ret = Y0 * std.to(DEVICE) + mu.to(DEVICE)
        reward = portfolio_validation_reward(Y0_ret, r_val_tensor)
        reward_vec = reward.unsqueeze(0).expand(BATCH)
        loss = -(reward_vec - eta * kl_cost).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ctrl_net.parameters(), max_norm=0.5)
        opt.step()
        if (epoch + 1) % 50 == 0:
            print(f"  [{tag}] epoch {epoch+1:4d}  reward={reward.item():.5f}"
                  f"  KL={kl_cost.mean():.4f}")
    torch.save(ctrl_net.state_dict(), f"checkpoints/ctrl_net_{tag}.pt")
    return ctrl_net


@torch.no_grad()
def generate(base_model, ctrl_net, d, mu, std, n_samples=2000):
    ctrl_net.eval()
    Y0, _ = rollout(base_model, ctrl_net, n_samples, d)
    return (Y0 * std.to(DEVICE) + mu.to(DEVICE)).cpu().numpy()


# ── Validation evaluation ────────────────────────────────────────────────────

def eval_on_validation(scenarios, val_returns, lam=LAMBDA):
    """
    Evaluate a fixed scenario set on 2021 VALIDATION returns.
    Computes Sharpe, CVaR, return, and HHI on val period only.
    NO test data is used.
    """
    import cvxpy as cp
    finite_mask = np.isfinite(scenarios).all(axis=1)
    if finite_mask.sum() < 10:
        d = scenarios.shape[1]
        pi = np.ones(d) / d
    else:
        sc = scenarios[finite_mask]
        mu = sc.mean(0)
        Sigma = np.cov(sc.T) + 1e-5 * np.eye(sc.shape[1])
        d  = len(mu)
        w  = cp.Variable(d)
        prob = cp.Problem(cp.Maximize(mu @ w - lam * cp.quad_form(w, Sigma)),
                          [cp.sum(w) == 1, w >= 0])
        prob.solve(solver=cp.CLARABEL)
        if w.value is None:
            pi = np.ones(d) / d
        else:
            wv = np.clip(w.value, 0, None)
            pi = wv / wv.sum()

    port_rets = val_returns @ pi
    ann_ret   = np.exp(port_rets.mean() * 252) - 1
    ann_vol   = port_rets.std() * np.sqrt(252)
    sharpe    = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cvar95    = float(-port_rets[port_rets <= np.quantile(port_rets, 0.05)].mean())
    hhi       = float(np.sum(pi ** 2))
    return {"val_sharpe": sharpe, "val_ann_ret": ann_ret, "val_ann_vol": ann_vol,
            "val_cvar95": cvar95, "val_hhi": hhi}


# ── Main sweep ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fast_mode = "--fast" in sys.argv
    epochs    = 30 if fast_mode else EPOCHS
    n_gen     = 500 if fast_mode else 2000

    # Load training data (for Sigma only — theory mode not used here)
    train_df = load_returns_csv("data/train_2014_2020.csv")

    # Load validation data (2021 only — for reward and selection metric)
    val_df   = load_returns_csv("data/val_2021.csv")
    r_val_np = val_df.values
    r_val_tensor = torch.tensor(r_val_np, dtype=torch.float32)
    print(f"Validation period: {val_df.index[0].date()} to {val_df.index[-1].date()}")
    print(f"Validation days: {len(val_df)}")
    print("NOTE: eta selection will use validation Sharpe only. Test data not loaded.")

    all_val_metrics = {}

    # Only base model in default config
    model_configs = {"base": "checkpoints/score_model_base.pt"}
    if os.path.exists("checkpoints/score_model_ot_augmented.pt"):
        model_configs["ot_augmented"] = "checkpoints/score_model_ot_augmented.pt"

    for model_name, ckpt_path in model_configs.items():
        print(f"\n{'='*60}")
        print(f"Base model: {model_name}  ({ckpt_path})")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        d    = ckpt["d"]
        base_model = ScoreNet(d)
        base_model.load_state_dict(ckpt["model"])
        base_model.to(DEVICE).eval()
        mu, std = ckpt["mu"].float(), ckpt["std"].float()

        for eta in ETA_VALUES:
            tag = f"{model_name}_eta{eta}"
            print(f"\n--- eta={eta}  tag={tag} ---")
            ctrl_net = train_one(base_model, d, mu, std, r_val_tensor, eta, tag, epochs)

            scenarios = generate(base_model, ctrl_net, d, mu, std, n_samples=n_gen)
            finite_frac = np.isfinite(scenarios).all(axis=1).mean()
            print(f"  Finite scenarios: {finite_frac:.1%}")

            np.save(f"data/ft_{tag}_scenarios.npy", scenarios)

            if finite_frac < 0.5:
                print(f"  WARNING: low finite fraction — recording NaN metrics")
                all_val_metrics[tag] = {
                    "model": model_name, "eta": eta,
                    "val_sharpe": float("nan"), "val_ann_ret": float("nan"),
                    "val_ann_vol": float("nan"), "val_cvar95": float("nan"),
                    "val_hhi": float("nan"),
                }
                continue

            vm = eval_on_validation(scenarios, r_val_np)
            all_val_metrics[tag] = {"model": model_name, "eta": eta, **vm}
            print(f"  Val Sharpe={vm['val_sharpe']:.3f}  Ann.Ret={vm['val_ann_ret']:.2%}"
                  f"  CVaR95={vm['val_cvar95']:.4f}  HHI={vm['val_hhi']:.4f}")

    # ── Save all validation metrics ───────────────────────────────────────────
    val_df_out = pd.DataFrame(all_val_metrics).T
    val_df_out.to_csv("results/eta_validation_metrics.csv")
    print(f"\nSaved results/eta_validation_metrics.csv")

    # ── Select best eta per model (on VALIDATION only) ───────────────────────
    selected = {}
    for model_name in {v["model"] for v in all_val_metrics.values() if "model" in v}:
        rows = {k: v for k, v in all_val_metrics.items() if v.get("model") == model_name}
        valid_rows = {k: v for k, v in rows.items() if not np.isnan(v.get("val_sharpe", np.nan))}
        if not valid_rows:
            continue
        best_tag = max(valid_rows, key=lambda k: valid_rows[k]["val_sharpe"])
        selected[model_name] = {
            "model": model_name,
            "selected_eta": valid_rows[best_tag]["eta"],
            "selection_metric": "validation_sharpe",
            "validation_score": valid_rows[best_tag]["val_sharpe"],
            "tag": best_tag,
            "scenarios_path": f"data/ft_{best_tag}_scenarios.npy",
            "val_start": "2021-01-01", "val_end": "2021-12-31",
            "test_metrics": "NOT AVAILABLE — use 06_compare.py after eta is frozen",
            "reward_mode": "portfolio_validation_reward",
            "note": "Eta selected on validation data only. No test data used.",
        }
        print(f"\n>> Selected eta for {model_name}: {valid_rows[best_tag]['eta']}"
              f"  (val_sharpe={valid_rows[best_tag]['val_sharpe']:.3f})")

    pd.DataFrame(selected).T.to_csv("results/eta_selected.csv")
    with open("results/eta_selected.json", "w") as f:
        json.dump(selected, f, indent=2)
    print("Saved results/eta_selected.csv  +  results/eta_selected.json")

    # ── Plot validation frontier ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Eta Sweep — Validation Period (2021) Only", fontsize=12)

    colors = {"base": "#2196F3", "ot_augmented": "#E91E63"}
    labels = {"base": "Base model + FT", "ot_augmented": "OT-augmented + FT"}

    ax = axes[0]
    ax.set_title("Validation Sharpe vs. Eta\n(eta selection criterion)")
    ax.set_xlabel("Eta (KL budget)"); ax.set_ylabel("Validation Sharpe (2021)")
    ax.set_xscale("log")
    for model_name in model_configs:
        subset = {k: v for k, v in all_val_metrics.items() if v.get("model") == model_name}
        if not subset:
            continue
        etas    = [v["eta"] for v in subset.values()]
        sharpes = [v["val_sharpe"] for v in subset.values()]
        ax.plot(etas, sharpes, "o-", color=colors.get(model_name, "#888"),
                label=labels.get(model_name, model_name), lw=2, ms=7)
        # Mark selected
        if model_name in selected:
            sel_eta = selected[model_name]["selected_eta"]
            sel_s   = selected[model_name]["validation_score"]
            ax.axvline(sel_eta, color=colors.get(model_name, "#888"), ls=":", lw=1.5,
                       label=f"Selected eta={sel_eta}")
            ax.scatter([sel_eta], [sel_s], s=120, zorder=5,
                       color=colors.get(model_name, "#888"), marker="*")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.02, "Selection on validation (2021) only — test data not used",
            transform=ax.transAxes, fontsize=7, color="gray")

    ax = axes[1]
    ax.set_title("Validation CVaR95 vs. Eta")
    ax.set_xlabel("Eta (KL budget)"); ax.set_ylabel("CVaR95 (2021, lower=better)")
    ax.set_xscale("log")
    for model_name in model_configs:
        subset = {k: v for k, v in all_val_metrics.items() if v.get("model") == model_name}
        if not subset:
            continue
        etas  = [v["eta"] for v in subset.values()]
        cvars = [v.get("val_cvar95", np.nan) for v in subset.values()]
        ax.plot(etas, cvars, "s-", color=colors.get(model_name, "#888"),
                label=labels.get(model_name, model_name), lw=2, ms=7)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/eta_validation_frontier.png", dpi=150, bbox_inches="tight")
    plt.savefig("figures/eta_sharpe_frontier.png",     dpi=150, bbox_inches="tight")
    print("Saved figures/eta_validation_frontier.png")
    print("\nEta sweep complete. Eta is now frozen based on validation.")
    print("Run 06_compare.py to evaluate on test (once only).")
