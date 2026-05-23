"""
Fine-tune the reverse SDE drift under a KL-penalized portfolio reward.

Objective:
  max_{u_phi}  E_{Y~p_phi}[ reward(pi_phi(Y), r_val) ]  -  eta * KL(p_phi || p_theta)

KL cost (Girsanov):
  KL = integral ||u_phi(Y_t, t)||^2 / (2*beta(t)) dt

Two reward modes:

A. theory_quadratic (label: quadratic proxy)
   U(r) = r^T Sigma_inv r / (2*lambda)
   Analytic / differentiable. Used for theory verification and ablations.
   Rewards large Mahalanobis moves regardless of sign — not decision-aligned.

B. portfolio_validation_reward (default)
   Compute differentiable MVO weights pi_S from generated scenarios S:
     pi_S = softmax( (Sigma_S + ridge*I)^{-1} mu_S )
   Evaluate on fixed 2021 VALIDATION returns (not test):
     reward = mean(pi_S^T r_val) - (lambda/2) * var(pi_S^T r_val)
            - gamma * CVaR95(pi_S^T r_val) - tau * HHI(pi_S)
   Gradient flows from r_val through pi_S to (mu_S, Sigma_S) to the samples Y.
   Approximation: differentiable soft-Markowitz with softmax normalization.

LEAKAGE CONTROLS:
  - Score model loaded from checkpoints/score_model_base.pt (trained 2014-2020).
  - Training data used only for Sigma_inv in theory_quadratic mode.
  - Validation returns (2021) used only in portfolio_validation_reward.
  - Test data is NEVER used for fine-tuning or reward.
  - Eta selection happens in 08_eta_sweep.py, not here.
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
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("figures",     exist_ok=True)

LAMBDA     = 5.0
ETA        = 0.5
N_STEPS    = 200
BATCH      = 128
LR_FT      = 3e-4
EPOCHS     = 300
CTRL_SCALE = 1.5
GAMMA_CVaR = 0.5   # CVaR weight
TAU_CONC   = 0.1   # HHI concentration weight
RIDGE      = 1e-4  # ridge for differentiable MVO


# ── Control network ───────────────────────────────────────────────────────────
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


# ── Reward functions ──────────────────────────────────────────────────────────

def theory_quadratic_reward(r, Sigma_inv, lam=LAMBDA):
    """
    Quadratic proxy reward: U = r^T Sigma_inv r / (2*lambda).
    Analytic and fully differentiable.
    WARNING: rewards large Mahalanobis moves regardless of sign.
    Use only for theory/ablation sections.
    """
    Sinv_r = r @ Sigma_inv.T
    return (r * Sinv_r).sum(-1) / (2 * lam)


def differentiable_mvo_weights(mu_S, Sigma_S, ridge=RIDGE):
    """
    Differentiable soft-Markowitz:
    1. Regularize Sigma: Sigma_reg = Sigma_S + ridge * I
    2. Compute unconstrained Markowitz direction: v = Sigma_reg^{-1} mu_S
    3. Normalize to simplex via softmax: pi = softmax(v)

    Gradient flows from (mu_S, Sigma_S) to pi.
    Approximation: does not enforce the hard long-only simplex constraint;
    softmax gives positive weights that sum to 1.
    """
    d = mu_S.shape[0]
    Sigma_reg = Sigma_S + ridge * torch.eye(d, device=mu_S.device)
    # v = Sigma_reg^{-1} mu_S  (differentiable through Sigma_S and mu_S)
    v  = torch.linalg.solve(Sigma_reg, mu_S)
    pi = torch.softmax(v, dim=0)
    return pi


def soft_cvar(port_rets, alpha=0.05):
    """
    Soft CVaR at level alpha (expected shortfall).
    Uses sorting-based empirical quantile — differentiable via sorted values.
    Returns a positive scalar (larger = worse tail).
    """
    n = port_rets.shape[0]
    k = max(1, int(n * alpha))
    sorted_rets, _ = torch.sort(port_rets)
    worst = sorted_rets[:k]
    return -worst.mean()


def portfolio_validation_reward(Y0, mu_S, Sigma_S, r_val_tensor,
                                lam=LAMBDA, gamma_cvar=GAMMA_CVaR, tau_conc=TAU_CONC,
                                ridge=RIDGE):
    """
    Decision-aligned reward evaluated on validation returns.

    1. Compute differentiable portfolio weights from generated scenario moments.
    2. Apply weights to 2021 validation returns.
    3. Return mean-variance + CVaR + concentration reward.

    r_val_tensor: (N_val, d) tensor of 2021 validation returns (FIXED, no grad).
    Y0: (B, d) generated scenarios — used to compute mu_S, Sigma_S.
    """
    # Compute batch mean and covariance from generated scenarios
    # (gradient flows from Y0 through mu_S, Sigma_S to pi)
    pi = differentiable_mvo_weights(mu_S, Sigma_S, ridge=ridge)  # (d,)

    # Portfolio returns on validation set
    port_rets = r_val_tensor @ pi   # (N_val,) — r_val has no grad

    ann_mean  = port_rets.mean() * 252
    ann_var   = port_rets.var()  * 252
    cvar_loss = soft_cvar(port_rets, alpha=0.05)    # positive = bad tail
    hhi       = (pi ** 2).sum()                      # concentration

    reward = ann_mean - (lam / 2) * ann_var - gamma_cvar * cvar_loss - tau_conc * hhi
    return reward, pi.detach()


# ── SDE rollout ───────────────────────────────────────────────────────────────

def rollout(base_model, ctrl_net, n_samples, n_steps=N_STEPS, ckpt_path="checkpoints/score_model_base.pt"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    d    = ckpt["d"]
    Y    = torch.randn(n_samples, d, device=DEVICE)
    dt   = 1.0 / n_steps
    ts   = torch.linspace(1.0, dt, n_steps, device=DEVICE)
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

def finetune(base_model, mu, std, reward_mode="portfolio_validation_reward",
             eta=ETA, epochs=EPOCHS, r_val_tensor=None, Sigma_inv_real=None,
             ckpt_path="checkpoints/score_model_base.pt", tag="default"):
    """
    Fine-tune ControlNet under KL-penalized reward.

    reward_mode: "portfolio_validation_reward" or "theory_quadratic"
    r_val_tensor: 2021 validation returns (required for portfolio mode)
    Sigma_inv_real: training data Sigma_inv (required for theory mode)
    """
    d        = mu.shape[0]
    ctrl_net = ControlNet(d).to(DEVICE)
    opt      = torch.optim.Adam(ctrl_net.parameters(), lr=LR_FT)

    reward_hist, kl_hist, loss_hist = [], [], []

    for epoch in range(epochs):
        ctrl_net.train()
        Y0, kl_cost = rollout(base_model, ctrl_net, BATCH, ckpt_path=ckpt_path)

        # Un-standardise to return scale
        Y0_ret = Y0 * std.to(DEVICE) + mu.to(DEVICE)

        if reward_mode == "theory_quadratic":
            reward = theory_quadratic_reward(Y0_ret, Sigma_inv_real.to(DEVICE))
        else:
            # portfolio_validation_reward
            mu_S    = Y0_ret.mean(0)
            Sigma_S = (Y0_ret - mu_S).T @ (Y0_ret - mu_S) / (BATCH - 1)
            reward, _ = portfolio_validation_reward(
                Y0_ret, mu_S, Sigma_S, r_val_tensor.to(DEVICE))
            reward = reward.unsqueeze(0).expand(BATCH)

        loss = -(reward - eta * kl_cost).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ctrl_net.parameters(), max_norm=0.5)
        opt.step()

        reward_hist.append(float(reward.mean().item()))
        kl_hist.append(float(kl_cost.mean().item()))
        loss_hist.append(float(loss.item()))

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:5d}  reward={reward.mean():.5f}"
                  f"  KL={kl_cost.mean():.4f}  loss={loss.item():.5f}")

    ckpt_out = f"checkpoints/control_base_{reward_mode}_eta{eta}.pt"
    torch.save(ctrl_net.state_dict(), ckpt_out)
    # Legacy name
    torch.save(ctrl_net.state_dict(), "checkpoints/ctrl_net.pt")
    return ctrl_net, reward_hist, kl_hist


@torch.no_grad()
def generate_finetuned(base_model, ctrl_net, mu, std, n_samples=2000,
                       ckpt_path="checkpoints/score_model_base.pt"):
    ctrl_net.eval()
    Y0, _ = rollout(base_model, ctrl_net, n_samples, ckpt_path=ckpt_path)
    return (Y0 * std.to(DEVICE) + mu.to(DEVICE)).cpu().numpy()


def plot_training(reward_hist, kl_hist, tag="default"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(reward_hist, color="#E91E63")
    ax1.set_title("Reward during Fine-tuning")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("E[reward]")
    ax2.plot(kl_hist, color="#FF9800")
    ax2.set_title("KL Cost (control effort)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("E[KL]")
    plt.tight_layout()
    out = f"figures/finetune_training_{tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig("figures/finetune_training.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    import sys

    reward_mode = "portfolio_validation_reward"
    if "--reward" in sys.argv:
        reward_mode = sys.argv[sys.argv.index("--reward") + 1]
    eta = ETA
    if "--eta" in sys.argv:
        eta = float(sys.argv[sys.argv.index("--eta") + 1])
    epochs = EPOCHS
    if "--epochs" in sys.argv:
        epochs = int(sys.argv[sys.argv.index("--epochs") + 1])
    if "--fast" in sys.argv:
        epochs = 50

    print(f"Fine-tuning  reward_mode={reward_mode}  eta={eta}  epochs={epochs}")

    # Load base model (trained 2014-2020)
    ckpt = torch.load("checkpoints/score_model_base.pt", map_location="cpu", weights_only=True)
    d    = ckpt["d"]
    base_model = ScoreNet(d)
    base_model.load_state_dict(ckpt["model"])
    base_model.to(DEVICE).eval()
    mu, std = ckpt["mu"].float(), ckpt["std"].float()

    # Sigma from training data only (for theory mode)
    train_df = load_returns_csv("data/train_2014_2020.csv")
    Sigma_real     = np.cov(train_df.values.T)
    Sigma_inv_real = torch.tensor(
        np.linalg.inv(Sigma_real + 1e-5 * np.eye(d)), dtype=torch.float32)

    # Validation returns (2021 only — for portfolio reward)
    val_df        = load_returns_csv("data/val_2021.csv")
    r_val_tensor  = torch.tensor(val_df.values, dtype=torch.float32)

    tag = f"base_{reward_mode}_eta{eta}"
    ctrl_net, reward_hist, kl_hist = finetune(
        base_model, mu, std,
        reward_mode=reward_mode, eta=eta, epochs=epochs,
        r_val_tensor=r_val_tensor, Sigma_inv_real=Sigma_inv_real,
        tag=tag,
    )
    plot_training(reward_hist, kl_hist, tag=tag)

    n_gen = 2000
    if "--fast" in sys.argv:
        n_gen = 500
    print(f"\nGenerating {n_gen} fine-tuned scenarios...")
    ft_samples = generate_finetuned(base_model, ctrl_net, mu, std, n_samples=n_gen)
    out_path   = f"data/scenarios_ft_base_{reward_mode}_eta{eta}.npy"
    np.save(out_path, ft_samples)
    np.save("data/finetuned_scenarios.npy", ft_samples)

    meta = {
        "checkpoint": f"checkpoints/control_base_{reward_mode}_eta{eta}.pt",
        "base_model": "score_model_base.pt",
        "reward_mode": reward_mode,
        "eta": eta,
        "epochs": epochs,
        "train_start": "2014-01-01", "train_end": "2020-12-31",
        "val_start": "2021-01-01",   "val_end": "2021-12-31",
        "test_start": "N/A — not used",
        "data_used_for_fit": "train_2014_2020 (score model); val_2021 (reward eval)",
        "data_used_for_selection": "none (eta selection in 08_eta_sweep.py)",
        "data_used_for_evaluation": "none",
        "n_samples": n_gen, "seed": 0,
    }
    with open(out_path.replace(".npy", "_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    pd.DataFrame({"reward": reward_hist, "kl": kl_hist}).to_csv(
        f"results/finetune_training_{tag}.csv", index=False)
    print(f"Saved {out_path}")
    print("Done. Run 08_eta_sweep.py to select eta on validation.")
