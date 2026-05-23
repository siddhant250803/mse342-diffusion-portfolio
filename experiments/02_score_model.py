"""
VP-SDE score model trained on 2014-2020 training data only.

Forward process (variance-preserving):
  dX_t = -0.5 * beta(t) * X_t dt + sqrt(beta(t)) dW_t,   t in [0, T]
  beta(t) = beta_min + t*(beta_max - beta_min)

Marginal:  X_t | X_0 ~ N(alpha_t * X_0, sigma_t^2 * I)
  alpha_t = exp(-0.5 * integral_0^t beta(s) ds)
  sigma_t^2 = 1 - alpha_t^2

Score target (denoising score matching):
  s_theta(x_t, t) ≈ -(x_t - alpha_t * x_0) / sigma_t^2

LEAKAGE CONTROLS:
  - Scaler (mu, std) fitted on 2014-2020 training data only.
  - Model trained on 2014-2020 training data only.
  - Validation data used only for loss monitoring (no model selection here;
    eta selection is in 08_eta_sweep.py).
  - Test data is never touched.
"""
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import json
import os
import hashlib
from csv_utils import load_returns_csv

torch.manual_seed(42)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── VP-SDE schedule ──────────────────────────────────────────────────────────
BETA_MIN, BETA_MAX, T_STEPS = 0.1, 20.0, 1000


def get_alpha_sigma(t):
    """t: tensor in [0,1]. Returns alpha_t, sigma_t."""
    log_alpha = -0.5 * (BETA_MIN * t + 0.5 * (BETA_MAX - BETA_MIN) * t ** 2)
    alpha = torch.exp(log_alpha)
    sigma = torch.sqrt(1 - alpha ** 2 + 1e-8)
    return alpha, sigma


# ── Score Network ─────────────────────────────────────────────────────────────
class ScoreNet(nn.Module):
    """Simple MLP score network. Input: (x_t, t_emb). Output: score."""
    def __init__(self, d, hidden=256, t_emb_dim=64):
        super().__init__()
        self.t_embed = nn.Sequential(
            nn.Linear(1, t_emb_dim), nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(d + t_emb_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d),
        )

    def forward(self, x, t):
        t_emb = self.t_embed(t.unsqueeze(-1))
        return self.net(torch.cat([x, t_emb], dim=-1))


# ── Training ─────────────────────────────────────────────────────────────────
def train_score_model(train_path="data/train_2014_2020.csv",
                      val_path="data/val_2021.csv",
                      epochs=2000, batch=256, lr=1e-3,
                      hidden=256, t_emb_dim=64, seed=42):
    """
    Train VP-SDE score model on training data only.
    Scaler parameters (mu, std) are computed from training data only.
    """
    torch.manual_seed(seed)

    # Load train — scaler fitted here and ONLY here
    df_train = load_returns_csv(train_path)
    X_train  = torch.tensor(df_train.values, dtype=torch.float32)
    mu  = X_train.mean(0)
    std = X_train.std(0) + 1e-8
    X_train_norm = (X_train - mu) / std
    N_train, d = X_train_norm.shape

    # Load val (transform with TRAINING scaler — no fitting)
    df_val = load_returns_csv(val_path)
    X_val  = torch.tensor(df_val.values, dtype=torch.float32)
    X_val_norm = (X_val - mu) / std

    print(f"Training samples: {N_train}  ({df_train.index[0].date()} to {df_train.index[-1].date()})")
    print(f"Validation samples: {len(X_val)}  ({df_val.index[0].date()} to {df_val.index[-1].date()})")
    print(f"Assets: {d}  Hidden: {hidden}")

    model = ScoreNet(d, hidden=hidden, t_emb_dim=t_emb_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        idx = torch.randint(0, N_train, (batch,))
        x0  = X_train_norm[idx].to(DEVICE)
        t   = torch.rand(batch, device=DEVICE)
        alpha, sigma = get_alpha_sigma(t)
        alpha = alpha.unsqueeze(-1); sigma = sigma.unsqueeze(-1)
        eps   = torch.randn_like(x0)
        x_t   = alpha * x0 + sigma * eps
        sigma_clamped = sigma.clamp(min=0.01)
        score_target  = -eps / sigma_clamped
        score_pred    = model(x_t, t)
        eps_pred      = -score_pred * sigma_clamped
        loss          = ((eps_pred - eps) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); scheduler.step()
        losses.append(loss.item())

        if (epoch + 1) % 200 == 0:
            # Validation loss (monitoring only, not used for model selection)
            model.eval()
            with torch.no_grad():
                N_v = len(X_val_norm)
                idx_v = torch.randint(0, N_v, (min(batch, N_v),))
                x0v  = X_val_norm[idx_v].to(DEVICE)
                tv   = torch.rand(len(idx_v), device=DEVICE)
                av, sv = get_alpha_sigma(tv)
                av = av.unsqueeze(-1); sv = sv.unsqueeze(-1)
                epsv = torch.randn_like(x0v)
                x_tv = av * x0v + sv * epsv
                sv_c = sv.clamp(min=0.01)
                ep_v = -model(x_tv, tv) * sv_c
                val_loss = ((ep_v - epsv) ** 2).mean().item()
                val_losses.append(val_loss)
            print(f"  Epoch {epoch+1:5d}  train_loss={loss.item():.5f}  "
                  f"val_loss={val_loss:.5f}")

    os.makedirs("checkpoints", exist_ok=True)
    cfg = {
        "epochs": epochs, "batch": batch, "lr": lr,
        "hidden": hidden, "t_emb_dim": t_emb_dim, "seed": seed,
        "beta_min": BETA_MIN, "beta_max": BETA_MAX,
        "train_start": "2014-01-01", "train_end": "2020-12-31",
        "val_start": "2021-01-01", "val_end": "2021-12-31",
        "test_start": "N/A — not used",
        "data_used_for_fit": "train_2014_2020",
        "data_used_for_selection": "val_2021 (monitoring only)",
        "data_used_for_evaluation": "none",
        "scaler_fitted_on": "train_2014_2020",
        "n_train": N_train, "d": d,
        "final_train_loss": float(losses[-1]),
    }

    torch.save({
        "model": model.state_dict(),
        "mu": mu, "std": std, "d": d,
        "train_start": "2014-01-01", "train_end": "2020-12-31",
        "scaler_fitted_on": "train_2014_2020",
    }, "checkpoints/score_model_base.pt")

    with open("checkpoints/score_model_base_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    np.save("checkpoints/score_model_base_losses.npy", np.array(losses))

    # Also save under legacy name for backward compat with scripts that load score_model.pt
    torch.save({
        "model": model.state_dict(),
        "mu": mu, "std": std, "d": d,
        "train_start": "2014-01-01", "train_end": "2020-12-31",
        "scaler_fitted_on": "train_2014_2020",
    }, "checkpoints/score_model.pt")

    print(f"\nSaved: checkpoints/score_model_base.pt  |  final loss={losses[-1]:.5f}")
    return model, mu, std


if __name__ == "__main__":
    import sys
    epochs = 2000
    if "--epochs" in sys.argv:
        epochs = int(sys.argv[sys.argv.index("--epochs") + 1])
    if "--fast" in sys.argv:
        epochs = 100
    train_score_model(epochs=epochs)
