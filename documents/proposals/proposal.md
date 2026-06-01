# MS&E 342 Project Proposal
## Reward-Conditioned Reverse Diffusion for Portfolio Optimization

**Author:** Siddhant Sukhani  
**Date:** May 22, 2026  
**Instructor:** Renyuan Xu

---

## 1. Problem Statement

Generative diffusion models have recently emerged as powerful tools for simulating financial return distributions. However, existing work uses the diffusion model as a **passive, frozen scenario generator**: paths are sampled from the learned distribution and fed into a downstream optimizer, with no feedback from the portfolio objective back to the generative model. This decoupling is a fundamental inefficiency. Scenarios that are statistically realistic in a generic sense need not be decision-relevant; a model that generates accurate average-day returns may poorly cover the tail events that dominate portfolio risk.

This project asks: **can we fine-tune the reverse SDE drift of a diffusion model under a portfolio-utility reward, using KL-penalized stochastic control, to generate scenarios that are both realistic and decision-relevant?**

---

## 2. Mathematical Framework

### 2.1 Forward VP-SDE

We model the joint daily log-return vector $X_0 \in \mathbb{R}^d$ under a **variance-preserving SDE**:

$$dX_t = -\frac{\beta(t)}{2} X_t \, dt + \sqrt{\beta(t)} \, dW_t, \qquad \beta(t) = \beta_{\min} + t(\beta_{\max} - \beta_{\min})$$

The marginal satisfies $X_t \mid X_0 \sim \mathcal{N}(\alpha_t X_0,\, \sigma_t^2 I)$ where
$$\alpha_t = \exp\!\left(-\tfrac{1}{2}\int_0^t \beta(s)\,ds\right), \qquad \sigma_t^2 = 1 - \alpha_t^2$$

and the Fokker–Planck equation governs the density evolution $\partial_t p_t = \nabla \cdot (\frac{\beta(t)}{2} x \, p_t) + \frac{\beta(t)}{2} \Delta p_t$.

### 2.2 Score Function and Reverse SDE

A score network $s_\theta(x,t) \approx \nabla \log p_t(x)$ is trained via **denoising score matching**:

$$\mathcal{L}(\theta) = \mathbb{E}_{t,X_0,\varepsilon}\!\left[\left\| s_\theta(X_t, t) + \frac{\varepsilon}{\sigma_t} \right\|^2\right], \quad \varepsilon \sim \mathcal{N}(0,I), \quad X_t = \alpha_t X_0 + \sigma_t \varepsilon$$

The **time-reversal theorem** gives the reverse SDE:

$$dY_t = \left[-\frac{\beta(T{-}t)}{2} Y_t + \beta(T{-}t)\, s_\theta(Y_t, T{-}t)\right]dt + \sqrt{\beta(T{-}t)}\, d\tilde{W}_t$$

Integrating from $Y_0 \sim \mathcal{N}(0,I)$ to $t = T$ yields $Y_T \approx X_0 \sim p_{\text{data}}$.

### 2.3 Portfolio Control Problem

Let $r = Y_T \in \mathbb{R}^d$ be a generated return scenario. The closed-form mean-variance utility under optimal Markowitz weights $\pi^* = \frac{1}{\lambda} \Sigma^{-1} \mu$ is:

$$U(r) = \frac{1}{2\lambda} r^\top \Sigma^{-1} r$$

### 2.4 Fine-Tuning as KL-Penalized Stochastic Control

Following **Han, Razaviyayn & Xu (ICML 2025)**, we introduce a control perturbation $u_\phi(Y_t, t)$ to the reverse drift and solve:

$$\max_\phi \; \mathbb{E}_{p_\phi}\!\left[U(Y_T)\right] - \eta \cdot \mathrm{KL}(p_\phi \,\|\, p_\theta)$$

where the controlled reverse SDE is:

$$dY_t = \underbrace{\left[-\frac{\beta(T{-}t)}{2} Y_t + \beta(T{-}t)\, s_\theta(Y_t, T{-}t)\right]}_{\text{base drift}} dt + \underbrace{u_\phi(Y_t, t)}_{\text{control}} \, dt + \sqrt{\beta(T{-}t)}\, d\tilde{W}_t$$

**Girsanov's theorem** connects the KL divergence to a quadratic control cost:

$$\mathrm{KL}(p_\phi \,\|\, p_\theta) = \mathbb{E}\!\left[\int_0^T \frac{\|u_\phi(Y_t, t)\|^2}{2\,\beta(T{-}t)}\, dt\right]$$

This is precisely an **entropy-regularized HJB structure**. The Hamiltonian maximization $\sup_u\{u \cdot \nabla_y V - \frac{\eta}{2\beta}\|u\|^2\}$ yields optimal control $u^*(t,y) = \frac{\beta(T-t)}{\eta}\nabla_y V(t,y)$, and $V$ solves:

$$\partial_t V + \mathcal{L}^{s_\theta} V + \frac{\beta(T-t)}{2\eta} \|\nabla_y V\|^2 = 0, \qquad V(T, y) = U(y)$$

The KL penalty η controls the **distributional budget**: small η makes the control cost cheap, so $u^*$ can be large (aggressive reward-seeking); large η keeps the fine-tuned distribution close to the base model (preserving realism). This is the continuous-time analogue of Wasserstein robustness (Blanchet, Chen & Zhou 2022).

---

## 3. Preliminary Experiments

### Data
Daily log-returns for 10 S&P 500 sector ETFs (XLB, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY) sourced via `yfinance`. Training: 2014–2021 (1,569 days). Test: 2022–2024 (752 days).

### Models
- **Score network**: 3-layer MLP with sinusoidal time embedding, trained via denoising score matching for 500 epochs (ε-prediction, MSE loss). Loss converges to ~0.19.
- **Control network**: smaller 2-layer MLP initialized near zero, fine-tuned for 300 epochs with $\eta = 0.5$, $\lambda = 5.0$.

### Preliminary results (monthly rebalancing, test 2022–2024)

| Strategy | Ann. Return | Sharpe | CVaR(95%) | Max Drawdown |
|---|---|---|---|---|
| Equal-weight | 5.59% | 0.363 | 0.0226 | -18.4% |
| Markowitz (plug-in) | 9.03% | **0.515** | 0.0264 | -17.1% |
| Wasserstein-Robust | 5.75% | 0.380 | **0.0222** | -17.9% |
| Diffusion-Base (no FT) | **11.48%** | 0.453 | 0.0354 | -33.1% |
| Diffusion-Finetuned (η=0.5) | 3.35% | 0.172 | 0.0275 | -26.5% |

### What the preliminary results reveal

The base diffusion model (500 training epochs) generates **underestimated-variance scenarios** (generated $\sigma \approx 0.007$ vs. historical $\sigma \approx 0.014$). This causes MVO to produce overconfident, concentrated weights — hence high returns in good years but catastrophic drawdowns (-33%). The naive fine-tuning exacerbates concentration by pushing the scenario distribution toward individually high-return paths without the KL constraint being tight enough to preserve covariance structure.

These preliminary results motivate **three concrete improvements** for the full project:

1. **More score model training** (2000+ epochs): bring generated $\sigma$ closer to historical before fine-tuning
2. **η calibration via held-out validation**: cross-validate η over {0.05, 0.1, 0.5, 1.0, 5.0} on a validation window
3. **CVaR-augmented reward**: replace $U(r) = r^\top \Sigma^{-1} r / (2\lambda)$ with $U(r) = \pi^{*\top} r - \gamma \cdot \text{CVaR}_{1-\alpha}(\pi^{*\top} r_{\text{historical}})$ to explicitly penalize tail exposure

---

## 4. Proposed Contributions

**Contribution 1 (Theory).** Derive the HJB equation for the KL-penalized portfolio fine-tuning problem explicitly. The correct optimal control is $u^*(t,y) = \frac{\beta(T-t)}{\eta}\nabla_y V(t,y)$ (η in the denominator, not numerator — see Section 2.4). In the linear-Gaussian special case ($p_\text{data} = \mathcal{N}(\mu,\Sigma)$ with exact score), the value function is quadratic and the optimal terminal control is $u^*(T,y) = \frac{\beta(T)}{\eta\lambda}\Sigma^{-1}y$ — a drift in the Markowitz direction of maximum Sharpe ratio, with magnitude inversely proportional to the KL budget η.

**Contribution 2 (Empirics).** Run the full experimental pipeline (2000 epochs, cross-validated η) and compare:

| Method | What it tests |
|---|---|
| Equal-weight | Naive benchmark |
| Markowitz plug-in | Classical baseline |
| Wasserstein-robust MVO | Robustness without dynamics |
| Diffusion-Base → MVO | Generative scenarios, no fine-tuning |
| Diffusion-Finetuned (MV reward) | Core contribution |
| Diffusion-Finetuned (CVaR reward) | Extension |

Evaluate over 2022–2024 test period (includes 2022 rate shock — the most severe multi-asset drawdown in the dataset). Primary metrics: Sharpe, CVaR(95%), max drawdown, turnover.

**Contribution 3 (Analysis).** Quantify how the KL budget η affects the generated scenario distribution:
- As η → 0: scenarios collapse toward reward-maximizing degenerate paths (look-ahead bias)
- As η → ∞: scenarios revert to base diffusion (no utility improvement)
- Plot the **η–Sharpe frontier** and the **η–KL cost curve** to identify the optimal operating point

---

## 5. Technical Ingredients

| Technical ingredient | Where it appears in this project |
|---|---|
| Fokker–Planck equation | Density evolution under VP-SDE forward process |
| Time-reversal theorem | Reverse SDE derivation for scenario generation |
| Denoising score matching | Score network training objective |
| Tweedie's formula | Denoised return estimate $\mathbb{E}[X_0 \mid X_t]$ |
| HJB equation | Optimality condition for fine-tuned control $u^*$ |
| Entropy-regularized HJB | KL-penalized objective structure |
| Girsanov theorem / KL as quadratic cost | KL = $\mathbb{E}[\int \|u\|^2 / 2\beta \, dt]$ |
| Reference-policy reward fine-tuning | Fine-tuning formulation and policy iteration |

---

## 6. Timeline

| Week | Milestone |
|---|---|
| Week 1 (done) | Data pipeline, VP-SDE score model, preliminary experiments |
| Week 2 | Retrain score model (2000 epochs); verify stylized facts match |
| Week 3 | Implement η cross-validation; run fine-tuning sweep over η ∈ {0.05,0.1,0.5,1,5} |
| Week 4 | Add CVaR reward variant; implement η–Sharpe frontier plots |
| Week 5 | Derive closed-form solution in linear-Gaussian special case; write theory section |
| Week 6 (Jun 12) | Write-up, finalize figures, submit |

---

## References

1. Aghapour, Bayraktar & Yuan. "Solving dynamic portfolio selection problems via score-based diffusion models." *NeurIPS 2025*. [arXiv:2507.09916](https://arxiv.org/abs/2507.09916)
2. Han, Razaviyayn & Xu. "Stochastic Control for Fine-tuning Diffusion Models: Optimality, Regularity, and Convergence." *ICML 2025*. [arXiv:2412.18164](https://arxiv.org/abs/2412.18164)
3. Domingo-Enrich et al. "Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control." *ICLR 2025*. [arXiv:2409.08861](https://arxiv.org/abs/2409.08861)
4. Gao, Zha & Zhou. "Data-driven generative simulation of SDEs using diffusion models." 2025. [arXiv:2509.08731](https://arxiv.org/abs/2509.08731)
5. Blanchet, Chen & Zhou. "Distributionally Robust Mean-Variance Portfolio Selection with Wasserstein Distances." *Management Science* 68(9), 2022.
6. Jia & Zhou. "Policy Evaluation and Temporal-Difference Learning in Continuous Time and Space." *JMLR* 23(154), 2022.
7. Tang & Zhao. "Score-based diffusion models via stochastic differential equations." *Statistic Surveys* 19, 2025. *(Course textbook)*
8. Pham. *Continuous-time stochastic control and optimization with financial applications.* Springer, 2009. *(Course textbook)*
