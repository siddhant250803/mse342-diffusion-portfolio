# Project Proposal: MS&E 342

**Title:** Reward-Conditioned Reverse Diffusion for Portfolio Optimization  
**Student:** Siddhant Sukhani  
**Course:** MS&E 342 — Stochastic Systems and Learning Theory with Applications in Finance  
**Instructor:** Prof. Renyuan Xu  
**Submission date:** May 22, 2026 · Final report due June 12, 2026

---

## Abstract

We propose a framework for fine-tuning a score-based diffusion model's reverse stochastic differential equation (SDE) drift under a portfolio-utility reward, using KL-penalized stochastic control. The key observation is that existing work treats the diffusion model as a frozen scenario generator with no feedback from the downstream portfolio objective; and the stochastic control theory for reward fine-tuning of diffusion models (Han, Razaviyayn & Xu, ICML 2025) has not yet been applied to financial decision-making. We bridge these two threads, derive the HJB optimality condition for the fine-tuning problem, introduce optimal transport (OT) calibration as a distributional correction step, and evaluate the full pipeline empirically on S&P 500 sector ETF data. The empirical goal is deliberately framed as exploratory: to characterize the trade-off between realism, reward-seeking, and portfolio performance, rather than to claim statistically significant outperformance from a short test window.

---

## 1. Motivation

### 1.1 The gap in the existing literature

Score-based diffusion models have recently been applied to financial scenario generation and portfolio optimization (Aghapour, Bayraktar & Yuan, NeurIPS 2025; Gao, Zha & Zhou, 2025). In these works the diffusion model is trained to approximate the historical return distribution and scenarios are sampled from it unconditionally. The portfolio optimizer then uses these scenarios as inputs. **The diffusion model and the portfolio objective are completely decoupled**: the generative model is never updated based on whether the scenarios it produces are useful for downstream decisions.

Concurrently, a rigorous stochastic control framework for reward fine-tuning of diffusion models has been developed (Han, Razaviyayn & Xu, ICML 2025; Domingo-Enrich et al., ICLR 2025). These works derive HJB equations, prove convergence of policy iteration, and show that fine-tuning the reverse SDE drift under a KL budget is well-posed. **Neither paper applies this theory to finance or to portfolio utility rewards.**

Our contribution is technically distinct from Aghapour, Bayraktar & Yuan (NeurIPS 2025): they model asset prices as a *diffusion-generated continuous-time process* and solve a dynamic HJB for the portfolio weight over the full price path, with state space $(t, \text{price}_t)$ and a path-dependent objective. Our setting is categorically different — the diffusion model is a *scenario generator* for a terminal return vector, and we fine-tune the generator's reverse SDE drift rather than the portfolio weight. The two problems have different state spaces, different HJB equations, and different solution methods.

The bridge between the two threads — fine-tuning a financial scenario-generating diffusion model toward a terminal portfolio utility using the stochastic control machinery of Han/Razaviyayn/Xu — is the contribution of this project.

### 1.2 Why the decoupling matters

A scenario generator optimised for generic statistical fidelity (e.g., denoising score matching loss) does not necessarily produce scenarios that are decision-relevant. Our preliminary experiments make this concrete: a VP-SDE score model trained on 10 sector ETFs generates scenarios with approximately half the historical volatility, causing mean-variance optimisation to allocate 100% of the portfolio to a single asset. The Bures-Wasserstein distance between the generated and real return distributions is $W_2 = 0.028$, and the Gaussian optimal transport map reveals that the model systematically underestimates cross-asset covariance. A scenario generator that is fine-tuned to be useful for portfolio decisions — while remaining close to the real distribution via a KL constraint — should correct this failure mode. The final evaluation will test this as a risk-performance frontier, with explicit attention to concentration, turnover, drawdown, and statistical uncertainty.

---

## 2. Mathematical Framework

### 2.1 Forward VP-SDE and score function

We model the joint daily log-return vector $X_0 \in \mathbb{R}^d$ under the variance-preserving SDE:

$$dX_t = -\tfrac{\beta(t)}{2}\, X_t\, dt + \sqrt{\beta(t)}\, dW_t, \qquad \beta(t) = \beta_{\min} + t(\beta_{\max} - \beta_{\min})$$

with marginal $X_t \mid X_0 \sim \mathcal{N}(\alpha_t X_0,\, \sigma_t^2 I)$ where $\alpha_t = \exp(-\tfrac{1}{2}\int_0^t \beta)$, $\sigma_t^2 = 1-\alpha_t^2$. A score network $s_\theta$ is trained via denoising score matching. By the time-reversal theorem, generating a scenario reduces to integrating the reverse SDE:

$$dY_t = \bigl[-\tfrac{\beta(T{-}t)}{2}\,Y_t + \beta(T{-}t)\,s_\theta(Y_t,T{-}t)\bigr]dt + \sqrt{\beta(T{-}t)}\,d\tilde{W}_t, \qquad Y_0 \sim \mathcal{N}(0,I)$$

### 2.2 Optimal transport calibration

Before fine-tuning, we align the generated distribution with the real data via optimal transport. By Brenier's theorem, the unique $W_2$-optimal map between two absolutely continuous measures is $T^* = \nabla\varphi$ for a convex potential $\varphi$. Under the Gaussian approximation, this is the closed-form Bures-Wasserstein map:

$$T^*(x) = \mu_q + A(x - \mu_p), \qquad A = \Sigma_p^{-1/2}\bigl(\Sigma_p^{1/2}\Sigma_q\Sigma_p^{1/2}\bigr)^{1/2}\Sigma_p^{-1/2}$$

For the non-Gaussian case, we will use the Sinkhorn algorithm (entropy-regularised discrete OT) to compute a sample-level transport plan and barycentric projection. This part is a planned extension rather than a result to overstate: the current empirical calibration result is Gaussian OT, while the final project will either implement a full out-of-sample Sinkhorn barycentric map or clearly report only the Gaussian calibration.

We will also test an OT-aware score model by augmenting the denoising score matching loss with a differentiable Sinkhorn penalty, so $p_\theta$ itself is trained to match $p_\text{data}$ in Wasserstein distance. This addresses a reference-measure issue: if we apply OT only post hoc as a sample transformation, then the KL penalty in problem $(\star)$ remains relative to the original miscalibrated base model. The final comparison will therefore separate three cases: raw score model, post-hoc Gaussian OT calibration, and OT-augmented score-model training.

### 2.3 Fine-tuning as KL-penalized stochastic control

We introduce a control perturbation $u_\phi(Y_t, t)$ to the reverse drift and solve:

$$\max_{\phi}\; \mathbb{E}_{p_\phi}\!\bigl[U(Y_T)\bigr] - \eta\cdot\mathrm{KL}(p_\phi \,\|\, p_\theta) \tag{$\star$}$$

where $U(Y_T)$ is a terminal portfolio reward, and the controlled reverse SDE is:

$$dY_t = \bigl[-\tfrac{\beta}{2}Y_t + \beta\,s_\theta(Y_t,t) + u_\phi(Y_t,t)\bigr]dt + \sqrt{\beta}\,d\tilde{W}_t$$

By Girsanov's theorem, the KL penalty equals the quadratic control cost:

$$\mathrm{KL}(p_\phi \,\|\, p_\theta) = \mathbb{E}\!\left[\int_0^T \frac{\|u_\phi(Y_t,t)\|^2}{2\,\beta(T-t)}\,dt\right]$$

This is an entropy-regularised stochastic control problem. Substituting the KL expression, problem $(\star)$ becomes:
$$\max_u \mathbb{E}\!\left[U(Y_T) - \eta\int_0^T \frac{\|u_t\|^2}{2\beta(T{-}t)}\,dt\right]$$
The running cost is $-\tfrac{\eta}{2\beta}\|u\|^2$. The Hamiltonian maximization $\sup_u\!\bigl\{u\cdot\nabla_y V - \tfrac{\eta}{2\beta}\|u\|^2\bigr\}$ yields first-order condition $\nabla_y V = \tfrac{\eta}{\beta} u$, so:
$$u^*(t,y) = \frac{\beta(T{-}t)}{\eta}\,\nabla_y V(t,y)$$
Substituting back, the supremum equals $\tfrac{\beta(T-t)}{2\eta}\|\nabla_y V\|^2$. The Hamilton-Jacobi-Bellman equation for $(\star)$ is therefore:

$$\partial_t V + \mathcal{L}^{s_\theta}V + \frac{\beta(T{-}t)}{2\eta}\|\nabla_y V\|^2 = 0, \qquad V(T,y) = U(y) \tag{HJB}$$

The scalar $\eta$ is the **distributional budget**: small $\eta$ makes the running cost cheap, so $u^*$ can be large (aggressive reward-seeking); large $\eta$ makes $u^*$ small, keeping $p_\phi$ close to $p_\theta$ (realism-preserving). This is the continuous-time analogue of reference-policy reward fine-tuning: the pre-trained diffusion model plays the role of the reference policy and the portfolio utility plays the role of the reward model.

The preliminary implementation used the quadratic proxy
$$U_{\text{quad}}(r)=\frac{1}{2\lambda}r^\top\Sigma^{-1}r,$$
which is analytically convenient but can reward large Mahalanobis moves regardless of sign. The final empirical specification will therefore use a decision-aligned reward based on portfolio returns under scenario-estimated weights:
$$U_{\text{port}}(\mathcal{S}) = \hat{\pi}(\mathcal{S})^\top \bar{r}_{\text{val}} - \frac{\lambda}{2}\hat{\pi}(\mathcal{S})^\top \Sigma_{\text{val}}\hat{\pi}(\mathcal{S}) - \gamma\,\mathrm{CVaR}_{95}(\hat{\pi}(\mathcal{S})^\top r_{\text{val}}),$$
where $\mathcal{S}$ is a generated scenario batch, $\hat{\pi}(\mathcal{S})$ is the long-only MVO portfolio implied by that batch, and the validation moments are estimated from data not used for the final test. The quadratic reward will remain in the theory section as the solvable linear-Gaussian special case.

---

## 3. Proposed Contributions

**Contribution 1 — Theoretical (small theorem).** In the linear-Gaussian special case ($p_\text{data} = \mathcal{N}(\mu, \Sigma)$ with perfect score $s_\theta(y,t) = -(y - \alpha_t\mu)/\sigma_t^2$, quadratic terminal reward $U_{\text{quad}}(r) = r^\top\Sigma^{-1}r/(2\lambda)$), we derive the closed-form solution to (HJB) via a quadratic ansatz $V(t,y) = \tfrac{1}{2}y^\top Q(t)y + b(t)^\top y + c(t)$ with terminal condition $Q(T)=\Sigma^{-1}/\lambda$, $b(T)=0$. The resulting ODEs for $Q(t), b(t)$ are tractable, and the optimal control at the terminal time is:
$$u^*(T, y) = \frac{\beta(T)}{\eta\lambda}\,\Sigma^{-1} y$$
This is a drift proportional to $\Sigma^{-1}y$, with magnitude controlled by the budget $\eta$: smaller $\eta$ allows a larger tilt. The fine-tuned terminal distribution remains Gaussian with covariance inflated in the principal eigendirections of $\Sigma^{-1}$, giving an interpretable geometric characterisation of how reward fine-tuning changes the generated scenario law. I will present this as the tractable benchmark case, not as the exact empirical reward.

**Contribution 2 — OT calibration layer.** Show that Gaussian OT reduces the Bures-Wasserstein gap between generated and real returns and recovers near-real portfolio weight vectors in the preliminary setup. Then test whether a true Sinkhorn barycentric map and/or Sinkhorn-augmented score training improves the root cause (underestimated variance) rather than only applying a post-hoc affine correction. I will report Gaussian OT and Sinkhorn OT separately, with no claim that the current Gaussian result proves non-Gaussian calibration.

**Contribution 3 — Empirical: validation-selected $\eta$ frontier.** On S&P 500 sector ETF data (yfinance), use 2014–2020 for score-model training, 2021 for validation/model selection, and 2022–2024 as a held-out test period. Select $\eta \in \{0.05, 0.1, 0.5, 1.0, 5.0\}$ on the validation window, then evaluate the selected model once on the test period. This avoids choosing $\eta$ using test-period Sharpe.

| Strategy | Tests |
|---|---|
| Equal-weight | Naive benchmark |
| Markowitz (plug-in) | Classical baseline |
| Wasserstein-robust MVO | Static robustness baseline (Blanchet et al. 2022) |
| Diffusion-Base → MVO | Generative model, no fine-tuning |
| Diffusion + OT → MVO | OT calibration only |
| Diffusion + OT + FT ($\eta$) | Full pipeline, validation-selected $\eta$ |

Primary metrics: Sharpe ratio, CVaR(95%), max drawdown, turnover, and portfolio concentration (Herfindahl index / max weight). The 2022 rate shock (largest multi-asset drawdown in the test window) is the main stress test. The $\eta$ frontier plot is the empirical core of the paper, but it will be framed as a realism-performance trade-off rather than a statistically decisive outperformance claim.

To make the comparison fair, the diffusion strategies will be evaluated in two modes:

1. **Fixed-scenario mode:** one generated scenario set from the 2014–2020 training window, useful for isolating the effect of calibration and fine-tuning.
2. **Rolling mode:** re-estimate or recalibrate scenario moments on the same rolling 252-day schedule used by the Markowitz and robust baselines. This is the main backtest comparison if computationally feasible.

---

## 4. Preliminary Results

Data: 10 S&P 500 sector ETFs, yfinance, 2014-2024.  
Score model: 3-layer MLP, VP-SDE, 500 training epochs, DSM loss converges to 0.19.  
OT: Gaussian map computed from training data moments.

**OT calibration (key finding):**

| | $W_2$ to real data | XLK weight (MVO) |
|---|---|---|
| Historical data | — | 35.8% |
| Generated (raw) | 0.0277 | 100% |
| After Gaussian OT | 0.0002 | 36.1% |

The base model severely underestimates volatility, causing MVO to concentrate entirely in one asset. The Gaussian OT map recovers near-real portfolio weights. *Note: $W_2 = 0.0002$ after Gaussian OT is near-zero by construction (the map minimises $W_2$ between Gaussian approximations); the meaningful diagnostic is the portfolio weight recovery, which is validated out-of-sample in the backtest.*

**Portfolio backtest — preliminary fixed-scenario pipeline (test 2022–2024, monthly rebalancing):**

| Strategy | Ann. Return | Sharpe | 95% CI Sharpe | Max Drawdown |
|---|---|---|---|---|
| Equal-weight | 5.6% | 0.36 | [−0.76, +1.46] | −18.4% |
| Markowitz | 9.0% | **0.52** | [−0.60, +1.57] | −17.1% |
| Wasserstein-Robust | 5.8% | 0.38 | [−0.74, +1.48] | −17.9% |
| Diffusion-Base | 11.5% | 0.45 | [−0.66, +1.55] | −33.1% |
| Diffusion + OT | 6.1% | 0.38 | [−0.72, +1.50] | −22.3% |
| Diffusion + FT ($\eta^*=0.1$) | 10.2% | 0.49 | [−0.63, +1.60] | −29.2% |
| Diffusion + OT + FT ($\eta^*=0.1$) | 8.8% | 0.47 | [−0.69, +1.57] | **−26.4%** |

This preliminary table is useful for diagnosing behavior, but it is not the final causal claim. In the current implementation, $\eta$ was swept directly on the 2022–2024 test window and the diffusion strategies used fixed scenario sets rather than the same rolling estimation protocol as the classical baselines. The final project will correct this by adding a 2021 validation window, selecting $\eta$ before test evaluation, reporting turnover and concentration, and adding a rolling diffusion-scenario backtest.

The main preliminary interpretation is therefore cautious: (1) the base diffusion model underestimates volatility and creates overly concentrated portfolios; (2) Gaussian OT fixes the first two moments and recovers diversified weights close to historical MVO; (3) KL-penalized fine-tuning changes the Sharpe/drawdown trade-off, but current point estimates are not statistically significant because bootstrap 95% confidence intervals are wide for a 3-year test window. The final write-up will present this as evidence of an interpretable frontier, not as proof of outperformance.

---

## 5. Technical Ingredients

The project uses three technical ingredients:

| Technical ingredient | Role in project |
|---|---|
| Fokker-Planck equation | Governs density evolution under VP-SDE |
| Time-reversal theorem | Reverse SDE for scenario generation |
| Denoising score matching | Score network training objective |
| Tweedie's formula | Denoised return estimate $\mathbb{E}[X_0\mid X_t]$ |
| HJB equation | Optimality condition for fine-tuned control $u^*$ |
| Entropy-regularised HJB | Exact structure of problem $(\star)$ |
| Girsanov / KL as quadratic cost | KL = $\mathbb{E}[\int \|u\|^2/2\beta\,dt]$ |
| Reference-policy reward fine-tuning | Fine-tuning formulation and policy iteration |

---

## 6. Timeline

| | Milestone |
|---|---|
| Week 1 (done) | Data pipeline, VP-SDE score model, Gaussian OT calibration, baseline backtest |
| Week 2 | Add 2014–2020 / 2021 / 2022–2024 train-validation-test split; retrain score model; verify stylized facts |
| Week 3 | Implement validation-selected $\eta$ sweep and report test metrics only after model selection |
| Week 4 | Add portfolio-aligned reward, CVaR penalty, turnover, concentration, and rolling diffusion backtest |
| Week 5 | Closed-form linear-Gaussian theory section; separate Gaussian OT from Sinkhorn OT claims |
| Week 6 (Jun 12) | Final write-up and submission |

---

## 7. References

1. Aghapour, Bayraktar & Yuan. "Solving dynamic portfolio selection problems via score-based diffusion models." *NeurIPS 2025.* arXiv:2507.09916
2. **Han, Razaviyayn & Xu.** "Stochastic Control for Fine-tuning Diffusion Models: Optimality, Regularity, and Convergence." *ICML 2025.* arXiv:2412.18164
3. Domingo-Enrich et al. "Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control." *ICLR 2025.* arXiv:2409.08861
4. Gao, Zha & Zhou. "Data-driven generative simulation of SDEs using diffusion models." 2025. arXiv:2509.08731
5. Blanchet, Chen & Zhou. "Distributionally Robust Mean-Variance Portfolio Selection with Wasserstein Distances." *Management Science* 68(9), 2022.
6. Jia & Zhou. "Policy Evaluation and Temporal-Difference Learning in Continuous Time and Space." *JMLR* 23(154), 2022.
7. Tang & Zhao. "Score-based diffusion models via stochastic differential equations." *Statistic Surveys* 19, 2025.
8. Pham. *Continuous-time stochastic control and optimization with financial applications.* Springer, 2009.
