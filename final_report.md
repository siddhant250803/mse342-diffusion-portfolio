# Reward-Conditioned Reverse Diffusion for Portfolio Optimization

**MS&E 342: Stochastic Control and Optimization in Continuous Time**
Siddhant Sukhani
Department of Management Science and Engineering, Stanford University
Spring Quarter 2026

---

## Abstract

We investigate whether a score-based diffusion model for financial return scenarios can be fine-tuned through KL-penalized stochastic control such that generated scenarios remain distributionally realistic while becoming more useful for downstream portfolio optimization. We train a variance-preserving stochastic differential equation (VP-SDE) score model on ten S&P 500 sector ETF daily log-returns from 2014 to 2020, apply Gaussian Bures-Wasserstein optimal transport calibration to align first-order moments, and fine-tune the reverse diffusion drift with a decision-aligned portfolio reward evaluated on 2021 validation data. The fine-tuning objective balances expected portfolio utility against a Girsanov KL divergence penalty, with the regularization strength (eta) selected exclusively on validation data to prevent test-period leakage. In the fixed-scenario backtest over 2022 to 2024, the fine-tuned strategy achieves a Sharpe ratio of 0.70 compared to 0.515 for rolling plug-in Markowitz and 0.363 for equal-weight, with a reduced maximum drawdown of 14.88%. Ninety-five percent bootstrap confidence intervals span approximately 2.1 Sharpe units over the three-year test window, reflecting limited statistical power. Results are presented as exploratory evidence of a realism-performance frontier rather than as statistically decisive claims of outperformance. Code and data splits are fully reproducible via a single command.

**Keywords:** score-based diffusion models, stochastic optimal control, KL divergence, optimal transport, portfolio optimization, mean-variance optimization

---

## 1. Introduction

Generative models of financial return distributions offer a principled alternative to parametric moment estimation for portfolio construction. Rather than fitting a multivariate Gaussian to historical returns and applying mean-variance optimization (MVO), one can sample scenarios from a learned generative model and solve a sample-average stochastic program. This scenario-based approach can in principle accommodate non-Gaussian features that are empirically prominent in financial data, including fat tails, left skewness, and volatility clustering (Cont, 2001; McNeil et al., 2015).

Score-based diffusion models have become the dominant class of high-dimensional generative models following the work of Ho et al. (2020) and Song et al. (2021). Their application to financial data is growing: recent work demonstrates that denoising score matching on return panels can partially recover stylized facts including excess kurtosis and cross-sectional correlation structure (Pelger & Zou, 2023; De Bortoli et al., 2021). However, a distributionally realistic generator is not automatically a useful one for portfolio construction: the score model may generate return moments that differ from historical moments in ways that systematically bias Markowitz weights, producing portfolios that are too concentrated, too volatile, or poorly calibrated to tail risk.

Two complementary tools address this gap. First, optimal transport (OT) calibration can align generated moments to historical moments post-hoc. The Gaussian Bures-Wasserstein map provides a closed-form linear correction that matches means and covariances under a Gaussian approximation (Bures, 1969; Wasserstein, 1969). Second, stochastic control fine-tuning can directly reshape the generative model to produce scenarios that improve a downstream portfolio objective. This second approach is grounded in the KL-penalized stochastic optimal control problem studied extensively in the course, connecting to the policy gradient and control-as-inference literature (Levine, 2018; Han et al., 2025).

The tension at the heart of this project is the trade-off between realism (staying close to the base generative model in KL sense) and performance (biasing scenarios toward high-utility portfolio outcomes). The KL penalty parameter eta parameterizes this trade-off: small eta allows large drift perturbations at the cost of distributional fidelity; large eta forces the fine-tuned model to remain close to the base generator.

This paper makes the following contributions:

1. A reproducible end-to-end pipeline with strict three-way data splits (2014-2020 train, 2021 validation, 2022-2024 test) and a formal leakage audit script.
2. A decision-aligned portfolio reward using validation returns for gradient signal, replacing the quadratic proxy reward that is standard in preliminary implementations.
3. Honest separation of Gaussian OT (first-two-moments calibration under Gaussian approximation) from Sinkhorn OT (subset-only diagnostic), addressing overstatement of OT calibration scope in prior versions.
4. Bootstrap confidence intervals that honestly reflect the limited statistical power of a three-year test window.

---

## 2. Related Work

### 2.1 Score-Based Generative Models

Song and Ermon (2019) introduced score matching for generative modeling, and Song et al. (2021) unified denoising diffusion probabilistic models (Ho et al., 2020) and score-based models under the stochastic differential equation (SDE) framework. The variance-preserving SDE used in this paper follows the VP formulation in Song et al. (2021), which has well-understood convergence properties and has been successfully applied to image synthesis, molecular generation, and time-series modeling.

For financial returns, Lim et al. (2023) and Vuletich and Cucuringu (2024) demonstrate that diffusion models can generate return vectors that match empirical stylized facts, including heavy tails and correlation structure. This project builds on that foundation but focuses on the downstream portfolio utility rather than distributional realism alone.

### 2.2 Distributionally Robust Portfolio Optimization

Blanchet et al. (2022) develop a tractable reformulation of the Wasserstein-ball distributionally robust MVO problem, which forms one of the baselines in this paper. Their approach adds a regularization term to the Markowitz objective that penalizes portfolio variance relative to the Wasserstein radius. Related work by Mohajerin Esfahani and Kuhn (2018) establishes the general theory of Wasserstein distributionally robust optimization, showing that the worst-case expectation over a W2 ball admits a tractable dual formulation.

### 2.3 KL-Penalized Stochastic Control

The fine-tuning objective in this paper follows the KL-penalized policy optimization framework studied in the reinforcement learning literature (Schulman et al., 2015; Ziegler et al., 2019). In the continuous-time SDE setting, Girsanov's theorem gives an explicit quadratic form for the KL cost between two path measures related by a drift perturbation (Oksendal, 2003). Han et al. (2025) apply this framework specifically to fine-tuning diffusion models under arbitrary reward functions, which is the direct methodological basis for this project. The course lectures (MS&E 342, Lectures 12-13) provide the mathematical foundation connecting the HJB equation to the Girsanov-cost stochastic control problem.

### 2.4 Optimal Transport for Distribution Calibration

The Bures-Wasserstein distance between Gaussian distributions has a closed-form expression in terms of the matrix square root (Bhatia et al., 2019), giving a tractable optimal transport map. Peyre and Cuturi (2019) provide a comprehensive review of computational OT methods including the Sinkhorn algorithm used for discrete entropy-regularized OT. For financial applications, OT has been used for distributional robustness (Blanchet et al., 2022), for scenario generation calibration (Backhoff-Veraguas et al., 2020), and for risk measure aggregation.

---

## 3. Mathematical Framework

### 3.1 Variance-Preserving SDE and Score Matching

Let X_0 ~ p_data be a d-dimensional vector of daily log-returns. The VP-SDE forward process is:

```
dX_t = -0.5 * beta(t) * X_t dt + sqrt(beta(t)) dW_t,   t in [0, 1]
beta(t) = beta_min + t * (beta_max - beta_min)
```

with beta_min = 0.1, beta_max = 20.0 following Song et al. (2021). The marginal distribution satisfies:

```
X_t | X_0 ~ N(alpha_t * X_0,  sigma_t^2 * I_d)
alpha_t = exp(-0.5 * integral_0^t beta(s) ds)
sigma_t^2 = 1 - alpha_t^2
```

At t = 1, X_1 is approximately standard Gaussian, forming the prior for the reverse process. We train a score network s_theta(x_t, t): R^d x [0,1] -> R^d via denoising score matching:

```
L_DSM(theta) = E_{t, X_0, epsilon} [||s_theta(alpha_t * X_0 + sigma_t * epsilon, t) + epsilon / sigma_t||^2]
```

where epsilon ~ N(0, I_d). Equivalently, defining epsilon_pred = -s_theta * sigma_t, we minimize mean squared error between predicted and true noise. Scenario generation uses the Euler-Maruyama discretization of the reverse SDE (Anderson, 1982):

```
Y_{k+1} = Y_k + [-0.5 * beta_k * Y_k + beta_k * s_theta(Y_k, t_k)] * dt + sqrt(beta_k * dt) * Z_k
```

for Z_k ~ N(0, I_d) and t_k decreasing from 1 to 0.

### 3.2 HJB Formulation and KL-Penalized Fine-Tuning

We parameterize a drift perturbation u_phi(y, t): R^d x [0,1] -> R^d and consider the controlled reverse SDE:

```
dY_t = [-0.5*beta*Y_t + beta*s_theta(Y_t,t) + u_phi(Y_t,t)] dt + sqrt(beta) dW_t
```

Let p_phi denote the path measure of the controlled process and p_theta the path measure of the uncontrolled (base) process. By Girsanov's theorem (Oksendal, 2003, Theorem 8.6.4), the KL divergence between these path measures is:

```
KL(p_phi || p_theta) = E_{Y~p_phi}[integral_0^T ||u_phi(Y_t, t)||^2 / (2 * beta(T-t)) dt]
```

This is a quadratic control cost in the style of linear-quadratic stochastic control (Anderson & Moore, 1990). The fine-tuning objective is:

```
max_{u_phi}  E_{Y_0 ~ p_phi}[R(Y_0)]  -  eta * KL(p_phi || p_theta)
```

where R: R^d -> R is a portfolio reward function and eta > 0 is the KL penalty weight. The associated Hamilton-Jacobi-Bellman equation (Fleming & Soner, 1993) for the value function V(y, t) satisfies:

```
-partial_t V = max_u {L_theta V + u^T nabla V - eta * ||u||^2 / (2*beta(t))}
            = L_theta V + beta(t) / (2*eta) * ||nabla V||^2
```

where L_theta is the generator of the uncontrolled reverse SDE. The optimal control takes the form:

```
u_phi*(y, t) = beta(t) / eta * nabla_y V(y, t)
```

which recovers a gradient-ascent form: the control steers the process in the direction of increasing value function. In the linear-Gaussian special case with quadratic reward, the value function is quadratic and the optimal control is linear-affine in y, providing a closed-form solution that validates the training procedure.

### 3.3 Gaussian Optimal Transport Calibration

Given empirical samples X_gen ~ p_gen (generated) and X_real ~ p_train (historical training returns), the Bures-Wasserstein W2 distance under the Gaussian approximation is:

```
W2^2(p_gen, p_train) = ||mu_gen - mu_train||^2 + tr(Sigma_gen) + tr(Sigma_train)
                      - 2 * tr((Sigma_gen^{1/2} Sigma_train Sigma_gen^{1/2})^{1/2})
```

The optimal transport map T*: R^d -> R^d that minimizes this is the linear-affine map (Brenier, 1991):

```
T*(x) = mu_train + A @ (x - mu_gen)
A = Sigma_gen^{-1/2} (Sigma_gen^{1/2} Sigma_train Sigma_gen^{1/2})^{1/2} Sigma_gen^{-1/2}
```

This map is applied to all generated scenarios before portfolio construction. It is an exact OT solution under the Gaussian approximation and corrects first-order moments (mean and covariance). Non-Gaussian features, including fat tails, skewness, and volatility clustering, are not corrected by this linear map.

---

## 4. Empirical Design

### 4.1 Data and Splits

We use daily log-returns for ten S&P 500 sector ETFs (XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE) sourced via yfinance (Ranaraja, 2019) with auto-adjustment for dividends and corporate actions. The strict data split is:

- **Training** (2014-01-01 to 2020-12-31): 1,317 trading days. Used exclusively for score model training and scaler fitting.
- **Validation** (2021-01-01 to 2021-12-31): 252 trading days. Used exclusively for eta selection and portfolio reward evaluation during fine-tuning.
- **Test** (2022-01-01 to 2024-12-31): 756 trading days. Evaluated exactly once after all model and hyperparameter choices are frozen.

A standardization scaler (per-asset mean and standard deviation) is computed on training data only and saved with every model checkpoint. No validation or test returns enter scaler computation at any stage.

### 4.2 Score Model Architecture and Training

The score network s_theta is a three-hidden-layer MLP with SiLU activations and sinusoidal time embedding. Architecture: hidden size 256, time embedding dimension 64, input dimension 10 (number of assets). Training: Adam optimizer (Kingma & Ba, 2015) with learning rate 1e-3, cosine annealing decay (Loshchilov & Hutter, 2017) over 2,000 epochs, batch size 256, seed 42. The full-run checkpoint is saved as score_model_base.pt with associated JSON metadata recording the training split and scaler provenance.

### 4.3 Scenario Generation

Scenarios are generated via Euler-Maruyama integration of the reverse SDE with 500 steps. The full production run generates 10,000 scenarios; the fast smoke-test mode uses 500 scenarios and 100 steps. Generated scenarios are de-standardized using the training-fitted scaler. Quality is assessed via Bures-Wasserstein distance, covariance Frobenius error, per-asset kurtosis comparison, QQ plots, and Herfindahl-Hirschman Index (HHI) of implied MVO weights.

### 4.4 Fine-Tuning Procedure

The ControlNet u_phi is a two-hidden-layer MLP (hidden size 128) with tanh output bounding: u_phi(y,t) = tanh(MLP(y,t)) * 1.5. This soft bound prevents unbounded control signals while preserving gradient flow everywhere. Training uses Adam with learning rate 3e-4, 200 epochs (eta sweep), 300 epochs (single run), batch size 128.

The **portfolio validation reward** for a batch of generated return scenarios Y_0 is computed as follows. Let mu_S = mean(Y_0) and Sigma_S = cov(Y_0) denote batch moments. The soft-Markowitz weight vector is:

```
pi_S = softmax((Sigma_S + ridge*I)^{-1} mu_S),   ridge = 1e-4
```

This is a differentiable approximation to the long-only MVO solution; the softmax ensures positive weights summing to one. The portfolio return series on 2021 validation returns r_val (fixed, no gradient) is:

```
port_val = r_val @ pi_S
```

The reward function is:

```
R(Y_0) = 252 * mean(port_val) - (lambda/2) * 252 * var(port_val)
        - gamma * CVaR95(port_val) - tau * HHI(pi_S)
```

with lambda = 5.0, gamma = 0.5, tau = 0.1. CVaR95 uses the empirical 5th percentile of validation portfolio returns. HHI = sum(pi_S^2) penalizes concentration. Gradient flows from r_val through pi_S to (mu_S, Sigma_S) and hence to the generated samples Y_0.

### 4.5 Eta Selection Protocol

Eta candidates are evaluated over {0.05, 0.1, 0.5, 1.0, 5.0}. For each candidate:
1. Train ControlNet for 200 epochs (30 in fast mode).
2. Generate 2,000 scenarios.
3. Compute long-only MVO weights from scenarios.
4. Evaluate validation Sharpe of those weights applied to 2021 returns.

The selected eta* = argmax validation Sharpe. Results saved in results/eta_selected.csv, which explicitly records "NOT AVAILABLE" for test metrics and "validation_sharpe" as the selection_metric field. This file is checked by the leakage audit.

### 4.6 Portfolio Strategies and Backtesting

All strategies rebalance at month-end, hold long-only weights summing to one, and are evaluated over 2022-2024. The full strategy set is:

| Strategy | Description |
|----------|-------------|
| Equal-Weight | 1/d uniform allocation |
| Markowitz | Plug-in MVO on rolling 252-day historical window |
| Wasserstein-Robust | MVO with Sigma_robust = Sigma_hat + (delta/lambda)*I, delta = 0.02 (Blanchet et al., 2022) |
| Diffusion-Base | MVO on base generated scenarios (no calibration) |
| Diffusion+GaussOT | MVO on Gaussian-OT-calibrated scenarios |
| Diffusion+FT(eta*) | MVO on fine-tuned scenarios, validation-selected eta |

**Fixed-scenario backtest:** one scenario set used throughout the test period. Isolates score model behavior from rolling moment adaptation.

**Rolling-diffusion backtest:** at each monthly rebalance date d, the rolling window [d-252, d] is used to compute mu_roll and Sigma_roll. Generated scenarios are recalibrated to (mu_roll, Sigma_roll) via Gaussian OT using only data available on or before d. This introduces no look-ahead.

### 4.7 Performance Metrics

Primary metrics: annualized return, annualized volatility, Sharpe ratio, CVaR at the 95% level, maximum drawdown. Secondary metrics: average turnover per rebalance (L1 weight change), average HHI (portfolio concentration), average maximum weight. Confidence intervals: 95% bootstrap Sharpe CI from 2,000 re-samples with replacement. Simple i.i.d. bootstrap is used; block bootstrap would give wider, more appropriate intervals for autocorrelated returns.

---

## 5. Results

### 5.1 Base Scenario Quality

Table 1 summarizes the scenario quality diagnostics for the fast-mode smoke test. The Bures-Wasserstein distance from generated to training scenarios is 0.030, reflecting a small distributional gap. After Gaussian OT calibration, the W2 distance reduces to 0.000258, confirming that the linear map perfectly matches the empirical first- and second-order moments. The base model generates scenarios with high HHI (0.629), reflecting concentrated implied MVO weights; Gaussian OT calibration reduces HHI to 0.260 by correcting the covariance structure.

**Table 1.** Scenario quality diagnostics (fast-mode, 500 generated scenarios vs. 1,317 training returns).

| Metric | Base Scenarios | Gaussian OT |
|--------|---------------|-------------|
| Bures-Wasserstein distance | 0.0303 | 0.0003 |
| Covariance Frobenius error | --- | reduced |
| Implied MVO HHI | 0.629 | 0.260 |
| NaN fraction | 0.000 | 0.000 |

Non-Gaussian structure (excess kurtosis, fat tails) remains present in generated scenarios and is not removed by the linear OT map. This is expected and correctly labeled in the methodology.

### 5.2 Eta Selection (Validation Period Only)

The validation Sharpe as a function of eta is reported in Table 2. The optimal eta* = 0.5 achieves validation Sharpe 2.30, substantially higher than the base diffusion (which would correspond to eta approaching infinity, i.e., no control). At very small eta (0.05), the control over-steers the scenarios and validation Sharpe falls. At eta = 1.0 and above, the model concentrates portfolio weights (HHI 0.74 at eta = 1.0), suggesting that the control amplifies individual asset exposures when the KL penalty is weak relative to reward.

**Table 2.** Eta sweep validation metrics (2021 validation period only; test data not used).

| Eta | Val Sharpe | Val Ann. Return | Val CVaR95 | Val HHI |
|-----|-----------|----------------|-----------|---------|
| 0.05 | 0.97 | --- | --- | --- |
| 0.10 | 1.95 | 21.85% | 0.0174 | 0.479 |
| **0.50** | **2.30** | **34.93%** | **0.0203** | **0.316** |
| 1.00 | 1.82 | 33.66% | 0.0257 | 0.738 |
| 5.00 | 2.14 | 32.39% | 0.0205 | 0.334 |

Selected: eta* = 0.50 based on validation Sharpe. No test metrics are available at this stage.

### 5.3 Fixed-Scenario Backtest (Test 2022-2024)

**Table 3.** Fixed-scenario backtest results (2022-2024 test period, evaluated once after eta frozen).

| Strategy | Ann. Return | Ann. Vol | Sharpe | 95% CI | CVaR(95%) | Max DD | HHI | Turnover |
|----------|-------------|----------|--------|--------|-----------|--------|-----|----------|
| Equal-Weight | 5.88% | 16.21% | 0.363 | [-0.76, +1.46] | 0.0226 | -18.41% | 0.100 | 0.000 |
| Markowitz | 9.77% | 18.97% | 0.515 | [-0.60, +1.57] | 0.0264 | -17.09% | 0.540 | 0.617 |
| Wasserstein-Robust | 5.75% | 15.13% | 0.380 | [-0.74, +1.48] | 0.0222 | -17.92% | 0.103 | 0.040 |
| Diffusion-Base | 8.69% | 22.52% | 0.386 | [-0.72, +1.49] | 0.0319 | -30.65% | 0.629 | 0.000 |
| Diffusion+GaussOT | 6.45% | 16.54% | 0.390 | [-0.69, +1.49] | 0.0242 | -22.98% | 0.260 | 0.000 |
| **Diffusion+FT(eta=0.5)** | **11.31%** | **16.16%** | **0.700** | **[-0.38, +1.76]** | **0.0241** | **-14.88%** | **0.316** | 0.000 |

The fine-tuned strategy (Diffusion+FT, eta* = 0.5) achieves the highest Sharpe ratio (0.70), the highest annualized return (11.31%), and the lowest maximum drawdown among all strategies (-14.88%). It also achieves the lowest CVaR95 among diffusion strategies (0.0241), comparable to rolling Markowitz.

The base Diffusion-MVO strategy has notably poor performance characteristics: high CVaR95 (0.0319), the largest drawdown (-30.65%), and extreme concentration (HHI 0.629). This reflects that the un-calibrated score model generates scenarios with biased moments that concentrate MVO weights on individual assets. The Gaussian OT calibration substantially improves all three metrics by correcting the covariance structure, but the fine-tuned strategy further improves performance by directly shaping the scenario distribution toward portfolio-useful outcomes.

However, the bootstrap confidence intervals are wide: the fine-tuned strategy's 95% CI is [-0.38, +1.76], spanning 2.1 Sharpe units. Most pairwise Sharpe differences are within one standard error of each other. We make no claim of statistical significance. The results constitute exploratory evidence that the fine-tuning approach moves the realism-performance frontier in the intended direction.

### 5.4 Rolling-Diffusion Backtest (Test 2022-2024)

**Table 4.** Rolling-diffusion backtest results (2022-2024 test period).

| Strategy | Ann. Return | Sharpe | Max DD | HHI | Turnover |
|----------|-------------|--------|--------|-----|----------|
| Equal-Weight | 5.59% | 0.363 | -18.41% | 0.100 | 0.000 |
| Markowitz | 9.03% | 0.515 | -17.09% | 0.541 | 0.617 |
| Wasserstein-Robust | 5.75% | 0.380 | -17.92% | 0.103 | 0.040 |
| Diffusion-Base | 8.89% | 0.510 | -17.09% | 0.496 | 0.585 |
| Diffusion+GaussOT | 8.87% | 0.510 | -17.09% | 0.498 | 0.586 |
| Diffusion+FT(eta=0.5) | 8.88% | 0.510 | -17.09% | 0.496 | 0.585 |

Rolling recalibration causes all diffusion strategies to converge to approximately the same performance as rolling Markowitz. This collapse is expected: at each rebalance, the Gaussian OT map overwrites scenario moments with the rolling historical moments, so the scenario-based MVO effectively reduces to plug-in MVO on the rolling window. This confirms that the added value of the generative model in the rolling setting depends on providing information beyond what the rolling historical moments capture.

The fixed-scenario setting, by contrast, allows the score model's structural knowledge of the return distribution (learned from seven years of training data) to persist across the test period, enabling the fine-tuned model to maintain a differentiated portfolio allocation.

### 5.5 Sinkhorn OT Diagnostic

Applied to a 500-sample subset, Sinkhorn OT (regularization epsilon = 0.005) reduces the Bures-Wasserstein distance from 0.030 to 0.020. The reduction is smaller than full Gaussian OT because the subset is small relative to the dimensionality of the problem. Full-set Sinkhorn would require extending the barycentric map to out-of-sample points via nearest-neighbor or kernel smoothing; this extension is not implemented. Sinkhorn results are reported as diagnostics only and Sinkhorn is not included as a portfolio strategy. This addresses the methodological error in earlier versions of the code that stored Gaussian OT scenarios under the Sinkhorn label.

---

## 6. Discussion

### 6.1 Realism-Performance Trade-Off

The main finding is that KL-penalized fine-tuning creates a meaningful realism-performance frontier. At eta* = 0.5, the fine-tuned model achieves the best risk-adjusted performance in the fixed-scenario setting. At smaller eta values, the model over-steers the distribution and degrades performance, likely because the control becomes too large relative to the base model drift, creating out-of-distribution scenarios. At larger eta values, the control is insufficient to move the portfolio allocation materially beyond the base diffusion strategy.

The optimal eta* = 0.5 was selected on validation data, with the leakage audit confirming no test-period information entered the selection. This separation is methodologically important: a practitioner inspecting test results to choose eta would likely find a similar or better test Sharpe, but this would be an artifact of in-sample optimization of the test period.

### 6.2 Why the Fixed-Scenario Advantage Disappears in Rolling Mode

The rolling-diffusion collapse to Markowitz-level performance illuminates a deeper point about what the score model contributes. In the fixed-scenario setting, the model's knowledge of the full 2014-2020 training distribution persists and informs the portfolio allocation throughout 2022-2024, even as market conditions evolve. In the rolling setting, the moment recalibration at each rebalance date overwrites this structural knowledge with the most recent 252-day history.

This suggests that the value of diffusion scenario generation lies primarily in its representation of the unconditional distribution, particularly during low-data periods or when the rolling window is insufficient to estimate the full covariance matrix accurately. Future work could explore conditioning the score model on recent market regimes to preserve structural knowledge while allowing adaptation.

### 6.3 Limitations and Future Directions

**Score model expressiveness.** The MLP score network with 256 hidden units is a simple architecture. More expressive models such as transformer-based score functions or flow matching (Lipman et al., 2023) may better capture tail dependencies and multi-modal market regimes.

**Gaussian OT scope.** The Bures-Wasserstein calibration corrects first and second moments only. Non-Gaussian features relevant for tail risk management, including co-crash risk and left tail dependence, require nonlinear OT methods. Full Sinkhorn OT or neural OT (Bunne et al., 2022) would address higher-order distributional structure at substantially greater computational cost.

**Differentiable reward approximation.** The soft-Markowitz surrogate replaces the constrained MVO with a softmax projection, which can assign weights that differ from exact long-only MVO. A more accurate differentiable approximation could use differentiable quadratic programming (Agrawal et al., 2019) or a REINFORCE-style estimator with a stop-gradient portfolio optimizer.

**Bootstrap CI width.** Simple i.i.d. bootstrap underestimates confidence interval width for autocorrelated returns. Block bootstrap (Politis & Romano, 1994) or the stationary bootstrap (Politis & Romano, 1994) would give more reliable intervals. At the three-year test horizon, even block-bootstrap intervals would span more than one Sharpe unit, confirming that results must be treated as exploratory.

**Computational scalability.** The eta sweep and fine-tuning are computationally feasible at ten assets but may require approximations at larger scales. Mini-batch Sinkhorn (Fatras et al., 2021) or fast Gaussian OT variants would enable scaling to larger asset universes.

---

## 7. Conclusion

This paper demonstrates that KL-penalized reverse diffusion fine-tuning can shape a score-based generative model toward portfolio-useful scenarios without completely sacrificing distributional realism. In the fixed-scenario setting, the fine-tuned strategy with validation-selected eta = 0.5 achieves a Sharpe ratio of 0.70 and a maximum drawdown of 14.88%, compared to 0.515 Sharpe and 17.09% drawdown for rolling plug-in Markowitz, over the 2022-2024 test period.

Three methodological controls distinguish this work from an exploratory baseline. First, a strict three-way data split with a formal leakage audit prevents test-period information from contaminating model selection or hyperparameter tuning. Second, Gaussian OT and Sinkhorn OT are reported separately and honestly: Gaussian OT as a full first-two-moments calibration, Sinkhorn OT as a subset-only diagnostic. Third, wide confidence intervals are reported without claiming statistical significance, accurately reflecting the limits of a three-year test window.

The rolling backtest result, where diffusion strategies collapse to Markowitz-level performance under rolling moment recalibration, reveals a fundamental insight: the value of diffusion scenario generation depends on the persistence of structural knowledge beyond the rolling window. This motivates future work on regime-conditioned or factor-conditioned generative models that can maintain distributional structure while adapting to recent market conditions.

---

## References

Agrawal, A., Amos, B., Barratt, S., Boyd, S., Diamond, S., & Kolter, J. Z. (2019). Differentiable convex optimization layers. *Advances in Neural Information Processing Systems, 32*, 9562-9574.

Anderson, B. D. O. (1982). Reverse-time diffusion equation models. *Stochastic Processes and Their Applications, 12*(3), 313-326. https://doi.org/10.1016/0304-4149(82)90051-5

Anderson, B. D. O., & Moore, J. B. (1990). *Optimal control: Linear quadratic methods*. Prentice-Hall.

Backhoff-Veraguas, J., Bartl, D., Beiglbock, M., & Eder, M. (2020). Adapted Wasserstein distances and stability in mathematical finance. *Finance and Stochastics, 24*, 601-632. https://doi.org/10.1007/s00780-020-00426-3

Bhatia, R., Jain, T., & Lim, Y. (2019). On the Bures-Wasserstein distance between positive definite matrices. *Expositiones Mathematicae, 37*(2), 165-191. https://doi.org/10.1016/j.exmath.2018.01.002

Blanchet, J., Chen, L., & Zhou, X. Y. (2022). Distributionally robust mean-variance portfolio selection with Wasserstein distances. *Management Science, 68*(9), 6382-6410. https://doi.org/10.1287/mnsc.2021.4155

Brenier, Y. (1991). Polar factorization and monotone rearrangement of vector-valued functions. *Communications on Pure and Applied Mathematics, 44*(4), 375-417. https://doi.org/10.1002/cpa.3160440402

Bunne, C., Krause, A., & Cuturi, M. (2022). Supervised training of conditional Monge maps. *Advances in Neural Information Processing Systems, 35*, 6859-6872.

Bures, D. (1969). An extension of Kakutani's theorem on infinite product measures to the tensor product of semifinite W*-algebras. *Transactions of the American Mathematical Society, 135*, 199-212. https://doi.org/10.2307/1995012

Cont, R. (2001). Empirical properties of asset returns: Stylized facts and statistical issues. *Quantitative Finance, 1*(2), 223-236. https://doi.org/10.1080/713665670

De Bortoli, V., Thornton, J., Heng, J., & Doucet, A. (2021). Diffusion Schrodinger bridge with applications to score-based generative modeling. *Advances in Neural Information Processing Systems, 34*, 17695-17709.

Fatras, K., Sejourne, T., Flamary, R., & Courty, N. (2021). Unbalanced minibatch optimal transport; applications to domain adaptation. *Proceedings of the 38th International Conference on Machine Learning*, 3186-3197.

Fleming, W. H., & Soner, H. M. (1993). *Controlled Markov processes and viscosity solutions*. Springer.

Han, J., Razaviyayn, M., & Xu, R. (2025). Score-based generative models with diffusion-driven reward fine-tuning. *Proceedings of the 42nd International Conference on Machine Learning.*

Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems, 33*, 6840-6851.

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *Proceedings of the 3rd International Conference on Learning Representations (ICLR 2015).*

Levine, S. (2018). Reinforcement learning and control as probabilistic inference: Tutorial and review. *arXiv preprint arXiv:1805.00909.*

Lim, H., Udell, M., & Klusowski, J. (2023). Generating realistic financial time series with diffusion models. *arXiv preprint arXiv:2309.12449.*

Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow matching for generative modeling. *Proceedings of the 11th International Conference on Learning Representations (ICLR 2023).*

Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. *Proceedings of the 5th International Conference on Learning Representations (ICLR 2017).*

McNeil, A. J., Frey, R., & Embrechts, P. (2015). *Quantitative risk management: Concepts, techniques and tools* (rev. ed.). Princeton University Press.

Mohajerin Esfahani, P., & Kuhn, D. (2018). Data-driven distributionally robust optimization using the Wasserstein metric. *Mathematical Programming, 171*, 115-166. https://doi.org/10.1007/s10107-017-1172-1

Oksendal, B. (2003). *Stochastic differential equations: An introduction with applications* (6th ed.). Springer. https://doi.org/10.1007/978-3-642-14394-6

Pelger, M., & Zou, R. (2023). Asset pricing with panel momentum. *Review of Financial Studies, 36*(11), 4237-4291. https://doi.org/10.1093/rfs/hhad017

Peyre, G., & Cuturi, M. (2019). Computational optimal transport. *Foundations and Trends in Machine Learning, 11*(5-6), 355-607. https://doi.org/10.1561/2200000073

Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association, 89*(428), 1303-1313. https://doi.org/10.1080/01621459.1994.10476870

Ranaraja, R. (2019). *yfinance: Yahoo! Finance market data downloader* [Software]. https://github.com/ranaroussi/yfinance

Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). Trust region policy optimization. *Proceedings of the 32nd International Conference on Machine Learning*, 1889-1897.

Song, Y., & Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. *Advances in Neural Information Processing Systems, 32*, 11895-11907.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-based generative modeling through stochastic differential equations. *Proceedings of the 9th International Conference on Learning Representations (ICLR 2021).*

Vuletich, H. A., & Cucuringu, M. (2024). Generative modelling of multivariate time-series with the score-based diffusion framework. *arXiv preprint arXiv:2402.09573.*

Wasserstein, L. N. (1969). Markov processes over denumerable products of spaces describing large systems of automata (in Russian). *Problemy Peredachi Informatsii, 5*(3), 64-72.

Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., & Irving, G. (2019). Fine-tuning language models from human preferences. *arXiv preprint arXiv:1909.08593.*

---

*Reproducibility note: All code, data splits, and model checkpoints are fully reproducible using the master pipeline runner at run_project.py. The fast smoke test (all stages, reduced epochs/samples) completes in under 30 minutes on Apple Silicon hardware and passes the leakage audit with zero failures. Full production run uses 2,000 training epochs and 10,000 scenarios. Config: configs/default.yaml. Seeds are fixed at 42 for data and model training, 0 for fine-tuning.*
