# One-Page Project Proposal: MS&E 342

**Title:** Reward-Conditioned Reverse Diffusion for Portfolio Optimization  
**Student:** Siddhant Sukhani  
**Instructor:** Prof. Renyuan Xu  
**Final report due:** June 12, 2026

## Objective

This project studies whether a score-based diffusion model for financial return scenarios can be fine-tuned through KL-penalized stochastic control so that generated scenarios remain distributionally realistic while becoming more useful for downstream portfolio decisions. Existing diffusion-based finance papers typically train a generative model to match historical returns and then pass generated scenarios to a portfolio optimizer. The portfolio objective does not feed back into the generator. This project addresses that decoupling by adding a controlled perturbation to the reverse SDE drift and optimizing a terminal portfolio reward subject to a KL penalty relative to the base model.

## Method

The base model is a variance-preserving SDE trained by denoising score matching on daily log returns for 10 S&P 500 sector ETFs. Generation follows the reverse SDE
$$
dY_t = \left[-\frac{\beta(T-t)}{2}Y_t+\beta(T-t)s_\theta(Y_t,T-t)\right]dt+\sqrt{\beta(T-t)}d\widetilde W_t.
$$
Fine-tuning introduces a control $u_\phi(Y_t,t)$ and solves
$$
\max_{\phi}\ \mathbb{E}_{p_\phi}[U(Y_T)]-\eta\,\mathrm{KL}(p_\phi\|p_\theta),
$$
where Girsanov's theorem converts the KL term into a quadratic control cost. The associated HJB equation is
$$
\partial_t V+\mathcal{L}^{s_\theta}V+\frac{\beta(T-t)}{2\eta}\|\nabla_y V\|^2=0,\qquad V(T,y)=U(y).
$$

## Flowchart

```text
Historical ETF Returns
        |
        v
VP-SDE Score Model
        |
        v
Base Return Scenarios -----> OT Calibration
        |                         |
        v                         v
KL-Control Fine-Tuning ----> Calibrated/Fine-Tuned Scenarios
        |                         |
        v                         v
Long-Only MVO Backtest ---> Sharpe, CVaR, Drawdown, Turnover
```

## Empirical Design

The data will be split into 2014-2020 for score-model training, 2021 for validation and eta selection, and 2022-2024 for final testing. Strategies include equal weight, rolling Markowitz, Wasserstein-robust MVO, base diffusion scenarios, Gaussian OT calibrated scenarios, and KL-fine-tuned diffusion scenarios. Metrics include Sharpe ratio, CVaR(95%), maximum drawdown, turnover, and portfolio concentration.

## Contributions

The theoretical contribution is a closed-form linear-Gaussian benchmark showing how the optimal control scales as $u^*(t,y)=\beta(T-t)\nabla V(t,y)/\eta$. The empirical contribution is an eta frontier that quantifies the trade-off between distributional realism and decision usefulness. Preliminary results show that the base diffusion model underestimates volatility and produces overly concentrated portfolios, while Gaussian OT restores first and second moments and recovers diversified weights. The final project will present these results cautiously as an interpretable realism-performance frontier, not as a statistically decisive outperformance claim.
