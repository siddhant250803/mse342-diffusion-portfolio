Subject: MS&E 342 project proposal revision

Hi Professor Xu,

I wanted to share a revised version of my MS&E 342 project proposal, "Reward-Conditioned Reverse Diffusion for Portfolio Optimization."

The core idea is to use the KL-penalized stochastic control framework for diffusion fine-tuning to make a financial scenario generator responsive to a downstream portfolio objective. I am combining three pieces: a VP-SDE score model for sector ETF returns, an HJB/control formulation for reward-conditioned reverse drift fine-tuning, and an optimal transport calibration layer to address the variance/covariance underestimation I observed in the base generator.

After reviewing the preliminary results, I made the empirical plan more conservative and rigorous. In particular, I now plan to:

- use a 2014-2020 / 2021 / 2022-2024 train-validation-test split, so eta is selected on validation rather than on the test period;
- distinguish Gaussian OT, true Sinkhorn barycentric OT, and OT-augmented score training instead of overstating the current Gaussian calibration result;
- add turnover and portfolio concentration metrics in addition to Sharpe, CVaR, and drawdown;
- compare fixed-scenario diffusion backtests with a rolling protocol closer to the Markowitz and robust-MVO baselines;
- keep the quadratic reward as the analytically tractable linear-Gaussian theory case, while using a more decision-aligned portfolio/CVaR reward in the final empirical section.

The preliminary results are suggestive rather than statistically decisive: the base diffusion model underestimates volatility and creates concentrated portfolios, Gaussian OT restores the first two moments and improves diversification, and KL fine-tuning changes the Sharpe/drawdown trade-off. I will frame the final result as an interpretable realism-performance frontier rather than as a claim of significant outperformance.

I would appreciate any feedback on whether this is the right level of scope for the final project, especially on the reward formulation and whether the OT calibration layer is a useful addition or should be kept secondary to the stochastic control/fine-tuning contribution.

Best,
Siddhant
