# MS&E 342 Project: Final Summary

Generated: 2026-05-22 23:48
Leakage audit status: **PASSED**

## Dataset Split

| Split | Start | End |
|-------|-------|-----|
| Training | 2014-01-01 | 2020-12-31 |
| Validation | 2021-01-01 | 2021-12-31 |
| Test | 2022-01-01 | 2024-12-31 |

Assets: XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE (10 S&P 500 sector ETFs)

## Leakage Controls

- Scaler (mean, std) fitted on training data only (2014-2020).
- Score model trained on 2014-2020 data only.
- OT calibration target moments computed from training data only.
- Eta selected on 2021 validation Sharpe. Test data not used for selection.
- Test data evaluated once for final metrics.
- Sinkhorn OT reported as subset-only diagnostic, not a full strategy.

## Methods Compared

- **Equal-Weight**: uniform 1/d weights, monthly rebalanced.
- **Markowitz**: plug-in MVO on rolling 252-day window.
- **Wasserstein-Robust**: robust MVO with W2 ball (Blanchet et al. 2022).
- **Diffusion-Base**: MVO using VP-SDE generated scenarios (no calibration).
- **Diffusion+GaussOT**: Gaussian OT calibration applied to generated scenarios.
  Corrects first two moments under Gaussian approximation only.
- **Diffusion+FT(eta*)**: fine-tuned with KL-penalized portfolio reward.
  Eta selected on 2021 validation Sharpe.

## Selected Eta

- Model `base`: eta* = 0.5 (val Sharpe = 2.301)

## Final Metrics (Fixed Backtest, 2022-2024)

Sharpe CI uses simple i.i.d. bootstrap (2000 draws). With a ~3-year test window, CI width is approximately 0.6-0.8 Sharpe units. Results should be interpreted as exploratory, not statistically decisive.

```
                           Ann. Return Ann. Volatility  Sharpe Sharpe CI Lo Sharpe CI Hi CVaR(95%) Max Drawdown Avg Turnover Avg HHI
Equal-Weight                     5.59%          15.41%  0.3629      -0.7591       1.4633    0.0226      -18.41%       0.0000  0.1000
Markowitz                        9.03%          17.54%  0.5151      -0.5976       1.5661    0.0264      -17.09%       0.6173  0.5408
Wasserstein-Robust               5.75%          15.12%  0.3802      -0.7398       1.4828    0.0222      -17.92%       0.0395  0.1027
Diffusion-Base                   8.69%          22.53%  0.3856      -0.7214       1.4926    0.0319      -30.65%       0.0000  0.6291
Diffusion+GaussOT                6.45%          16.56%  0.3897      -0.6949       1.4924    0.0242      -22.98%       0.0000  0.2599
Diffusion+FT(eta=0.5,base)      11.31%          16.16%  0.6997      -0.3826       1.7609    0.0241      -14.88%       0.0000  0.3161
```

## Final Metrics (Rolling Backtest, 2022-2024)

Sharpe CI uses simple i.i.d. bootstrap (2000 draws). With a ~3-year test window, CI width is approximately 0.6-0.8 Sharpe units. Results should be interpreted as exploratory, not statistically decisive.

```
                           Ann. Return Ann. Volatility  Sharpe Sharpe CI Lo Sharpe CI Hi CVaR(95%) Max Drawdown Avg Turnover Avg HHI
Equal-Weight                     5.59%          15.41%  0.3629      -0.7591       1.4633    0.0226      -18.41%       0.0000  0.1000
Markowitz                        9.03%          17.54%  0.5151      -0.5976       1.5661    0.0264      -17.09%       0.6173  0.5408
Wasserstein-Robust               5.75%          15.12%  0.3802      -0.7398       1.4828    0.0222      -17.92%       0.0395  0.1027
Diffusion-Base                   8.89%          17.41%  0.5105      -0.6090       1.5708    0.0263      -17.09%       0.5846  0.4962
Diffusion+GaussOT                8.87%          17.40%  0.5096      -0.6106       1.5699    0.0263      -17.09%       0.5855  0.4975
Diffusion+FT(eta=0.5,base)       8.88%          17.40%  0.5101      -0.6095       1.5699    0.0263      -17.09%       0.5847  0.4961
```

## Key Diagnostics

- Bures-Wasserstein distance (generated vs. train): 0.0303
- Covariance Frobenius error: 0.0014
- W2 before Gaussian OT: 0.0303
- W2 after  Gaussian OT: 0.0003

## Limitations

- Score model is a simple MLP; more expressive architectures may improve realism.
- Gaussian OT corrects first two moments only. Non-Gaussian tail behavior is not corrected by this calibration.
- Sinkhorn OT applied to a subset only. Full-set extension was not implemented.
- The portfolio reward uses a differentiable soft-Markowitz approximation, not a hard constrained optimizer. Weights may differ from exact MVO.
- Bootstrap Sharpe CIs use simple i.i.d. bootstrap. With autocorrelated returns, block bootstrap would give more reliable intervals.
- A 3-year test window (~756 trading days) provides limited statistical power. Pairwise Sharpe differences are not individually statistically significant at conventional levels.
- Rolling backtests use Gaussian OT recalibration, not full model retraining.

## Reproducibility

Full pipeline: `python run_project.py all`
Fast smoke test: `python run_project.py all --fast`
Config: `configs/default.yaml`
Seeds: see config `seeds` section.