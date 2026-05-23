# Importable shim for 04_portfolio_baseline.py
import importlib.util, sys, os
_spec = importlib.util.spec_from_file_location(
    "portfolio_baseline_impl",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_portfolio_baseline.py")
)
_m = importlib.util.module_from_spec(_spec)
sys.modules["portfolio_baseline_impl"] = _m
_spec.loader.exec_module(_m)
from portfolio_baseline_impl import (
    mvo, compute_metrics, bootstrap_sharpe_ci, backtest,
    LAMBDA, robust_mvo, diffusion_mvo, gaussian_ot_apply,
)
