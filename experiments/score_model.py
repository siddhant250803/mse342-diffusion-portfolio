# Importable shim — re-exports everything from the numbered training script.
import importlib, sys
_mod = importlib.util.spec_from_file_location(
    "score_model_impl",
    __file__.replace("score_model.py", "02_score_model.py")
)
_m = importlib.util.module_from_spec(_mod)
sys.modules["score_model_impl"] = _m
_mod.loader.exec_module(_m)

from score_model_impl import ScoreNet, get_alpha_sigma, BETA_MIN, BETA_MAX, train_score_model
