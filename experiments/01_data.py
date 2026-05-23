"""
Pull S&P 500 sector ETF daily returns (2014-2024) and save explicit
train / validation / test splits.

Split:
  Train:      2014-01-01 through 2020-12-31
  Validation: 2021-01-01 through 2021-12-31
  Test:       2022-01-01 through 2024-12-31

Old files (data/train.csv, data/test.csv) are kept for backward compat
but final scripts use the new explicit-split files.

LEAKAGE NOTE: no scaler, model, or hyperparameter is fitted here;
this script only saves raw log-returns.
"""
import yfinance as yf
import numpy as np
import pandas as pd
import json
import os

TICKERS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE"]
START, END = "2014-01-01", "2024-12-31"

TRAIN_START, TRAIN_END = "2014-01-01", "2020-12-31"
VAL_START,   VAL_END   = "2021-01-01", "2021-12-31"
TEST_START,  TEST_END  = "2022-01-01", "2024-12-31"

os.makedirs("data", exist_ok=True)

_dl = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
# Handle both old single-level and new multi-level yfinance column formats
if isinstance(_dl.columns, pd.MultiIndex):
    raw = _dl["Close"]
else:
    raw = _dl["Close"]
raw.columns.name = None   # remove yfinance 'Ticker' label so to_csv() writes a clean single-header CSV
raw.index.name = "Date"
raw = raw.dropna()

returns = np.log(raw / raw.shift(1)).dropna()
returns.to_csv("data/returns.csv")
returns.to_csv("data/all_returns.csv")   # explicit alias

train = returns[(returns.index >= TRAIN_START) & (returns.index <= TRAIN_END)]
val   = returns[(returns.index >= VAL_START)   & (returns.index <= VAL_END)]
test  = returns[(returns.index >= TEST_START)  & (returns.index <= TEST_END)]

train.to_csv("data/train_2014_2020.csv")
val.to_csv("data/val_2021.csv")
test.to_csv("data/test_2022_2024.csv")

# Backward-compatible files (old split: train=2014-2021, test=2022-2024)
old_train = returns[returns.index < TEST_START]
old_train.to_csv("data/train.csv")
test.to_csv("data/test.csv")

# Save split metadata
split_meta = {
    "train_start": TRAIN_START, "train_end": TRAIN_END,
    "val_start":   VAL_START,   "val_end":   VAL_END,
    "test_start":  TEST_START,  "test_end":  TEST_END,
    "tickers": TICKERS,
    "total_days": int(len(returns)),
    "train_days": int(len(train)),
    "val_days":   int(len(val)),
    "test_days":  int(len(test)),
    "data_used_for_fit": "none — raw returns only",
    "data_used_for_selection": "none",
    "data_used_for_evaluation": "none",
}
with open("data/split_metadata.json", "w") as f:
    json.dump(split_meta, f, indent=2)

print(f"Total days:      {len(returns)}")
print(f"Train days:      {len(train)}  ({TRAIN_START} to {TRAIN_END})")
print(f"Validation days: {len(val)}   ({VAL_START} to {VAL_END})")
print(f"Test days:       {len(test)}  ({TEST_START} to {TEST_END})")
print(f"Assets:          {list(returns.columns)}")
print(f"\nTrain stats (2014-2020):")
print(train.describe().loc[["mean", "std", "min", "max"]].round(5))
print(f"\nVal stats (2021):")
print(val.describe().loc[["mean", "std", "min", "max"]].round(5))
print(f"\nSaved: data/all_returns.csv, data/train_2014_2020.csv, "
      f"data/val_2021.csv, data/test_2022_2024.csv")
print(f"Saved: data/split_metadata.json")
