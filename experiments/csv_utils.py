"""
Shared CSV loading utility for robust date parsing across pandas versions and yfinance formats.
"""
import pandas as pd


def load_returns_csv(path):
    """
    Load a returns CSV, robust to:
    - pandas 2.x dropping parse_dates=True auto-inference
    - yfinance 0.2.x writing a 'Ticker' column-name header row
    - any stray non-numeric / non-date rows
    """
    df = pd.read_csv(path, index_col=0)
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    return df
