"""Loading price series from CSV / Excel or (optionally) Yahoo Finance."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def load_prices(path: str, column: Optional[str] = None) -> np.ndarray:
    """Read a price column from ``.csv``/``.xlsx``.

    If ``column`` is not given, the first of ``Price``, ``price``, ``Adj Close``,
    ``Close`` that exists is used; for headerless numeric CSVs the first column
    that is not a time grid is used.
    """
    p = Path(path)
    df = pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p)
    if column is None:
        for c in ("Price", "price", "Adj Close", "Close", "close"):
            if c in df.columns:
                column = c
                break
    if column is None:
        num = df.select_dtypes("number")
        cols = [c for c in num.columns if str(c).lower() not in ("t", "time")]
        column = cols[0]
    s = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)
    if np.any(s <= 0):
        raise ValueError("prices must be positive")
    return s


def download_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Adjusted close prices from Yahoo Finance (requires ``yfinance``)."""
    import yfinance as yf

    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError(f"no data returned for {ticker}")
    s = data["Adj Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.dropna()
