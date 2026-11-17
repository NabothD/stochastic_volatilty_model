"""Loading price series from CSV / Excel or (optionally) Yahoo Finance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
_PRICE_COLUMNS = ("Price", "price", "Adj Close", "Close", "close")


def load_prices(path: str | Path, column: str | None = None) -> np.ndarray:
    """Read a positive price series from a ``.csv`` or ``.xlsx`` file.

    Parameters
    ----------
    path : file path.
    column : column to read; default is the first of ``Price``, ``Adj Close``,
        ``Close`` found, else the first numeric column that is not a time grid.

    Returns
    -------
    Prices, shape (n,).
    """
    p = Path(path)
    df = pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p)
    if column is None:
        column = next((c for c in _PRICE_COLUMNS if c in df.columns), None)
    if column is None:
        numeric = df.select_dtypes("number").columns
        column = next(c for c in numeric if str(c).lower() not in ("t", "time"))
    prices = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)
    if np.any(prices <= 0):
        raise ValueError("prices must be positive")
    return prices


def download_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Download adjusted close prices from Yahoo Finance (requires ``yfinance``).

    Parameters
    ----------
    ticker : Yahoo ticker, e.g. ``"^OEX"``.
    start, end : ISO dates.

    Returns
    -------
    Price series indexed by date.
    """
    import yfinance as yf

    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError(f"no data returned for {ticker}")
    prices = data["Adj Close"]
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    return prices.dropna()
