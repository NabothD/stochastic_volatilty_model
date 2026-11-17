#!/usr/bin/env python
r"""Fit a model to a price history and forecast volatility, optionally streaming new prices.

The last ``--holdout`` prices are fed one at a time through the online filter
(as they would arrive live), printing the now-cast after each, followed by
volatility forecasts for the requested horizons.

Example::

    python scripts/forecast_volatility.py --data data/raw/sp500_historical.xlsx --model heston \
        --holdout 10 --horizons 1 5 21 63
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from svuq import Estimator, Priors, SamplerConfig, VolatilityForecaster
from svuq.data import load_prices


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV/XLSX file with a price column")
    ap.add_argument("--column", default=None)
    ap.add_argument("--model", choices=["heston", "bates"], default="heston")
    ap.add_argument("--iter", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--particles", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=100, help="posterior draws in the forecaster")
    ap.add_argument("--holdout", type=int, default=10, help="prices streamed online at the end")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 21, 63])
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    """Fit, initialise the forecaster, stream the hold-out prices and print forecasts."""
    a = parse_args()
    prices = load_prices(a.data, a.column)
    history, stream = prices[:-a.holdout or None], prices[len(prices) - a.holdout:]
    dt = 1 / 252
    rv = np.diff(np.log(history)).var() / dt
    priors = Priors.from_beliefs(dt, theta=(rv, rv), v0=(rv, rv), lam=(1.0, 1.0),
                                 mu_j=(-0.05, 0.1), sigma_j=0.05)
    post = Estimator(a.model, priors, SamplerConfig(n_iter=a.iter, n_particles=a.particles)).fit(
        history, dt, n_chains=a.chains, seed=a.seed, progress=True)
    print(post.summary().round(4).to_string())

    forecaster = VolatilityForecaster(post, n_draws=a.draws, seed=a.seed).initialise(history)
    rows = []
    for price in stream:
        t0 = time.perf_counter()
        now = forecaster.update(price)
        rows.append({"price": price, "vol": now.vol, "vol_lo": now.vol_lo, "vol_hi": now.vol_hi,
                     "jump_prob": now.jump_prob, "update_ms": 1e3 * (time.perf_counter() - t0)})
    if rows:
        print("\nonline now-casts:\n", pd.DataFrame(rows).round(4).to_string(index=False))
    print("\nforecast (annualised vol averaged over the horizon):")
    print(forecaster.forecast(a.horizons).round(4).to_string())


if __name__ == "__main__":
    main()
