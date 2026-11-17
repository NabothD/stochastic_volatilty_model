#!/usr/bin/env python
r"""Estimate Heston and/or Bates on historical prices and price an option chain.

Prices come from a CSV/XLSX file (``--data``) or Yahoo Finance (``--ticker``,
requires ``yfinance``). With ``--options`` (CSV with ``strike``,
``expiry_days``, ``quoted`` and optionally ``S0``), model prices with 95%
credible bands and implied volatilities are produced for every quote. The
thesis' 11-Jun-1997 S&P 100 quotes are in ``data/oex_options_1997-06-11.csv``.

Example (thesis setup, needs internet)::

    python scripts/run_real_data.py --ticker ^OEX --start 1994-06-11 --end 1997-06-12 \
        --options data/oex_options_1997-06-11.csv --rate 0.05 --dividend 0.017
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from svuq import (
    Estimator,
    OptionSpec,
    PosteriorPricer,
    Priors,
    SamplerConfig,
    implied_volatility,
    posterior_draws,
)
from svuq.data import download_prices, load_prices


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="CSV/XLSX file with a price column")
    source.add_argument("--ticker", help="Yahoo Finance ticker, e.g. ^OEX")
    ap.add_argument("--column", default=None)
    ap.add_argument("--start", default="1994-06-11")
    ap.add_argument("--end", default="1997-06-12")
    ap.add_argument("--models", nargs="+", choices=["heston", "bates"], default=["heston", "bates"])
    ap.add_argument("--iter", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--particles", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--options", help="CSV of option quotes (strike, expiry_days, quoted[, S0])")
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--dividend", type=float, default=0.0, help="continuous dividend yield")
    ap.add_argument("--vol-risk-premium", type=float, default=0.0)
    ap.add_argument("--out", default="results/real")
    return ap.parse_args()


def price_quotes(post, model: str, quotes: pd.DataFrame, a: argparse.Namespace) -> pd.DataFrame:
    """Price every quoted option at all posterior draws and compare with the quote."""
    draws = posterior_draws(post, seed=a.seed)
    pricer = PosteriorPricer(model, vol_risk_premium=a.vol_risk_premium)
    rows = []
    for (days, S0), group in quotes.groupby(["expiry_days", "S0"]):
        T = days / 365.0
        option = OptionSpec(S0, group["strike"].to_numpy(float), T, a.rate, a.dividend)
        summary = pricer.price(draws, option).summary()
        for (_, quote), (_, model_row) in zip(group.iterrows(), summary.iterrows(), strict=True):
            K = model_row["strike"]
            rows.append({
                "expiry_days": days, "strike": K, "quoted": quote["quoted"],
                "model_mean": model_row["mean"], "model_q2.5": model_row["q2.5"],
                "model_q97.5": model_row["q97.5"],
                "iv_quoted": implied_volatility(quote["quoted"], S0, K, T, a.rate, a.dividend),
                "iv_model": implied_volatility(model_row["mean"], S0, K, T, a.rate, a.dividend),
            })
    return pd.DataFrame(rows)


def main() -> None:
    """Run the estimation (and option comparison) and write results to ``--out``."""
    a = parse_args()
    if a.data:
        prices, name = load_prices(a.data, a.column), Path(a.data).stem
    else:
        prices, name = download_prices(a.ticker, a.start, a.end).to_numpy(), a.ticker.strip("^")
    out = Path(a.out) / name
    out.mkdir(parents=True, exist_ok=True)
    dt = 1 / 252
    realised_var = np.diff(np.log(prices)).var() / dt
    print(f"{prices.size} prices, realised vol {np.sqrt(realised_var):.3f}")

    priors = Priors.from_beliefs(dt, mu=(0.08, 0.5), kappa=(2.0, 2.0),
                                 theta=(realised_var, realised_var), sigma=0.3, rho=(-0.5, 0.4),
                                 lam=(1.0, 1.0), mu_j=(-0.05, 0.1), sigma_j=0.05,
                                 v0=(realised_var, realised_var))
    config = SamplerConfig(n_iter=a.iter, n_particles=a.particles)
    quotes = pd.read_csv(a.options) if a.options else None
    if quotes is not None and "S0" not in quotes:
        quotes["S0"] = prices[-1]
    report = {}
    for model in a.models:
        print(f"\n=== {model} ===")
        post = Estimator(model, priors, config).fit(prices, dt, n_chains=a.chains, n_jobs=a.jobs,
                                                    seed=a.seed, progress=True)
        summary = post.summary()
        print(summary.round(4).to_string())
        post.save(str(out / model))
        report[model] = {"summary": summary.to_dict(orient="index"), "v_final": post.v_final,
                         "runtime_s": post.settings["runtime_s"]}
        if quotes is not None:
            table = price_quotes(post, model, quotes, a)
            table.to_csv(out / f"{model}_option_prices.csv", index=False)
            print(table.round(3).to_string(index=False))
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
