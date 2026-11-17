#!/usr/bin/env python
"""Estimate Heston and/or Bates on a historical price series and price options
with parameter uncertainty (thesis Section 6.2).

Prices come from a CSV/XLSX file (``--data``) or Yahoo Finance (``--ticker``,
needs ``yfinance``).  If ``--options`` is given (CSV with columns ``strike``,
``expiry_days``, ``quoted`` and optionally ``S0``), model prices with 95%
credible bands and model-implied volatility smiles are produced for every
quote.  The thesis' 11-Jun-1997 S&P 100 quotes are in
``data/oex_options_1997-06-11.csv``.

Example (thesis setup, requires internet)::

    python scripts/run_real_data.py --ticker ^OEX --start 1994-06-11 --end 1997-06-12 \
        --options data/oex_options_1997-06-11.csv --rate 0.05 --dividend 0.017
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from svuq import Priors, estimate  # noqa: E402
from svuq.data import download_prices, load_prices  # noqa: E402
from svuq.pricing import implied_volatility  # noqa: E402
from svuq.uq import posterior_mc_prices, posterior_table  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="CSV/XLSX file with a price column")
    src.add_argument("--ticker", help="Yahoo Finance ticker, e.g. ^OEX")
    ap.add_argument("--column", default=None)
    ap.add_argument("--start", default="1994-06-11")
    ap.add_argument("--end", default="1997-06-12")
    ap.add_argument("--models", nargs="+", choices=["heston", "bates"], default=["heston", "bates"])
    ap.add_argument("--iter", type=int, default=2000)
    ap.add_argument("--particles", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--options", help="CSV of option quotes (strike, expiry_days, quoted[, S0])")
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--dividend", type=float, default=0.0, help="continuous dividend yield q")
    ap.add_argument("--vol-risk-premium", type=float, default=0.0,
                    help="eps in kappa_Q = kappa + eps, theta_Q = kappa theta / (kappa + eps)")
    ap.add_argument("--draws", type=int, default=300, help="posterior draws used for pricing")
    ap.add_argument("--out", default="results/real")
    a = ap.parse_args(argv)

    if a.data:
        prices = load_prices(a.data, a.column)
        name = Path(a.data).stem
    else:
        s = download_prices(a.ticker, a.start, a.end)
        prices = s.to_numpy()
        name = a.ticker.strip("^")
    out = Path(a.out) / name
    out.mkdir(parents=True, exist_ok=True)
    dt = 1 / 252
    logret = np.diff(np.log(prices))
    rv = logret.var() / dt
    print(f"{prices.size} prices, annualised realised variance {rv:.4f} (vol {np.sqrt(rv):.3f})")

    # weakly informative priors centred on the realised variance of the sample
    priors = Priors.from_beliefs(
        dt, mu=(0.08, 0.5), kappa=(2.0, 2.0), theta=(rv, rv), sigma=0.3, rho=(-0.5, 0.4),
        lam=(1.0, 1.0), mu_j=(-0.05, 0.1), sigma_j=0.05, v0=(rv, rv),
    )
    quotes = pd.read_csv(a.options) if a.options else None
    report = {}
    for model in a.models:
        print(f"\n=== {model} ===")
        res = estimate(prices, dt, model, priors, n_iter=a.iter, n_particles=a.particles,
                       seed=a.seed, progress=True)
        summ = res.summary()
        print(summ.round(4).to_string())
        res.save(str(out / model))
        report[model] = {"summary": summ.to_dict(orient="index"), "v_final": res.v_final,
                         "runtime_s": res.settings["runtime_s"]}
        if quotes is None:
            continue
        if "S0" not in quotes:
            quotes["S0"] = prices[-1]
        draws = posterior_table(res)
        rows = []
        for (days, S0), grp in quotes.groupby(["expiry_days", "S0"]):
            T = days / 365.0
            K = grp["strike"].to_numpy(float)
            pd_ = posterior_mc_prices(draws, model, S0, K, T, a.rate, n_draws=a.draws,
                                      vol_risk_premium=a.vol_risk_premium, q=a.dividend)
            summ_p = pd_.summary()
            for i, (_, q) in enumerate(grp.iterrows()):
                m = summ_p["mean"].iloc[i]
                rows.append({"expiry_days": days, "strike": K[i], "quoted": q["quoted"], "model_mean": m,
                             "model_q2.5": summ_p["q2.5"].iloc[i], "model_q97.5": summ_p["q97.5"].iloc[i],
                             "iv_quoted": implied_volatility(q["quoted"], S0, K[i], T, a.rate, a.dividend),
                             "iv_model": implied_volatility(m, S0, K[i], T, a.rate, a.dividend)})
        tab = pd.DataFrame(rows)
        tab.to_csv(out / f"{model}_option_prices.csv", index=False)
        print(tab.round(3).to_string(index=False))
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
