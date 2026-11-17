#!/usr/bin/env python
"""End-to-end UQ experiment on synthetic data (thesis Sections 4 and 6.1).

1. simulate a Heston or Bates price path with known parameters,
2. estimate the posterior with the particle-filter / Gibbs sampler,
3. propagate the posterior into a European call price (exact Monte Carlo over
   posterior draws and a polynomial-chaos surrogate) and compare with the
   price under the true parameters.

Example::

    python scripts/run_synthetic.py --model bates --preset thesis --years 3 --iter 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from svuq import BatesParams, HestonParams, Priors, estimate, simulate  # noqa: E402
from svuq.diagnostics import mse  # noqa: E402
from svuq.uq import make_pricer, pce_prices, posterior_mc_prices, posterior_table  # noqa: E402

PRESETS = {
    # Table 2 / Section 6.1 of the thesis (very small vol-of-vol)
    "thesis": dict(mu=0.1, kappa=1.1, theta=0.04, sigma=0.01, rho=-0.5, lam=3.0, mu_j=-0.3, sigma_j=0.2),
    # equity-like values (vol-of-vol of the size usually found in option calibrations)
    "realistic": dict(mu=0.08, kappa=3.0, theta=0.04, sigma=0.4, rho=-0.7, lam=1.0, mu_j=-0.1, sigma_j=0.1),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["heston", "bates"], default="heston")
    ap.add_argument("--preset", choices=PRESETS, default="realistic")
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--iter", type=int, default=2000, help="Gibbs sweeps")
    ap.add_argument("--burn-in", type=int, default=None)
    ap.add_argument("--particles", type=int, default=1000)
    ap.add_argument("--path", choices=["ffbs", "mean"], default="ffbs")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--strike", type=float, default=110.0)
    ap.add_argument("--maturity", type=float, default=1 / 12, help="option maturity in years")
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--pce-order", type=int, default=3)
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)

    dt = 1 / 252
    n_steps = int(round(a.years * 252))
    preset = PRESETS[a.preset]
    if a.model == "bates":
        truth = BatesParams(**preset)
    else:
        truth = HestonParams(**{k: preset[k] for k in ("mu", "kappa", "theta", "sigma", "rho")})
    sim = simulate(truth, S0=100.0, v0=truth.theta, T=n_steps * dt, n_steps=n_steps, seed=a.seed)

    # weakly informative priors centred away from the truth
    priors = Priors.from_beliefs(
        dt, mu=(0.05, 0.5), kappa=(2.0, 2.0), theta=(0.05, 0.05),
        sigma=max(2 * truth.sigma, 0.05), rho=(-0.3, 0.4),
        lam=(1.0, 1.0), mu_j=(-0.05, 0.3), sigma_j=0.1,
    )
    out = Path(a.out) / f"synthetic_{a.model}_{a.preset}_{a.years:g}y_seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"simulated {n_steps} steps, {int(sim.jumps.sum())} jumps -> {out}")

    res = estimate(sim.prices, dt, a.model, priors, n_iter=a.iter, burn_in=a.burn_in,
                   n_particles=a.particles, path=a.path, seed=a.seed + 1000, progress=True)
    summary = res.summary(truth)
    print(summary.round(4).to_string())
    res.save(str(out / "posterior"))
    pd.DataFrame({"price": sim.prices, "true_variance": sim.variance}).to_csv(out / "data.csv", index_label="k")
    vol_mse = mse(res.variance_path, sim.variance)
    print(f"latent variance MSE = {vol_mse:.3e} (variance of true path {sim.variance.var():.3e})")

    # --- option price uncertainty --------------------------------------------
    S0, K, T, r = sim.prices[-1], a.strike * sim.prices[-1] / 100.0, a.maturity, a.rate
    draws = posterior_table(res)
    true_kw = {**truth.to_dict(), "v0": sim.variance[-1]}
    true_price = float(make_pricer(a.model, S0, K, T, r)(**true_kw))
    mc = posterior_mc_prices(draws, a.model, S0, K, T, r, n_draws=min(1000, len(draws)))
    pce = pce_prices(draws, a.model, S0, K, T, r, order=a.pce_order)
    report = {
        "S0": S0, "K": K, "T": T, "r": r, "true_price": true_price,
        "mc_mean": float(mc.prices.mean()), "mc_sd": float(mc.prices.std(ddof=1)),
        "mc_q2.5": float(np.percentile(mc.prices, 2.5)), "mc_q97.5": float(np.percentile(mc.prices, 97.5)),
        "pce_mean": float(pce.mean[0]), "pce_sd": float(pce.std[0]), "pce_evals": pce.n_evals,
        "pce_loo_error": float(pce.loo_error[0]), "latent_variance_mse": vol_mse,
        "sobol_total": pce.sobol_total.iloc[:, 0].to_dict(),
        "settings": {k: v for k, v in res.settings.items() if k != "priors"},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: report[k] for k in ("true_price", "mc_mean", "mc_sd", "pce_mean", "pce_sd")}, indent=2))

    if not a.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        from svuq import plotting
        plotting.plot_posteriors(res, truth).savefig(out / "posteriors.png", dpi=150)
        plotting.plot_traces(res).savefig(out / "traces.png", dpi=120)
        plotting.plot_variance(res, sim.variance).savefig(out / "variance.png", dpi=150)
        plotting.plot_price_distribution(mc.prices, true_price).savefig(out / "option_price.png", dpi=150)


if __name__ == "__main__":
    main()
