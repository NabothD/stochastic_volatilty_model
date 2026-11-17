#!/usr/bin/env python
"""End-to-end UQ experiment on synthetic data (thesis Sections 4 and 6.1).

1. Simulate a Heston or Bates path with known parameters.
2. Estimate the posterior with parallel particle-filter / Gibbs chains.
3. Propagate the posterior into a European call price (posterior Monte Carlo
   and a PCE surrogate) and compare with the price under the true parameters.

Example::

    python scripts/run_synthetic.py --model bates --preset thesis --iter 2000 --chains 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

from svuq import (
    BatesParams,
    Estimator,
    FourierPricer,
    HestonParams,
    OptionSpec,
    PCESurrogate,
    PosteriorPricer,
    Priors,
    SamplerConfig,
    plotting,
    posterior_draws,
    simulate,
)
from svuq.diagnostics import mse

PRESETS = {
    # thesis Table 2 / Section 6.1 (very small vol-of-vol)
    "thesis": dict(mu=0.1, kappa=1.1, theta=0.04, sigma=0.01, rho=-0.5, lam=3.0, mu_j=-0.3,
                   sigma_j=0.2),
    # equity-like values
    "realistic": dict(mu=0.08, kappa=3.0, theta=0.04, sigma=0.4, rho=-0.7, lam=1.0, mu_j=-0.1,
                      sigma_j=0.1),
}


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["heston", "bates"], default="heston")
    ap.add_argument("--preset", choices=PRESETS, default="realistic")
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--iter", type=int, default=2000, help="Gibbs sweeps per chain")
    ap.add_argument("--burn-in", type=int, default=None)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=None, help="worker processes (default: all cores)")
    ap.add_argument("--particles", type=int, default=1000)
    ap.add_argument("--path", choices=["ffbs", "mean"], default="ffbs")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--moneyness", type=float, default=1.10, help="strike / spot")
    ap.add_argument("--maturity", type=float, default=1 / 12, help="option maturity in years")
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--pce-order", type=int, default=3)
    ap.add_argument("--out", default="results")
    return ap.parse_args()


def main() -> None:
    """Run the experiment and write results to ``--out``."""
    matplotlib.use("Agg")
    a = parse_args()
    dt = 1 / 252
    n_steps = int(round(a.years * 252))
    preset = PRESETS[a.preset]
    truth = BatesParams(**preset) if a.model == "bates" else HestonParams(
        **{k: preset[k] for k in HestonParams.__dataclass_fields__})
    sim = simulate(truth, S0=100.0, T=n_steps * dt, n_steps=n_steps, seed=a.seed)
    out = Path(a.out) / f"synthetic_{a.model}_{a.preset}_{a.years:g}y_seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"simulated {n_steps} steps with {int(sim.jumps.sum())} jumps -> {out}")

    priors = Priors.from_beliefs(dt, mu=(0.05, 0.5), kappa=(2.0, 2.0), theta=(0.05, 0.05),
                                 sigma=max(2 * truth.sigma, 0.05), rho=(-0.3, 0.4),
                                 lam=(1.0, 1.0), mu_j=(-0.05, 0.3), sigma_j=0.1)
    config = SamplerConfig(n_iter=a.iter, burn_in=a.burn_in, n_particles=a.particles, path=a.path)
    post = Estimator(a.model, priors, config).fit(sim.prices, dt, n_chains=a.chains,
                                                  n_jobs=a.jobs, seed=a.seed, progress=True)
    summary = post.summary(truth)
    print(summary.round(4).to_string())
    post.save(str(out / "posterior"))
    pd.DataFrame({"price": sim.prices, "true_variance": sim.variance}).to_csv(
        out / "data.csv", index_label="k")

    S0 = float(sim.prices[-1])
    option = OptionSpec(S0, [a.moneyness * S0], a.maturity, a.rate)
    true_price = float(FourierPricer().call(S0, option.K, option.T, option.r, truth,
                                            v0=sim.variance[-1])[0])
    draws = posterior_draws(post)
    mc = PosteriorPricer(a.model).price(draws, option)
    pce = PCESurrogate(a.model, order=a.pce_order).fit(draws, option)
    report = {
        "true_price": true_price, "mc_mean": float(mc.prices.mean()),
        "mc_sd": float(mc.prices.std(ddof=1)), "pce_mean": float(pce.mean[0]),
        "pce_sd": float(pce.std[0]), "pce_loo_error": float(pce.loo_error[0]),
        "latent_variance_mse": mse(post.variance_path, sim.variance),
        "sobol_total": pce.sobol_total.iloc[:, 0].to_dict(),
        "runtime_s": post.settings["runtime_s"],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "sobol_total"}, indent=2))

    plotting.plot_posteriors(post, truth).savefig(out / "posteriors.png", dpi=150)
    plotting.plot_traces(post).savefig(out / "traces.png", dpi=120)
    plotting.plot_variance(post, sim.variance).savefig(out / "variance.png", dpi=150)
    fig = plotting.plot_price_distribution(mc.prices, true_price)
    fig.savefig(out / "option_price.png", dpi=150)


if __name__ == "__main__":
    main()
