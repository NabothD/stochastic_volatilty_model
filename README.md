# Uncertainty quantification for stochastic-volatility models

Code for the MEng thesis *Uncertainty Quantification of Stochastic Volatility
Models*. It does three things:

1. **Estimation.** It estimates the parameters of the **Heston** and
   **Bates** (Heston + log-normal price jumps) models, together with the
   latent variance path and the jumps, from asset prices alone. The method
   is Bayesian: a particle filter (Sequential Importance Resampling) inside a
   Gibbs sampler with conjugate regression updates.
2. **Pricing.** It prices European options semi-analytically under both
   models.
3. **Propagation.** It pushes the joint parameter posterior into option
   prices, either exactly (posterior Monte Carlo) or with a polynomial chaos
   surrogate that also gives Sobol sensitivity indices.

The original research scripts are kept unchanged in [`legacy/`](legacy). The
problems found in them, and what was changed, are listed in
[`docs/CODE_REVIEW.md`](docs/CODE_REVIEW.md). Read that first if you are
comparing with the thesis results.

## Layout

```
src/svuq/
  models.py        HestonParams / BatesParams (frozen dataclasses), simulate()
  priors.py        Priors: conjugate hyper-parameters from intuitive (mean, sd) beliefs
  conjugate.py     ConjugateSampler: posterior draws of mu, (kappa, theta, sigma), rho, jumps
  kernels.py       numba-compiled filter / resampling / backward-sampling kernels (nogil, cached)
  filtering.py     ParticleFilter: typed wrapper around the kernels (batch and online step)
  estimation.py    SamplerConfig, GibbsSampler (one chain), Estimator (parallel chains),
                   PosteriorResult (summary with ESS and R-hat, CSV export)
  pricing.py       FourierPricer: vectorised Heston / Bates call prices; Black–Scholes, implied vol
  uq.py            PosteriorPricer (exact, vectorised), PCESurrogate (Hermite PCE + copula, Sobol)
  forecasting.py   VolatilityForecaster: online now-casts and volatility forecasts with bands
  parallel.py      process_map / thread_map helpers
  diagnostics.py   ESS, R-hat, autocorrelation, MSE
  plotting.py      posterior / trace / variance-path / price figures
  data.py          CSV/XLSX loading, optional Yahoo Finance download
scripts/
  run_synthetic.py       full pipeline on simulated data with known parameters
  run_real_data.py       estimate on market prices and price an option chain
  forecast_volatility.py fit, then stream new prices through the online forecaster
tests/                   pytest suite (67 tests, ~15 s)
data/                    raw price data and the 11-Jun-1997 S&P 100 option quotes
results/thesis/          parameter / volatility samples produced for the thesis
figures/thesis/          figures used in the thesis
legacy/                  original Python and MATLAB scripts (for reference only)
docs/                    CODE_REVIEW.md (issues fixed), RESULTS.md (synthetic experiments)
```

## Install

Requires Python ≥ 3.10.

```bash
pip install -e ".[dev]"
```

Add the extras to cross-check prices against QuantLib, use chaospy
distributions, and download prices:

```bash
pip install -e ".[dev,quantlib,chaospy,data]"
```

Run the tests and the linter:

```bash
pytest
```

```bash
ruff check src scripts tests
```

## Quick start

```python
import numpy as np
from svuq import (BatesParams, Estimator, OptionSpec, PCESurrogate, PosteriorPricer, Priors,
                  SamplerConfig, VolatilityForecaster, posterior_draws, simulate)

if __name__ == "__main__":          # needed: chains run in separate processes
    dt = 1 / 252
    truth = BatesParams(mu=0.08, kappa=3.0, theta=0.04, sigma=0.4, rho=-0.7,
                        lam=1.0, mu_j=-0.1, sigma_j=0.1)
    sim = simulate(truth, S0=100, T=3, n_steps=756, seed=1)

    # 1. Estimate: 4 chains in parallel
    priors = Priors.from_beliefs(dt, kappa=(2, 2), theta=(0.05, 0.05), sigma=0.3,
                                 rho=(-0.3, 0.4), lam=(1, 1), mu_j=(-0.05, 0.3))
    estimator = Estimator("bates", priors, SamplerConfig(n_iter=2000, n_particles=1000))
    post = estimator.fit(sim.prices, dt, n_chains=4, n_jobs=4, seed=0, progress=True)
    print(post.summary(truth))      # mean, sd, 95% CI, ESS, R-hat, error, coverage

    # 2. Option-price uncertainty
    draws = posterior_draws(post)   # adds v0 = filtered variance at the last date
    option = OptionSpec(S0=sim.prices[-1], strikes=[95, 100, 110], T=1 / 12, r=0.05)
    mc = PosteriorPricer("bates").price(draws, option)          # exact, all draws
    pce = PCESurrogate("bates", order=3).fit(draws, option)      # surrogate + Sobol
    print(mc.summary(), pce.mean, pce.std, pce.sobol_total, sep="\n")

    # 3. Volatility now-cast and forecast, updated online
    forecaster = VolatilityForecaster(post, n_draws=100).initialise(sim.prices)
    print(forecaster.forecast([1, 5, 21]))
    # live: forecaster.update(new_price) -> Nowcast   (a few ms per price)
```

From the command line:

```bash
python scripts/run_synthetic.py --model heston --preset realistic --iter 2000 --chains 4
```

```bash
python scripts/run_real_data.py --ticker ^OEX --start 1994-06-11 --end 1997-06-12 --options data/oex_options_1997-06-11.csv --rate 0.05 --dividend 0.017
```

```bash
python scripts/forecast_volatility.py --data data/raw/sp500_historical.xlsx --model heston --holdout 10
```

Each run writes its chains, summary, variance path, reports and figures to
`results/`.

## Performance

Timings are for 3 years of daily data on one core of the development
machine.

| | original scripts | `svuq` |
|---|---|---|
| Gibbs sweep, 200 particles | 1.44 s | 9–14 ms (**100–150×**) |
| Gibbs sweep, 1000 particles | 32.7 s | 44–85 ms (**390–740×**) |
| 4 chains × 2000 sweeps, 1000 particles | ~3 days (serial) | ~3 min (4 processes) |
| 500 posterior draws × 36 strikes | ~20–60 s (adaptive quadrature per price) | ~0.15 s |
| Online volatility update (100 draws × 500 particles) | not available | ~3–7 ms |

The sweep timings include drawing a full variance path by backward sampling,
which the original code did not do. The lower figures are with
`resampling="systematic"`; the default `"smooth"` matches the thesis.

How the speed was achieved:
- **Compiled kernels.** The particle filter, both resampling schemes and the
  backward sampler are numba kernels. They are compiled once and cached on
  disk, and release the GIL.
- **Parallel chains.** `Estimator.fit(n_chains=…, n_jobs=…)` runs chains in
  separate processes. That also gives R-hat convergence checks.
- **Threaded forecaster.** The `VolatilityForecaster` filters one cloud per
  posterior draw in threads.
- **Vectorised pricing.** All strikes and parameter sets share one composite
  Gauss–Legendre grid, so a whole posterior × option-chain grid costs two
  matrix products. It agrees with adaptive quadrature to 1e-8 and with
  QuantLib to 1e-8 (tests).
- **No chaospy in the PCE.** It uses an orthonormal Hermite basis in numpy;
  mean, variance and Sobol indices come straight from the coefficients.
- **Reproducibility.** Every kernel call is seeded from a
  `numpy.random.Generator`, so results are reproducible for a given seed.

## Method in brief

The thesis' Euler discretisation, in gross returns `R_k = S_k / S_{k-1}`:

```
R_k = (1 + mu dt + sqrt(V_{k-1} dt) eps^S_k) * exp(J_k Z_k),   J_k ~ Bernoulli(lam dt), Z_k ~ N(mu_J, sigma_J^2)
V_k = V_{k-1} + kappa (theta - V_{k-1}) dt + sigma sqrt(V_{k-1} dt) eps^V_k,   corr(eps^S, eps^V) = rho
```

Each Gibbs sweep does four things:

1. **Particle filter** (`ParticleFilter`). Particles of `V_{k-1}` (and, for Bates, a proposed
   jump `(J_k, Z_k)`) are weighted by `p(R_k | ·)`. They are then resampled
   with refined (continuous) resampling and propagated to `V_k`, using the
   variance shock correlated with the price shock implied by `R_k`. The
   filter also returns the log-likelihood and the per-day jump probabilities.
2. **Jumps (Bates).** Jump times and sizes are drawn from the filter. Then
   `lam ~ Gamma`, and `(mu_J, sigma_J^2) ~ Normal–inverse-gamma`. The sampled
   jumps are removed from the returns.
3. **Variance path.** A trajectory is drawn by backward sampling
   (`path="ffbs"`). The thesis used the filtered mean (`path="mean"`); see
   the warning below.
4. **Regressions.** Conjugate regression draws of
   - `eta = 1 + mu dt`;
   - `beta = (kappa theta dt, 1 - kappa dt)` with `sigma^2`
     (Normal–inverse-gamma);
   - `(psi, omega) = (sigma rho, sigma^2 (1 - rho^2))`, which gives `rho`.

After burn-in the stored draws form the posterior. Option prices are
computed under the risk-neutral measure with the estimated parameters. An
optional variance risk premium `eps` (thesis Eq. 79) shifts `kappa` and
`theta`. The spot variance `v0` is drawn from the filtered distribution at
the last date.

### Things to know when using it

- **Use `path="ffbs"`.** The filtered-mean path is too smooth, and its
  increments are mostly the `rho`-correlated propagation term. With it,
  `sigma` is biased low and `rho` drifts towards −1. With weakly informative
  priors the chain can diverge outright.
- **Check the ESS column.** `sigma` and `rho` mix slowly because they are
  strongly coupled to the latent path. Typical values are ESS ≈ 10–20 per
  1000 sweeps. Run 2000+ sweeps, look at `plot_traces`, and ideally run
  several seeds.
- **Very small vol-of-vol is not identifiable from prices.** With
  `sigma = 0.01` (the thesis' synthetic setup), the variance hardly moves.
  The returns then carry almost no information on `sigma` or `rho`, and the
  posterior is whatever the prior says. That is the honest answer, not a bug.
- **Jump parameters need jumps.** With 1 year of data and only 1–2 jumps,
  the posteriors of `lam` and `sigma_J` stay close to the prior. Three
  years with about 8 jumps already pins down `mu_J`.
- **Physical vs risk-neutral.** Parameters estimated from historical prices
  are under the physical measure P. Pricing with them ignores the variance
  and jump risk premia (thesis Section 6.2). Use `vol_risk_premium` to
  explore that effect.

## Results on synthetic data

See [`docs/RESULTS.md`](docs/RESULTS.md) for the output of
`scripts/run_synthetic.py` with 2000 sweeps and 1000 particles.

## References

- Heston, S. L. (1993). A closed-form solution for options with stochastic
  volatility. *Review of Financial Studies* 6(2).
- Bates, D. S. (1996). Jumps and stochastic volatility: exchange rate
  processes implicit in Deutsche Mark options. *Review of Financial Studies*
  9(1).
- Jacquier, E., Polson, N. G., Rossi, P. E. (1994). Bayesian analysis of
  stochastic volatility models. *JBES* 12(4).
- Malik, S., Pitt, M. K. (2011). Particle filters for continuous likelihood
  evaluation and maximisation. *Journal of Econometrics* 165(2).
- Lewis, A. (2001). A simple option formula for general jump-diffusion and
  other exponential Lévy processes.
- Albrecher, H. et al. (2007). The little Heston trap. *Wilmott Magazine*.

## Author

NabothD
