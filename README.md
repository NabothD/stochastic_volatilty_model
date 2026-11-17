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
src/svuq/            the package
  models.py          parameter classes, Euler simulator (Heston / Bates)
  priors.py          conjugate priors, built from intuitive (mean, sd) beliefs
  bayes.py           conjugate posterior draws: mu, (kappa, theta, sigma), rho, jumps
  particle_filter.py vectorised SIR filter, refined resampling, backward sampling
  estimation.py      Gibbs sampler -> PosteriorResult (summary, CSV export)
  pricing.py         Heston / Bates call prices (Lewis formula), implied vol
  uq.py              posterior Monte Carlo pricing, PCE surrogate, Sobol indices
  diagnostics.py     ESS, R-hat, autocorrelation
  plotting.py        posterior / trace / variance-path / price figures
  data.py            CSV/XLSX loading, optional Yahoo Finance download
scripts/
  run_synthetic.py   full pipeline on simulated data with known parameters
  run_real_data.py   estimate on market prices and price an option chain
tests/               pytest suite (pricing vs QuantLib, filter, conjugate updates, PCE, sampler)
data/                raw price data and the 11-Jun-1997 S&P 100 option quotes (thesis Table 7)
results/thesis/      parameter / volatility samples produced for the thesis
figures/thesis/      figures used in the thesis
legacy/              original Python and MATLAB scripts (for reference only)
docs/CODE_REVIEW.md  issues found in the original code and how they were fixed
```

## Install

```bash
pip install -e ".[dev]"
```

Add the `quantlib` extra to cross-check prices against QuantLib, and the
`data` extra to download prices with yfinance:

```bash
pip install -e ".[dev,quantlib,data]"
```

Run the tests:

```bash
pytest
```

The suite runs in about 30 s.

## Quick start

```python
import numpy as np
from svuq import BatesParams, Priors, simulate, estimate
from svuq.uq import posterior_table, posterior_mc_prices, pce_prices

dt = 1 / 252
truth = BatesParams(mu=0.08, kappa=3.0, theta=0.04, sigma=0.4, rho=-0.7,
                    lam=1.0, mu_j=-0.1, sigma_j=0.1)
sim = simulate(truth, S0=100, T=3, n_steps=756, seed=1)

priors = Priors.from_beliefs(dt, kappa=(2, 2), theta=(0.05, 0.05), sigma=0.3,
                             rho=(-0.3, 0.4), lam=(1, 1), mu_j=(-0.05, 0.3))
post = estimate(sim.prices, dt, model="bates", priors=priors,
                n_iter=2000, n_particles=1000, seed=0, progress=True)
print(post.summary(truth))          # mean, sd, 95% CI, ESS, error, coverage

draws = posterior_table(post)        # adds v0 = filtered variance at the last date
mc  = posterior_mc_prices(draws, "bates", S0=sim.prices[-1], K=[95, 100, 110], T=1/12, r=0.05)
pce = pce_prices(draws, "bates", S0=sim.prices[-1], K=[95, 100, 110], T=1/12, r=0.05, order=3)
print(mc.summary());  print(pce.mean, pce.std);  print(pce.sobol_total)
```

From the command line:

```bash
python scripts/run_synthetic.py --model heston --preset realistic --iter 2000
```

```bash
python scripts/run_synthetic.py --model bates  --preset thesis    --iter 2000
```

```bash
python scripts/run_real_data.py --ticker ^OEX --start 1994-06-11 --end 1997-06-12 --options data/oex_options_1997-06-11.csv --rate 0.05 --dividend 0.017
```

```bash
python scripts/run_real_data.py --data data/raw/sp500_historical.xlsx --models heston
```

Each run writes its chain, summary, variance path, option-price report and
figures to `results/`.

## Method in brief

The thesis' Euler discretisation, in gross returns `R_k = S_k / S_{k-1}`:

```
R_k = (1 + mu dt + sqrt(V_{k-1} dt) eps^S_k) * exp(J_k Z_k),   J_k ~ Bernoulli(lam dt), Z_k ~ N(mu_J, sigma_J^2)
V_k = V_{k-1} + kappa (theta - V_{k-1}) dt + sigma sqrt(V_{k-1} dt) eps^V_k,   corr(eps^S, eps^V) = rho
```

Each Gibbs sweep does four things:

1. **Particle filter.** Particles of `V_{k-1}` (and, for Bates, a proposed
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
