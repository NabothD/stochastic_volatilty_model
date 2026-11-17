"""Online volatility now-casting and forecasting from a fitted posterior.

:class:`VolatilityForecaster` keeps one particle cloud per posterior draw, so
its forecasts carry both state uncertainty (where is the variance now?) and
parameter uncertainty (how fast does it revert, to what level?). New prices
are processed one at a time with :meth:`VolatilityForecaster.update`, which
is cheap enough for live use (a few milliseconds per observation).

Forecasts use the conditional expectation of the Heston variance::

    E[V_{t+h} | V_t] = theta + (V_t - theta) exp(-kappa h)
    E[(1/H) int_0^H V_{t+s} ds | V_t] = theta + (V_t - theta) (1 - exp(-kappa H)) / (kappa H)

plus the jump variance ``lam (mu_j^2 + sigma_j^2)`` for Bates. The bands are
quantiles of these expectations across draws and particles; they describe
uncertainty about the *expected* volatility, not about the realised path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .estimation import PosteriorResult
from .filtering import ParticleFilter
from .models import Params, make_params
from .parallel import thread_map


@dataclass(frozen=True)
class Nowcast:
    """Current-variance estimate after the latest observation.

    Attributes
    ----------
    variance : posterior mean of the current variance.
    vol : posterior mean of the current volatility ``sqrt(V)`` (annualised).
    vol_lo, vol_hi : credible band of the current volatility.
    jump_prob : posterior probability that the latest return contained a jump.
    """

    variance: float
    vol: float
    vol_lo: float
    vol_hi: float
    jump_prob: float


class VolatilityForecaster:
    """Particle-filter volatility forecaster built from a posterior sample.

    Parameters
    ----------
    posterior : output of :meth:`svuq.Estimator.fit`.
    n_draws : posterior parameter draws to carry (more = smoother bands, slower).
    n_particles : particles per draw.
    resampling : ``"smooth"`` or ``"systematic"``.
    n_jobs : threads used by :meth:`initialise` (``None`` = all cores).
    seed : random seed.
    """

    def __init__(self, posterior: PosteriorResult, n_draws: int = 100, n_particles: int = 500,
                 resampling: str = "smooth", n_jobs: int | None = None,
                 seed: int | None = None) -> None:
        """Select parameter draws and set up one filter state per draw."""
        rng = np.random.default_rng(seed)
        samples = posterior.samples
        rows = rng.choice(len(samples), size=min(n_draws, len(samples)), replace=False)
        self.params: list[Params] = [make_params(posterior.model, samples.iloc[i].to_dict())
                                     for i in rows]
        self.dt = posterior.dt
        self.filter = ParticleFilter(n_particles, resampling)
        self.n_jobs = n_jobs
        children = np.random.SeedSequence(seed).spawn(len(rows))
        self._rngs = [np.random.default_rng(s) for s in children]
        self._clouds: list[np.ndarray] | None = None
        self._last_price: float | None = None
        self._jump_prob = 0.0

    @property
    def is_ready(self) -> bool:
        """Whether :meth:`initialise` has been called."""
        return self._clouds is not None

    def initialise(self, prices: Sequence[float]) -> VolatilityForecaster:
        """Filter a price history to set the current variance distribution.

        Parameters
        ----------
        prices : historical prices ``S_0..S_n`` (same spacing as the fit).

        Returns
        -------
        self, for chaining.
        """
        prices = np.asarray(prices, dtype=np.float64)
        returns = prices[1:] / prices[:-1]

        def run_one(i: int) -> np.ndarray:
            return self.filter.run(returns, self.params[i], self.dt, self._rngs[i]).final_particles

        self._clouds = thread_map(run_one, range(len(self.params)), self.n_jobs)
        self._last_price = float(prices[-1])
        return self

    def update(self, price: float) -> Nowcast:
        """Process one new price and return the updated now-cast.

        Parameters
        ----------
        price : the latest price (one ``dt`` after the previous one).

        Returns
        -------
        Nowcast
        """
        if not self.is_ready:
            raise RuntimeError("call initialise(prices) first")
        r = float(price) / self._last_price
        jump_probs = np.empty(len(self.params))
        for i, params in enumerate(self.params):
            step = self.filter.step(self._clouds[i], r, params, self.dt, self._rngs[i])
            self._clouds[i] = step.particles
            jump_probs[i] = step.jump_prob
        self._last_price = float(price)
        self._jump_prob = float(jump_probs.mean())
        return self.nowcast()

    def _all_particles(self) -> np.ndarray:
        """Return the particle clouds stacked, shape (n_draws, N)."""
        if not self.is_ready:
            raise RuntimeError("call initialise(prices) first")
        return np.vstack(self._clouds)

    def nowcast(self, ci: float = 0.9) -> Nowcast:
        """Return the current-variance estimate.

        Parameters
        ----------
        ci : credible-band mass.

        Returns
        -------
        Nowcast
        """
        V = self._all_particles().ravel()
        vol = np.sqrt(V)
        lo, hi = np.quantile(vol, [0.5 - ci / 2, 0.5 + ci / 2])
        return Nowcast(float(V.mean()), float(vol.mean()), float(lo), float(hi), self._jump_prob)

    def forecast(self, horizons_days: Sequence[int] = (1, 5, 21), ci: float = 0.9,
                 include_jumps: bool = True) -> pd.DataFrame:
        """Forecast the annualised volatility averaged over each horizon.

        Parameters
        ----------
        horizons_days : horizons in observation steps (trading days for daily data).
        ci : credible-band mass.
        include_jumps : add the jump variance (Bates) to the diffusive variance.

        Returns
        -------
        DataFrame indexed by horizon with ``vol_mean``, ``vol_lo``, ``vol_hi`` and
        ``variance_mean`` (all annualised).
        """
        V = self._all_particles()
        kappa = np.array([p.kappa for p in self.params])[:, None]
        theta = np.array([p.theta for p in self.params])[:, None]
        jump_var = np.array([p.jump_variance for p in self.params])[:, None]
        if not include_jumps:
            jump_var = np.zeros_like(jump_var)
        rows = []
        for h in horizons_days:
            H = h * self.dt
            decay = (1.0 - np.exp(-kappa * H)) / (kappa * H)
            avg_var = theta + (V - theta) * decay + jump_var
            vol = np.sqrt(np.maximum(avg_var, 0.0)).ravel()
            lo, hi = np.quantile(vol, [0.5 - ci / 2, 0.5 + ci / 2])
            rows.append({"horizon_days": h, "vol_mean": vol.mean(), "vol_lo": lo, "vol_hi": hi,
                         "variance_mean": float(avg_var.mean())})
        return pd.DataFrame(rows).set_index("horizon_days")
