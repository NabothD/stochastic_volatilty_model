"""Particle filter (SIR) for the latent variance and jumps.

:class:`ParticleFilter` is a thin, typed wrapper around the compiled kernels
in :mod:`svuq.kernels`. Relative to the original scripts it fixes the
observation timing, re-initialises the cloud each sweep, resamples
``(V, J, Z)`` jointly and uses a return-informed jump proposal (see
``docs/CODE_REVIEW.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import kernels
from .models import Params

_SEED_MAX = 2 ** 31 - 1


def draw_seed(rng: np.random.Generator) -> int:
    """Draw an integer seed for a compiled kernel from a numpy generator.

    Parameters
    ----------
    rng : random generator.

    Returns
    -------
    Non-negative integer seed.
    """
    return int(rng.integers(_SEED_MAX))


@dataclass
class FilterResult:
    """Output of one full filter pass.

    Attributes
    ----------
    v_mean, v_sd : filtered mean / sd of ``V_k``, shape (n+1,).
    loglik : particle estimate of ``log p(R_1..R_n | params)``.
    ess : effective sample size at each step, shape (n,).
    jump_prob : ``P(J_k = 1 | R_1..R_k)``, shape (n,).
    jump_mean, jump_m2 : moments of ``Z_k`` given a jump, shape (n,).
    jump_draw, size_draw : one draw of ``(J_k, Z_k)`` per step, shape (n,).
    cloud : filtered particle clouds, shape (n+1, N), if stored.
    final_particles : particles of ``V_n``, shape (N,).
    """

    v_mean: np.ndarray
    v_sd: np.ndarray
    loglik: float
    ess: np.ndarray
    jump_prob: np.ndarray
    jump_mean: np.ndarray
    jump_m2: np.ndarray
    jump_draw: np.ndarray
    size_draw: np.ndarray
    cloud: np.ndarray | None
    final_particles: np.ndarray


@dataclass
class FilterStep:
    """Output of one online filter step.

    Attributes
    ----------
    particles : particles of the new variance, shape (N,).
    loglik : log predictive density of the new return.
    ess : effective sample size of the weights.
    jump_prob : posterior probability that the return contained a jump.
    """

    particles: np.ndarray
    loglik: float
    ess: float
    jump_prob: float


class ParticleFilter:
    """SIR particle filter for Heston / Bates under the Euler discretisation.

    Parameters
    ----------
    n_particles : number of particles.
    resampling : ``"smooth"`` (refined, thesis) or ``"systematic"``.
    jump_proposal_prob : minimum probability of proposing a jump (Bates only).
    """

    def __init__(self, n_particles: int = 1000, resampling: str = "smooth",
                 jump_proposal_prob: float = 0.1) -> None:
        """Validate and store the filter settings."""
        if resampling not in ("smooth", "systematic"):
            raise ValueError("resampling must be 'smooth' or 'systematic'")
        if n_particles < 2:
            raise ValueError("n_particles must be at least 2")
        self.n_particles = n_particles
        self.resampling = resampling
        self.jump_proposal_prob = jump_proposal_prob

    def _jump_probs(self, params: Params, dt: float) -> tuple:
        """Return ``(p_jump, q_jump)``: true and proposal jump probability per step."""
        p = min(params.lam * dt, 0.999)
        return p, (max(p, self.jump_proposal_prob) if p > 0 else 0.0)

    def initial_particles(self, params: Params, rng: np.random.Generator,
                          v0_mean: float | None = None,
                          v0_sd: float | None = None) -> np.ndarray:
        """Draw particles of ``V_0``.

        Uses a Gamma prior with the given mean / sd, otherwise the stationary
        CIR distribution ``Gamma(2 kappa theta / sigma^2, sigma^2 / (2 kappa))``.

        Parameters
        ----------
        params : model parameters.
        rng : random generator.
        v0_mean, v0_sd : optional prior mean / sd of ``V_0``.

        Returns
        -------
        Particles, shape (N,).
        """
        if v0_mean is not None and v0_sd is not None and v0_sd > 0:
            shape, mean = (v0_mean / v0_sd) ** 2, v0_mean
        else:
            shape = min(2.0 * params.kappa * params.theta / params.sigma ** 2, 1e6)
            mean = params.theta
        return np.maximum(rng.gamma(shape, mean / shape, self.n_particles), kernels.V_FLOOR)

    def run(self, returns: np.ndarray, params: Params, dt: float, rng: np.random.Generator,
            store_cloud: bool = False, v0_mean: float | None = None,
            v0_sd: float | None = None) -> FilterResult:
        """Filter a whole return series.

        Parameters
        ----------
        returns : gross returns ``S_k / S_{k-1}``, shape (n,).
        params : model parameters.
        dt : time step in years.
        rng : random generator.
        store_cloud : keep all clouds (required by :meth:`backward_sample`).
        v0_mean, v0_sd : optional prior of ``V_0``.

        Returns
        -------
        FilterResult
        """
        p, q = self._jump_probs(params, dt)
        V0 = self.initial_particles(params, rng, v0_mean, v0_sd)
        out = kernels.run_filter(
            np.ascontiguousarray(returns, dtype=np.float64), V0, dt, params.mu, params.kappa,
            params.theta, params.sigma, params.rho, p, q, params.mu_j, params.sigma_j,
            self.resampling == "smooth", store_cloud, draw_seed(rng))
        v_mean, v_sd, ll, ess, jp, jm, j2, jd, zd, cloud, final = out
        return FilterResult(v_mean, v_sd, float(ll), ess, jp, jm, j2, jd, zd,
                            cloud if store_cloud else None, final)

    def step(self, particles: np.ndarray, gross_return: float, params: Params, dt: float,
             rng: np.random.Generator) -> FilterStep:
        """Advance a particle cloud by one new observation (online use).

        Parameters
        ----------
        particles : particles of the current variance, shape (N,).
        gross_return : new gross return ``S_t / S_{t-1}``.
        params : model parameters.
        dt : time step in years.
        rng : random generator.

        Returns
        -------
        FilterStep
        """
        p, q = self._jump_probs(params, dt)
        kernels.seed_rng(draw_seed(rng))
        V, ll, ess, pj, *_ = kernels.filter_step(
            particles, float(gross_return), dt, params.mu, params.kappa, params.theta,
            params.sigma, params.rho, p, q, params.mu_j, params.sigma_j,
            self.resampling == "smooth")
        return FilterStep(V, float(ll), float(ess), float(pj))

    @staticmethod
    def backward_sample(result: FilterResult, adjusted_returns: np.ndarray, params: Params,
                        dt: float, rng: np.random.Generator) -> np.ndarray:
        """Draw one variance trajectory by backward sampling.

        Parameters
        ----------
        result : output of :meth:`run` with ``store_cloud=True``.
        adjusted_returns : returns with the sampled jumps removed, shape (n,).
        params : model parameters.
        dt : time step in years.
        rng : random generator.

        Returns
        -------
        Variance path ``V_0..V_n``, shape (n+1,).
        """
        if result.cloud is None:
            raise ValueError("backward sampling needs run(..., store_cloud=True)")
        return kernels.backward_sample(
            result.cloud, np.ascontiguousarray(adjusted_returns, dtype=np.float64), dt,
            params.mu, params.kappa, params.theta, params.sigma, params.rho, draw_seed(rng))
