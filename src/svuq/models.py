"""Model parameter containers and synthetic-data simulation for Heston / Bates.

Discretisation
--------------
All estimation code in this package works with the Euler–Maruyama
discretisation used in the thesis (Eqs. 23–26),
written in terms of the gross return ``R_k = S_k / S_{k-1}``::

    R_k = (1 + mu*dt + sqrt(V_{k-1}*dt) * eps^S_k) * exp(J_k * Z_k)
    V_k = V_{k-1} + kappa*(theta - V_{k-1})*dt + sigma*sqrt(V_{k-1}*dt) * eps^V_k
    corr(eps^S_k, eps^V_k) = rho
    J_k ~ Bernoulli(lam*dt),  Z_k ~ N(mu_j, sigma_j^2)       (Bates only)

``simulate`` with ``scheme="euler"`` generates data from exactly this
discretisation, so on synthetic data the estimator is correctly specified
and the "true" parameters are the ones you pass in.  ``scheme="log"`` uses a
log-Euler step with the jump compensator, which is closer to the
continuous-time model but changes the meaning of ``mu`` (see docstring).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Dict, Optional

import numpy as np

HESTON_PARAMS = ("mu", "kappa", "theta", "sigma", "rho")
JUMP_PARAMS = ("lam", "mu_j", "sigma_j")
BATES_PARAMS = HESTON_PARAMS + JUMP_PARAMS


@dataclass(frozen=True)
class HestonParams:
    """Physical-measure Heston parameters (annualised)."""

    mu: float = 0.1
    kappa: float = 1.1
    theta: float = 0.04
    sigma: float = 0.01
    rho: float = -0.5

    def __post_init__(self) -> None:
        if self.kappa <= 0 or self.theta <= 0 or self.sigma <= 0:
            raise ValueError("kappa, theta and sigma must be strictly positive")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie in (-1, 1)")

    @property
    def feller_satisfied(self) -> bool:
        """``2*kappa*theta >= sigma^2`` (variance stays strictly positive)."""
        return 2.0 * self.kappa * self.theta >= self.sigma ** 2

    @property
    def lam(self) -> float:  # convenience so Heston can be treated as Bates with no jumps
        return 0.0

    @property
    def mu_j(self) -> float:
        return 0.0

    @property
    def sigma_j(self) -> float:
        return 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def replace(self, **kw) -> "HestonParams":
        return replace(self, **kw)


@dataclass(frozen=True)
class BatesParams(HestonParams):
    """Heston parameters plus Merton log-normal jumps in the price."""

    lam: float = 1.0       # jump intensity (expected jumps per year)
    mu_j: float = -0.1     # mean log-jump size
    sigma_j: float = 0.2   # std of log-jump size

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.lam < 0 or self.sigma_j < 0:
            raise ValueError("lam and sigma_j must be non-negative")


@dataclass
class SimulatedPath:
    t: np.ndarray
    prices: np.ndarray
    variance: np.ndarray
    jumps: np.ndarray = field(default_factory=lambda: np.zeros(0))       # J_k (len n)
    jump_sizes: np.ndarray = field(default_factory=lambda: np.zeros(0))  # J_k * Z_k (len n)

    @property
    def returns(self) -> np.ndarray:
        return self.prices[1:] / self.prices[:-1]


def simulate(
    params: HestonParams,
    S0: float = 100.0,
    v0: Optional[float] = None,
    T: float = 1.0,
    n_steps: int = 252,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    scheme: str = "euler",
) -> SimulatedPath:
    """Simulate one Heston (``HestonParams``) or Bates (``BatesParams``) path.

    Parameters
    ----------
    scheme:
        ``"euler"``  – the discretisation assumed by the estimator (default).
        ``"log"``    – log-Euler for the price with the jump compensator
        ``lam*(exp(mu_j + sigma_j^2/2) - 1)`` subtracted from the drift.  Under
        this scheme the expected *diffusive* simple return is no longer
        ``mu*dt``, so ``mu`` estimated by the Euler-based sampler will differ
        from the simulation input by roughly the compensator.

    The variance uses a full-truncation Euler step (``max(V, 0)`` inside the
    drift and diffusion), which is the standard low-bias fix for the CIR process.
    """
    if scheme not in ("euler", "log"):
        raise ValueError("scheme must be 'euler' or 'log'")
    rng = rng if rng is not None else np.random.default_rng(seed)
    dt = T / n_steps
    v0 = params.theta if v0 is None else v0

    eps_s = rng.standard_normal(n_steps)
    eps_v = params.rho * eps_s + np.sqrt(1.0 - params.rho ** 2) * rng.standard_normal(n_steps)
    jumps = rng.random(n_steps) < params.lam * dt
    jump_sizes = np.where(jumps, rng.normal(params.mu_j, params.sigma_j if params.sigma_j > 0 else 0.0, n_steps), 0.0)

    S = np.empty(n_steps + 1)
    V = np.empty(n_steps + 1)
    S[0], V[0] = S0, v0
    comp = params.lam * (np.exp(params.mu_j + 0.5 * params.sigma_j ** 2) - 1.0)
    for k in range(n_steps):
        vp = max(V[k], 0.0)
        sq = np.sqrt(vp * dt)
        if scheme == "euler":
            S[k + 1] = S[k] * (1.0 + params.mu * dt + sq * eps_s[k]) * np.exp(jump_sizes[k])
        else:
            S[k + 1] = S[k] * np.exp((params.mu - comp - 0.5 * vp) * dt + sq * eps_s[k] + jump_sizes[k])
        V[k + 1] = V[k] + params.kappa * (params.theta - vp) * dt + params.sigma * sq * eps_v[k]
    t = np.linspace(0.0, T, n_steps + 1)
    return SimulatedPath(t=t, prices=S, variance=np.maximum(V, 0.0), jumps=jumps.astype(int), jump_sizes=jump_sizes)
