"""Model parameters and synthetic-data simulation for Heston and Bates.

Discretisation (thesis Eqs. 23–26), in gross returns ``R_k = S_k / S_{k-1}``::

    R_k = (1 + mu dt + sqrt(V_{k-1} dt) eps^S_k) * exp(J_k Z_k)
    V_k = V_{k-1} + kappa (theta - V_{k-1}) dt + sigma sqrt(V_{k-1} dt) eps^V_k
    corr(eps^S, eps^V) = rho,  J_k ~ Bernoulli(lam dt),  Z_k ~ N(mu_j, sigma_j^2)

``simulate(..., scheme="euler")`` produces data from exactly this scheme, so on
synthetic data the estimator is correctly specified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np

HESTON_PARAMS: tuple[str, ...] = ("mu", "kappa", "theta", "sigma", "rho")
JUMP_PARAMS: tuple[str, ...] = ("lam", "mu_j", "sigma_j")
BATES_PARAMS: tuple[str, ...] = HESTON_PARAMS + JUMP_PARAMS


@dataclass(frozen=True)
class HestonParams:
    """Physical-measure Heston parameters (annualised).

    Attributes
    ----------
    mu : drift of the price.
    kappa : speed of mean reversion of the variance.
    theta : long-run variance.
    sigma : volatility of the variance (vol-of-vol).
    rho : correlation between price and variance shocks.
    """

    mu: float = 0.1
    kappa: float = 1.1
    theta: float = 0.04
    sigma: float = 0.01
    rho: float = -0.5

    def __post_init__(self) -> None:
        """Validate the parameter ranges."""
        if self.kappa <= 0 or self.theta <= 0 or self.sigma <= 0:
            raise ValueError("kappa, theta and sigma must be strictly positive")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie in (-1, 1)")

    @property
    def model(self) -> str:
        """Model name: ``"heston"``."""
        return "heston"

    @property
    def names(self) -> tuple[str, ...]:
        """Parameter names of this model."""
        return HESTON_PARAMS

    @property
    def lam(self) -> float:
        """Jump intensity (always 0 for Heston)."""
        return 0.0

    @property
    def mu_j(self) -> float:
        """Mean log-jump (always 0 for Heston)."""
        return 0.0

    @property
    def sigma_j(self) -> float:
        """Std of log-jump (always 0 for Heston)."""
        return 0.0

    @property
    def feller_satisfied(self) -> bool:
        """Whether ``2 kappa theta >= sigma^2`` (variance stays strictly positive)."""
        return 2.0 * self.kappa * self.theta >= self.sigma ** 2

    @property
    def jump_variance(self) -> float:
        """Annual return variance contributed by jumps, ``lam (mu_j^2 + sigma_j^2)``."""
        return self.lam * (self.mu_j ** 2 + self.sigma_j ** 2)

    def to_dict(self) -> dict[str, float]:
        """Return the parameters as a ``{name: value}`` dict."""
        return asdict(self)

    def replace(self, **changes: float) -> HestonParams:
        """Return a copy with some parameters changed."""
        return replace(self, **changes)


@dataclass(frozen=True)
class BatesParams(HestonParams):
    """Heston parameters plus Merton log-normal jumps in the price.

    Attributes
    ----------
    lam : jump intensity (expected jumps per year).
    mu_j : mean log-jump size.
    sigma_j : std of the log-jump size.
    """

    lam: float = 1.0
    mu_j: float = -0.1
    sigma_j: float = 0.2

    def __post_init__(self) -> None:
        """Validate the parameter ranges."""
        super().__post_init__()
        if self.lam < 0 or self.sigma_j < 0:
            raise ValueError("lam and sigma_j must be non-negative")

    @property
    def model(self) -> str:
        """Model name: ``"bates"``."""
        return "bates"

    @property
    def names(self) -> tuple[str, ...]:
        """Parameter names of this model."""
        return BATES_PARAMS


Params = HestonParams | BatesParams


def make_params(model: str, values: dict[str, float]) -> Params:
    """Build a parameter object for ``model`` from a dict (extra keys are ignored).

    Parameters
    ----------
    model : ``"heston"`` or ``"bates"``.
    values : mapping containing at least the model's parameter names.

    Returns
    -------
    ``HestonParams`` or ``BatesParams``.
    """
    if model == "heston":
        return HestonParams(**{k: float(values[k]) for k in HESTON_PARAMS})
    if model == "bates":
        return BatesParams(**{k: float(values[k]) for k in BATES_PARAMS})
    raise ValueError("model must be 'heston' or 'bates'")


@dataclass
class SimulatedPath:
    """A simulated price / variance path.

    Attributes
    ----------
    t : time grid in years, shape (n+1,).
    prices : asset prices, shape (n+1,).
    variance : latent variance, shape (n+1,).
    jumps : jump indicators ``J_k``, shape (n,).
    jump_sizes : realised log-jumps ``J_k Z_k``, shape (n,).
    """

    t: np.ndarray
    prices: np.ndarray
    variance: np.ndarray
    jumps: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    jump_sizes: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def returns(self) -> np.ndarray:
        """Gross returns ``S_k / S_{k-1}``, shape (n,)."""
        return self.prices[1:] / self.prices[:-1]


def simulate(
    params: Params,
    S0: float = 100.0,
    v0: float | None = None,
    T: float = 1.0,
    n_steps: int = 252,
    seed: int | None = None,
    scheme: str = "euler",
) -> SimulatedPath:
    """Simulate one Heston or Bates path.

    The variance uses a full-truncation Euler step (``max(V, 0)`` inside drift
    and diffusion).

    Parameters
    ----------
    params : ``HestonParams`` or ``BatesParams``.
    S0 : initial price.
    v0 : initial variance (defaults to ``theta``).
    T : horizon in years.
    n_steps : number of time steps.
    seed : random seed.
    scheme : ``"euler"`` (matches the estimator) or ``"log"`` (log-Euler with
        the jump compensator; then ``mu`` no longer equals the drift the
        estimator recovers).

    Returns
    -------
    SimulatedPath
    """
    if scheme not in ("euler", "log"):
        raise ValueError("scheme must be 'euler' or 'log'")
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    v0 = params.theta if v0 is None else v0

    eps_s = rng.standard_normal(n_steps)
    eps_v = params.rho * eps_s + np.sqrt(1.0 - params.rho ** 2) * rng.standard_normal(n_steps)
    jumps = rng.random(n_steps) < params.lam * dt
    sizes = np.where(jumps, rng.normal(params.mu_j, params.sigma_j, n_steps), 0.0)
    compensator = params.lam * (np.exp(params.mu_j + 0.5 * params.sigma_j ** 2) - 1.0)

    S = np.empty(n_steps + 1)
    V = np.empty(n_steps + 1)
    S[0], V[0] = S0, v0
    for k in range(n_steps):
        vp = max(V[k], 0.0)
        sq = np.sqrt(vp * dt)
        if scheme == "euler":
            S[k + 1] = S[k] * (1.0 + params.mu * dt + sq * eps_s[k]) * np.exp(sizes[k])
        else:
            S[k + 1] = S[k] * np.exp((params.mu - compensator - 0.5 * vp) * dt
                                     + sq * eps_s[k] + sizes[k])
        V[k + 1] = V[k] + params.kappa * (params.theta - vp) * dt + params.sigma * sq * eps_v[k]
    return SimulatedPath(np.linspace(0.0, T, n_steps + 1), S, np.maximum(V, 0.0),
                         jumps.astype(int), sizes)
