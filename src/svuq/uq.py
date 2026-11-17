"""Propagation of parameter uncertainty into option prices.

Two routes are provided.

``posterior_mc_prices``
    Prices the option at (a subset of) the joint posterior draws.  This is
    exact up to Monte-Carlo error and keeps the posterior *correlations*
    between parameters (``kappa`` and ``theta`` are typically strongly
    negatively correlated), so it is the reference answer.

``pce_prices`` (thesis method)
    Fits a parametric marginal to each parameter, builds an independent joint
    distribution in chaospy and fits a polynomial chaos surrogate.  The
    surrogate gives mean/variance and Sobol sensitivity indices cheaply.

Fixes relative to the original scripts:

* ``scipy_to_chaospy`` dropped the fitted ``loc`` (shift) of ``lognorm``,
  ``gamma`` and ``invgamma``.  E.g. a gamma fitted to rho-samples around
  ``-0.5`` has ``loc ~ -0.7``; dropping it produced a *positive* rho
  distribution (``cp.Gamma(2.09, 0.0057)``, mean +0.012, in
  ``Chaospy_Heston.py``) and a sigma distribution 10x too small.  The shift is
  now kept and the mapping is unit-tested against scipy moments.
* Marginals were fitted to samples trimmed at the 2.5/97.5% quantiles, which
  removes the tails and understates the uncertainty being propagated.
  Fitting now uses all samples by default.
* The PCE is built on an independent standard-normal germ and mapped to the
  parameters through the fitted marginals (inverse-CDF, clipped to the
  admissible domain: ``rho`` in (-1, 1), positive ``kappa, theta, sigma, ...``)
  with an optional Gaussian copula, so posterior correlations are kept and
  quadrature nodes never hit invalid parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as st

from .estimation import PosteriorResult
from .pricing import bates_call, heston_call

DOMAINS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "mu": (None, None), "kappa": (0.0, None), "theta": (0.0, None), "sigma": (0.0, None),
    "rho": (-0.999, 0.999), "lam": (0.0, None), "mu_j": (None, None), "sigma_j": (0.0, None),
    "v0": (0.0, None), "r": (None, None),
}
DEFAULT_CANDIDATES = ("norm", "lognorm", "gamma")


# ----------------------------------------------------------------------------
# Marginal fitting
# ----------------------------------------------------------------------------
def scipy_to_chaospy(dist_name: str, params: Sequence[float]):
    """Map fitted ``scipy.stats`` parameters to the equivalent chaospy distribution."""
    import chaospy as cp

    if dist_name == "norm":
        loc, scale = params
        return cp.Normal(loc, scale)
    if dist_name == "lognorm":
        s, loc, scale = params
        return cp.LogNormal(mu=np.log(scale), sigma=s, shift=loc)
    if dist_name == "gamma":
        a, loc, scale = params
        return cp.Gamma(shape=a, scale=scale, shift=loc)
    if dist_name == "invgamma":
        a, loc, scale = params
        return cp.InverseGamma(shape=a, scale=scale, shift=loc)
    if dist_name == "beta":
        a, b, loc, scale = params
        return cp.Beta(a, b, lower=loc, upper=loc + scale)
    if dist_name == "uniform":
        loc, scale = params
        return cp.Uniform(loc, loc + scale)
    raise ValueError(f"No chaospy mapping for '{dist_name}'")


@dataclass
class MarginalFit:
    name: str
    dist_name: str
    params: tuple
    aic: float
    ks_pvalue: float

    @property
    def frozen(self):
        """The fitted ``scipy.stats`` distribution."""
        return getattr(st, self.dist_name)(*self.params)

    @property
    def cp_dist(self):
        """Equivalent chaospy distribution (shift/loc preserved)."""
        return scipy_to_chaospy(self.dist_name, self.params)

    def ppf_clipped(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF, clipped to the admissible domain of the parameter."""
        x = self.frozen.ppf(np.clip(u, 1e-12, 1 - 1e-12))
        lo, hi = DOMAINS.get(self.name, (None, None))
        if lo is not None:
            x = np.maximum(x, lo + 1e-10)
        if hi is not None:
            x = np.minimum(x, hi)
        return x


def fit_marginal(samples: np.ndarray, name: str = "x",
                 candidates: Sequence[str] = DEFAULT_CANDIDATES,
                 trim: Optional[Tuple[float, float]] = None) -> MarginalFit:
    """Fit candidate distributions by maximum likelihood and keep the lowest AIC."""
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if trim is not None:
        lo, hi = np.percentile(x, trim)
        x = x[(x >= lo) & (x <= hi)]
    if np.ptp(x) == 0:
        raise ValueError(f"samples of {name} are constant")
    best = None
    for dn in candidates:
        dist = getattr(st, dn)
        try:
            with np.errstate(all="ignore"):
                params = dist.fit(x)
                ll = np.sum(dist.logpdf(x, *params))
        except Exception:
            continue
        if not np.isfinite(ll):
            continue
        aic = 2 * len(params) - 2 * ll
        if best is None or aic < best[2]:
            best = (dn, params, aic)
    if best is None:
        raise RuntimeError(f"no candidate distribution could be fitted to {name}")
    dn, params, aic = best
    ks = st.kstest(x, getattr(st, dn)(*params).cdf).pvalue
    return MarginalFit(name, dn, tuple(float(p) for p in params), float(aic), float(ks))


# ----------------------------------------------------------------------------
# Pricing helpers
# ----------------------------------------------------------------------------
def risk_neutral(kappa: float, theta: float, vol_risk_premium: float = 0.0) -> Tuple[float, float]:
    """Thesis Eq. 79: ``kappa_Q = kappa + eps``, ``theta_Q = kappa*theta/(kappa + eps)``."""
    kq = kappa + vol_risk_premium
    return kq, kappa * theta / kq


def make_pricer(model: str, S0: float, K: Union[float, np.ndarray], T: float, r: float,
                vol_risk_premium: float = 0.0, q: float = 0.0) -> Callable[..., np.ndarray]:
    """Return ``f(**params) -> price(s)`` with model inputs as keyword arguments."""
    def f(kappa, theta, sigma, rho, v0, lam=0.0, mu_j=0.0, sigma_j=0.0, **_):
        kq, tq = risk_neutral(kappa, theta, vol_risk_premium)
        if model == "bates" and lam > 0:
            return bates_call(S0, K, T, r, v0, kq, tq, sigma, rho, lam, mu_j, max(sigma_j, 1e-8), q)
        return heston_call(S0, K, T, r, v0, kq, tq, sigma, rho, q)
    return f


def _param_names(model: str) -> List[str]:
    base = ["kappa", "theta", "sigma", "rho", "v0"]
    return base + (["lam", "mu_j", "sigma_j"] if model == "bates" else [])


def posterior_table(result: PosteriorResult, include_v0: bool = True,
                    rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
    """Posterior draws with a ``v0`` column (the filtered variance at the last date).

    ``v0`` for each draw is sampled from that sweep's filtered distribution
    ``N(v_T, v_T_sd^2)`` truncated at 0, so the spot-variance uncertainty from the
    particle filter is propagated as well (the thesis fixed ``v0`` or guessed
    its spread).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    s = result.samples.copy()
    if include_v0:
        v = s["v_T"].to_numpy() + s["v_T_sd"].to_numpy() * rng.standard_normal(len(s))
        s["v0"] = np.abs(v)
    return s


@dataclass
class PriceDistribution:
    prices: np.ndarray       # (n_draws, n_strikes)
    strikes: np.ndarray

    def summary(self, ci: float = 0.95) -> pd.DataFrame:
        lo, hi = 50 * (1 - ci), 50 * (1 + ci)
        return pd.DataFrame({
            "strike": self.strikes, "mean": self.prices.mean(0), "sd": self.prices.std(0, ddof=1),
            f"q{lo:g}": np.percentile(self.prices, lo, axis=0),
            f"q{hi:g}": np.percentile(self.prices, hi, axis=0),
        })


def posterior_mc_prices(draws: pd.DataFrame, model: str, S0: float, K, T: float, r: float,
                        n_draws: Optional[int] = 500, vol_risk_premium: float = 0.0,
                        seed: int = 0, q: float = 0.0) -> PriceDistribution:
    """Price at ``n_draws`` joint posterior draws (keeps parameter correlations)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(draws))
    if n_draws is not None and n_draws < len(draws):
        idx = rng.choice(idx, n_draws, replace=False)
    f = make_pricer(model, S0, K, T, r, vol_risk_premium, q)
    names = _param_names(model)
    rows = draws.iloc[idx][names].to_dict("records")
    prices = np.array([np.atleast_1d(f(**row)) for row in rows])
    return PriceDistribution(prices, np.atleast_1d(np.asarray(K, dtype=float)))


# ----------------------------------------------------------------------------
# Polynomial chaos
# ----------------------------------------------------------------------------
@dataclass
class InputModel:
    """Marginals + Gaussian copula mapping a standard-normal germ to parameters.

    ``x_p = F_p^{-1}(Phi(z_p))`` with ``z = L xi``, ``xi ~ N(0, I)`` and ``L L' = C``
    the correlation of the normal scores of the posterior draws (Nataf
    transform).  With ``C = I`` this reduces to independent marginals, which is
    what the thesis used.
    """
    names: List[str]
    marginals: Dict[str, MarginalFit]
    corr: np.ndarray

    @classmethod
    def from_draws(cls, draws: pd.DataFrame, names: Sequence[str], dependence: str = "gaussian_copula",
                   candidates: Sequence[str] = DEFAULT_CANDIDATES,
                   trim: Optional[Tuple[float, float]] = None) -> "InputModel":
        names = list(names)
        marg = {p: fit_marginal(draws[p].to_numpy(), p, candidates, trim) for p in names}
        if dependence == "independent":
            corr = np.eye(len(names))
        elif dependence == "gaussian_copula":
            u = np.column_stack([np.clip(marg[p].frozen.cdf(draws[p].to_numpy()), 1e-9, 1 - 1e-9)
                                 for p in names])
            corr = np.atleast_2d(np.corrcoef(st.norm.ppf(u), rowvar=False))
        else:
            raise ValueError("dependence must be 'gaussian_copula' or 'independent'")
        return cls(names, marg, corr)

    def transform(self, xi: np.ndarray) -> np.ndarray:
        """Map germ samples ``xi`` (d, n) to parameter samples (d, n)."""
        L = np.linalg.cholesky(self.corr + 1e-12 * np.eye(len(self.names)))
        u = st.norm.cdf(L @ np.atleast_2d(xi))
        return np.vstack([self.marginals[p].ppf_clipped(u[i]) for i, p in enumerate(self.names)])

    def sample(self, n: int, seed: int = 0) -> pd.DataFrame:
        xi = np.random.default_rng(seed).standard_normal((len(self.names), n))
        return pd.DataFrame(self.transform(xi).T, columns=self.names)


@dataclass
class PCEResult:
    inputs: InputModel
    germ: object
    expansion: object
    n_evals: int
    mean: np.ndarray
    std: np.ndarray
    sobol_first: pd.DataFrame
    sobol_total: pd.DataFrame
    strikes: np.ndarray
    loo_error: Optional[np.ndarray] = None

    @property
    def names(self) -> List[str]:
        return self.inputs.names

    @property
    def marginals(self) -> Dict[str, MarginalFit]:
        return self.inputs.marginals

    def sample(self, n: int = 2000, seed: int = 0) -> np.ndarray:
        """Draw option prices from the surrogate, shape (n, n_strikes)."""
        xi = self.germ.sample(n, seed=seed, rule="random")
        out = np.asarray(self.expansion(*np.atleast_2d(xi)))
        return np.maximum(np.atleast_2d(out).reshape(-1, n).T, 0.0)  # prices are non-negative


def pce_prices(draws: pd.DataFrame, model: str, S0: float, K, T: float, r: float,
               order: int = 3, method: str = "regression", n_samples: Optional[int] = None,
               dependence: str = "gaussian_copula",
               candidates: Sequence[str] = DEFAULT_CANDIDATES,
               fixed: Optional[Dict[str, float]] = None, vol_risk_premium: float = 0.0,
               trim: Optional[Tuple[float, float]] = None, seed: int = 0,
               q: float = 0.0) -> PCEResult:
    """Polynomial-chaos surrogate of the option price over the posterior.

    The expansion is built in an independent standard-normal germ ``xi``
    (probabilists' Hermite polynomials, exactly orthonormal and well
    conditioned) and mapped to the parameters through the fitted marginals
    and, by default, a Gaussian copula that keeps the posterior correlations
    (see :class:`InputModel`).  Building orthogonal polynomials directly for
    shifted / truncated fitted marginals (the original approach) is
    numerically fragile and ignores correlations.

    method : ``"regression"`` (least squares on ``n_samples`` Sobol points,
        default ``3 x #terms``; scales to the 8 Bates inputs),
        ``"quadrature"`` (tensor Gauss–Hermite grid, ``(order+1)^d`` model runs,
        as in the thesis) or ``"sparse"`` (Smolyak sparse grid).
    dependence : ``"gaussian_copula"`` (default) or ``"independent"`` (thesis).
        With the copula, Sobol indices refer to the *independent germ
        components* in the order of ``inputs.names`` (Rosenblatt ordering).
    fixed : parameters to hold constant (e.g. ``{"v0": 0.04}``).
    """
    import chaospy as cp

    fixed = dict(fixed or {})
    names = [p for p in _param_names(model) if p not in fixed]
    inputs = InputModel.from_draws(draws, names, dependence, candidates, trim)
    d = len(names)
    germ = cp.Iid(cp.Normal(0, 1), d)
    expansion = cp.generate_expansion(order, germ, normed=True)
    f = make_pricer(model, S0, K, T, r, vol_risk_premium, q)

    def evaluate(xi):
        x = inputs.transform(xi)
        return np.array([np.atleast_1d(f(**dict(zip(names, col)), **fixed)) for col in x.T])

    loo = None
    if method == "regression":
        n = n_samples or 3 * len(expansion)
        nodes = germ.sample(n, rule="sobol", seed=seed)
        evals = evaluate(nodes)
        surrogate, coef = cp.fit_regression(expansion, nodes, evals, retall=1)
        # leave-one-out error (relative to output variance), via the hat matrix
        A = np.asarray(expansion(*nodes)).T
        H = A @ np.linalg.pinv(A)
        resid = evals - np.asarray(surrogate(*nodes)).reshape(evals.shape[1], -1).T
        loo_res = resid / (1.0 - np.clip(np.diag(H), None, 1 - 1e-10))[:, None]
        loo = np.mean(loo_res ** 2, axis=0) / np.maximum(evals.var(axis=0), 1e-300)
    elif method in ("quadrature", "sparse"):
        nodes, weights = cp.generate_quadrature(order, germ, rule="gaussian",
                                                sparse=(method == "sparse"))
        evals = evaluate(nodes)
        surrogate = cp.fit_quadrature(expansion, nodes, weights, evals)
    else:
        raise ValueError("method must be 'regression', 'quadrature' or 'sparse'")

    strikes = np.atleast_1d(np.asarray(K, dtype=float))
    mean = np.atleast_1d(cp.E(surrogate, germ))
    std = np.atleast_1d(cp.Std(surrogate, germ))
    s1 = np.atleast_2d(cp.Sens_m(surrogate, germ)).reshape(d, -1)
    stt = np.atleast_2d(cp.Sens_t(surrogate, germ)).reshape(d, -1)
    cols = [f"K={k:g}" for k in strikes]
    return PCEResult(inputs, germ, surrogate, evals.shape[0], mean, std,
                     pd.DataFrame(s1, index=names, columns=cols),
                     pd.DataFrame(stt, index=names, columns=cols), strikes, loo)
