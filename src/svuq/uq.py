"""Propagation of parameter uncertainty into option prices.

* :class:`PosteriorPricer` prices the option at joint posterior draws. It is
  exact up to Monte-Carlo error, keeps parameter correlations, and is the
  reference answer. Thanks to the batched pricer it is also fast.
* :class:`PCESurrogate` (thesis method, corrected) fits marginals, links them
  with a Gaussian copula and fits a polynomial chaos expansion in a standard
  normal germ. Mean, variance and Sobol indices follow directly from the
  coefficients. Only numpy/scipy are needed (chaospy is optional, used solely
  by :func:`scipy_to_chaospy`).

Corrections relative to the original scripts: the fitted ``loc`` of
``lognorm``/``gamma``/``invgamma`` is kept (it was dropped, giving a positive
``rho`` distribution), marginals are fitted to all samples (not tail-trimmed),
and posterior correlations are kept.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import qmc

from .estimation import PosteriorResult
from .pricing import FourierPricer

DOMAINS: dict[str, tuple[float | None, float | None]] = {
    "mu": (None, None), "kappa": (0.0, None), "theta": (0.0, None), "sigma": (0.0, None),
    "rho": (-0.999, 0.999), "lam": (0.0, None), "mu_j": (None, None), "sigma_j": (0.0, None),
    "v0": (0.0, None),
}
DEFAULT_CANDIDATES: tuple[str, ...] = ("norm", "lognorm", "gamma")


def pricing_inputs(model: str) -> list[str]:
    """Return the parameter names that enter the option price.

    Parameters
    ----------
    model : ``"heston"`` or ``"bates"``.

    Returns
    -------
    List of names (``mu`` is excluded: pricing is risk-neutral).
    """
    base = ["kappa", "theta", "sigma", "rho", "v0"]
    return base + (["lam", "mu_j", "sigma_j"] if model == "bates" else [])


def posterior_draws(result: PosteriorResult, seed: int = 0) -> pd.DataFrame:
    """Posterior draws with a ``v0`` column (the variance at the last observed date).

    ``v0`` of each draw is sampled from that sweep's filtered distribution
    ``N(v_T, v_T_sd^2)`` reflected at 0, so spot-variance uncertainty is
    propagated too.

    Parameters
    ----------
    result : estimator output.
    seed : random seed.

    Returns
    -------
    DataFrame of draws.
    """
    rng = np.random.default_rng(seed)
    draws = result.samples.copy()
    draws["v0"] = np.abs(draws["v_T"] + draws["v_T_sd"] * rng.standard_normal(len(draws)))
    return draws


def risk_neutral(kappa: np.ndarray, theta: np.ndarray, vol_risk_premium: float = 0.0):
    """Apply a variance risk premium (thesis Eq. 79).

    Parameters
    ----------
    kappa, theta : physical-measure parameters.
    vol_risk_premium : ``eps`` in ``kappa_Q = kappa + eps``.

    Returns
    -------
    ``(kappa_Q, theta_Q)`` with ``theta_Q = kappa theta / (kappa + eps)``.
    """
    kq = kappa + vol_risk_premium
    return kq, kappa * theta / kq


@dataclass(frozen=True)
class OptionSpec:
    """A European call chain to price.

    Attributes
    ----------
    S0 : spot price.
    strikes : strike(s).
    T : maturity in years.
    r : risk-free rate.
    q : dividend yield.
    """

    S0: float
    strikes: Sequence[float]
    T: float
    r: float
    q: float = 0.0

    @property
    def K(self) -> np.ndarray:
        """Strikes as an array."""
        return np.atleast_1d(np.asarray(self.strikes, dtype=float))


@dataclass
class PriceDistribution:
    """Option prices across posterior draws.

    Attributes
    ----------
    prices : prices, shape (n_draws, n_strikes).
    strikes : strikes, shape (n_strikes,).
    """

    prices: np.ndarray
    strikes: np.ndarray

    def summary(self, ci: float = 0.95) -> pd.DataFrame:
        """Return mean, sd and credible interval per strike.

        Parameters
        ----------
        ci : credible-interval mass.

        Returns
        -------
        DataFrame with one row per strike.
        """
        lo, hi = 50 * (1 - ci), 50 * (1 + ci)
        return pd.DataFrame({
            "strike": self.strikes, "mean": self.prices.mean(0), "sd": self.prices.std(0, ddof=1),
            f"q{lo:g}": np.percentile(self.prices, lo, axis=0),
            f"q{hi:g}": np.percentile(self.prices, hi, axis=0),
        })


class PosteriorPricer:
    """Prices options for a table of parameter draws in one vectorised call.

    Parameters
    ----------
    model : ``"heston"`` or ``"bates"``.
    pricer : Fourier pricer (default: fixed quadrature).
    vol_risk_premium : variance risk premium applied to ``kappa, theta``.
    """

    def __init__(self, model: str, pricer: FourierPricer | None = None,
                 vol_risk_premium: float = 0.0) -> None:
        """Store the model, pricer and risk premium."""
        self.model = model
        self.pricer = pricer or FourierPricer()
        self.vol_risk_premium = vol_risk_premium

    def price_table(self, table: pd.DataFrame, option: OptionSpec) -> np.ndarray:
        """Price ``option`` for every row of a parameter table.

        Parameters
        ----------
        table : columns ``kappa, theta, sigma, rho, v0`` (+ jump columns for Bates).
        option : option chain specification.

        Returns
        -------
        Prices, shape (n_rows, n_strikes).
        """
        kq, tq = risk_neutral(table["kappa"].to_numpy(), table["theta"].to_numpy(),
                              self.vol_risk_premium)
        cols = dict(v0=table["v0"].to_numpy(), kappa=kq, theta=tq,
                    sigma=table["sigma"].to_numpy(), rho=table["rho"].to_numpy())
        if self.model == "bates":
            cols.update({k: table[k].to_numpy() for k in ("lam", "mu_j", "sigma_j")})
        return self.pricer.call_batch(option.S0, option.K, option.T, option.r, cols, option.q)

    def price(self, draws: pd.DataFrame, option: OptionSpec, n_draws: int | None = None,
              seed: int = 0) -> PriceDistribution:
        """Price at (a random subset of) the posterior draws.

        Parameters
        ----------
        draws : output of :func:`posterior_draws`.
        option : option chain specification.
        n_draws : number of draws to use (default: all).
        seed : random seed for the subset.

        Returns
        -------
        PriceDistribution
        """
        if n_draws is not None and n_draws < len(draws):
            draws = draws.sample(n_draws, random_state=seed)
        return PriceDistribution(self.price_table(draws, option), option.K)


# ----------------------------------------------------------------------------
# Marginals and dependence
# ----------------------------------------------------------------------------
def scipy_to_chaospy(dist_name: str, params: Sequence[float]):
    """Map fitted ``scipy.stats`` parameters to the equivalent chaospy distribution.

    Parameters
    ----------
    dist_name : scipy distribution name.
    params : parameters as returned by ``scipy.stats.<dist>.fit``.

    Returns
    -------
    chaospy distribution with the same law (location shift preserved).
    """
    import chaospy as cp

    if dist_name == "norm":
        return cp.Normal(*params)
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


@dataclass(frozen=True)
class MarginalFit:
    """A fitted marginal distribution of one parameter.

    Attributes
    ----------
    name : parameter name.
    dist_name : scipy distribution name.
    params : fitted scipy parameters.
    aic : Akaike information criterion of the fit.
    ks_pvalue : Kolmogorov–Smirnov p-value of the fit.
    """

    name: str
    dist_name: str
    params: tuple[float, ...]
    aic: float
    ks_pvalue: float

    @property
    def frozen(self):
        """The fitted ``scipy.stats`` distribution."""
        return getattr(st, self.dist_name)(*self.params)

    @property
    def cp_dist(self):
        """Equivalent chaospy distribution."""
        return scipy_to_chaospy(self.dist_name, self.params)

    def ppf_clipped(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF clipped to the admissible domain of the parameter.

        Parameters
        ----------
        u : probabilities in (0, 1).

        Returns
        -------
        Quantiles.
        """
        x = self.frozen.ppf(np.clip(u, 1e-12, 1 - 1e-12))
        lo, hi = DOMAINS.get(self.name, (None, None))
        if lo is not None:
            x = np.maximum(x, lo + 1e-10)
        if hi is not None:
            x = np.minimum(x, hi)
        return x

    @classmethod
    def fit(cls, samples: np.ndarray, name: str = "x",
            candidates: Sequence[str] = DEFAULT_CANDIDATES,
            trim: tuple[float, float] | None = None) -> MarginalFit:
        """Fit candidate distributions by maximum likelihood and keep the lowest AIC.

        Parameters
        ----------
        samples : posterior draws of one parameter.
        name : parameter name (used for domain clipping).
        candidates : scipy distribution names to try.
        trim : optional (lower, upper) percentiles to drop before fitting.

        Returns
        -------
        MarginalFit
        """
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
                    ll = float(np.sum(dist.logpdf(x, *params)))
            except (ValueError, RuntimeError, FloatingPointError):
                continue
            aic = 2 * len(params) - 2 * ll
            if np.isfinite(aic) and (best is None or aic < best[2]):
                best = (dn, params, aic)
        if best is None:
            raise RuntimeError(f"no candidate distribution could be fitted to {name}")
        dn, params, aic = best
        ks = st.kstest(x, getattr(st, dn)(*params).cdf).pvalue
        return cls(name, dn, tuple(float(p) for p in params), float(aic), float(ks))


@dataclass
class InputModel:
    """Marginals + Gaussian copula mapping a standard-normal germ to parameters.

    ``x_p = F_p^{-1}(Phi(z_p))`` with ``z = L xi``, ``xi ~ N(0, I)``, ``L L' = C``
    (Nataf transform). ``C = I`` gives independent marginals (thesis).

    Attributes
    ----------
    names : parameter names, in germ order.
    marginals : fitted marginal per parameter.
    corr : correlation matrix of the normal scores.
    """

    names: list[str]
    marginals: dict[str, MarginalFit]
    corr: np.ndarray

    @classmethod
    def from_draws(cls, draws: pd.DataFrame, names: Sequence[str],
                   dependence: str = "gaussian_copula",
                   candidates: Sequence[str] = DEFAULT_CANDIDATES,
                   trim: tuple[float, float] | None = None) -> InputModel:
        """Fit marginals (and the copula) to posterior draws.

        Parameters
        ----------
        draws : posterior draws.
        names : parameters to include.
        dependence : ``"gaussian_copula"`` or ``"independent"``.
        candidates : scipy distribution names to try for each marginal.
        trim : optional percentile trimming before fitting.

        Returns
        -------
        InputModel
        """
        names = list(names)
        marg = {p: MarginalFit.fit(draws[p].to_numpy(), p, candidates, trim) for p in names}
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
        """Map germ samples to parameter samples.

        Parameters
        ----------
        xi : standard-normal germ samples, shape (d, n).

        Returns
        -------
        Parameter samples, shape (d, n).
        """
        L = np.linalg.cholesky(self.corr + 1e-12 * np.eye(len(self.names)))
        u = st.norm.cdf(L @ np.atleast_2d(xi))
        return np.vstack([self.marginals[p].ppf_clipped(u[i]) for i, p in enumerate(self.names)])

    def sample(self, n: int, seed: int = 0) -> pd.DataFrame:
        """Draw parameter samples from the fitted input model.

        Parameters
        ----------
        n : number of samples.
        seed : random seed.

        Returns
        -------
        DataFrame with one column per parameter.
        """
        xi = np.random.default_rng(seed).standard_normal((len(self.names), n))
        return pd.DataFrame(self.transform(xi).T, columns=self.names)


# ----------------------------------------------------------------------------
# Polynomial chaos
# ----------------------------------------------------------------------------
class HermiteBasis:
    """Orthonormal probabilists' Hermite polynomials of total degree <= ``order``.

    Basis functions are ``psi_a(xi) = prod_i He_{a_i}(xi_i) / sqrt(a_i!)``, which
    are orthonormal under the standard normal distribution.

    Parameters
    ----------
    dim : number of input variables.
    order : maximum total degree.
    """

    def __init__(self, dim: int, order: int) -> None:
        """Enumerate the multi-indices, sorted by total degree."""
        indices = [a for a in itertools.product(range(order + 1), repeat=dim) if sum(a) <= order]
        self.indices = np.array(sorted(indices, key=lambda a: (sum(a), a)), dtype=int)
        self.dim = dim
        self.order = order

    def __len__(self) -> int:
        """Return the number of basis functions."""
        return len(self.indices)

    def evaluate(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate every basis function at the given points.

        Parameters
        ----------
        xi : germ points, shape (dim, n).

        Returns
        -------
        Design matrix, shape (n, n_terms).
        """
        xi = np.atleast_2d(xi)
        he = np.empty((self.order + 1,) + xi.shape)
        he[0] = 1.0
        if self.order >= 1:
            he[1] = xi
        for k in range(1, self.order):
            he[k + 1] = xi * he[k] - k * he[k - 1]
        norms = np.sqrt([math.factorial(k) for k in range(self.order + 1)])
        he /= norms[:, None, None]
        out = np.ones((xi.shape[1], len(self)))
        for i in range(self.dim):
            out *= he[self.indices[:, i], i, :].T
        return out


@dataclass
class PCEResult:
    """A fitted PCE surrogate of option prices.

    Attributes
    ----------
    inputs : the input model (marginals + copula).
    basis : orthonormal Hermite basis.
    coef : expansion coefficients, shape (n_terms, n_strikes).
    n_evals : number of pricing-model evaluations used.
    strikes : strikes.
    loo_error : leave-one-out error relative to output variance (regression only).
    """

    inputs: InputModel
    basis: HermiteBasis
    coef: np.ndarray
    n_evals: int
    strikes: np.ndarray
    loo_error: np.ndarray | None = None

    @property
    def names(self) -> list[str]:
        """Names of the uncertain inputs (germ order)."""
        return self.inputs.names

    @property
    def mean(self) -> np.ndarray:
        """Surrogate mean per strike (the constant coefficient)."""
        return self.coef[0]

    @property
    def variance(self) -> np.ndarray:
        """Surrogate variance per strike (sum of squared non-constant coefficients)."""
        return np.sum(self.coef[1:] ** 2, axis=0)

    @property
    def std(self) -> np.ndarray:
        """Surrogate standard deviation per strike."""
        return np.sqrt(self.variance)

    def _sobol(self, mask: np.ndarray) -> pd.DataFrame:
        """Sum squared coefficients selected by ``mask`` (dim x terms), relative to variance."""
        share = (mask.astype(float) @ self.coef ** 2) / np.maximum(self.variance, 1e-300)
        return pd.DataFrame(share, index=self.names, columns=[f"K={k:g}" for k in self.strikes])

    @property
    def sobol_first(self) -> pd.DataFrame:
        """First-order Sobol indices (germ component x strike)."""
        idx = self.basis.indices
        only = (idx > 0) & ((idx > 0).sum(axis=1) == 1)[:, None]
        return self._sobol(only.T)

    @property
    def sobol_total(self) -> pd.DataFrame:
        """Total Sobol indices (germ component x strike)."""
        return self._sobol((self.basis.indices > 0).T)

    def predict(self, xi: np.ndarray) -> np.ndarray:
        """Evaluate the surrogate at germ points.

        Parameters
        ----------
        xi : germ points, shape (dim, n).

        Returns
        -------
        Prices, shape (n, n_strikes).
        """
        return self.basis.evaluate(xi) @ self.coef

    def sample(self, n: int = 2000, seed: int = 0) -> np.ndarray:
        """Draw option prices from the surrogate.

        Parameters
        ----------
        n : number of samples.
        seed : random seed.

        Returns
        -------
        Non-negative prices, shape (n, n_strikes).
        """
        xi = np.random.default_rng(seed).standard_normal((self.basis.dim, n))
        return np.maximum(self.predict(xi), 0.0)


class PCESurrogate:
    """Polynomial-chaos surrogate of option prices over the posterior.

    The expansion uses orthonormal Hermite polynomials in an independent
    standard-normal germ, mapped to the parameters by :class:`InputModel`.

    Parameters
    ----------
    model : ``"heston"`` or ``"bates"``.
    order : total polynomial degree.
    method : ``"regression"`` (least squares on scrambled Sobol points) or
        ``"quadrature"`` (tensor Gauss–Hermite projection, ``(order+1)^d`` runs, thesis).
    dependence : ``"gaussian_copula"`` (default) or ``"independent"`` (thesis).
    n_samples : regression points (default ``3 x #terms``, rounded up to a power of 2).
    candidates : scipy distributions tried for each marginal.
    pricer : posterior pricer (risk premium, quadrature settings).
    """

    def __init__(self, model: str, order: int = 3, method: str = "regression",
                 dependence: str = "gaussian_copula", n_samples: int | None = None,
                 candidates: Sequence[str] = DEFAULT_CANDIDATES,
                 pricer: PosteriorPricer | None = None) -> None:
        """Validate and store the surrogate settings."""
        if method not in ("regression", "quadrature"):
            raise ValueError("method must be 'regression' or 'quadrature'")
        self.model = model
        self.order = order
        self.method = method
        self.dependence = dependence
        self.n_samples = n_samples
        self.candidates = candidates
        self.pricer = pricer or PosteriorPricer(model)

    def _nodes(self, dim: int, n_terms: int, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
        """Return germ nodes, shape (dim, n), and quadrature weights (or ``None``)."""
        if self.method == "regression":
            n = self.n_samples or 3 * n_terms
            m = int(np.ceil(np.log2(n)))  # Sobol balance properties need a power of two
            u = qmc.Sobol(dim, scramble=True, seed=seed).random_base2(m)
            return st.norm.ppf(np.clip(u, 1e-12, 1 - 1e-12)).T, None
        x, w = np.polynomial.hermite_e.hermegauss(self.order + 1)
        w = w / np.sqrt(2.0 * np.pi)
        nodes = np.array(list(itertools.product(x, repeat=dim))).T
        weights = np.prod(np.array(list(itertools.product(w, repeat=dim))), axis=1)
        return nodes, weights

    def fit(self, draws: pd.DataFrame, option: OptionSpec,
            fixed: dict[str, float] | None = None, seed: int = 0) -> PCEResult:
        """Fit the surrogate.

        Parameters
        ----------
        draws : posterior draws (see :func:`posterior_draws`).
        option : option chain specification.
        fixed : parameters to hold constant instead of treating as uncertain.
        seed : random seed for the regression points.

        Returns
        -------
        PCEResult
        """
        fixed = dict(fixed or {})
        names = [p for p in pricing_inputs(self.model) if p not in fixed]
        inputs = InputModel.from_draws(draws, names, self.dependence, self.candidates)
        basis = HermiteBasis(len(names), self.order)
        nodes, weights = self._nodes(len(names), len(basis), seed)
        table = pd.DataFrame(inputs.transform(nodes).T, columns=names).assign(**fixed)
        evals = self.pricer.price_table(table, option)
        A = basis.evaluate(nodes)
        loo = None
        if weights is None:
            coef = np.linalg.lstsq(A, evals, rcond=None)[0]
            loo = self._loo_error(A, coef, evals)
        else:
            coef = A.T @ (weights[:, None] * evals)
        return PCEResult(inputs, basis, coef, evals.shape[0], option.K, loo)

    @staticmethod
    def _loo_error(A: np.ndarray, coef: np.ndarray, evals: np.ndarray) -> np.ndarray:
        """Leave-one-out error of a least-squares fit, relative to the output variance."""
        H = A @ np.linalg.pinv(A)
        resid = (evals - A @ coef) / (1.0 - np.clip(np.diag(H), None, 1 - 1e-10))[:, None]
        return np.mean(resid ** 2, axis=0) / np.maximum(evals.var(axis=0), 1e-300)
