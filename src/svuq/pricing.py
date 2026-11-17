"""Vectorised European option pricing under Heston and Bates (risk-neutral).

Prices use the Lewis (2001) single-integral formula::

    C = S e^{-qT} - sqrt(S K) e^{-(r+q)T/2} / pi
        * int_0^inf Re[e^{iux} phi(u - i/2)] / (u^2 + 1/4) du,   x = ln(S/K) + (r - q) T

with ``phi`` the characteristic function of ``ln(S_T/S_0) - (r-q)T``. The
Heston CF uses the stable "little Heston trap" form. The Bates CF includes
the jump compensator (thesis Eq. 53 omits it).

:class:`FourierPricer` with ``method="fixed"`` evaluates the integral on a
composite Gauss–Legendre grid shared by all strikes and parameter sets, so a
whole option chain for hundreds of posterior draws costs two matrix products.
``method="quad"`` uses adaptive quadrature and serves as the reference.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize
from scipy.stats import norm

from .models import Params

ArrayLike = float | np.ndarray


def heston_cf(u, T, v0, kappa, theta, sigma, rho):
    """Characteristic function of ``ln(S_T/S_0) - (r-q)T`` under Heston.

    Parameters
    ----------
    u : complex argument (array, broadcasts with the parameters).
    T : maturity in years.
    v0, kappa, theta, sigma, rho : Heston parameters (scalars or broadcastable arrays).

    Returns
    -------
    Complex array of ``E[exp(i u X_T)]``.
    """
    iu = 1j * u
    a = kappa - rho * sigma * iu
    d = np.sqrt(a * a + sigma ** 2 * (iu + u * u))
    g = (a - d) / (a + d)
    edt = np.exp(-d * T)
    C = kappa * theta / sigma ** 2 * ((a - d) * T - 2.0 * np.log((1.0 - g * edt) / (1.0 - g)))
    D = (a - d) / sigma ** 2 * (1.0 - edt) / (1.0 - g * edt)
    return np.exp(C + D * v0)


def bates_cf(u, T, v0, kappa, theta, sigma, rho, lam, mu_j, sigma_j):
    """Characteristic function of ``ln(S_T/S_0) - (r-q)T`` under Bates.

    Parameters
    ----------
    u : complex argument.
    T : maturity in years.
    v0, kappa, theta, sigma, rho : Heston parameters.
    lam, mu_j, sigma_j : jump intensity, mean and std of the log-jump.

    Returns
    -------
    Complex array of ``E[exp(i u X_T)]``.
    """
    iu = 1j * u
    jump = lam * T * (np.exp(iu * mu_j - 0.5 * sigma_j ** 2 * u * u) - 1.0) \
        - iu * lam * T * (np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0)
    return heston_cf(u, T, v0, kappa, theta, sigma, rho) * np.exp(jump)


def _gauss_legendre_panels(u_max: float, max_width: float, n_nodes: int, max_panels: int):
    """Composite Gauss–Legendre rule on ``[0, u_max]`` with geometrically growing panels.

    Panels start narrow (the integrand peaks at ``u = 0`` with width ~0.5) and
    grow by 30% per panel up to ``max_width`` (a fraction of the oscillation period).

    Parameters
    ----------
    u_max : upper integration limit.
    max_width : largest allowed panel width.
    n_nodes : nodes per panel.
    max_panels : cap on the number of panels.

    Returns
    -------
    nodes, weights : arrays of equal length.
    """
    edges = [0.0]
    h = min(0.25, max_width)
    while edges[-1] < u_max and len(edges) <= max_panels:
        edges.append(min(edges[-1] + h, u_max))
        h = min(1.3 * h, max_width)
    edges = np.asarray(edges)
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[1:] + edges[:-1])
    return (mid[:, None] + half[:, None] * x).ravel(), (half[:, None] * w).ravel()


class FourierPricer:
    """European call pricer for Heston / Bates via the Lewis formula.

    Parameters
    ----------
    method : ``"fixed"`` (vectorised composite Gauss–Legendre) or ``"quad"`` (adaptive).
    nodes_per_panel : Gauss–Legendre nodes per panel (fixed method).
    tol : truncation tolerance on the integrand (fixed method).
    max_panels : cap on the number of panels (fixed method).
    """

    def __init__(self, method: str = "fixed", nodes_per_panel: int = 16, tol: float = 1e-13,
                 max_panels: int = 2048) -> None:
        """Validate and store the quadrature settings."""
        if method not in ("fixed", "quad"):
            raise ValueError("method must be 'fixed' or 'quad'")
        self.method = method
        self.nodes_per_panel = nodes_per_panel
        self.tol = tol
        self.max_panels = max_panels

    @staticmethod
    def _cf(u, T, table: dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate the CF for a batch of parameter sets (rows) at nodes ``u`` (columns)."""
        p = {k: np.asarray(v, dtype=float)[:, None] for k, v in table.items()}
        heston = (p["v0"], p["kappa"], p["theta"], p["sigma"], p["rho"])
        if "lam" in p and np.any(p["lam"] > 0):
            return bates_cf(u, T, *heston, p["lam"], p["mu_j"], np.maximum(p["sigma_j"], 1e-8))
        return heston_cf(u, T, *heston)

    def _truncation(self, T: float, table: dict[str, np.ndarray]) -> float:
        """Return an upper integration limit beyond which the integrand is negligible."""
        grid = np.geomspace(0.1, 1e5, 160)
        mag = np.abs(self._cf(grid[None, :] - 0.5j, T, table)) / (grid ** 2 + 0.25)
        above = np.flatnonzero((mag > self.tol).any(axis=0))
        return float(grid[min(above[-1] + 1, grid.size - 1)]) if above.size else float(grid[0])

    def call_batch(self, S: float, K: ArrayLike, T: float, r: float,
                   table: dict[str, ArrayLike], q: float = 0.0) -> np.ndarray:
        """Price calls for many parameter sets and strikes at once.

        Parameters
        ----------
        S : spot price.
        K : strike or array of strikes, shape (n_K,).
        T : maturity in years.
        r : risk-free rate (continuous).
        table : parameter arrays of length B: ``v0, kappa, theta, sigma, rho`` and
            optionally ``lam, mu_j, sigma_j``.
        q : continuous dividend yield.

        Returns
        -------
        Call prices, shape (B, n_K).
        """
        K = np.atleast_1d(np.asarray(K, dtype=float))
        table = {k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in table.items()}
        B = table["v0"].size
        x = np.log(S / K) + (r - q) * T
        if self.method == "quad":
            integral = np.array([[self._quad(T, {k: v[b:b + 1] for k, v in table.items()}, xi)
                                  for xi in x] for b in range(B)])
        else:
            u_max = self._truncation(T, table)
            max_width = min(np.pi / (np.abs(x).max() + 1e-12), max(u_max / 16.0, 1.0))
            u, w = _gauss_legendre_panels(u_max, max_width, self.nodes_per_panel, self.max_panels)
            phi = self._cf(u[None, :] - 0.5j, T, table) * (w / (u * u + 0.25))[None, :]
            ux = np.outer(u, x)
            integral = phi.real @ np.cos(ux) - phi.imag @ np.sin(ux)
        price = S * np.exp(-q * T) - np.sqrt(S * K) * np.exp(-0.5 * (r + q) * T) * integral / np.pi
        lower = np.maximum(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        return np.clip(price, lower, S * np.exp(-q * T))

    def _quad(self, T: float, table: dict[str, np.ndarray], x: float) -> float:
        """Adaptive-quadrature value of the Lewis integral for one parameter set and strike."""
        def integrand(u: float) -> float:
            return float((np.exp(1j * u * x) * self._cf(np.array([u - 0.5j]), T, table)[0, 0]).real
                         / (u * u + 0.25))
        return integrate.quad(integrand, 0.0, np.inf, limit=500, epsabs=1e-12, epsrel=1e-10)[0]

    def call(self, S: float, K: ArrayLike, T: float, r: float, params: Params, v0: float,
             q: float = 0.0) -> ArrayLike:
        """Price calls for one parameter set.

        Parameters
        ----------
        S : spot price.
        K : strike or array of strikes.
        T : maturity in years.
        r : risk-free rate.
        params : ``HestonParams`` or ``BatesParams`` (``mu`` is ignored: risk-neutral pricing).
        v0 : current variance.
        q : dividend yield.

        Returns
        -------
        Price (float) or array of prices matching ``K``.
        """
        table = {"v0": v0, **{k: getattr(params, k) for k in
                              ("kappa", "theta", "sigma", "rho", "lam", "mu_j", "sigma_j")}}
        out = self.call_batch(S, K, T, r, table, q)[0]
        return out if np.ndim(K) else float(out[0])


_DEFAULT_PRICER = FourierPricer()


def heston_call(S: float, K: ArrayLike, T: float, r: float, v0: float, kappa: float,
                theta: float, sigma: float, rho: float, q: float = 0.0,
                pricer: FourierPricer | None = None) -> ArrayLike:
    """Price a European call under Heston.

    Parameters
    ----------
    S, K, T, r : spot, strike(s), maturity (years), risk-free rate.
    v0, kappa, theta, sigma, rho : Heston parameters (risk-neutral).
    q : dividend yield.
    pricer : optional :class:`FourierPricer` (default: fixed quadrature).

    Returns
    -------
    Price (float) or array matching ``K``.
    """
    table = dict(v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
    out = (pricer or _DEFAULT_PRICER).call_batch(S, K, T, r, table, q)[0]
    return out if np.ndim(K) else float(out[0])


def bates_call(S: float, K: ArrayLike, T: float, r: float, v0: float, kappa: float,
               theta: float, sigma: float, rho: float, lam: float, mu_j: float,
               sigma_j: float, q: float = 0.0, pricer: FourierPricer | None = None) -> ArrayLike:
    """Price a European call under Bates (Heston + log-normal jumps).

    Parameters
    ----------
    S, K, T, r : spot, strike(s), maturity (years), risk-free rate.
    v0, kappa, theta, sigma, rho : Heston parameters (risk-neutral).
    lam, mu_j, sigma_j : jump intensity, mean and std of the log-jump.
    q : dividend yield.
    pricer : optional :class:`FourierPricer`.

    Returns
    -------
    Price (float) or array matching ``K``.
    """
    table = dict(v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho, lam=lam, mu_j=mu_j,
                 sigma_j=sigma_j)
    out = (pricer or _DEFAULT_PRICER).call_batch(S, K, T, r, table, q)[0]
    return out if np.ndim(K) else float(out[0])


def black_scholes_call(S: float, K: ArrayLike, T: float, r: float, vol: ArrayLike,
                       q: float = 0.0) -> ArrayLike:
    """Price a European call under Black–Scholes.

    Parameters
    ----------
    S, K, T, r : spot, strike(s), maturity (years), risk-free rate.
    vol : volatility.
    q : dividend yield.

    Returns
    -------
    Price (float) or array.
    """
    K = np.asarray(K, dtype=float)
    sq = vol * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / sq
    out = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - sq)
    return out if np.ndim(out) else float(out)


def implied_volatility(price: float, S: float, K: float, T: float, r: float, q: float = 0.0,
                       lo: float = 1e-6, hi: float = 5.0) -> float:
    """Black–Scholes implied volatility by Brent's method.

    Parameters
    ----------
    price : call price.
    S, K, T, r, q : spot, strike, maturity, rate, dividend yield.
    lo, hi : search bracket for the volatility.

    Returns
    -------
    Implied volatility, or ``nan`` if the price violates no-arbitrage bounds.
    """
    lower = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    if not lower < price < S * np.exp(-q * T):
        return float("nan")
    try:
        return float(optimize.brentq(lambda s: black_scholes_call(S, K, T, r, s, q) - price,
                                     lo, hi, xtol=1e-12))
    except ValueError:
        return float("nan")


def intrinsic_value(S_T: float, K: ArrayLike, T: float, r: float) -> ArrayLike:
    """Discounted realised payoff ``e^{-rT} max(S_T - K, 0)`` (thesis Eq. 78).

    Parameters
    ----------
    S_T : price at expiry.
    K : strike(s).
    T : maturity in years.
    r : risk-free rate.

    Returns
    -------
    Discounted payoff(s).
    """
    return np.exp(-r * T) * np.maximum(S_T - np.asarray(K, dtype=float), 0.0)
