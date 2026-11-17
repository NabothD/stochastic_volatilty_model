"""Semi-analytic European option pricing under Heston and Bates (risk-neutral).

Prices use the Lewis (2001) single-integral representation

    C = S e^{-qT} - sqrt(S K) e^{-(r+q)T/2} / pi
        * int_0^inf Re[ e^{i u x} phi(u - i/2) ] / (u^2 + 1/4) du,
    x = ln(S/K) + (r - q) T,

where ``phi`` is the characteristic function of ``ln(S_T/S_0) - (r-q)T``.
The Heston characteristic function is written in the "little Heston trap"
form (Albrecher et al. 2007), which is numerically stable for long
maturities.  The Bates characteristic function includes the jump
*compensator* ``-i u lam T (exp(mu_j + sigma_j^2/2) - 1)``, which keeps the
discounted price a martingale (note: thesis Eq. 53 omits this term; QuantLib's
``BatesEngine`` – used for the thesis results – does include it).

No QuantLib dependency is needed; :func:`heston_call_quantlib` /
:func:`bates_call_quantlib` are provided only for cross-checking.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy import integrate, optimize
from scipy.stats import norm

ArrayLike = Union[float, np.ndarray]


def heston_cf(u, T, v0, kappa, theta, sigma, rho):
    """Characteristic function of ``ln(S_T/S_0) - (r-q)T`` under Heston."""
    iu = 1j * u
    a = kappa - rho * sigma * iu
    d = np.sqrt(a * a + sigma ** 2 * (iu + u * u))
    g = (a - d) / (a + d)
    edt = np.exp(-d * T)
    C = kappa * theta / sigma ** 2 * ((a - d) * T - 2.0 * np.log((1.0 - g * edt) / (1.0 - g)))
    D = (a - d) / sigma ** 2 * (1.0 - edt) / (1.0 - g * edt)
    return np.exp(C + D * v0)


def bates_cf(u, T, v0, kappa, theta, sigma, rho, lam, mu_j, sigma_j):
    iu = 1j * u
    jump = lam * T * (np.exp(iu * mu_j - 0.5 * sigma_j ** 2 * u * u) - 1.0) \
        - iu * lam * T * (np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0)
    return heston_cf(u, T, v0, kappa, theta, sigma, rho) * np.exp(jump)


def _lewis_call(cf, S, K, T, r, q):
    x = np.log(S / K) + (r - q) * T

    def integrand(u):
        return (np.exp(1j * u * x) * cf(u - 0.5j)).real / (u * u + 0.25)

    val, _ = integrate.quad(integrand, 0.0, np.inf, limit=500, epsabs=1e-12, epsrel=1e-10)
    price = S * np.exp(-q * T) - np.sqrt(S * K) * np.exp(-0.5 * (r + q) * T) * val / np.pi
    # clamp tiny negative values from quadrature error to the no-arbitrage bounds
    lower = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    return float(min(max(price, lower), S * np.exp(-q * T)))


def _vectorise(fn, S, K, T, *args):
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))
    out = np.array([fn(S, k, T, *args) for k in K_arr])
    return out if np.ndim(K) else float(out[0])


def heston_call(S: float, K: ArrayLike, T: float, r: float, v0: float, kappa: float,
                theta: float, sigma: float, rho: float, q: float = 0.0) -> ArrayLike:
    """European call under Heston (``K`` may be an array)."""
    def one(S, k, T):
        cf = lambda u: heston_cf(u, T, v0, kappa, theta, sigma, rho)
        return _lewis_call(cf, S, k, T, r, q)
    return _vectorise(one, S, K, T)


def bates_call(S: float, K: ArrayLike, T: float, r: float, v0: float, kappa: float,
               theta: float, sigma: float, rho: float, lam: float, mu_j: float,
               sigma_j: float, q: float = 0.0) -> ArrayLike:
    """European call under Bates (Heston + log-normal jumps)."""
    def one(S, k, T):
        cf = lambda u: bates_cf(u, T, v0, kappa, theta, sigma, rho, lam, mu_j, sigma_j)
        return _lewis_call(cf, S, k, T, r, q)
    return _vectorise(one, S, K, T)


def black_scholes_call(S: float, K: ArrayLike, T: float, r: float, vol: float,
                       q: float = 0.0) -> ArrayLike:
    K = np.asarray(K, dtype=float)
    sq = vol * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / sq
    d2 = d1 - sq
    out = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return out if out.ndim else float(out)


def implied_volatility(price: float, S: float, K: float, T: float, r: float,
                       q: float = 0.0, lo: float = 1e-6, hi: float = 5.0) -> float:
    """Black–Scholes implied volatility by Brent's method (``nan`` if out of bounds)."""
    lower = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    if not lower < price < S * np.exp(-q * T):
        return float("nan")
    f = lambda s: black_scholes_call(S, K, T, r, s, q) - price
    try:
        return float(optimize.brentq(f, lo, hi, xtol=1e-12))
    except ValueError:
        return float("nan")


def intrinsic_value(S_T: float, K: ArrayLike, T: float, r: float) -> ArrayLike:
    """Discounted realised payoff ``e^{-rT} max(S_T - K, 0)`` (thesis Eq. 78)."""
    return np.exp(-r * T) * np.maximum(S_T - np.asarray(K, dtype=float), 0.0)


# ----------------------------------------------------------------------------
# Optional QuantLib cross-check
# ----------------------------------------------------------------------------
def _ql_setup(S, T, r, q):
    import QuantLib as ql

    today = ql.Date(1, 1, 2020)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    # exact maturity in days (the old scripts truncated with int(T*365))
    maturity = today + max(int(round(T * 365)), 1)
    spot = ql.QuoteHandle(ql.SimpleQuote(S))
    rts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc))
    dts = ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc))
    return ql, maturity, spot, rts, dts


def heston_call_quantlib(S, K, T, r, v0, kappa, theta, sigma, rho, q=0.0):
    ql, mat, spot, rts, dts = _ql_setup(S, T, r, q)
    proc = ql.HestonProcess(rts, dts, spot, v0, kappa, theta, sigma, rho)
    opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, float(K)), ql.EuropeanExercise(mat))
    opt.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(proc)))
    return opt.NPV()


def bates_call_quantlib(S, K, T, r, v0, kappa, theta, sigma, rho, lam, mu_j, sigma_j, q=0.0):
    ql, mat, spot, rts, dts = _ql_setup(S, T, r, q)
    proc = ql.BatesProcess(rts, dts, spot, v0, kappa, theta, sigma, rho, lam, mu_j, sigma_j)
    opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, float(K)), ql.EuropeanExercise(mat))
    opt.setPricingEngine(ql.BatesEngine(ql.BatesModel(proc)))
    return opt.NPV()
