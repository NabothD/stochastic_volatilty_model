"""Conjugate Bayesian-regression updates used inside the Gibbs sweep.

Each function takes the current latent variance path ``v`` (length n+1,
``v[0]`` = V_0) and the (jump-adjusted) gross returns ``R`` (length n,
``R[k-1] = S_k / S_{k-1}``) and returns one posterior draw.

Index convention (this fixes an off-by-one in the original scripts, which
regressed ``R_k`` on the *end-of-interval* variance ``v_k`` instead of the
variance ``v_{k-1}`` that actually drives the return over ``[k-1, k]``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .priors import Priors

_V_FLOOR = 1e-12


def _scale(v_prev: np.ndarray, dt: float) -> np.ndarray:
    return np.sqrt(dt * np.maximum(v_prev, _V_FLOOR))


def draw_mu(R: np.ndarray, v: np.ndarray, dt: float, pr: Priors, rng: np.random.Generator) -> float:
    """Drift via ``y = eta * x + eps``, ``eta = 1 + mu dt`` (thesis Eqs. 28–34)."""
    s = _scale(v[:-1], dt)
    x = 1.0 / s
    y = R / s
    xx, xy = x @ x, x @ y
    tau = pr.eta_prec + xx
    mean = (pr.eta_prec * pr.eta_mean + xy) / tau
    eta = rng.normal(mean, 1.0 / np.sqrt(tau))
    return (eta - 1.0) / dt


@dataclass
class VarianceDraw:
    kappa: float
    theta: float
    sigma: float


def draw_kappa_theta_sigma(v: np.ndarray, dt: float, pr: Priors, rng: np.random.Generator,
                           max_tries: int = 200, theta_max: float = 4.0) -> VarianceDraw:
    """Normal–inverse-gamma regression for ``(beta1, beta2, sigma^2)`` (Eqs. 35–40).

    ``sigma^2`` is drawn from its marginal posterior first and then ``beta``
    conditionally on it, which is the correct joint draw (the original code
    used the *previous* sigma for beta).  Draws with ``kappa <= 0`` or
    ``theta <= 0`` are rejected, i.e. the prior is truncated to the admissible
    region, instead of clipping (clipping piles mass on the boundary).
    """
    s = _scale(v[:-1], dt)
    y = v[1:] / s
    X = np.column_stack((1.0 / s, v[:-1] / s))
    n = y.size
    L0, m0 = pr.beta_prec, pr.beta_mean
    Ln = X.T @ X + L0
    Ln_inv = np.linalg.inv(Ln)
    mn = Ln_inv @ (L0 @ m0 + X.T @ y)
    a = pr.sigma_a + 0.5 * n
    b = pr.sigma_b + 0.5 * (y @ y + m0 @ L0 @ m0 - mn @ Ln @ mn)
    b = max(b, 1e-300)
    chol = np.linalg.cholesky(0.5 * (Ln_inv + Ln_inv.T))
    for _ in range(max_tries):
        sigma2 = b / rng.gamma(a)                       # IG(a, b)
        beta = mn + np.sqrt(sigma2) * (chol @ rng.standard_normal(2))
        kappa = (1.0 - beta[1]) / dt
        if 0.0 < kappa < 1.0 / dt:
            theta = beta[0] / (kappa * dt)
            if 0.0 < theta < theta_max:
                return VarianceDraw(kappa, theta, np.sqrt(sigma2))
    # posterior mass is essentially outside the admissible region: fall back to clipping
    kappa = float(np.clip((1.0 - mn[1]) / dt, 1e-6, 1.0 / dt - 1e-6))
    theta = float(np.clip(mn[0] / (kappa * dt), 1e-8, theta_max))
    return VarianceDraw(kappa, theta, np.sqrt(b / (a - 1.0) if a > 1 else b / a))


def draw_rho(R: np.ndarray, v: np.ndarray, dt: float, mu: float, kappa: float, theta: float,
             pr: Priors, rng: np.random.Generator) -> float:
    """Correlation via ``e2 = psi * e1 + sqrt(omega) * e`` (Eqs. 41–42, Jacquier et al.).

    Uses the exact Normal–inverse-gamma posterior, including the prior term
    ``tau0*psi0^2 - tau_n*m_n^2`` in the omega scale that the original code
    omitted.
    """
    vp = v[:-1]
    s = _scale(vp, dt)
    e1 = (R - 1.0 - mu * dt) / s
    e2 = (v[1:] - vp - kappa * (theta - vp) * dt) / s
    A11, A12, A22 = e1 @ e1, e1 @ e2, e2 @ e2
    tau = pr.psi_prec + A11
    m = (pr.psi_prec * pr.psi_mean + A12) / tau
    a = pr.omega_a + 0.5 * e1.size
    b = pr.omega_b + 0.5 * (A22 + pr.psi_prec * pr.psi_mean ** 2 - tau * m ** 2)
    omega = max(b, 1e-300) / rng.gamma(a)
    psi = rng.normal(m, np.sqrt(omega / tau))
    return float(psi / np.sqrt(psi ** 2 + omega))


@dataclass
class JumpDraw:
    lam: float
    mu_j: float
    sigma_j: float


def draw_jump_params(J: np.ndarray, Z: np.ndarray, T: float, pr: Priors,
                     rng: np.random.Generator) -> JumpDraw:
    """Conjugate draw of ``(lam, mu_j, sigma_j)`` given sampled jump times/sizes.

    * ``lam | J ~ Gamma(lam_a + sum J, rate = lam_b + T)`` (Poisson count over T years)
    * ``(mu_j, sigma_j^2) | Z`` – Normal–inverse-gamma on the sizes of the jumps that occurred
    """
    J = np.asarray(J, dtype=bool)
    nj = int(J.sum())
    lam = rng.gamma(pr.lam_a + nj, 1.0 / (pr.lam_b + T))
    z = np.asarray(Z)[J]
    k0, m0 = pr.mu_j_prec, pr.mu_j_mean
    kn = k0 + nj
    zbar = z.mean() if nj else 0.0
    mn = (k0 * m0 + nj * zbar) / kn
    a = pr.sigma_j_a + 0.5 * nj
    ss = ((z - zbar) ** 2).sum() if nj else 0.0
    b = pr.sigma_j_b + 0.5 * ss + 0.5 * k0 * nj * (zbar - m0) ** 2 / kn
    sj2 = b / rng.gamma(a)
    mu_j = rng.normal(mn, np.sqrt(sj2 / kn))
    return JumpDraw(float(lam), float(mu_j), float(np.sqrt(sj2)))
