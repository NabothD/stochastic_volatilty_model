"""Conjugate Bayesian-regression draws used inside each Gibbs sweep.

Inputs are the latent variance path ``v`` (length n+1, ``v[0] = V_0``) and the
jump-adjusted gross returns ``R`` (length n, ``R[k-1] = S_k / S_{k-1}``). The
return over ``[k-1, k]`` is regressed on ``v[k-1]``, the variance that drives it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .priors import Priors

_V_FLOOR = 1e-12


@dataclass(frozen=True)
class VarianceDraw:
    """One draw of the variance-equation parameters."""

    kappa: float
    theta: float
    sigma: float


@dataclass(frozen=True)
class JumpDraw:
    """One draw of the jump parameters."""

    lam: float
    mu_j: float
    sigma_j: float


class ConjugateSampler:
    """Draws model parameters from their conjugate full conditionals.

    Parameters
    ----------
    priors : prior hyper-parameters.
    dt : time step in years.
    theta_max : upper bound of the admissible long-run variance.
    max_tries : rejection attempts before falling back to the posterior mean.
    """

    def __init__(self, priors: Priors, dt: float, theta_max: float = 4.0,
                 max_tries: int = 200) -> None:
        """Store the priors and sampler settings."""
        self.priors = priors
        self.dt = dt
        self.theta_max = theta_max
        self.max_tries = max_tries

    def _scale(self, v_prev: np.ndarray) -> np.ndarray:
        """Return ``sqrt(dt * v_{k-1})`` with a small floor."""
        return np.sqrt(self.dt * np.maximum(v_prev, _V_FLOOR))

    def draw_mu(self, R: np.ndarray, v: np.ndarray, rng: np.random.Generator) -> float:
        """Draw the drift from ``R_k/s_k = eta / s_k + eps`` with ``eta = 1 + mu dt``.

        Parameters
        ----------
        R : jump-adjusted gross returns, shape (n,).
        v : variance path, shape (n+1,).
        rng : random generator.

        Returns
        -------
        Draw of ``mu``.
        """
        pr = self.priors
        s = self._scale(v[:-1])
        x, y = 1.0 / s, R / s
        tau = pr.eta_prec + x @ x
        mean = (pr.eta_prec * pr.eta_mean + x @ y) / tau
        return float((rng.normal(mean, 1.0 / np.sqrt(tau)) - 1.0) / self.dt)

    def draw_variance_params(self, v: np.ndarray, rng: np.random.Generator) -> VarianceDraw:
        """Draw ``(kappa, theta, sigma)`` from the Normal–inverse-gamma regression.

        ``sigma^2`` is drawn from its marginal posterior and then
        ``beta = (kappa theta dt, 1 - kappa dt)`` given ``sigma^2``. Draws outside
        ``0 < kappa < 1/dt``, ``0 < theta < theta_max`` are rejected (truncated prior).

        Parameters
        ----------
        v : variance path, shape (n+1,).
        rng : random generator.

        Returns
        -------
        VarianceDraw
        """
        pr, dt = self.priors, self.dt
        s = self._scale(v[:-1])
        y = v[1:] / s
        X = np.column_stack((1.0 / s, v[:-1] / s))
        L0, m0 = pr.beta_prec, pr.beta_mean
        Ln = X.T @ X + L0
        Ln_inv = np.linalg.inv(Ln)
        mn = Ln_inv @ (L0 @ m0 + X.T @ y)
        a = pr.sigma_a + 0.5 * y.size
        b = max(pr.sigma_b + 0.5 * (y @ y + m0 @ L0 @ m0 - mn @ Ln @ mn), 1e-300)
        chol = np.linalg.cholesky(0.5 * (Ln_inv + Ln_inv.T))
        for _ in range(self.max_tries):
            sigma2 = b / rng.gamma(a)
            beta = mn + np.sqrt(sigma2) * (chol @ rng.standard_normal(2))
            kappa = (1.0 - beta[1]) / dt
            if 0.0 < kappa < 1.0 / dt:
                theta = beta[0] / (kappa * dt)
                if 0.0 < theta < self.theta_max:
                    return VarianceDraw(float(kappa), float(theta), float(np.sqrt(sigma2)))
        kappa = float(np.clip((1.0 - mn[1]) / dt, 1e-6, 1.0 / dt - 1e-6))
        theta = float(np.clip(mn[0] / (kappa * dt), 1e-8, self.theta_max))
        return VarianceDraw(kappa, theta, float(np.sqrt(b / max(a - 1.0, 1.0))))

    def draw_rho(self, R: np.ndarray, v: np.ndarray, mu: float, kappa: float, theta: float,
                 rng: np.random.Generator) -> float:
        """Draw the correlation from ``e2 = psi e1 + sqrt(omega) e`` (Jacquier et al.).

        Parameters
        ----------
        R : jump-adjusted gross returns, shape (n,).
        v : variance path, shape (n+1,).
        mu, kappa, theta : current parameter values.
        rng : random generator.

        Returns
        -------
        Draw of ``rho = psi / sqrt(psi^2 + omega)``.
        """
        pr, dt = self.priors, self.dt
        vp = v[:-1]
        s = self._scale(vp)
        e1 = (R - 1.0 - mu * dt) / s
        e2 = (v[1:] - vp - kappa * (theta - vp) * dt) / s
        tau = pr.psi_prec + e1 @ e1
        m = (pr.psi_prec * pr.psi_mean + e1 @ e2) / tau
        a = pr.omega_a + 0.5 * e1.size
        b = pr.omega_b + 0.5 * (e2 @ e2 + pr.psi_prec * pr.psi_mean ** 2 - tau * m ** 2)
        omega = max(b, 1e-300) / rng.gamma(a)
        psi = rng.normal(m, np.sqrt(omega / tau))
        return float(psi / np.sqrt(psi ** 2 + omega))

    def draw_jump_params(self, J: np.ndarray, Z: np.ndarray, T: float,
                         rng: np.random.Generator) -> JumpDraw:
        """Draw ``(lam, mu_j, sigma_j)`` given sampled jump times and sizes.

        ``lam ~ Gamma(lam_a + #jumps, rate = lam_b + T)`` and
        ``(mu_j, sigma_j^2)`` from the Normal–inverse-gamma posterior of the sizes.

        Parameters
        ----------
        J : jump indicators, shape (n,).
        Z : jump sizes (only entries with ``J = 1`` are used), shape (n,).
        T : time span of the sample in years.
        rng : random generator.

        Returns
        -------
        JumpDraw
        """
        pr = self.priors
        mask = np.asarray(J, dtype=bool)
        nj = int(mask.sum())
        z = np.asarray(Z)[mask]
        lam = rng.gamma(pr.lam_a + nj, 1.0 / (pr.lam_b + T))
        zbar = z.mean() if nj else 0.0
        kn = pr.mu_j_prec + nj
        mn = (pr.mu_j_prec * pr.mu_j_mean + nj * zbar) / kn
        ss = ((z - zbar) ** 2).sum() if nj else 0.0
        b = pr.sigma_j_b + 0.5 * ss + 0.5 * pr.mu_j_prec * nj * (zbar - pr.mu_j_mean) ** 2 / kn
        sj2 = b / rng.gamma(pr.sigma_j_a + 0.5 * nj)
        mu_j = rng.normal(mn, np.sqrt(sj2 / kn))
        return JumpDraw(float(lam), float(mu_j), float(np.sqrt(sj2)))
