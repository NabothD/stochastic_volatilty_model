"""Conjugate prior hyper-parameters for the Heston / Bates Gibbs sampler.

The sampler works on re-parametrised quantities (thesis Section 4.3):

* ``eta = 1 + mu*dt``                       ~ N(eta_mean, 1/eta_prec)
* ``beta = (kappa*theta*dt, 1 - kappa*dt)`` | sigma^2 ~ N(beta_mean, sigma^2 * beta_prec^-1)
* ``sigma^2``                               ~ IG(sigma_a, sigma_b)
* ``psi = sigma*rho`` | omega               ~ N(psi_mean, omega / psi_prec)
* ``omega = sigma^2 (1 - rho^2)``           ~ IG(omega_a, omega_b)
* ``lam``                                   ~ Gamma(lam_a, rate=lam_b)          (Bates)
* ``mu_j`` | sigma_j^2                      ~ N(mu_j_mean, sigma_j^2 / mu_j_prec)(Bates)
* ``sigma_j^2``                             ~ IG(sigma_j_a, sigma_j_b)          (Bates)

Choosing these numbers directly is unintuitive (Table 3 of the thesis), so
:meth:`Priors.from_beliefs` maps statements such as "kappa is about 1 +/- 1"
onto them using the change-of-variables / delta-method arguments of
Section 4.6.  Two corrections relative to the thesis text are built in:

* Precisions are ``1/sd**2``.  The original scripts used ``1/sqrt(sd**2)``
  (i.e. ``1/sd``), which made the drift prior ~1000x too wide and the psi
  prior ~4x too wide relative to what was written in the report.
* Eq. (70) sets ``tau_psi = w0^2 / (sd_rho^2 (psi0^2 + w0)^3)``, but because the
  conditional prior is ``psi | omega ~ N(psi0, omega/tau_psi)`` the delta method
  actually requires ``tau_psi = w0^3 / (sd_rho^2 (psi0^2 + w0)^3)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .models import BatesParams, HestonParams, Params

MeanSd = tuple[float, float]


@dataclass
class Priors:
    """Conjugate prior hyper-parameters (see the module docstring for their meaning)."""

    eta_mean: float
    eta_prec: float
    beta_mean: np.ndarray
    beta_prec: np.ndarray
    sigma_a: float
    sigma_b: float
    psi_mean: float
    psi_prec: float
    omega_a: float
    omega_b: float
    # jump priors (ignored for Heston)
    lam_a: float = 2.0
    lam_b: float = 1.0
    mu_j_mean: float = 0.0
    mu_j_prec: float = 0.1
    sigma_j_a: float = 3.0
    sigma_j_b: float = 0.02
    # initial variance V_0 ~ Gamma with this mean / sd (None -> use theta)
    v0_mean: float | None = None
    v0_sd: float | None = None

    def __post_init__(self) -> None:
        """Coerce array shapes and check that scale / shape parameters are positive."""
        self.beta_mean = np.asarray(self.beta_mean, dtype=float).reshape(2)
        self.beta_prec = np.asarray(self.beta_prec, dtype=float).reshape(2, 2)
        for name in ("eta_prec", "sigma_a", "sigma_b", "psi_prec", "omega_a", "omega_b",
                     "lam_a", "lam_b", "mu_j_prec", "sigma_j_a", "sigma_j_b"):
            if getattr(self, name) <= 0:
                raise ValueError(f"prior hyper-parameter {name} must be > 0")

    def to_dict(self) -> dict:
        """Return the hyper-parameters as a JSON-serialisable dict."""
        d = asdict(self)
        d["beta_mean"] = self.beta_mean.tolist()
        d["beta_prec"] = self.beta_prec.tolist()
        return d

    def initial_params(self, model: str, dt: float) -> Params:
        """Return a starting point for the sampler: the prior means.

        Parameters
        ----------
        model : ``"heston"`` or ``"bates"``.
        dt : time step in years.

        Returns
        -------
        ``HestonParams`` or ``BatesParams``.
        """
        kappa = max((1.0 - self.beta_mean[1]) / dt, 1e-2)
        w0 = self.omega_b / max(self.omega_a - 1.0, 1e-3)
        base = dict(
            mu=(self.eta_mean - 1.0) / dt,
            kappa=kappa,
            theta=max(self.beta_mean[0] / (kappa * dt), 1e-4),
            sigma=float(np.sqrt(self.sigma_b / max(self.sigma_a - 1.0, 1e-3))),
            rho=float(np.clip(self.psi_mean / np.sqrt(self.psi_mean ** 2 + w0), -0.95, 0.95)),
        )
        if model == "heston":
            return HestonParams(**base)
        sigma_j = float(np.sqrt(self.sigma_j_b / max(self.sigma_j_a - 1.0, 1e-3)))
        return BatesParams(**base, lam=self.lam_a / self.lam_b, mu_j=self.mu_j_mean,
                           sigma_j=sigma_j)

    @classmethod
    def from_beliefs(
        cls,
        dt: float,
        mu: MeanSd = (0.05, 0.5),
        kappa: MeanSd = (2.0, 2.0),
        theta: MeanSd = (0.04, 0.04),
        sigma: float = 0.3,
        sigma_strength: float = 3.0,
        rho: MeanSd = (-0.5, 0.3),
        omega_mean: float | None = None,
        lam: MeanSd = (1.0, 1.0),
        mu_j: MeanSd = (-0.05, 0.2),
        sigma_j: float = 0.1,
        sigma_j_strength: float = 3.0,
        v0: MeanSd | None = None,
    ) -> Priors:
        """Build hyper-parameters from intuitive (mean, sd) beliefs.

        Parameters
        ----------
        dt : time step in years.
        mu, kappa, theta, rho, lam, mu_j : (mean, sd) of each parameter.
        sigma, sigma_j : prior mean of the vol-of-vol / jump-size std.
        sigma_strength, sigma_j_strength : inverse-gamma shapes (> 1).
        omega_mean : prior mean of ``omega`` (default ``sigma^2 (1 - rho^2)``).
        v0 : optional (mean, sd) of the initial variance.

        Returns
        -------
        Priors

        ``sigma_strength`` / ``sigma_j_strength`` are the inverse-gamma shape
        parameters (must be > 1).  Shape ``a`` is worth roughly ``2a``
        observations, so small values (2–5) are weakly informative.  The
        thesis used ``a = 149`` (synthetic) and ``a = 250`` (S&P 100), i.e. a
        prior worth 300–500 observations, which largely fixed sigma at its
        prior mean.
        """
        if sigma_strength <= 1 or sigma_j_strength <= 1:
            raise ValueError("inverse-gamma shape must be > 1 for the mean to exist")
        # --- drift: eta = 1 + mu dt (Eq. 56) ---------------------------------
        eta_mean = 1.0 + mu[0] * dt
        eta_prec = 1.0 / (mu[1] * dt) ** 2

        # --- beta = (kappa theta dt, 1 - kappa dt) via the delta method -------
        k0, sk = kappa
        t0, st_ = theta
        beta_mean = np.array([k0 * t0 * dt, 1.0 - k0 * dt])
        cov = dt ** 2 * np.array([
            [t0 ** 2 * sk ** 2 + k0 ** 2 * st_ ** 2, -t0 * sk ** 2],
            [-t0 * sk ** 2, sk ** 2],
        ])
        # beta | sigma^2 ~ N(m0, sigma^2 Lambda0^-1)  =>  Lambda0 = sigma0^2 cov^-1
        beta_prec = sigma ** 2 * np.linalg.inv(cov)

        # --- sigma^2 ~ IG(a, b) with E[sigma^2] = sigma0^2 -------------------
        sigma_a = sigma_strength
        sigma_b = sigma ** 2 * (sigma_a - 1.0)

        # --- rho through (psi, omega), thesis Eq. 71 (corrected) -------------
        m_rho, s_rho = rho
        w0 = omega_mean if omega_mean is not None else sigma ** 2 * (1.0 - m_rho ** 2)
        omega_a = 2.0
        omega_b = w0 * (omega_a - 1.0)
        psi_mean = m_rho * np.sqrt(w0) / np.sqrt(1.0 - m_rho ** 2)
        psi_prec = w0 ** 3 / (s_rho ** 2 * (psi_mean ** 2 + w0) ** 3)

        # --- jumps -----------------------------------------------------------
        lam_a = (lam[0] / lam[1]) ** 2
        lam_b = lam[0] / lam[1] ** 2
        sigma_j_b = sigma_j ** 2 * (sigma_j_strength - 1.0)
        mu_j_prec = sigma_j ** 2 / mu_j[1] ** 2

        v0_mean, v0_sd = (v0 if v0 is not None else (None, None))
        return cls(eta_mean, eta_prec, beta_mean, beta_prec, sigma_a, sigma_b,
                   psi_mean, psi_prec, omega_a, omega_b,
                   lam_a, lam_b, mu_j[0], mu_j_prec, sigma_j_strength, sigma_j_b,
                   v0_mean, v0_sd)

    @classmethod
    def thesis_synthetic(cls, dt: float = 1 / 252) -> Priors:
        """Return the hyper-parameters of thesis Table 3 (precisions as written there).

        Parameters
        ----------
        dt : time step in years (unused, kept for a uniform signature).

        Returns
        -------
        Priors
        """
        return cls(
            eta_mean=1.00125, eta_prec=1.0 / 0.001 ** 2,
            beta_mean=[35e-6, 0.988], beta_prec=[[10.0, 0.0], [0.0, 5.0]],
            sigma_a=149.0, sigma_b=0.025,
            psi_mean=-0.45, psi_prec=1.0 / 0.3 ** 2,
            omega_a=1.03, omega_b=0.05,
            lam_a=2.0, lam_b=1.0, mu_j_mean=-0.3, mu_j_prec=1.0,
            sigma_j_a=3.0, sigma_j_b=0.08,
        )
