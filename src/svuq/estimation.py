"""Bayesian estimation of Heston / Bates parameters from a price series.

The sampler follows the two-stage MCMC + SIR scheme used
in the thesis (Section 4.7): each sweep

1. runs the particle filter for the latent variance (and jumps) given the
   current parameters,
2. (Bates) draws jump times/sizes and the jump parameters, and removes the
   sampled jumps from the returns,
3. picks a variance path – either the filtered mean (``path="mean"``, as in
   the thesis) or a trajectory drawn by backward sampling
   (``path="ffbs"``, the default) – and
4. draws ``mu``, ``(kappa, theta, sigma)`` and ``rho`` from their conjugate
   regression posteriors given that path.

The stored draws after ``burn_in`` are the posterior sample used for
uncertainty quantification.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from . import bayes
from .diagnostics import effective_sample_size
from .models import BATES_PARAMS, HESTON_PARAMS, BatesParams, HestonParams
from .particle_filter import backward_sample, run_filter
from .priors import Priors


@dataclass
class PosteriorResult:
    model: str
    dt: float
    burn_in: int
    chain: pd.DataFrame                 # every sweep (incl. burn-in)
    vol_paths: np.ndarray               # variance path used in each sweep, (n_iter, n+1)
    filtered_mean: np.ndarray           # filtered E[V_k] in each sweep, (n_iter, n+1)
    jump_prob: Optional[np.ndarray]     # filtered P(J_k=1) in each sweep (Bates)
    settings: Dict = field(default_factory=dict)

    @property
    def param_names(self) -> Sequence[str]:
        return BATES_PARAMS if self.model == "bates" else HESTON_PARAMS

    @property
    def samples(self) -> pd.DataFrame:
        """Posterior draws after burn-in."""
        return self.chain.iloc[self.burn_in:].reset_index(drop=True)

    @property
    def variance_path(self) -> np.ndarray:
        """Posterior mean of the latent variance path."""
        return self.filtered_mean[self.burn_in:].mean(axis=0)

    @property
    def v_final(self) -> tuple:
        """Mean and sd of the latent variance at the last observation (use as v0 for pricing).

        The sd combines the within-sweep filter spread and the between-sweep
        (parameter) uncertainty.
        """
        s = self.samples
        m = s["v_T"].to_numpy()
        sd = s["v_T_sd"].to_numpy()
        return float(m.mean()), float(np.sqrt(np.mean(sd ** 2) + m.var()))

    def point_estimate(self) -> Union[HestonParams, BatesParams]:
        s = self.samples
        kw = {p: float(s[p].mean()) for p in self.param_names}
        return BatesParams(**kw) if self.model == "bates" else HestonParams(**kw)

    def summary(self, true_values: Optional[Union[Dict[str, float], HestonParams]] = None,
                ci: float = 0.95) -> pd.DataFrame:
        if isinstance(true_values, HestonParams):
            true_values = true_values.to_dict()
        lo, hi = 50 * (1 - ci), 50 * (1 + ci)
        rows = []
        for p in self.param_names:
            x = self.samples[p].to_numpy()
            row = {
                "param": p, "mean": x.mean(), "sd": x.std(ddof=1), "median": np.median(x),
                f"q{lo:g}": np.percentile(x, lo), f"q{hi:g}": np.percentile(x, hi),
                "ess": effective_sample_size(x),
            }
            if true_values is not None and p in true_values:
                tv = true_values[p]
                row["true"] = tv
                row["rel_err_%"] = 100 * abs(x.mean() - tv) / abs(tv) if tv != 0 else np.nan
                row["in_ci"] = bool(row[f"q{lo:g}"] <= tv <= row[f"q{hi:g}"])
            rows.append(row)
        return pd.DataFrame(rows).set_index("param")

    def save(self, prefix: str) -> None:
        """Write ``<prefix>_chain.csv``, ``<prefix>_variance.csv`` and ``<prefix>_summary.csv``."""
        self.chain.to_csv(f"{prefix}_chain.csv", index_label="iter")
        pd.DataFrame({
            "posterior_mean_variance": self.variance_path,
            "posterior_sd_variance": self.filtered_mean[self.burn_in:].std(axis=0),
        }).to_csv(f"{prefix}_variance.csv", index_label="k")
        self.summary().to_csv(f"{prefix}_summary.csv")


def _initial_params(model: str, pr: Priors, dt: float) -> Union[HestonParams, BatesParams]:
    mu = (pr.eta_mean - 1.0) / dt
    kappa = max((1.0 - pr.beta_mean[1]) / dt, 1e-2)
    theta = max(pr.beta_mean[0] / (kappa * dt), 1e-4)
    sigma = float(np.sqrt(pr.sigma_b / max(pr.sigma_a - 1.0, 1e-3)))
    w0 = pr.omega_b / max(pr.omega_a - 1.0, 1e-3)
    rho = float(np.clip(pr.psi_mean / np.sqrt(pr.psi_mean ** 2 + w0), -0.95, 0.95))
    base = dict(mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
    if model == "bates":
        return BatesParams(**base, lam=pr.lam_a / pr.lam_b, mu_j=pr.mu_j_mean,
                           sigma_j=float(np.sqrt(pr.sigma_j_b / max(pr.sigma_j_a - 1.0, 1e-3))))
    return HestonParams(**base)


def estimate(
    prices: np.ndarray,
    dt: float,
    model: str = "heston",
    priors: Optional[Priors] = None,
    n_iter: int = 1000,
    burn_in: Optional[int] = None,
    n_particles: int = 1000,
    init: Optional[Union[HestonParams, BatesParams]] = None,
    path: str = "ffbs",
    resampling: str = "smooth",
    jump_update: str = "gibbs",
    jump_proposal_prob: float = 0.1,
    seed: Optional[int] = None,
    progress: Union[bool, Callable[[int, Dict], None]] = False,
) -> PosteriorResult:
    """Run the particle-filter / Gibbs sampler.

    Parameters
    ----------
    prices : array of asset prices ``S_0..S_n`` (equally spaced by ``dt`` years).
    model : ``"heston"`` or ``"bates"``.
    priors : :class:`Priors`; defaults to weakly informative ``Priors.from_beliefs(dt)``.
    n_iter, burn_in : number of sweeps and how many to discard (default ``n_iter // 4``).
    n_particles : particles in the filter.
    path : ``"ffbs"`` (backward-sampled trajectory, recommended) or ``"mean"``
        (filtered mean path, as in the thesis – tends to under-estimate ``sigma``
        because the filtered mean is smoother than the true variance).
    resampling : ``"smooth"`` (thesis' refined resampling) or ``"systematic"``.
    jump_update : ``"gibbs"`` – sample jump times/sizes and draw ``(lam, mu_j,
        sigma_j)`` from conjugate posteriors (gives genuine posterior
        uncertainty), or ``"moments"`` – the thesis' moment estimates
        (Eqs. 80–83), corrected to divide by the time span and to use within-
        step jump-size variance.
    """
    if model not in ("heston", "bates"):
        raise ValueError("model must be 'heston' or 'bates'")
    if path not in ("ffbs", "mean"):
        raise ValueError("path must be 'ffbs' or 'mean'")
    if jump_update not in ("gibbs", "moments"):
        raise ValueError("jump_update must be 'gibbs' or 'moments'")
    if path == "mean":
        warnings.warn(
            "path='mean' regresses on the filtered-mean variance path (thesis method). Its "
            "increments are dominated by the rho-correlated propagation step, so rho drifts "
            "towards -1/+1 and sigma is biased low unless the priors are very tight. "
            "Use path='ffbs' for inference.", stacklevel=2)
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or prices.size < 10 or np.any(prices <= 0):
        raise ValueError("prices must be a 1-D array of > 10 positive values")
    rng = np.random.default_rng(seed)
    pr = priors if priors is not None else Priors.from_beliefs(dt)
    burn_in = n_iter // 4 if burn_in is None else burn_in
    if not 0 <= burn_in < n_iter:
        raise ValueError("burn_in must be in [0, n_iter)")

    R = prices[1:] / prices[:-1]
    n = R.size
    T = n * dt
    params = init if init is not None else _initial_params(model, pr, dt)
    if model == "bates" and not isinstance(params, BatesParams):
        params = BatesParams(**params.to_dict(), lam=pr.lam_a / pr.lam_b, mu_j=pr.mu_j_mean, sigma_j=0.1)
    if model == "heston" and isinstance(params, BatesParams):
        params = HestonParams(**{p: getattr(params, p) for p in HESTON_PARAMS})

    names = list(BATES_PARAMS if model == "bates" else HESTON_PARAMS)
    cols = names + ["loglik", "v_T", "v_T_sd", "min_ess"] + (["n_jumps"] if model == "bates" else [])
    chain = np.empty((n_iter, len(cols)))
    vol_paths = np.empty((n_iter, n + 1))
    filt_mean = np.empty((n_iter, n + 1))
    jprob = np.empty((n_iter, n)) if model == "bates" else None
    t0 = time.time()
    n_rejected = 0

    for it in range(n_iter):
        fr = run_filter(R, params, dt, n_particles, rng, resampling, jump_proposal_prob,
                        store_cloud=(path == "ffbs"), v0_mean=pr.v0_mean, v0_sd=pr.v0_sd)

        jump_kw = {}
        n_jumps = 0.0
        if model == "bates":
            if jump_update == "gibbs":
                Jd, Zd = fr.jump_draw, fr.size_draw
                R_adj = R * np.exp(-Jd * Zd)
                jd = bayes.draw_jump_params(Jd, Zd, T, pr, rng)
                jump_kw = dict(lam=jd.lam, mu_j=jd.mu_j, sigma_j=max(jd.sigma_j, 1e-6))
                n_jumps = float(Jd.sum())
            else:
                p, zm, z2 = fr.jump_prob, fr.jump_size_mean, fr.jump_size_m2
                # E[exp(-J Z)] with Z at its posterior mean
                R_adj = R * (1.0 - p * (1.0 - np.exp(-zm)))
                sp = p.sum()
                lam = sp / T
                if sp > 1e-12:
                    mu_j = float(p @ zm / sp)
                    var_j = float(p @ (z2 - 2 * mu_j * zm + mu_j ** 2) / sp)
                else:
                    mu_j, var_j = params.mu_j, params.sigma_j ** 2
                jump_kw = dict(lam=lam, mu_j=mu_j, sigma_j=float(np.sqrt(max(var_j, 1e-12))))
                n_jumps = float(sp)
                Jd = (rng.random(n) < p).astype(int)
            jprob[it] = fr.jump_prob
        else:
            R_adj = R
            Jd = None

        if path == "ffbs":
            v = backward_sample(fr.cloud, R_adj, params, dt, rng)
        else:
            v = fr.v_mean

        mu = bayes.draw_mu(R_adj, v, dt, pr, rng)
        vd = bayes.draw_kappa_theta_sigma(v, dt, pr, rng)
        rho = bayes.draw_rho(R_adj, v, dt, mu, vd.kappa, vd.theta, pr, rng)
        base = dict(mu=mu, kappa=vd.kappa, theta=vd.theta, sigma=vd.sigma,
                    rho=float(np.clip(rho, -0.999, 0.999)))
        new = {**base, **jump_kw}
        if all(np.isfinite(x) for x in new.values()) and new["sigma"] > 0:
            params = BatesParams(**new) if model == "bates" else HestonParams(**base)
        else:  # numerically degenerate sweep: keep the previous state
            n_rejected += 1

        row = [getattr(params, p) for p in names] + [fr.loglik, fr.v_mean[-1], fr.v_sd[-1], fr.ess.min()]
        if model == "bates":
            row.append(n_jumps)
        chain[it] = row
        vol_paths[it] = v
        filt_mean[it] = fr.v_mean

        if progress:
            info = dict(zip(cols, row), elapsed=time.time() - t0)
            if callable(progress):
                progress(it, info)
            elif (it + 1) % max(1, n_iter // 20) == 0 or it == 0:
                txt = " ".join(f"{p}={info[p]:.4g}" for p in names)
                print(f"[{it + 1:5d}/{n_iter}] {txt}  ({info['elapsed']:.0f}s)", flush=True)

    settings = dict(n_iter=n_iter, burn_in=burn_in, n_particles=n_particles, path=path,
                    resampling=resampling, jump_update=jump_update, seed=seed,
                    priors=pr.to_dict(), runtime_s=time.time() - t0, rejected_sweeps=n_rejected)
    return PosteriorResult(model, dt, burn_in, pd.DataFrame(chain, columns=cols),
                           vol_paths, filt_mean, jprob, settings)
