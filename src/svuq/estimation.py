"""Bayesian estimation of Heston / Bates parameters from a price series.

Each Gibbs sweep (thesis Section 4.7, corrected):

1. run the particle filter for the latent variance (and jumps),
2. (Bates) draw jump times/sizes and the jump parameters; remove the jumps
   from the returns,
3. take a variance path: a backward-sampled trajectory (``path="ffbs"``,
   default) or the filtered mean (``path="mean"``, thesis method),
4. draw ``mu``, ``(kappa, theta, sigma)`` and ``rho`` from their conjugate
   regression posteriors.

:class:`Estimator` runs one or more independent chains (in parallel
processes) and returns a :class:`PosteriorResult`.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .conjugate import ConjugateSampler
from .diagnostics import effective_sample_size, gelman_rubin
from .filtering import ParticleFilter
from .models import BATES_PARAMS, HESTON_PARAMS, HestonParams, Params, make_params
from .parallel import process_map
from .priors import Priors

ProgressFn = Callable[[int, dict[str, float]], None]


@dataclass(frozen=True)
class SamplerConfig:
    """Settings of the particle-filter Gibbs sampler.

    Attributes
    ----------
    n_iter : number of Gibbs sweeps per chain.
    burn_in : sweeps discarded at the start of each chain (default ``n_iter // 4``).
    n_particles : particles in the filter.
    path : ``"ffbs"`` (backward-sampled path) or ``"mean"`` (filtered mean, thesis).
    resampling : ``"smooth"`` (refined, thesis) or ``"systematic"``.
    jump_update : ``"gibbs"`` (conjugate draws) or ``"moments"`` (thesis Eqs. 80–83).
    jump_proposal_prob : minimum probability of proposing a jump in the filter.
    """

    n_iter: int = 2000
    burn_in: int | None = None
    n_particles: int = 1000
    path: str = "ffbs"
    resampling: str = "smooth"
    jump_update: str = "gibbs"
    jump_proposal_prob: float = 0.1

    def __post_init__(self) -> None:
        """Validate the settings."""
        if self.path not in ("ffbs", "mean"):
            raise ValueError("path must be 'ffbs' or 'mean'")
        if self.jump_update not in ("gibbs", "moments"):
            raise ValueError("jump_update must be 'gibbs' or 'moments'")
        if not 0 <= self.effective_burn_in < self.n_iter:
            raise ValueError("burn_in must be in [0, n_iter)")

    @property
    def effective_burn_in(self) -> int:
        """Burn-in actually used (``n_iter // 4`` if not set)."""
        return self.n_iter // 4 if self.burn_in is None else self.burn_in


@dataclass
class ChainOutput:
    """Raw output of one chain.

    Attributes
    ----------
    draws : one row per sweep: parameters and diagnostics.
    vol_paths : variance path used in each sweep, shape (n_iter, n+1).
    filtered_mean : filtered ``E[V_k]`` in each sweep, shape (n_iter, n+1).
    jump_prob : filtered jump probabilities per sweep (Bates), shape (n_iter, n).
    rejected : number of numerically degenerate sweeps that were skipped.
    runtime_s : wall time of the chain in seconds.
    """

    draws: pd.DataFrame
    vol_paths: np.ndarray
    filtered_mean: np.ndarray
    jump_prob: np.ndarray | None
    rejected: int
    runtime_s: float


class GibbsSampler:
    """One chain of the particle-filter Gibbs sampler.

    Parameters
    ----------
    model : ``"heston"`` or ``"bates"``.
    priors : prior hyper-parameters.
    config : sampler settings.
    """

    def __init__(self, model: str, priors: Priors, config: SamplerConfig) -> None:
        """Store the model, priors and settings."""
        if model not in ("heston", "bates"):
            raise ValueError("model must be 'heston' or 'bates'")
        self.model = model
        self.priors = priors
        self.config = config
        self.filter = ParticleFilter(config.n_particles, config.resampling,
                                     config.jump_proposal_prob)

    @property
    def param_names(self) -> tuple[str, ...]:
        """Names of the sampled parameters."""
        return BATES_PARAMS if self.model == "bates" else HESTON_PARAMS

    def _jump_step(self, fr, R: np.ndarray, params: Params, T: float,
                   conj: ConjugateSampler, rng: np.random.Generator):
        """Update jumps: return (adjusted returns, jump parameters, number of jumps)."""
        if self.config.jump_update == "gibbs":
            J, Z = fr.jump_draw, fr.size_draw
            jd = conj.draw_jump_params(J, Z, T, rng)
            new = dict(lam=jd.lam, mu_j=jd.mu_j, sigma_j=max(jd.sigma_j, 1e-6))
            return R * np.exp(-J * Z), new, float(J.sum())
        p, zm, z2 = fr.jump_prob, fr.jump_mean, fr.jump_m2
        total = p.sum()
        if total > 1e-12:
            mu_j = float(p @ zm / total)
            var_j = float(p @ (z2 - 2.0 * mu_j * zm + mu_j ** 2) / total)
        else:
            mu_j, var_j = params.mu_j, params.sigma_j ** 2
        new = dict(lam=total / T, mu_j=mu_j, sigma_j=float(np.sqrt(max(var_j, 1e-12))))
        return R * (1.0 - p * (1.0 - np.exp(-zm))), new, float(total)

    def run(self, prices: np.ndarray, dt: float, seed: int | None = None,
            init: Params | None = None, progress: bool | ProgressFn = False,
            label: str = "") -> ChainOutput:
        """Run the chain.

        Parameters
        ----------
        prices : asset prices ``S_0..S_n`` equally spaced by ``dt``.
        dt : time step in years.
        seed : random seed.
        init : starting parameters (default: prior means).
        progress : ``True`` to print progress, or a callback ``f(iteration, info)``.
        label : prefix for printed progress lines.

        Returns
        -------
        ChainOutput
        """
        cfg, pr = self.config, self.priors
        rng = np.random.default_rng(seed)
        conj = ConjugateSampler(pr, dt)
        R = prices[1:] / prices[:-1]
        n = R.size
        T = n * dt
        start = pr.initial_params(self.model, dt).to_dict()
        if init is not None:
            start.update(init.to_dict())
        params = make_params(self.model, start)
        names = list(self.param_names)
        extra = ["loglik", "v_T", "v_T_sd", "min_ess"]
        if self.model == "bates":
            extra.append("n_jumps")
        rows = np.empty((cfg.n_iter, len(names) + len(extra)))
        vol_paths = np.empty((cfg.n_iter, n + 1))
        filt_mean = np.empty((cfg.n_iter, n + 1))
        jump_prob = np.empty((cfg.n_iter, n)) if self.model == "bates" else None
        rejected, t0 = 0, time.time()

        for it in range(cfg.n_iter):
            fr = self.filter.run(R, params, dt, rng, store_cloud=cfg.path == "ffbs",
                                 v0_mean=pr.v0_mean, v0_sd=pr.v0_sd)
            jump_kw, n_jumps, R_adj = {}, 0.0, R
            if self.model == "bates":
                R_adj, jump_kw, n_jumps = self._jump_step(fr, R, params, T, conj, rng)
                jump_prob[it] = fr.jump_prob

            if cfg.path == "ffbs":
                v = self.filter.backward_sample(fr, R_adj, params, dt, rng)
            else:
                v = fr.v_mean

            mu = conj.draw_mu(R_adj, v, rng)
            vd = conj.draw_variance_params(v, rng)
            rho = conj.draw_rho(R_adj, v, mu, vd.kappa, vd.theta, rng)
            proposal = dict(mu=mu, kappa=vd.kappa, theta=vd.theta, sigma=vd.sigma,
                            rho=float(np.clip(rho, -0.999, 0.999)), **jump_kw)
            if all(np.isfinite(x) for x in proposal.values()) and proposal["sigma"] > 0:
                params = make_params(self.model, proposal)
            else:
                rejected += 1

            row = [getattr(params, p) for p in names] + [fr.loglik, fr.v_mean[-1], fr.v_sd[-1],
                                                          fr.ess.min()]
            rows[it] = row + ([n_jumps] if self.model == "bates" else [])
            vol_paths[it] = v
            filt_mean[it] = fr.v_mean
            self._report(progress, it, dict(zip(names + extra, rows[it], strict=False)), t0, label)

        return ChainOutput(pd.DataFrame(rows, columns=names + extra), vol_paths, filt_mean,
                           jump_prob, rejected, time.time() - t0)

    def _report(self, progress: bool | ProgressFn, it: int, info: dict[str, float],
                t0: float, label: str) -> None:
        """Print or forward progress information for sweep ``it``."""
        if not progress:
            return
        info = {**info, "elapsed": time.time() - t0}
        if callable(progress):
            progress(it, info)
            return
        n = self.config.n_iter
        if it == 0 or (it + 1) % max(1, n // 20) == 0:
            txt = " ".join(f"{p}={info[p]:.4g}" for p in self.param_names)
            print(f"{label}[{it + 1:5d}/{n}] {txt}  ({info['elapsed']:.0f}s)", flush=True)


def _run_chain(job: tuple) -> ChainOutput:
    """Run one chain in a worker process (module-level so it can be pickled)."""
    sampler, prices, dt, seed, init, progress, label = job
    return sampler.run(prices, dt, seed, init, progress, label)


@dataclass
class PosteriorResult:
    """Posterior sample from one or more chains.

    Attributes
    ----------
    model : ``"heston"`` or ``"bates"``.
    dt : time step in years.
    burn_in : sweeps discarded per chain.
    chains : raw output of each chain.
    settings : sampler settings and priors, for the record.
    """

    model: str
    dt: float
    burn_in: int
    chains: list[ChainOutput]
    settings: dict = field(default_factory=dict)

    @property
    def param_names(self) -> tuple[str, ...]:
        """Names of the model parameters."""
        return BATES_PARAMS if self.model == "bates" else HESTON_PARAMS

    @property
    def n_chains(self) -> int:
        """Number of chains."""
        return len(self.chains)

    @property
    def chain(self) -> pd.DataFrame:
        """All sweeps of all chains, with ``chain`` and ``iter`` columns."""
        frames = [c.draws.assign(chain=i, iter=np.arange(len(c.draws)))
                  for i, c in enumerate(self.chains)]
        return pd.concat(frames, ignore_index=True)

    @property
    def samples(self) -> pd.DataFrame:
        """Posterior draws after burn-in, chains stacked."""
        return pd.concat([c.draws.iloc[self.burn_in:] for c in self.chains], ignore_index=True)

    def _stack(self, attr: str) -> np.ndarray:
        """Stack a per-sweep array attribute of all chains after burn-in."""
        return np.concatenate([getattr(c, attr)[self.burn_in:] for c in self.chains])

    @property
    def vol_paths(self) -> np.ndarray:
        """Variance paths used after burn-in, shape (n_draws, n+1)."""
        return self._stack("vol_paths")

    @property
    def variance_path(self) -> np.ndarray:
        """Posterior mean of the filtered variance, shape (n+1,)."""
        return self._stack("filtered_mean").mean(axis=0)

    @property
    def jump_prob(self) -> np.ndarray | None:
        """Posterior mean jump probability per day (Bates), shape (n,)."""
        return self._stack("jump_prob").mean(axis=0) if self.model == "bates" else None

    @property
    def v_final(self) -> tuple[float, float]:
        """Mean and sd of the variance at the last date (filter + parameter uncertainty)."""
        s = self.samples
        m, sd = s["v_T"].to_numpy(), s["v_T_sd"].to_numpy()
        return float(m.mean()), float(np.sqrt(np.mean(sd ** 2) + m.var()))

    def point_estimate(self) -> Params:
        """Return the posterior-mean parameters."""
        return make_params(self.model, self.samples[list(self.param_names)].mean().to_dict())

    def summary(self, true_values: dict[str, float] | Params | None = None,
                ci: float = 0.95) -> pd.DataFrame:
        """Posterior summary table.

        Parameters
        ----------
        true_values : optional true parameters (adds error and CI coverage).
        ci : credible-interval mass.

        Returns
        -------
        DataFrame indexed by parameter with mean, sd, median, quantiles, ESS
        (summed over chains) and R-hat (if more than one chain).
        """
        if isinstance(true_values, HestonParams):
            true_values = true_values.to_dict()
        lo, hi = 50 * (1 - ci), 50 * (1 + ci)
        rows = []
        for p in self.param_names:
            per_chain = [c.draws[p].to_numpy()[self.burn_in:] for c in self.chains]
            x = np.concatenate(per_chain)
            row = {"param": p, "mean": x.mean(), "sd": x.std(ddof=1), "median": np.median(x),
                   f"q{lo:g}": np.percentile(x, lo), f"q{hi:g}": np.percentile(x, hi),
                   "ess": sum(effective_sample_size(c) for c in per_chain)}
            if self.n_chains > 1:
                row["r_hat"] = gelman_rubin(per_chain)
            if true_values is not None and p in true_values:
                tv = true_values[p]
                row["true"] = tv
                row["rel_err_%"] = 100 * abs(x.mean() - tv) / abs(tv) if tv else np.nan
                row["in_ci"] = bool(row[f"q{lo:g}"] <= tv <= row[f"q{hi:g}"])
            rows.append(row)
        return pd.DataFrame(rows).set_index("param")

    def save(self, prefix: str) -> None:
        """Write ``<prefix>_chain.csv``, ``<prefix>_variance.csv`` and ``<prefix>_summary.csv``.

        Parameters
        ----------
        prefix : output path prefix.
        """
        self.chain.to_csv(f"{prefix}_chain.csv", index=False)
        fm = self._stack("filtered_mean")
        pd.DataFrame({"posterior_mean_variance": fm.mean(axis=0),
                      "posterior_sd_variance": fm.std(axis=0)}).to_csv(
            f"{prefix}_variance.csv", index_label="k")
        self.summary().to_csv(f"{prefix}_summary.csv")


class Estimator:
    """User-facing estimator: runs independent chains, optionally in parallel.

    Parameters
    ----------
    model : ``"heston"`` or ``"bates"``.
    priors : prior hyper-parameters (default: weakly informative, built from ``dt`` at fit time).
    config : sampler settings.

    Examples
    --------
    >>> est = Estimator("heston", config=SamplerConfig(n_iter=2000))
    >>> post = est.fit(prices, dt=1/252, n_chains=4, n_jobs=4, seed=0)
    >>> post.summary()
    """

    def __init__(self, model: str = "heston", priors: Priors | None = None,
                 config: SamplerConfig | None = None) -> None:
        """Store the model, priors and settings."""
        if model not in ("heston", "bates"):
            raise ValueError("model must be 'heston' or 'bates'")
        self.model = model
        self.priors = priors
        self.config = config or SamplerConfig()

    def fit(self, prices: Sequence[float], dt: float, n_chains: int = 1,
            n_jobs: int | None = None, seed: int | None = None,
            init: Params | None = None, progress: bool | ProgressFn = False
            ) -> PosteriorResult:
        """Estimate the posterior from a price series.

        Parameters
        ----------
        prices : asset prices ``S_0..S_n`` equally spaced by ``dt`` years.
        dt : time step in years (1/252 for daily data).
        n_chains : number of independent chains (>1 enables R-hat).
        n_jobs : worker processes for the chains (``None`` = all cores).
        seed : master seed; each chain gets an independent child seed.
        init : starting parameters for every chain (default: prior means).
        progress : ``True`` prints progress of each chain; a callback is only
            supported with a single chain or ``n_jobs=1``.

        Returns
        -------
        PosteriorResult
        """
        prices = np.asarray(prices, dtype=np.float64)
        if prices.ndim != 1 or prices.size < 10 or np.any(prices <= 0):
            raise ValueError("prices must be a 1-D array of more than 10 positive values")
        if self.config.path == "mean":
            warnings.warn(
                "path='mean' regresses on the filtered-mean variance path (thesis method); rho "
                "drifts towards +/-1 and sigma is biased low unless priors are tight. "
                "Use path='ffbs' for inference.", stacklevel=2)
        priors = self.priors or Priors.from_beliefs(dt)
        sampler = GibbsSampler(self.model, priors, self.config)
        seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(n_chains)]
        jobs = [(sampler, prices, dt, s, init, progress, f"chain {i} " if n_chains > 1 else "")
                for i, s in enumerate(seeds)]
        if callable(progress) and n_chains > 1 and n_jobs != 1:
            raise ValueError("progress callbacks need n_chains=1 or n_jobs=1")
        t0 = time.time()
        chains = process_map(_run_chain, jobs, n_jobs) if n_chains > 1 else [_run_chain(jobs[0])]
        settings = {**{k: getattr(self.config, k) for k in self.config.__dataclass_fields__},
                    "burn_in": self.config.effective_burn_in, "n_chains": n_chains,
                    "seed": seed, "priors": priors.to_dict(), "runtime_s": time.time() - t0,
                    "rejected_sweeps": sum(c.rejected for c in chains)}
        return PosteriorResult(self.model, dt, self.config.effective_burn_in, chains, settings)

    def with_config(self, **changes) -> Estimator:
        """Return a copy of this estimator with some sampler settings changed."""
        return Estimator(self.model, self.priors, replace(self.config, **changes))
