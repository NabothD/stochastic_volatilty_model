"""Matplotlib figures for posterior, latent variance and option-price results."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg") if matplotlib.get_backend().lower() == "agg" else None
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from .estimation import PosteriorResult

LABELS = {"mu": r"$\mu$", "kappa": r"$\kappa$", "theta": r"$\theta$", "sigma": r"$\sigma$",
          "rho": r"$\rho$", "lam": r"$\lambda$", "mu_j": r"$\mu_J$", "sigma_j": r"$\sigma_J$"}


def plot_posteriors(result: PosteriorResult, true_values: Optional[Dict[str, float]] = None,
                    names: Optional[Sequence[str]] = None, ncols: int = 4):
    names = list(names or result.param_names)
    if true_values is not None and hasattr(true_values, "to_dict"):
        true_values = true_values.to_dict()
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), squeeze=False)
    for ax, p in zip(axes.flat, names):
        x = result.samples[p].to_numpy()
        ax.hist(x, bins=40, density=True, alpha=0.35, color="C0")
        if np.ptp(x) > 0:
            grid = np.linspace(x.min(), x.max(), 300)
            ax.plot(grid, gaussian_kde(x)(grid), color="C0", lw=1.5, label="posterior")
        lo, hi = np.percentile(x, [2.5, 97.5])
        ax.axvspan(lo, hi, color="C0", alpha=0.08, label="95% CI")
        if true_values and p in true_values:
            ax.axvline(true_values[p], color="k", ls="--", lw=1.2, label="true")
        ax.set_xlabel(LABELS.get(p, p))
        ax.set_yticks([])
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_traces(result: PosteriorResult, names: Optional[Sequence[str]] = None):
    names = list(names or result.param_names)
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 1.5 * len(names)), sharex=True)
    for ax, p in zip(np.atleast_1d(axes), names):
        ax.plot(result.chain[p].to_numpy(), lw=0.6)
        ax.axvline(result.burn_in, color="r", ls=":", lw=1)
        ax.set_ylabel(LABELS.get(p, p))
    np.atleast_1d(axes)[-1].set_xlabel("sweep")
    fig.tight_layout()
    return fig


def plot_variance(result: PosteriorResult, true_variance: Optional[np.ndarray] = None,
                  t: Optional[np.ndarray] = None, as_vol: bool = True):
    paths = result.vol_paths[result.burn_in:]
    f = np.sqrt if as_vol else (lambda z: z)
    lo, hi = np.percentile(f(paths), [2.5, 97.5], axis=0)
    t = np.arange(paths.shape[1]) if t is None else t
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.fill_between(t, lo, hi, color="C0", alpha=0.2, label="95% band")
    ax.plot(t, f(result.variance_path), color="C0", lw=1.2, label="posterior mean")
    if true_variance is not None:
        ax.plot(t, f(np.asarray(true_variance)[: len(t)]), color="k", lw=0.9, label="true")
    ax.set_ylabel("volatility $\\sqrt{V_t}$" if as_vol else "variance $V_t$")
    ax.set_xlabel("time step")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_price_distribution(prices: np.ndarray, true_price: Optional[float] = None,
                            label: str = "option price"):
    prices = np.asarray(prices).ravel()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(prices, bins=40, alpha=0.7, color="C0", edgecolor="white")
    ax.axvline(prices.mean(), color="C3", ls="--", label=f"mean {prices.mean():.3f}")
    if true_price is not None:
        ax.axvline(true_price, color="k", lw=1.5, label=f"true {true_price:.3f}")
    ax.set_xlabel(label)
    ax.set_ylabel("count")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
