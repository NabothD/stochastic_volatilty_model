"""Matplotlib figures for posterior, latent variance and option-price results."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from .estimation import PosteriorResult

LABELS = {"mu": r"$\mu$", "kappa": r"$\kappa$", "theta": r"$\theta$", "sigma": r"$\sigma$",
          "rho": r"$\rho$", "lam": r"$\lambda$", "mu_j": r"$\mu_J$", "sigma_j": r"$\sigma_J$"}


def plot_posteriors(result: PosteriorResult, true_values: dict[str, float] | None = None,
                    names: Sequence[str] | None = None, ncols: int = 4) -> plt.Figure:
    """Plot posterior histograms with KDE, 95% interval and optional true values.

    Parameters
    ----------
    result : estimator output.
    true_values : optional true parameters (dict or parameter object).
    names : parameters to plot (default: all).
    ncols : panels per row.

    Returns
    -------
    matplotlib Figure.
    """
    names = list(names or result.param_names)
    if true_values is not None and hasattr(true_values, "to_dict"):
        true_values = true_values.to_dict()
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), squeeze=False)
    for ax, p in zip(axes.flat, names, strict=False):
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


def plot_traces(result: PosteriorResult, names: Sequence[str] | None = None) -> plt.Figure:
    """Plot the trace of each parameter for every chain, with the burn-in marked.

    Parameters
    ----------
    result : estimator output.
    names : parameters to plot (default: all).

    Returns
    -------
    matplotlib Figure.
    """
    names = list(names or result.param_names)
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 1.5 * len(names)), sharex=True)
    for ax, p in zip(np.atleast_1d(axes), names, strict=False):
        for chain in result.chains:
            ax.plot(chain.draws[p].to_numpy(), lw=0.6)
        ax.axvline(result.burn_in, color="r", ls=":", lw=1)
        ax.set_ylabel(LABELS.get(p, p))
    np.atleast_1d(axes)[-1].set_xlabel("sweep")
    fig.tight_layout()
    return fig


def plot_variance(result: PosteriorResult, true_variance: np.ndarray | None = None,
                  t: np.ndarray | None = None, as_vol: bool = True) -> plt.Figure:
    """Plot the posterior latent volatility (or variance) path with a 95% band.

    Parameters
    ----------
    result : estimator output.
    true_variance : optional true variance path to overlay.
    t : optional x-axis values.
    as_vol : plot ``sqrt(V)`` instead of ``V``.

    Returns
    -------
    matplotlib Figure.
    """
    paths = result.vol_paths
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


def plot_price_distribution(prices: np.ndarray, true_price: float | None = None,
                            label: str = "option price") -> plt.Figure:
    """Plot a histogram of option prices across posterior draws.

    Parameters
    ----------
    prices : prices (any shape, flattened).
    true_price : optional reference price.
    label : x-axis label.

    Returns
    -------
    matplotlib Figure.
    """
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
