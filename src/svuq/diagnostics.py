"""MCMC diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Compute the normalised autocorrelation function via FFT.

    Parameters
    ----------
    x : 1-D chain.

    Returns
    -------
    Autocorrelations at lags ``0..n-1``.
    """
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = x.size
    if n < 2 or np.allclose(x, 0):
        return np.ones(1)
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    return acf / acf[0]


def effective_sample_size(x: np.ndarray) -> float:
    """Estimate the effective sample size with Geyer's initial positive sequence.

    Parameters
    ----------
    x : 1-D chain.

    Returns
    -------
    Effective number of independent draws.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4 or np.ptp(x) == 0:
        return float(n)
    rho = autocorrelation(x)
    tau = 1.0
    for k in range(1, n // 2):
        pair = rho[2 * k - 1] + (rho[2 * k] if 2 * k < n else 0.0)
        if pair < 0:
            break
        tau += 2.0 * pair
    return float(n / tau)


def gelman_rubin(chains: Sequence[np.ndarray]) -> float:
    """Compute the potential scale reduction factor R-hat.

    Parameters
    ----------
    chains : equally long chains of one parameter.

    Returns
    -------
    R-hat (values close to 1 indicate convergence).
    """
    c = np.asarray([np.asarray(ch, dtype=float) for ch in chains])
    _, n = c.shape
    within = c.var(axis=1, ddof=1).mean()
    between = n * c.mean(axis=1).var(ddof=1)
    var_hat = (n - 1) / n * within + between / n
    return float(np.sqrt(var_hat / within)) if within > 0 else float("nan")


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the mean squared error over the common length of two series.

    Parameters
    ----------
    a, b : series to compare.

    Returns
    -------
    Mean squared difference.
    """
    a, b = np.asarray(a), np.asarray(b)
    m = min(a.size, b.size)
    return float(np.mean((a[:m] - b[:m]) ** 2))
