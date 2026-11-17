"""MCMC diagnostics and posterior summaries."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation function via FFT."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = x.size
    if n < 2 or np.allclose(x, 0):
        return np.ones(1)
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    return acf / acf[0]


def effective_sample_size(x: np.ndarray) -> float:
    """ESS using Geyer's initial positive sequence estimator."""
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
    """Potential scale reduction factor R-hat for several chains of one parameter."""
    c = np.asarray([np.asarray(ch, dtype=float) for ch in chains])
    m, n = c.shape
    W = c.var(axis=1, ddof=1).mean()
    B = n * c.mean(axis=1).var(ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W)) if W > 0 else np.nan


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    m = min(a.size, b.size)
    return float(np.mean((a[:m] - b[:m]) ** 2))
