"""Numba-compiled kernels for the particle filter and backward sampler.

All hot loops of the estimator live here. The kernels are compiled with
``cache=True`` (compiled once, reused across runs and worker processes) and
``nogil=True`` (several threads can run them concurrently).

Randomness: each public kernel takes an integer ``seed`` and reseeds numba's
generator for the calling thread, so results are reproducible and
independent between threads and processes. Seeds come from a
``numpy.random.Generator`` on the Python side.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

V_FLOOR = 1e-10
_LOG2PI = math.log(2.0 * math.pi)
_JIT = dict(cache=True, nogil=True, fastmath=False)


@nb.njit(**_JIT)
def seed_rng(seed: int) -> None:
    """Seed numba's random generator for the calling thread.

    Parameters
    ----------
    seed : non-negative integer seed.
    """
    np.random.seed(seed)


@nb.njit(**_JIT)
def _pick(w: np.ndarray, u: float) -> int:
    """Return the index of the cumulative weight bracket containing ``u``."""
    c = 0.0
    for i in range(w.size):
        c += w[i]
        if u < c:
            return i
    return w.size - 1


@nb.njit(**_JIT)
def systematic_resample(w: np.ndarray) -> np.ndarray:
    """Systematic resampling.

    Parameters
    ----------
    w : normalised weights, shape (N,).

    Returns
    -------
    Indices of the selected particles, shape (N,).
    """
    n = w.size
    idx = np.empty(n, np.int64)
    u = np.random.random() / n
    c = w[0]
    j = 0
    for i in range(n):
        target = u + i / n
        while target > c and j < n - 1:
            j += 1
            c += w[j]
        idx[i] = j
    return idx


@nb.njit(**_JIT)
def smooth_resample(x: np.ndarray, w: np.ndarray):
    """Continuous ("refined") resampling of a scalar state (thesis Eqs. 67–73).

    The weighted empirical CDF is replaced by a piecewise-linear CDF through
    the sorted particles and inverted at systematic uniforms ``(i + U)/N``
    (Malik & Pitt). One random number per call keeps it cheap.

    Parameters
    ----------
    x : particle values, shape (N,).
    w : normalised weights, shape (N,).

    Returns
    -------
    x_new : resampled values in ascending order, shape (N,).
    idx : index of the nearest bracketing original particle for each new value,
        used to carry auxiliary variables (jumps) along, shape (N,).
    """
    n = x.size
    order = np.argsort(x)
    knots = np.empty(n)
    acc = 0.0
    for i in range(n):
        wi = w[order[i]]
        knots[i] = acc + 0.5 * wi
        acc += wi
    knots[0] = 0.0
    knots[n - 1] = 1.0
    for i in range(1, n):
        if knots[i] < knots[i - 1]:
            knots[i] = knots[i - 1]
    x_new = np.empty(n)
    idx = np.empty(n, np.int64)
    offset = np.random.random()
    j = 0
    for i in range(n):
        u = (i + offset) / n
        while j < n - 2 and knots[j + 1] <= u:
            j += 1
        width = knots[j + 1] - knots[j]
        frac = (u - knots[j]) / width if width > 0.0 else 0.0
        frac = min(max(frac, 0.0), 1.0)
        lo, hi = x[order[j]], x[order[j + 1]]
        x_new[i] = lo + frac * (hi - lo)
        idx[i] = order[j + 1] if frac > 0.5 else order[j]
    return x_new, idx


@nb.njit(**_JIT)
def filter_step(V, r, dt, mu, kappa, theta, sigma, rho, p_jump, q_jump, mu_j, sigma_j, smooth):
    """Advance the particle cloud by one observation.

    Particles of ``V_{k-1}`` (plus a proposed jump for Bates) are weighted by
    ``p(R_k | V_{k-1}, J, Z)``, resampled jointly with their jumps, and then
    propagated to ``V_k`` with the variance shock correlated with the price
    shock implied by ``R_k``.

    Parameters
    ----------
    V : particles of the variance at the previous date, shape (N,).
    r : gross return ``S_k / S_{k-1}``.
    dt : time step in years.
    mu, kappa, theta, sigma, rho : Heston parameters.
    p_jump : jump probability per step (``lam * dt``; 0 for Heston).
    q_jump : proposal probability for a jump (>= ``p_jump``).
    mu_j, sigma_j : jump-size parameters.
    smooth : True for refined resampling, False for systematic.

    Returns
    -------
    V_new : propagated particles, shape (N,).
    loglik : log ``p(R_k | R_1..R_{k-1})`` estimate.
    ess : effective sample size of the weights.
    jump_prob : ``P(J_k = 1 | R_1..R_k)``.
    jump_mean, jump_m2 : first and second moment of ``Z_k`` given a jump.
    jump_draw, size_draw : one joint draw of ``(J_k, Z_k)``.
    """
    n = V.size
    drift = 1.0 + mu * dt
    rho_c = math.sqrt(max(1.0 - rho * rho, 1e-12))
    has_jumps = p_jump > 0.0
    sj = max(sigma_j, 1e-8)
    lw = np.empty(n)
    J = np.zeros(n, np.bool_)
    Z = np.zeros(n)
    log_nojump = math.log((1.0 - p_jump) / (1.0 - q_jump)) if has_jumps else 0.0
    log_jump = math.log(p_jump / q_jump) if has_jumps else 0.0
    y = math.log(r / drift)

    for i in range(n):
        var = V[i] * dt
        if has_jumps and np.random.random() < q_jump:
            prec = 1.0 / (sj * sj) + 1.0 / var
            mq = (mu_j / (sj * sj) + y / var) / prec
            sq = 1.5 / math.sqrt(prec)
            z = mq + sq * np.random.standard_normal()
            ez2 = math.exp(2.0 * z)
            J[i] = True
            Z[i] = z
            lw[i] = (-0.5 * (_LOG2PI + math.log(ez2 * var))
                     - 0.5 * (r - math.exp(z) * drift) ** 2 / (ez2 * var)
                     + log_jump
                     - 0.5 * ((z - mu_j) / sj) ** 2 - math.log(sj)
                     + 0.5 * ((z - mq) / sq) ** 2 + math.log(sq))
        else:
            lw[i] = -0.5 * (_LOG2PI + math.log(var)) - 0.5 * (r - drift) ** 2 / var + log_nojump

    m = lw.max()
    w = np.exp(lw - m)
    sw = w.sum()
    loglik = m + math.log(sw / n)
    w /= sw
    ess = 1.0 / np.sum(w * w)

    jump_prob, jump_mean, jump_m2 = 0.0, mu_j, mu_j * mu_j + sj * sj
    jump_draw, size_draw = 0, 0.0
    if has_jumps:
        s0 = s1 = s2 = 0.0
        for i in range(n):
            if J[i]:
                s0 += w[i]
                s1 += w[i] * Z[i]
                s2 += w[i] * Z[i] * Z[i]
        jump_prob = s0
        if s0 > 1e-300:
            jump_mean, jump_m2 = s1 / s0, s2 / s0
        k = _pick(w, np.random.random())
        if J[k]:
            jump_draw, size_draw = 1, Z[k]
        else:
            size_draw = mu_j + sj * np.random.standard_normal()

    if smooth:
        Vr, idx = smooth_resample(V, w)
    else:
        idx = systematic_resample(w)
        Vr = V[idx]

    V_new = np.empty(n)
    for i in range(n):
        r_adj = r * math.exp(-Z[idx[i]]) if J[idx[i]] else r
        sqv = math.sqrt(Vr[i] * dt)
        eps_s = (r_adj - drift) / sqv
        eps_v = rho * eps_s + rho_c * np.random.standard_normal()
        V_new[i] = max(Vr[i] + kappa * (theta - Vr[i]) * dt + sigma * sqv * eps_v, V_FLOOR)
    return V_new, loglik, ess, jump_prob, jump_mean, jump_m2, jump_draw, size_draw


@nb.njit(**_JIT)
def run_filter(R, V0, dt, mu, kappa, theta, sigma, rho, p_jump, q_jump, mu_j, sigma_j,
               smooth, store_cloud, seed):
    """Run the particle filter over a whole return series.

    Parameters
    ----------
    R : gross returns, shape (n,).
    V0 : initial particles of ``V_0``, shape (N,).
    dt, mu, kappa, theta, sigma, rho, p_jump, q_jump, mu_j, sigma_j, smooth :
        see :func:`filter_step`.
    store_cloud : keep every filtered cloud (needed for backward sampling).
    seed : random seed for this run.

    Returns
    -------
    v_mean, v_sd : filtered mean / sd of ``V_k``, shape (n+1,).
    loglik : total log-likelihood estimate.
    ess : effective sample size per step, shape (n,).
    jump_prob, jump_mean, jump_m2 : per-step jump statistics, shape (n,).
    jump_draw, size_draw : one draw of ``(J_k, Z_k)`` per step, shape (n,).
    cloud : filtered clouds, shape (n+1, N), or (1, 1) if not stored.
    final : particles of ``V_n``, shape (N,).
    """
    seed_rng(seed)
    n, N = R.size, V0.size
    v_mean = np.empty(n + 1)
    v_sd = np.empty(n + 1)
    ess = np.empty(n)
    jump_prob = np.zeros(n)
    jump_mean = np.zeros(n)
    jump_m2 = np.zeros(n)
    jump_draw = np.zeros(n, np.int64)
    size_draw = np.zeros(n)
    cloud = np.empty((n + 1, N)) if store_cloud else np.empty((1, 1))
    V = V0.copy()
    v_mean[0], v_sd[0] = V.mean(), V.std()
    if store_cloud:
        cloud[0] = V
    loglik = 0.0
    for k in range(n):
        V, ll, e, pj, zm, z2, jd, zd = filter_step(V, R[k], dt, mu, kappa, theta, sigma, rho,
                                                   p_jump, q_jump, mu_j, sigma_j, smooth)
        loglik += ll
        ess[k] = e
        jump_prob[k], jump_mean[k], jump_m2[k] = pj, zm, z2
        jump_draw[k], size_draw[k] = jd, zd
        v_mean[k + 1], v_sd[k + 1] = V.mean(), V.std()
        if store_cloud:
            cloud[k + 1] = V
    return (v_mean, v_sd, loglik, ess, jump_prob, jump_mean, jump_m2, jump_draw, size_draw,
            cloud, V)


@nb.njit(**_JIT)
def backward_sample(cloud, R_adj, dt, mu, kappa, theta, sigma, rho, seed):
    """Draw one variance trajectory by forward-filter backward-sampling.

    Backward weights are ``p(R_{k+1} | V_k) * p(V_{k+1} | V_k, R_{k+1})``.

    Parameters
    ----------
    cloud : equally weighted filtered particles, shape (n+1, N).
    R_adj : returns with the sampled jumps removed, shape (n,).
    dt, mu, kappa, theta, sigma, rho : Heston parameters.
    seed : random seed.

    Returns
    -------
    Sampled path ``V_0..V_n``, shape (n+1,).
    """
    seed_rng(seed)
    n, N = R_adj.size, cloud.shape[1]
    rho_c2 = max(1.0 - rho * rho, 1e-12)
    drift = 1.0 + mu * dt
    path = np.empty(n + 1)
    path[n] = cloud[n, np.random.randint(N)]
    lw = np.empty(N)
    for k in range(n - 1, -1, -1):
        target = path[k + 1]
        for j in range(N):
            v = cloud[k, j]
            var = v * dt
            sq = math.sqrt(var)
            eps_s = (R_adj[k] - drift) / sq
            mean_next = v + kappa * (theta - v) * dt + sigma * sq * rho * eps_s
            var_next = sigma * sigma * var * rho_c2
            lw[j] = (-0.5 * math.log(var) - 0.5 * eps_s * eps_s
                     - 0.5 * math.log(var_next) - 0.5 * (target - mean_next) ** 2 / var_next)
        w = np.exp(lw - lw.max())
        w /= w.sum()
        path[k] = cloud[k, _pick(w, np.random.random())]
    return path
