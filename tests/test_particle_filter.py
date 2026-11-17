import numpy as np

from svuq import BatesParams, HestonParams, run_filter, simulate
from svuq.particle_filter import smooth_resample, systematic_resample


def test_systematic_resample_counts():
    rng = np.random.default_rng(0)
    w = np.array([0.5, 0.25, 0.25, 0.0])
    idx = systematic_resample(w, rng)
    counts = np.bincount(idx, minlength=4)
    np.testing.assert_array_equal(counts, [2, 1, 1, 0])


def test_smooth_resample_preserves_distribution():
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    w = np.exp(-0.5 * (x - 1.0) ** 2)
    w /= w.sum()
    new, idx = smooth_resample(x, w, rng)
    assert new.min() >= x.min() and new.max() <= x.max()
    assert abs(new.mean() - w @ x) < 0.03
    # the carried index points to a neighbouring particle
    assert np.all(np.abs(x[idx] - new) <= np.ptp(x))


def test_filter_tracks_true_variance():
    p = HestonParams(0.05, 3.0, 0.04, 0.5, -0.7)
    sim = simulate(p, T=3, n_steps=756, seed=4)
    fr = run_filter(sim.returns, p, 1 / 252, 1000, np.random.default_rng(0))
    assert np.isfinite(fr.loglik)
    assert np.corrcoef(fr.v_mean, sim.variance)[0, 1] > 0.8


def test_filter_detects_large_jumps():
    p = BatesParams(0.05, 3.0, 0.04, 0.3, -0.5, lam=4.0, mu_j=-0.25, sigma_j=0.05)
    sim = simulate(p, T=2, n_steps=504, seed=2)
    fr = run_filter(sim.returns, p, 1 / 252, 1000, np.random.default_rng(0))
    true_k = np.flatnonzero(sim.jumps)
    assert sim.jumps.sum() >= 3
    assert np.all(fr.jump_prob[true_k] > 0.5)
    assert fr.jump_prob[sim.jumps == 0].mean() < 0.02
