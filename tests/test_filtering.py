import numpy as np
import pytest

from svuq import BatesParams, HestonParams, ParticleFilter, kernels, simulate

DT = 1 / 252


def test_systematic_resample_counts():
    kernels.seed_rng(0)
    counts = np.bincount(kernels.systematic_resample(np.array([0.5, 0.25, 0.25, 0.0])), minlength=4)
    np.testing.assert_array_equal(counts, [2, 1, 1, 0])


def test_smooth_resample_preserves_distribution():
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    w = np.exp(-0.5 * (x - 1.0) ** 2)
    w /= w.sum()
    kernels.seed_rng(0)
    new, idx = kernels.smooth_resample(x, w)
    assert np.all(np.diff(new) >= 0)                      # output is sorted
    assert x.min() <= new.min() and new.max() <= x.max()
    assert abs(new.mean() - w @ x) < 0.02
    assert np.max(np.abs(x[idx] - new)) < 0.5             # carried index is a neighbour


@pytest.fixture(scope="module")
def heston_path():
    p = HestonParams(0.05, 3.0, 0.04, 0.5, -0.7)
    return p, simulate(p, T=3, n_steps=756, seed=4)


@pytest.mark.parametrize("resampling", ["smooth", "systematic"])
def test_filter_tracks_true_variance(heston_path, resampling):
    p, sim = heston_path
    fr = ParticleFilter(1000, resampling).run(sim.returns, p, DT, np.random.default_rng(0))
    assert np.isfinite(fr.loglik)
    assert np.corrcoef(fr.v_mean, sim.variance)[0, 1] > 0.8
    assert fr.final_particles.shape == (1000,)


def test_filter_is_reproducible(heston_path):
    p, sim = heston_path
    pf = ParticleFilter(300)
    a = pf.run(sim.returns, p, DT, np.random.default_rng(5))
    b = pf.run(sim.returns, p, DT, np.random.default_rng(5))
    np.testing.assert_array_equal(a.v_mean, b.v_mean)


def test_backward_sample_requires_cloud(heston_path):
    p, sim = heston_path
    pf = ParticleFilter(200)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        pf.backward_sample(pf.run(sim.returns, p, DT, rng), sim.returns, p, DT, rng)
    result = pf.run(sim.returns, p, DT, rng, store_cloud=True)
    path = pf.backward_sample(result, sim.returns, p, DT, rng)
    assert path.shape == (sim.returns.size + 1,) and np.all(path > 0)


def test_online_step_matches_batch_statistics(heston_path):
    p, sim = heston_path
    pf, rng = ParticleFilter(2000), np.random.default_rng(0)
    cloud = pf.initial_particles(p, rng)
    means = []
    for r in sim.returns[:200]:
        cloud = pf.step(cloud, r, p, DT, rng).particles
        means.append(cloud.mean())
    batch = pf.run(sim.returns[:200], p, DT, rng).v_mean[1:]
    assert np.corrcoef(means, batch)[0, 1] > 0.95


def test_filter_detects_large_jumps():
    p = BatesParams(0.05, 3.0, 0.04, 0.3, -0.5, lam=4.0, mu_j=-0.25, sigma_j=0.05)
    sim = simulate(p, T=2, n_steps=504, seed=2)
    fr = ParticleFilter(1000).run(sim.returns, p, DT, np.random.default_rng(0))
    assert sim.jumps.sum() >= 3
    assert np.all(fr.jump_prob[sim.jumps == 1] > 0.5)
    assert fr.jump_prob[sim.jumps == 0].mean() < 0.02


def test_invalid_settings():
    with pytest.raises(ValueError):
        ParticleFilter(100, resampling="multinomial")
