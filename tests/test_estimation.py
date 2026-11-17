import numpy as np
import pytest

from svuq import (
    BatesParams,
    Estimator,
    HestonParams,
    Priors,
    SamplerConfig,
    VolatilityForecaster,
    simulate,
)

DT = 1 / 252


@pytest.fixture(scope="module")
def heston_sim():
    p = HestonParams(0.1, 3.0, 0.04, 0.4, -0.7)
    return p, simulate(p, T=3, n_steps=756, seed=1)


@pytest.fixture(scope="module")
def heston_post(heston_sim):
    _, sim = heston_sim
    est = Estimator("heston", Priors.from_beliefs(DT), SamplerConfig(n_iter=150, burn_in=50,
                                                                     n_particles=300))
    return est.fit(sim.prices, DT, seed=0)


def test_heston_sampler_lands_near_truth(heston_sim, heston_post):
    p, _ = heston_sim
    s = heston_post.summary(p)
    # a 150-sweep chain is far too short for inference; this checks the region only
    assert heston_post.settings["rejected_sweeps"] == 0
    assert s.loc["theta", "in_ci"] and s.loc["kappa", "in_ci"]
    assert abs(s.loc["sigma", "mean"] - p.sigma) < 0.15
    assert s.loc["rho", "mean"] < -0.3
    assert len(heston_post.samples) == 100
    m, sd = heston_post.v_final
    assert 0 < m < 0.2 and sd > 0


def test_same_seed_same_chain(heston_sim):
    _, sim = heston_sim
    est = Estimator("heston", config=SamplerConfig(n_iter=5, burn_in=1, n_particles=100))
    a = est.fit(sim.prices, DT, seed=3).chain
    b = est.fit(sim.prices, DT, seed=3).chain
    assert a.equals(b)


def test_parallel_chains(heston_sim):
    _, sim = heston_sim
    est = Estimator("heston", config=SamplerConfig(n_iter=12, burn_in=2, n_particles=100))
    post = est.fit(sim.prices, DT, n_chains=2, n_jobs=2, seed=0)
    assert post.n_chains == 2 and len(post.samples) == 20
    assert "r_hat" in post.summary().columns
    assert not np.allclose(post.chains[0].draws["kappa"], post.chains[1].draws["kappa"])


def test_bates_sampler_finds_jumps():
    p = BatesParams(0.1, 3.0, 0.04, 0.3, -0.6, lam=4.0, mu_j=-0.25, sigma_j=0.05)
    sim = simulate(p, T=2, n_steps=504, seed=2)
    pr = Priors.from_beliefs(DT, lam=(1, 1), mu_j=(-0.1, 0.3), sigma_j=0.1)
    post = Estimator("bates", pr, SamplerConfig(n_iter=80, burn_in=30, n_particles=300)).fit(
        sim.prices, DT, seed=0)
    assert post.summary(p).loc["mu_j", "in_ci"]
    assert post.samples["n_jumps"].median() == pytest.approx(sim.jumps.sum(), abs=2)
    assert post.jump_prob.shape == (504,)


def test_forecaster(heston_sim, heston_post):
    p, sim = heston_sim
    fc = VolatilityForecaster(heston_post, n_draws=20, n_particles=300, seed=0)
    with pytest.raises(RuntimeError):
        fc.forecast()
    fc.initialise(sim.prices[:-10])
    for price in sim.prices[-10:]:
        now = fc.update(price)
    assert now.vol_lo < now.vol < now.vol_hi
    assert abs(now.vol - np.sqrt(sim.variance[-1])) < 0.1
    table = fc.forecast([1, 21, 2520])
    long_run = np.sqrt(np.mean([q.theta for q in fc.params]))
    assert table.loc[2520, "vol_mean"] == pytest.approx(long_run, rel=0.15)


def test_input_validation():
    with pytest.raises(ValueError):
        Estimator().fit(np.ones(5), DT)
    with pytest.raises(ValueError):
        Estimator("svjj")
    with pytest.raises(ValueError):
        SamplerConfig(n_iter=10, burn_in=10)
