import numpy as np
import pytest

from svuq import BatesParams, HestonParams, Priors, estimate, simulate


def test_heston_sampler_short_run():
    p = HestonParams(0.1, 3.0, 0.04, 0.4, -0.7)
    sim = simulate(p, T=3, n_steps=756, seed=1)
    res = estimate(sim.prices, 1 / 252, "heston", Priors.from_beliefs(1 / 252),
                   n_iter=120, burn_in=40, n_particles=300, seed=0)
    s = res.summary(p)
    assert res.settings["rejected_sweeps"] == 0
    # a 120-sweep chain is far too short for inference (sigma's ESS is tiny);
    # this only checks that the sampler lands in the right region
    assert s.loc["theta", "in_ci"] and s.loc["kappa", "in_ci"]
    assert abs(s.loc["sigma", "mean"] - p.sigma) < 0.15
    assert s.loc["rho", "mean"] < -0.3
    assert res.samples.shape[0] == 80
    m, sd = res.v_final
    assert 0 < m < 0.2 and sd > 0


def test_bates_sampler_finds_jumps():
    p = BatesParams(0.1, 3.0, 0.04, 0.3, -0.6, lam=4.0, mu_j=-0.25, sigma_j=0.05)
    sim = simulate(p, T=2, n_steps=504, seed=2)
    pr = Priors.from_beliefs(1 / 252, lam=(1, 1), mu_j=(-0.1, 0.3), sigma_j=0.1)
    res = estimate(sim.prices, 1 / 252, "bates", pr, n_iter=80, burn_in=30, n_particles=300, seed=0)
    s = res.summary(p)
    assert s.loc["mu_j", "in_ci"]
    assert res.samples["n_jumps"].median() == pytest.approx(sim.jumps.sum(), abs=2)


def test_input_validation():
    with pytest.raises(ValueError):
        estimate(np.ones(5), 1 / 252)
    with pytest.raises(ValueError):
        estimate(np.linspace(1, 2, 100), 1 / 252, model="svjj")
