import numpy as np
import pytest

from svuq import HestonParams, Priors, simulate
from svuq.conjugate import ConjugateSampler

DT = 1 / 252


@pytest.fixture(scope="module")
def long_path():
    # Feller ratio 2.7, so the Euler path stays away from zero
    p = HestonParams(0.1, 3.0, 0.04, 0.3, -0.7)
    sim = simulate(p, T=40, n_steps=40 * 252, seed=10)
    assert sim.variance.min() > 1e-3
    return p, sim


def test_posteriors_concentrate_on_truth_given_true_variance(long_path):
    p, sim = long_path
    rng = np.random.default_rng(0)
    cs = ConjugateSampler(Priors.from_beliefs(DT), DT)
    R, v = sim.returns, sim.variance
    vd = [cs.draw_variance_params(v, rng) for _ in range(400)]
    assert np.mean([d.sigma for d in vd]) == pytest.approx(p.sigma, rel=0.05)
    assert np.mean([d.theta for d in vd]) == pytest.approx(p.theta, rel=0.25)
    assert np.mean([d.kappa for d in vd]) == pytest.approx(p.kappa, rel=0.35)
    rho = [cs.draw_rho(R, v, p.mu, p.kappa, p.theta, rng) for _ in range(400)]
    assert np.mean(rho) == pytest.approx(p.rho, abs=0.05)
    mu = [cs.draw_mu(R, v, rng) for _ in range(400)]
    assert abs(np.mean(mu) - p.mu) < 3 * np.std(mu)


def test_jump_posterior():
    rng = np.random.default_rng(0)
    cs = ConjugateSampler(Priors.from_beliefs(DT), DT)
    J = np.zeros(2520, int)
    J[rng.choice(2520, 30, replace=False)] = 1
    Z = np.where(J, rng.normal(-0.2, 0.1, 2520), 0.0)
    draws = [cs.draw_jump_params(J, Z, 10.0, rng) for _ in range(2000)]
    assert np.mean([d.lam for d in draws]) == pytest.approx(3.0, rel=0.15)
    assert np.mean([d.mu_j for d in draws]) == pytest.approx(-0.2, abs=0.05)
    assert np.mean([d.sigma_j for d in draws]) == pytest.approx(0.1, rel=0.3)


def test_rho_prior_matches_requested_moments():
    pr = Priors.from_beliefs(DT, sigma=0.3, rho=(-0.5, 0.15))
    w0 = pr.omega_b / (pr.omega_a - 1)
    psi = np.random.default_rng(0).normal(pr.psi_mean, np.sqrt(w0 / pr.psi_prec), 200_000)
    rho = psi / np.sqrt(psi ** 2 + w0)
    assert rho.mean() == pytest.approx(-0.5, abs=0.03)
    assert rho.std() == pytest.approx(0.15, rel=0.25)


def test_beta_prior_reproduces_kappa_theta_means():
    pr = Priors.from_beliefs(DT, kappa=(2.0, 0.5), theta=(0.05, 0.01))
    kappa = (1 - pr.beta_mean[1]) / DT
    assert kappa == pytest.approx(2.0)
    assert pr.beta_mean[0] / (kappa * DT) == pytest.approx(0.05)


def test_initial_params_follow_priors():
    pr = Priors.from_beliefs(DT, kappa=(2.0, 1.0), theta=(0.05, 0.02), sigma=0.3, rho=(-0.5, 0.2))
    p = pr.initial_params("bates", DT)
    assert p.kappa == pytest.approx(2.0) and p.theta == pytest.approx(0.05)
    assert p.sigma == pytest.approx(0.3) and p.rho == pytest.approx(-0.5, abs=1e-6)


def test_thesis_precisions_are_one_over_variance():
    assert Priors.thesis_synthetic().eta_prec == pytest.approx(1e6)
