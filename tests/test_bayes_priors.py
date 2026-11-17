import numpy as np
import pytest

from svuq import HestonParams, Priors, simulate
from svuq import bayes


@pytest.fixture(scope="module")
def long_path():
    # Feller ratio 2*kappa*theta/sigma^2 = 2.7 so the Euler path stays away from zero
    p = HestonParams(0.1, 3.0, 0.04, 0.3, -0.7)
    sim = simulate(p, T=40, n_steps=40 * 252, seed=10)
    assert sim.variance.min() > 1e-3
    return p, sim


def test_conjugate_updates_recover_truth_given_true_variance(long_path):
    """With the true latent path, the regression posteriors concentrate on the truth."""
    p, sim = long_path
    dt, rng = 1 / 252, np.random.default_rng(0)
    pr = Priors.from_beliefs(dt)
    R, v = sim.returns, sim.variance
    vd = [bayes.draw_kappa_theta_sigma(v, dt, pr, rng) for _ in range(400)]
    assert np.mean([d.sigma for d in vd]) == pytest.approx(p.sigma, rel=0.05)
    assert np.mean([d.theta for d in vd]) == pytest.approx(p.theta, rel=0.25)
    assert np.mean([d.kappa for d in vd]) == pytest.approx(p.kappa, rel=0.35)
    rho = [bayes.draw_rho(R, v, dt, p.mu, p.kappa, p.theta, pr, rng) for _ in range(400)]
    assert np.mean(rho) == pytest.approx(p.rho, abs=0.05)
    mu = [bayes.draw_mu(R, v, dt, pr, rng) for _ in range(400)]
    assert abs(np.mean(mu) - p.mu) < 3 * np.std(mu)


def test_rho_prior_matches_requested_moments():
    pr = Priors.from_beliefs(1 / 252, sigma=0.3, rho=(-0.5, 0.15))
    rng = np.random.default_rng(0)
    w0 = pr.omega_b / (pr.omega_a - 1)
    psi = rng.normal(pr.psi_mean, np.sqrt(w0 / pr.psi_prec), 200000)  # psi | omega = w0
    rho = psi / np.sqrt(psi ** 2 + w0)
    assert rho.mean() == pytest.approx(-0.5, abs=0.03)
    assert rho.std() == pytest.approx(0.15, rel=0.25)


def test_beta_prior_reproduces_kappa_theta_means():
    dt = 1 / 252
    pr = Priors.from_beliefs(dt, kappa=(2.0, 0.5), theta=(0.05, 0.01))
    kappa = (1 - pr.beta_mean[1]) / dt
    assert kappa == pytest.approx(2.0)
    assert pr.beta_mean[0] / (kappa * dt) == pytest.approx(0.05)


def test_jump_posterior():
    rng = np.random.default_rng(0)
    pr = Priors.from_beliefs(1 / 252)
    J = np.zeros(2520, int)
    J[rng.choice(2520, 30, replace=False)] = 1
    Z = np.where(J, rng.normal(-0.2, 0.1, 2520), 0.0)
    d = [bayes.draw_jump_params(J, Z, 10.0, pr, rng) for _ in range(2000)]
    assert np.mean([x.lam for x in d]) == pytest.approx(3.0, rel=0.15)
    assert np.mean([x.mu_j for x in d]) == pytest.approx(-0.2, abs=0.05)
    assert np.mean([x.sigma_j for x in d]) == pytest.approx(0.1, rel=0.3)


def test_thesis_precision_bug_documented():
    # the original scripts used 1/sqrt(sd^2) instead of 1/sd^2
    assert Priors.thesis_synthetic().eta_prec == pytest.approx(1e6)
