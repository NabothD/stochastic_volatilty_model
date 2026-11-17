import numpy as np
import pandas as pd
import pytest
import scipy.stats as st

cp = pytest.importorskip("chaospy")

from svuq.uq import InputModel, fit_marginal, pce_prices, posterior_mc_prices, scipy_to_chaospy


@pytest.mark.parametrize("name,params", [
    ("norm", (-0.5, 0.1)),
    ("lognorm", (0.3, -0.7, 0.2)),
    ("gamma", (3.0, -0.8, 0.1)),        # negative shift, as for rho
    ("invgamma", (5.0, 0.01, 0.2)),
    ("beta", (2.0, 3.0, -1.0, 2.0)),
])
def test_scipy_to_chaospy_keeps_shift(name, params):
    d_sp = getattr(st, name)(*params)
    d_cp = scipy_to_chaospy(name, params)
    x = d_cp.sample(200000, seed=1)
    assert x.mean() == pytest.approx(d_sp.mean(), abs=4 * d_sp.std() / np.sqrt(200000) + 1e-6)
    assert x.std() == pytest.approx(d_sp.std(), rel=0.02)


def test_fit_marginal_negative_parameter_stays_negative():
    rho = np.random.default_rng(0).normal(-0.5, 0.08, 2000)
    fit = fit_marginal(rho, "rho")
    assert fit.frozen.mean() == pytest.approx(-0.5, abs=0.01)
    assert fit.ppf_clipped(np.array([0.5]))[0] < 0


def test_copula_preserves_correlation():
    rng = np.random.default_rng(0)
    z = rng.multivariate_normal([0, 0], [[1, -0.8], [-0.8, 1]], 5000)
    df = pd.DataFrame({"kappa": np.exp(0.5 + 0.3 * z[:, 0]), "theta": 0.04 + 0.005 * z[:, 1]})
    im = InputModel.from_draws(df, ["kappa", "theta"])
    s = im.sample(20000, seed=1)
    assert np.corrcoef(s.kappa, s.theta)[0, 1] == pytest.approx(np.corrcoef(df.kappa, df.theta)[0, 1], abs=0.03)
    indep = InputModel.from_draws(df, ["kappa", "theta"], dependence="independent").sample(20000, seed=1)
    assert abs(np.corrcoef(indep.kappa, indep.theta)[0, 1]) < 0.03


def test_pce_matches_monte_carlo():
    rng = np.random.default_rng(0)
    n = 400
    draws = pd.DataFrame({
        "kappa": rng.gamma(20, 0.1, n), "theta": rng.normal(0.04, 0.004, n),
        "sigma": rng.gamma(30, 0.01, n), "rho": rng.normal(-0.6, 0.05, n),
        "v0": rng.normal(0.04, 0.005, n),
    })
    K = [95.0, 105.0]
    mc = posterior_mc_prices(draws, "heston", 100, K, 0.25, 0.02, n_draws=None)
    pce = pce_prices(draws, "heston", 100, K, 0.25, 0.02, order=2, method="regression")
    np.testing.assert_allclose(pce.mean, mc.prices.mean(0), rtol=0.01)
    np.testing.assert_allclose(pce.std, mc.prices.std(0), rtol=0.1)
    assert np.all(pce.loo_error < 0.05)
    # v0 dominates short-dated price uncertainty
    assert pce.sobol_total.loc["v0"].min() > 0.5
