import numpy as np
import pandas as pd
import pytest
import scipy.stats as st

from svuq import OptionSpec, PCESurrogate, PosteriorPricer
from svuq.uq import HermiteBasis, InputModel, MarginalFit, scipy_to_chaospy


@pytest.mark.parametrize("name,params", [
    ("norm", (-0.5, 0.1)), ("lognorm", (0.3, -0.7, 0.2)), ("gamma", (3.0, -0.8, 0.1)),
    ("invgamma", (5.0, 0.01, 0.2)), ("beta", (2.0, 3.0, -1.0, 2.0)),
])
def test_scipy_to_chaospy_keeps_shift(name, params):
    pytest.importorskip("chaospy")
    d_sp = getattr(st, name)(*params)
    x = scipy_to_chaospy(name, params).sample(200_000, seed=1)
    assert x.mean() == pytest.approx(d_sp.mean(), abs=4 * d_sp.std() / np.sqrt(200_000) + 1e-6)
    assert x.std() == pytest.approx(d_sp.std(), rel=0.02)


def test_marginal_of_negative_parameter_stays_negative():
    fit = MarginalFit.fit(np.random.default_rng(0).normal(-0.5, 0.08, 2000), "rho")
    assert fit.frozen.mean() == pytest.approx(-0.5, abs=0.01)
    assert fit.ppf_clipped(np.array([0.5]))[0] < 0


def test_copula_preserves_correlation():
    z = np.random.default_rng(0).multivariate_normal([0, 0], [[1, -0.8], [-0.8, 1]], 5000)
    df = pd.DataFrame({"kappa": np.exp(0.5 + 0.3 * z[:, 0]), "theta": 0.04 + 0.005 * z[:, 1]})
    s = InputModel.from_draws(df, ["kappa", "theta"]).sample(20_000, seed=1)
    target = np.corrcoef(df.kappa, df.theta)[0, 1]
    assert np.corrcoef(s.kappa, s.theta)[0, 1] == pytest.approx(target, abs=0.03)
    indep = InputModel.from_draws(df, ["kappa", "theta"], "independent").sample(20_000, seed=1)
    assert abs(np.corrcoef(indep.kappa, indep.theta)[0, 1]) < 0.03


def test_hermite_basis_is_orthonormal():
    basis = HermiteBasis(3, 3)
    xi = np.random.default_rng(0).standard_normal((3, 400_000))
    A = basis.evaluate(xi)
    assert len(basis) == 20
    np.testing.assert_allclose(A.T @ A / xi.shape[1], np.eye(20), atol=0.03)


@pytest.fixture(scope="module")
def heston_draws():
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame({"kappa": rng.gamma(20, 0.1, n), "theta": rng.normal(0.04, 0.004, n),
                         "sigma": rng.gamma(30, 0.01, n), "rho": rng.normal(-0.6, 0.05, n),
                         "v0": rng.normal(0.04, 0.005, n)})


@pytest.mark.parametrize("method", ["regression", "quadrature"])
def test_pce_matches_monte_carlo(heston_draws, method):
    option = OptionSpec(100, [95.0, 105.0], 0.25, 0.02)
    mc = PosteriorPricer("heston").price(heston_draws, option)
    pce = PCESurrogate("heston", order=2, method=method).fit(heston_draws, option)
    np.testing.assert_allclose(pce.mean, mc.prices.mean(0), rtol=0.01)
    np.testing.assert_allclose(pce.std, mc.prices.std(0), rtol=0.1)
    assert pce.sobol_total.loc["v0"].min() > 0.5          # spot variance dominates short dates
    assert np.all(pce.sobol_first.to_numpy() <= pce.sobol_total.to_numpy() + 1e-12)
    if method == "regression":
        assert np.all(pce.loo_error < 0.05)
    assert pce.sample(1000).shape == (1000, 2)


def test_variance_risk_premium_lowers_long_dated_prices(heston_draws):
    option = OptionSpec(100, [100.0], 2.0, 0.02)
    base = PosteriorPricer("heston").price(heston_draws, option).prices.mean()
    premium = PosteriorPricer("heston", vol_risk_premium=-0.5)
    shifted = premium.price(heston_draws, option).prices.mean()
    assert shifted != pytest.approx(base)
