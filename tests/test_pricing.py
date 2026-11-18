import itertools

import numpy as np
import pytest

from svuq import (
    BatesParams,
    FourierPricer,
    bates_call,
    black_scholes_call,
    heston_call,
    implied_volatility,
    intrinsic_value,
)

PARAM_SETS = [(0.04, 1.5, 0.04, 0.3, -0.7), (0.01, 3.0, 0.09, 1.0, -0.9),
              (0.2, 0.5, 0.05, 0.1, 0.3), (0.04, 1.1, 0.04, 0.01, -0.5)]


CASES = list(itertools.product([1 / 365, 0.25, 5.0], PARAM_SETS, [0.0, 3.0]))


@pytest.mark.parametrize("T,params,lam", CASES)
def test_fixed_quadrature_matches_adaptive(T, params, lam):
    names = ("v0", "kappa", "theta", "sigma", "rho")
    table = dict(zip(names, params, strict=True), lam=lam, mu_j=-0.2, sigma_j=0.15)
    K = 100 * np.exp(np.linspace(-0.6, 0.6, 9) * np.sqrt(max(T, 0.02)))
    fixed = FourierPricer("fixed").call_batch(100, K, T, 0.03, table, 0.01)
    quad = FourierPricer("quad").call_batch(100, K, T, 0.03, table, 0.01)
    np.testing.assert_allclose(fixed, quad, atol=1e-8)


def test_batch_matches_single_calls():
    rng = np.random.default_rng(0)
    table = dict(v0=rng.uniform(0.02, 0.06, 5), kappa=rng.uniform(1, 4, 5),
                 theta=rng.uniform(0.02, 0.06, 5), sigma=rng.uniform(0.2, 0.6, 5),
                 rho=rng.uniform(-0.8, -0.4, 5))
    batch = FourierPricer().call_batch(100, [90, 110], 0.5, 0.02, table)
    single = [heston_call(100, [90, 110], 0.5, 0.02, *[v[i] for v in table.values()])
              for i in range(5)]
    # a batch shares one quadrature grid, so agreement is to quadrature accuracy
    np.testing.assert_allclose(batch, single, atol=1e-8)


def test_heston_reduces_to_black_scholes():
    K = np.array([80.0, 100.0, 120.0])
    np.testing.assert_allclose(heston_call(100, K, 1.0, 0.05, 0.04, 1.0, 0.04, 1e-5, 0.0),
                               black_scholes_call(100, K, 1.0, 0.05, 0.2), rtol=1e-6)


def test_black_scholes_reference_value():
    assert black_scholes_call(100, 120, 1.0, 0.05, 0.2) == pytest.approx(3.2475, abs=1e-4)


def test_bates_without_jumps_equals_heston():
    args = (100, 105, 0.5, 0.03, 0.05, 2.0, 0.04, 0.5, -0.7)
    assert bates_call(*args, 0.0, -0.1, 0.2) == pytest.approx(heston_call(*args), rel=1e-10)


def test_bates_is_martingale():
    # a call with K -> 0 is worth S only if the jump compensator is included
    assert bates_call(100, 1e-6, 1.0, 0.02, 0.04, 1.5, 0.04, 0.3, -0.5, 3.0, -0.3, 0.2) == \
        pytest.approx(100.0, abs=1e-4)


def test_pricer_call_with_params_object():
    p = BatesParams(0.1, 2.0, 0.04, 0.3, -0.6, lam=1.0, mu_j=-0.1, sigma_j=0.1)
    assert FourierPricer().call(100, 100, 0.5, 0.01, p, v0=0.04) == pytest.approx(
        bates_call(100, 100, 0.5, 0.01, 0.04, 2.0, 0.04, 0.3, -0.6, 1.0, -0.1, 0.1))


def test_implied_vol_round_trip():
    p = black_scholes_call(100, 110, 0.25, 0.01, 0.27)
    assert implied_volatility(p, 100, 110, 0.25, 0.01) == pytest.approx(0.27, abs=1e-8)
    assert np.isnan(implied_volatility(0.0, 100, 110, 0.25, 0.01))


def test_intrinsic_value():
    np.testing.assert_allclose(intrinsic_value(110, [100, 120], 1.0, 0.0), [10.0, 0.0])


@pytest.mark.parametrize("S,K,T,r,v0,kappa,theta,sigma,rho", [
    (100, 110, 30 / 365, 0.1, 0.049, 1.1, 0.04, 0.3, -0.5),
    (423.95, 405, 24 / 365, 0.05, 0.05, 0.7, 0.19, 0.5, -0.9),
    (100, 100, 5.0, 0.02, 0.04, 1.5, 0.06, 0.8, -0.7),
])
def test_against_quantlib(S, K, T, r, v0, kappa, theta, sigma, rho):
    pytest.importorskip("QuantLib")
    from quantlib_reference import quantlib_call

    T = round(T * 365) / 365
    assert heston_call(S, K, T, r, v0, kappa, theta, sigma, rho) == pytest.approx(
        quantlib_call(S, K, T, r, 0.0, v0, kappa, theta, sigma, rho), abs=1e-8)
    assert bates_call(S, K, T, r, v0, kappa, theta, sigma, rho, 2.0, -0.1, 0.2) == pytest.approx(
        quantlib_call(S, K, T, r, 0.0, v0, kappa, theta, sigma, rho, 2.0, -0.1, 0.2), abs=1e-8)
