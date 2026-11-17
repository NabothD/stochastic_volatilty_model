import numpy as np
import pytest

from svuq.pricing import (bates_call, black_scholes_call, heston_call, implied_volatility,
                          intrinsic_value)


def test_heston_reduces_to_black_scholes_when_vol_of_vol_vanishes():
    K = np.array([80.0, 100.0, 120.0])
    h = heston_call(100, K, 1.0, 0.05, 0.04, 1.0, 0.04, 1e-5, 0.0)
    bs = black_scholes_call(100, K, 1.0, 0.05, 0.2)
    np.testing.assert_allclose(h, bs, rtol=1e-6)


def test_black_scholes_reference_value():
    # thesis Section 5.2: S=100, K=120, r=5%, T=1, vol=20% -> 3.25
    assert black_scholes_call(100, 120, 1.0, 0.05, 0.2) == pytest.approx(3.2475, abs=1e-4)


def test_bates_without_jumps_equals_heston():
    args = (100, 105, 0.5, 0.03, 0.05, 2.0, 0.04, 0.5, -0.7)
    assert bates_call(*args, 0.0, -0.1, 0.2) == pytest.approx(heston_call(*args), rel=1e-10)


def test_bates_put_call_parity_and_martingale():
    # the call with K -> 0 must equal S (requires the jump compensator)
    c = bates_call(100, 1e-6, 1.0, 0.02, 0.04, 1.5, 0.04, 0.3, -0.5, 3.0, -0.3, 0.2)
    assert c == pytest.approx(100.0, abs=1e-4)


def test_implied_vol_round_trip():
    p = black_scholes_call(100, 110, 0.25, 0.01, 0.27)
    assert implied_volatility(p, 100, 110, 0.25, 0.01) == pytest.approx(0.27, abs=1e-8)


def test_intrinsic_value():
    np.testing.assert_allclose(intrinsic_value(110, [100, 120], 1.0, 0.0), [10.0, 0.0])


@pytest.mark.parametrize("args", [
    (100, 110, 30 / 365, 0.1, 0.049, 1.1, 0.04, 0.3, -0.5),
    (423.95, 405, 24 / 365, 0.05, 0.05, 0.7, 0.19, 0.5, -0.9),
    (100, 100, 5.0, 0.02, 0.04, 1.5, 0.06, 0.8, -0.7),
])
def test_against_quantlib(args):
    ql = pytest.importorskip("QuantLib")  # noqa: F841
    from svuq.pricing import bates_call_quantlib, heston_call_quantlib
    assert heston_call(*args) == pytest.approx(heston_call_quantlib(*args), rel=1e-8, abs=1e-8)
    jumps = (2.0, -0.1, 0.2)
    assert bates_call(*args, *jumps) == pytest.approx(bates_call_quantlib(*args, *jumps), rel=1e-8, abs=1e-8)
