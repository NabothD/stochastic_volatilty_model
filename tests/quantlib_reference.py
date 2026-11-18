"""QuantLib reference prices used to cross-check the Fourier pricer."""

from __future__ import annotations


def quantlib_call(S, K, T, r, q, v0, kappa, theta, sigma, rho, lam=0.0, mu_j=0.0, sigma_j=0.0):
    """Price a European call with QuantLib's analytic Heston / Bates engines."""
    import QuantLib as ql  # noqa: N813

    today = ql.Date(1, 1, 2020)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    maturity = today + int(round(T * 365))
    spot = ql.QuoteHandle(ql.SimpleQuote(S))
    rts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc))
    dts = ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc))
    if lam > 0:
        process = ql.BatesProcess(rts, dts, spot, v0, kappa, theta, sigma, rho, lam, mu_j, sigma_j)
        engine = ql.BatesEngine(ql.BatesModel(process))
    else:
        process = ql.HestonProcess(rts, dts, spot, v0, kappa, theta, sigma, rho)
        engine = ql.AnalyticHestonEngine(ql.HestonModel(process))
    option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, float(K)),
                              ql.EuropeanExercise(maturity))
    option.setPricingEngine(engine)
    return option.NPV()
