import scipy.stats as stats
import numpy as np
import QuantLib as ql


today = ql.Date(4, 12, 2024)
# Inputs
S = 100  # Spot price
K = 120  # Strike price
r = 0.05  # Risk-free rate
T = 1     # Time to maturity
sigma = 0.2  # Volatility

# Analytical Black-Scholes
d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
N = stats.norm.cdf
bs_price = S * N(d1) - K * np.exp(-r * T) * N(d2)

# QuantLib
payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
exercise = ql.EuropeanExercise(today + ql.Period(int(T * 365), ql.Days))
option = ql.VanillaOption(payoff, exercise)
spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
curve_handle = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))
vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), sigma, ql.Actual365Fixed()))
process = ql.BlackScholesProcess(spot_handle, curve_handle, vol_handle)
option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
quantlib_price = option.NPV()

print(f"The analytical solution is: {bs_price}")

print(f"The quantlib solution is: {quantlib_price}")

assert np.isclose(bs_price, quantlib_price), "Option pricing mismatch"