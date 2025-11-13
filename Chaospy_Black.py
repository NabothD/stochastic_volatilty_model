import QuantLib as ql
import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt 
import tkinter


# Define Black-Scholes function using QuantLib
def black_scholes_quantlib(S0, K, T, r, sigma, option_type='call'):
    # Option and market parameters
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if option_type == 'call' else ql.Option.Put, K)

    # Set the maturity date to T years from today
    todays_date = ql.Date().todaysDate()
    maturity_date = todays_date + int(T * 365)
    exercise = ql.EuropeanExercise(maturity_date)

    # Construct the option
    option = ql.VanillaOption(payoff, exercise)

    # Set up the Black-Scholes process
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, r, day_count))
    vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, calendar, sigma, day_count))
    process = ql.BlackScholesProcess(spot_handle, rate_handle, vol_handle)

    # Use analytic engine
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option.NPV()

# Parameters for Black-Scholes model
S0 = 72.34      # Initial stock price
K = 74       # Strike price
T = 1/52         # Time to maturity (1 year)

#---------------------------------------------------------------------------------------------------------------------------

## Define uncertainty distributions for sigma and r
sigma_dist = cp.Normal(0.2, 0.02)  # Volatility (mean 0.2, std 0.02)
r_dist = cp.Normal(0.05, 0.005)    # Risk-free rate (mean 0.05, std 0.005)
joint_dist = cp.J(sigma_dist, r_dist)

# Polynomial Chaos Expansion setup
order = 3
polynomials = cp.orth_ttr(order, joint_dist)
samples, weights = cp.generate_quadrature(order, joint_dist, rule='G')

# Evaluate the Black-Scholes model at each sample point
option_prices = [black_scholes_quantlib(S0, K, T, r, sigma) for sigma, r in samples.T]

# Fit PCE model
expansion_coeffs = cp.fit_quadrature(polynomials, samples, weights, option_prices)

# Now `expansion_coeffs` is a PCE model we can evaluate directly
# Calculate mean and variance using Chaospy's statistical functions
mean_price = cp.E(expansion_coeffs, joint_dist)
var_price = cp.Var(expansion_coeffs, joint_dist)

print(f"Mean Option Price: {mean_price}")
print(f"Variance of Option Price: {var_price}")



#------------------------------------------------------------------------------------------------------------------------------

# Generate random samples from the joint distribution
random_samples = joint_dist.sample(1000)
option_prices_samples = [expansion_coeffs(*sample) for sample in random_samples.T]

# Plot the distribution of option prices
plt.hist(option_prices_samples, bins=30, edgecolor='black', alpha=0.7)
plt.title("Distribution of Option Prices (Black-Scholes with Uncertainty)")
plt.xlabel("Option Price")
plt.ylabel("Frequency")
plt.axvline(mean_price, color='red', linestyle='--', label=f"Mean Price: {mean_price:.2f}")
plt.legend()
plt.show()

