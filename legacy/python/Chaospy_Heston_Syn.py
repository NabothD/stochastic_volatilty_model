import QuantLib as ql
import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt 
import tkinter
import pandas as pd

def heston_quantlib(S0, K, T, r, kappa, theta, sigma, rho, v0, option_type='call'):
    # Set up QuantLib dates
    calendar = ql.NullCalendar()
    todays_date = ql.Date().todaysDate()
    maturity_date = todays_date + int(T * 365)
    day_count = ql.Actual365Fixed() 
    
    # Market data handles
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, r, day_count))
    dividend_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, 0.0, day_count))  # Zero dividend yield
    
    # Heston process parameters
    heston_process = ql.HestonProcess(rate_handle, dividend_handle, spot_handle,
                                      v0, kappa, theta, sigma, rho)
    
    # Option setup
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if option_type == 'call' else ql.Option.Put, K)
    exercise = ql.EuropeanExercise(maturity_date)
    option = ql.VanillaOption(payoff, exercise)
    
    # Pricing engine
    engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process))
    option.setPricingEngine(engine)
    
    # Calculate the option price
    return option.NPV()

# Parameters
S0 = 100
K = 110
r  = 0.1
T=1/12



# Define distributions for uncertain parameters - This is where I can come up with proposed distributions 
# kappa_dist = cp.LogNormal(0.001,0.4)   # Mean reversion rate
theta_dist = cp.InverseGamma(16.318952322228498, scale=0.7424758605345488) # Long-run average variance
sigma_dist = cp.Gamma(3.0038953426024926, scale=0.00034025916519986657)  # Volatility of volatility
rho_dist = cp.Normal(mu=-0.5435687969031379, sigma=0.08037196364061963)        # Correlation
v0_dist = cp.Normal(0.046, 0.005)     # Initial variance


kappa_dist = cp.TruncNormal(0,np.inf, 1.008, 0.9517)   # Mean reversion rate
# theta_dist = cp.Uniform(0.13,0.2) # Long-run average variance
# sigma_dist = cp.Uniform(0.002,0.003)   # Volatility of volatility
# rho_dist = cp.Uniform(-1,-0.95)        # Correlation
# v0_dist = cp.Uniform(0.05, 0.08)     # Initial variance

# Joint distribution of all uncertain parameters
joint_dist = cp.J(kappa_dist, theta_dist, sigma_dist, rho_dist, v0_dist)

# Set up Polynomial Chaos Expansion (PCE)
order = 2
polynomials = cp.orth_ttr(order, joint_dist)
samples, weights = cp.generate_quadrature(order, joint_dist, rule='G')

# Evaluate Heston model at each sample point
option_prices = [heston_quantlib(S0, K, T, r, kappa, theta, sigma, rho, v0) 
                 for kappa, theta, sigma, rho, v0 in samples.T]

# Fit the PCE model
expansion_coeffs = cp.fit_quadrature(polynomials, samples, weights, option_prices)

# Calculate mean and variance of the option price
mean_price = cp.E(expansion_coeffs, joint_dist)
var_price = cp.Var(expansion_coeffs, joint_dist)

print(f"Mean Option Price (Heston Model): {mean_price}")
print(f"Variance of Option Price (Heston Model): {var_price}")

True_price = 30.75

# Generate random samples from the joint distribution for visualization
random_samples = joint_dist.sample(1000)
option_prices_samples = [expansion_coeffs(*sample) for sample in random_samples.T]

# Plot the distribution of option prices
plt.hist(option_prices_samples, bins=30, edgecolor='black', alpha=0.7)
plt.title("Distribution of Option Prices (Heston Model with Uncertainty)")
plt.xlabel("Option Price")
plt.ylabel("Frequency")
plt.axvline(mean_price, color='red', linestyle='--', label=f"Mean Price: {mean_price:.2f}")
# plt.axvline(True_price, color='Blue', linestyle='-', label=f"True Price")
plt.legend()
plt.show()


df_samples = pd.DataFrame({
    'OptionPrice': option_prices_samples
})
with pd.ExcelWriter('heston_prices.xlsx', engine='openpyxl') as writer:
    df_samples.to_excel(writer, sheet_name='samples', index=False)

    # 2) write the summaries into sheet “summary”
    df_summary = pd.DataFrame({
        'MeanPrice': [mean_price],
        'VariancePrice': [var_price],
        'TruePrice': [True_price]
    })
    df_summary.to_excel(writer, sheet_name='summary', index=False)

print("Wrote heston_prices.xlsx with sheets: samples, summary")
