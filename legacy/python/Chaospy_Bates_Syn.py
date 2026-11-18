import QuantLib as ql
import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt 
import tkinter
import pandas as pd


def bates_quantlib(S0, K, T, r, kappa, theta, sigma, rho, v0, mu_j, sigma_j, lambd, option_type='call'):
    K = float(K)
    calendar = ql.NullCalendar()
    todays_date = ql.Date().todaysDate()
    maturity_date = todays_date + int(T * 365)
    day_count = ql.Actual365Fixed()

    # Market handles
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, r, day_count))
    dividend_handle = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, 0.0, day_count))

    # Bates process parameters
    bates_process = ql.BatesProcess(rate_handle, dividend_handle, spot_handle,
                                    v0, kappa, theta, sigma, rho,
                                    lambd, mu_j, sigma_j)

    payoff = ql.PlainVanillaPayoff(ql.Option.Call if option_type == 'call' else ql.Option.Put, K)
    exercise = ql.EuropeanExercise(maturity_date)
    option = ql.VanillaOption(payoff, exercise)

    # Pricing engine
    engine = ql.BatesEngine(ql.BatesModel(bates_process))
    option.setPricingEngine(engine)

    return option.NPV()

# Parameters
S0 = 100
K = 110
r  = 0.1
T=1/12

# Bates model uncertainty distributions
kappa_dist = cp.TruncNormal(0,np.inf, 1.05, 0.9751) 
theta_dist = cp.Normal(mu=0.03733713633494201, sigma=0.006331966503020463)
sigma_dist = cp.Gamma(2.817766584536681, scale=0.0003934711935465393)
rho_dist = cp.Gamma(11.630775511393672, scale=0.01576758170665299)
v0_dist = cp.Normal(0.049, 0.005)

# Jump parameter distributions (suggested examples)
mu_j_dist = cp.Normal(mu=-0.07828551338172796, sigma=3.320017457586208e-08)     # Mean jump size
sigma_j_dist = cp.Uniform(0.00029, 0.00033) # Volatility of jump size
lambda_dist = cp.Uniform(0.0001, 0.00012)  # Jump intensity (frequency)

joint_dist_bates = cp.J(kappa_dist, theta_dist, sigma_dist,
                       rho_dist, v0_dist,
                       mu_j_dist, sigma_j_dist, lambda_dist)

# Build PCE basis & nodes once
order = 2
polynomials_bates = cp.orth_ttr(order, joint_dist_bates)
samples_bates, weights_bates = cp.generate_quadrature(order, joint_dist_bates, rule='G')
# Define strikes and expiries
# 1) Evaluate model at quadrature nodes
option_prices = [
    bates_quantlib(S0, K, T, r,
                    kappa, theta, sigma, rho, v0,
                    mu_j, sigma_j, lambd)
    for (kappa, theta, sigma,
            rho, v0, mu_j,
            sigma_j, lambd) in samples_bates.T
]

# 2) Fit PCE & extract moments
expansion_coeffs_bates = cp.fit_quadrature(
    polynomials_bates,
    samples_bates,
    weights_bates,
    option_prices
)
mean_price = cp.E(expansion_coeffs_bates, joint_dist_bates)
var_price  = cp.Var(expansion_coeffs_bates, joint_dist_bates)

print(f"Mean Option Price (Bates): {mean_price}")
print(f"Variance of Option Price (Bates Model): {var_price}")

True_price = 30.75

# Generate random samples from the joint distribution for visualization
random_samples = joint_dist_bates.sample(1000)
option_prices_samples = [expansion_coeffs_bates(*sample) for sample in random_samples.T]

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
with pd.ExcelWriter('bates_prices.xlsx', engine='openpyxl') as writer:
    df_samples.to_excel(writer, sheet_name='samples', index=False)

    # 2) write the summaries into sheet “summary”
    df_summary = pd.DataFrame({
        'MeanPrice': [mean_price],
        'VariancePrice': [var_price],
        'TruePrice': [True_price]
    })
    df_summary.to_excel(writer, sheet_name='summary', index=False)

print("Wrote bates_prices.xlsx with sheets: samples, summary")
