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

# Parameters for Heston model
S0 = 423.95     # Initial stock price
K = 405       # Strike price
T = 1/15.21         # Time to maturity (24 days)
# r = 0.0475      # Risk-free interest rate - this can also be perhaps an uncertain parameters. 
r= 0.2933



# Define distributions for uncertain parameters - This is where I can come up with proposed distributions 
# kappa_dist = cp.LogNormal(0.001,0.4)   # Mean reversion rate
theta_dist = cp.Gamma(2.756933963599759, scale=0.01805773096024975) # Long-run average variance
sigma_dist = cp.Gamma(11.18896657103237, scale=5.17413985956255e-05)   # Volatility of volatility
rho_dist = cp.Gamma(2.090290352181386, scale=0.005721873840276443)        # Correlation
v0_dist = cp.Normal(0.049, 0.009)     # Initial variance


kappa_dist = cp.TruncNormal(0,np.inf, 0.6203, 0.6415)   # Mean reversion rate
# theta_dist = cp.Uniform(0.13,0.2) # Long-run average variance
# sigma_dist = cp.Uniform(0.002,0.003)   # Volatility of volatility
# rho_dist = cp.Uniform(-1,-0.95)        # Correlation
# v0_dist = cp.Uniform(0.05, 0.08)     # Initial variance

# Joint distribution of all uncertain parameters
joint_dist = cp.J(kappa_dist, theta_dist, sigma_dist, rho_dist, v0_dist)

# Set up Polynomial Chaos Expansion (PCE)
order = 4
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


# -------------------------------------------------------------
# 1)  Sobol sensitivity indices from the PCE surrogate
# -------------------------------------------------------------
par_names = ["kappa", "theta", "sigma", "rho", "v0"]

# First‐order indices  S_i   (main effects)
S_first = cp.Sens_m(expansion_coeffs, joint_dist)

# Total indices  S_i^T  (main + all interactions)
S_total = cp.Sens_t(expansion_coeffs, joint_dist)

# -------------------------------------------------------------
# 1a)  Drop the last element (v0) for reporting
# -------------------------------------------------------------
# keep only parameters 0–3
keep = slice(0, 4)
par_names_no_v0   = [par_names[i] for i in range(4)]
S_first_no_v0     = S_first[keep]
S_total_no_v0     = S_total[keep]

# -------------------------------------------------------------
# 2)  Pretty print and save to Excel (without v0)
# -------------------------------------------------------------
df_sobol = pd.DataFrame({
    "parameter":   par_names_no_v0,
    "first_order": S_first_no_v0,
    "total":       S_total_no_v0
})
print("\nSobol indices for option price (excluding v0):")
print(df_sobol.round(4))

with pd.ExcelWriter("heston_prices.xlsx", engine="openpyxl",
                    mode="a", if_sheet_exists="replace") as writer:
    df_sobol.to_excel(writer, sheet_name="sobol_indices", index=False)

# -------------------------------------------------------------
# 3)  Quick bar‐chart visualisation (without v0)
# -------------------------------------------------------------
plt.figure(figsize=(6,4))
x = np.arange(len(par_names_no_v0))
width = 0.35

plt.bar(x - width/2, S_total_no_v0, label="Total",  linewidth=1.2)
plt.bar(x + width/2, S_first_no_v0, label="First",  linewidth=1.2)
plt.xticks(x, par_names_no_v0, fontsize=12)
plt.ylabel("Sobol index", fontsize=14)
plt.title("Global sensitivity of option price\n(excluding $v_0$)", fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()



# import QuantLib as ql
# import chaospy as cp
# import numpy as np
# import matplotlib.pyplot as plt 
# import tkinter
# import pandas as pd

# def heston_quantlib(S0, K, T, r,
#                     kappa, theta, sigma, rho, v0,
#                     option_type='call'):
#     # ensure K is float for QuantLib
#     K = float(K)

#     calendar      = ql.NullCalendar()
#     todays_date   = ql.Date().todaysDate()
#     maturity_date = todays_date + int(T * 365)
#     day_count     = ql.Actual365Fixed() 
    
#     spot_handle     = ql.QuoteHandle(ql.SimpleQuote(S0))
#     rate_handle     = ql.YieldTermStructureHandle(
#                          ql.FlatForward(0, calendar, r, day_count))
#     dividend_handle = ql.YieldTermStructureHandle(
#                          ql.FlatForward(0, calendar, 0.0, day_count))

#     heston_process = ql.HestonProcess(rate_handle, dividend_handle,
#                                       spot_handle, v0, kappa,
#                                       theta, sigma, rho)
    
#     payoff   = ql.PlainVanillaPayoff(
#                    ql.Option.Call if option_type=='call'
#                                  else ql.Option.Put,
#                    K)
#     exercise = ql.EuropeanExercise(maturity_date)
#     option   = ql.VanillaOption(payoff, exercise)
#     engine   = ql.AnalyticHestonEngine(ql.HestonModel(heston_process))
#     option.setPricingEngine(engine)
    
#     return option.NPV()

# # Fixed market params
# S0 = 423.95
# # r  = 0.2933  # keep r constant
# # r = 0.0475
# # r_dist = cp.Normal(0.0475, 0.005)   # mean=4.75%, σ=0.5% 
# r_dist = cp.Normal(mu=0.2933, sigma=0.12671266590058367) 

# # Uncertainty distributions
# kappa_dist   = cp.TruncNormal(0, np.inf, 0.6203, 0.6415)
# theta_dist   = cp.Gamma(2.756933963599759, scale=0.01805773096024975)
# sigma_dist   = cp.Gamma(11.18896657103237, scale=5.17413985956255e-05)
# rho_dist     = cp.Gamma(2.090290352181386, scale=0.005721873840276443)
# v0_dist      = cp.Normal(0.049, 0.009)

# joint_dist = cp.J(r_dist, kappa_dist, theta_dist, sigma_dist, rho_dist, v0_dist)

# # Build PCE basis & nodes once
# order = 3
# polynomials = cp.orth_ttr(order, joint_dist)
# samples, weights = cp.generate_quadrature(order, joint_dist, rule='G')

# # Strikes & expiries
# strikes = np.arange(395, 451, 5)
# expiries_days = [24, 87, 115]

# # Loop over expiries and strikes
# results = []  # to collect (expiry, strike, mean, var)

# for expiry in expiries_days:
#     T = expiry / 365.0
#     print(f"\n=== Expiry = {expiry} days (T={T:.4f}) ===")
#     for K in strikes:
#         # 1) Evaluate Heston at each quadrature node
#         price_evals = [
#             heston_quantlib(S0, K, T, r,
#                             kappa, theta, sigma, rho, v0)
#             for r, kappa, theta, sigma, rho, v0 in samples.T
#         ]
#         # 2) Fit PCE & extract moments
#         expansion = cp.fit_quadrature(polynomials,
#                                       samples,
#                                       weights,
#                                       price_evals)
#         mean_price = cp.E(expansion, joint_dist)
#         var_price  = cp.Var(expansion, joint_dist)

#         print(f"Strike {K:3d} → Mean = {mean_price:6.2f}, Var = {var_price:6.2f}")
#         results.append((expiry, K, float(mean_price), float(var_price)))

# # Optional: dump to DataFrame/Excel
# df = pd.DataFrame(results,
#                   columns=['ExpiryDays','Strike','MeanPrice','VarPrice'])
# df.to_excel('heston_pce_grid.xlsx', index=False)
# print("\nWrote heston_pce_grid.xlsx with results for all expiries & strikes.")
