import QuantLib as ql
import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt 
import tkinter


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
S0 = 423.95
r  = 0.0475

# Bates model uncertainty distributions
kappa_dist = cp.TruncNormal(0,np.inf, 2.9149, 2.2101) 
theta_dist = cp.Normal(mu=0.03805291552417714, sigma=0.0028068270073092743)
sigma_dist = cp.Gamma(3.2077619579097507, scale=0.0003541020807128636)
rho_dist = cp.Normal(mu=-0.4776912239774035, sigma=0.0371667598502587)
v0_dist = cp.Normal(0.037, 0.005)

# Jump parameter distributions (suggested examples)
mu_j_dist = cp.Normal(-0.9, 0.0001)     # Mean jump size
sigma_j_dist = cp.Uniform(0.029, 0.033) # Volatility of jump size
lambda_dist = cp.Uniform(0.0001, 0.00012)  # Jump intensity (frequency)

joint_dist_bates = cp.J(kappa_dist, theta_dist, sigma_dist,
                       rho_dist, v0_dist,
                       mu_j_dist, sigma_j_dist, lambda_dist)

# Build PCE basis & nodes once
order = 2
polynomials_bates = cp.orth_ttr(order, joint_dist_bates)
samples_bates, weights_bates = cp.generate_quadrature(order, joint_dist_bates, rule='G')
# Define strikes and expiries
strikes = np.arange(395, 451, 5)
expiries_days = [24, 87, 115]

# Loop over expiries and strikes
for expiry in expiries_days:
    T = expiry / 365.0
    print(f"\n=== Expiry = {expiry} days (T={T:.4f}) ===")
    for K in strikes:
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

        # 3) Print
        print(f"Strike {K:3d} →  Mean = {mean_price:6.2f},  Var = {var_price:6.2f}")
