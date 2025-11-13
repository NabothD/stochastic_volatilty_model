#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import chaospy as cp
import QuantLib as ql

###############################################################################
# 1. Define Parameter Distributions via Chaospy
###############################################################################

# Example: approximate means and standard deviations for Heston parameters
kappa_mean,   kappa_std   = 2.0,   0.3
theta_mean,   theta_std   = 0.04,  0.01
sigma_mean,   sigma_std   = 0.5,   0.1
rho_mean,     rho_std     = -0.5,  0.05
v0_mean,      v0_std      = 0.04,  0.01

# Example: approximate means and std dev for jump parameters (Bates model)
lambda_mean,  lambda_std  = 0.1,   0.02
muJ_mean,     muJ_std     = 0.0,   0.1
sigmaJ_mean,  sigmaJ_std  = 0.2,   0.05

# For strictly positive parameters, one might prefer a LogNormal or truncated normal.
# For demonstration, we'll keep it simple here.
dist_kappa   = cp.Normal(kappa_mean, kappa_std)
dist_theta   = cp.LogNormal(theta_mean, theta_std)
dist_sigma   = cp.LogNormal(sigma_mean, sigma_std)
dist_rho     = cp.Normal(rho_mean, rho_std)       # Must ensure -1 < rho < 1 if sampling
dist_v0      = cp.LogNormal(v0_mean, v0_std)

dist_lambda  = cp.LogNormal(lambda_mean, lambda_std)
dist_muJ     = cp.Normal(muJ_mean, muJ_std)
dist_sigmaJ  = cp.LogNormal(sigmaJ_mean, sigmaJ_std)

# Joint distribution for the pure Heston model (no jumps)
dist_no_jump = cp.J(dist_kappa, dist_theta, dist_sigma, dist_rho, dist_v0)

# Joint distribution for the Bates (Heston + jumps) model
dist_with_jump = cp.J(
    dist_kappa,
    dist_theta,
    dist_sigma,
    dist_rho,
    dist_v0,
    dist_lambda,
    dist_muJ,
    dist_sigmaJ
)

###############################################################################
# 2. Helper Functions: QuantLib Pricing for Heston and Bates
###############################################################################

def heston_call_price(
    S0, strike, r, T,
    kappa, theta, sigma, rho, v0,
    day_count=ql.Actual365Fixed(),
    steps=200,  # for the time discretization if needed
    engine_type="Analytic"
):
    """
    Price a European call option using Heston model in QuantLib.
    
    Arguments:
    ----------
    S0       : Spot price
    strike   : Option strike
    r        : Risk-free rate
    T        : Time to maturity (in years)
    kappa    : Mean reversion speed
    theta    : Long-term variance
    sigma    : Vol of vol
    rho      : Correlation
    v0       : Initial variance
    day_count: QuantLib day count convention
    steps    : If using a numerical method (not analytic)
    engine_type : "Analytic" for AnalyticHestonEngine or e.g. "FD" for FiniteDifferences
    
    Returns:
    --------
    Price of the European call option under Heston.
    """
    # 1) Set up the evaluation date
    todays_date = ql.Date(1, 1, 2025)
    ql.Settings.instance().evaluationDate = todays_date

    # 2) Construct yield/dividend/spot handles
    # Flat yield curve for the sake of example
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(todays_date, r, day_count)
    )
    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(todays_date, 0.0, day_count)
    )
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    
    # 3) Heston process
    heston_process = ql.HestonProcess(
        flat_ts, dividend_ts, spot_handle, v0, kappa, theta, sigma, rho
    )
    heston_model = ql.HestonModel(heston_process)
    
    if engine_type == "Analytic":
        engine = ql.AnalyticHestonEngine(heston_model)
    else:
        # Example of a numerical engine (just for reference)
        engine = ql.FdHestonVanillaEngine(heston_model, steps, steps, steps)
    
    # 4) Set up the European option
    maturity_date = todays_date + int(365 * T)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    exercise = ql.EuropeanExercise(maturity_date)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)
    
    # 5) Return the option NPV
    return option.NPV()


def bates_call_price(
    S0, strike, r, T,
    kappa, theta, sigma, rho, v0,
    lambda_j, muJ, sigmaJ,
    day_count=ql.Actual365Fixed(),
    steps=200,
    engine_type="Analytic"
):
    """
    Price a European call option using Bates model (Heston + jumps) in QuantLib.
    
    Bates extends Heston with a jump component:
    - jump intensity = lambda_j
    - jump size distribution ~ Lognormal with mean ln(1 + muJ) - or similar
    - jump vol = sigmaJ
    
    You may need to confirm the sign conventions or any shift for muJ if 
    you are calibrating a different parametrization.
    """
    todays_date = ql.Date(1, 1, 2025)
    ql.Settings.instance().evaluationDate = todays_date

    # Flat yield curve
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(todays_date, r, day_count)
    )
    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(todays_date, 0.0, day_count)
    )
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    
    # Bates process
    # Note: The Bates constructor typically expects the log of jump distribution
    # parameters. Consult QuantLib doc or examples for the exact usage.
    bates_process = ql.BatesProcess(
        flat_ts, dividend_ts, spot_handle,
        v0, kappa, theta, sigma, rho,
        lambda_j, muJ, sigmaJ
    )
    bates_model = ql.BatesModel(bates_process)
    
    if engine_type == "Analytic":
        engine = ql.FdBatesVanillaEngine(bates_model, steps, steps, steps) # you can adjust the grids as needed
    else:
        engine = ql.FdBatesVanillaEngine(bates_model, steps, steps, steps)
    
    maturity_date = todays_date + int(365 * T)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    exercise = ql.EuropeanExercise(maturity_date)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)
    
    return option.NPV()

###############################################################################
# 3. Generate Samples and Evaluate Model
###############################################################################

def main():
    # Configuration
    num_samples = 200   # Number of parameter samples
    poly_order  = 3     # Polynomial chaos expansion order

    # Underlying option setup
    S0 = 100.0   # e.g., current S&P 500 level
    K  = 105.0   # Strike
    r  = 0.02     # Risk-free rate
    T  = 3.0      # Maturity in years

    # ---------- NO-JUMP (Pure Heston) ----------
    # (1) Sample parameters
    samples_no_jump = dist_no_jump.sample(num_samples, rule="sobol")
    # shape = (5, num_samples) with parameters: [kappa, theta, sigma, rho, v0]

    # (2) Evaluate the Heston model for each sample
    prices_no_jump = []
    for i in range(num_samples):
        kappa_i, theta_i, sigma_i, rho_i, v0_i = samples_no_jump[:, i]
        price_i = heston_call_price(
            S0, K, r, T,
            kappa_i, theta_i, sigma_i, rho_i, v0_i,
            engine_type="Analytic"
        )
        prices_no_jump.append(price_i)

    prices_no_jump = np.array(prices_no_jump)

    # (3) Construct polynomial chaos expansion (orthogonal polynomials)
    orth_poly_no_jump = cp.orth_ttr(poly_order, dist_no_jump)

    # # (4) Fit regression-based PCE surrogate
    # pce_no_jump = cp.fit_regression(
    #     orth_poly_no_jump,
    #     samples_no_jump,
    #     prices_no_jump,
    #     rule="LS"
    # )

    pce_no_jump = cp.fit_regression(orth_poly_no_jump, samples_no_jump, prices_no_jump)

    # (5) Compute mean, variance, Sobol indices
    mean_no_jump = cp.E(pce_no_jump, dist_no_jump)
    var_no_jump  = cp.Var(pce_no_jump, dist_no_jump)

    first_order_sobol_no_jump = cp.Sens_m(pce_no_jump, dist_no_jump)
    total_order_sobol_no_jump = cp.Sens_t(pce_no_jump, dist_no_jump)

    print("===== Heston (No Jump) Results =====")
    print(f"Mean call price: {mean_no_jump:.4f}")
    print(f"Var call price:  {var_no_jump:.4e}")
    print("First-order Sobol indices:", first_order_sobol_no_jump)
    print("Total-order  Sobol indices:", total_order_sobol_no_jump, "\n")

    # --------------------------- WITH JUMP (Bates) ------------------------------------------
    # (1) Sample parameters
    samples_with_jump = dist_with_jump.sample(num_samples, rule="sobol")
    # shape = (8, num_samples)

    # (2) Evaluate the Bates model for each sample
    prices_with_jump = []
    for i in range(num_samples):
        (kappa_i, theta_i, sigma_i, rho_i, v0_i,
         lambda_i, muJ_i, sigmaJ_i) = samples_with_jump[:, i]
        price_i = bates_call_price(
            S0, K, r, T,
            kappa_i, theta_i, sigma_i, rho_i, v0_i,
            lambda_i, muJ_i, sigmaJ_i,
            engine_type="Analytic"
        )
        prices_with_jump.append(price_i)

    prices_with_jump = np.array(prices_with_jump)

    # (3) Construct PCE for Bates model
    # orth_poly_with_jump = cp.orth_ttr(poly_order, dist_with_jump)
    # pce_with_jump = cp.fit_regression(
    #     orth_poly_with_jump,
    #     samples_with_jump,
    #     prices_with_jump,
    #     rule="LS"
    # )

    pce_with_jump = cp.fit_regression(orth_poly_with_jump, samples_with_jump, prices_with_jump)

    # (4) Compute mean, variance, Sobol indices
    mean_with_jump = cp.E(pce_with_jump, dist_with_jump)
    var_with_jump  = cp.Var(pce_with_jump, dist_with_jump)

    first_order_sobol_jump = cp.Sens_m(pce_with_jump, dist_with_jump)
    total_order_sobol_jump = cp.Sens_t(pce_with_jump, dist_with_jump)

    print("===== Bates (Heston + Jump) Results =====")
    print(f"Mean call price: {mean_with_jump:.4f}")
    print(f"Var call price:  {var_with_jump:.4e}")
    print("First-order Sobol indices:", first_order_sobol_jump)
    print("Total-order  Sobol indices:", total_order_sobol_jump, "\n")


if __name__ == "__main__":
    main()
