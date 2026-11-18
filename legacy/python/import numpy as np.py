import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm, invgamma, bernoulli
import random

# ----------------------- Black-Scholes Pricing and Implied Volatility -----------------------
def bs_call_price(S, K, T, r, sigma):
    """
    Compute the Black-Scholes European call price.
    """
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def implied_vol_call(C_market, S, K, T, r, tol=1e-6, max_iter=100):
    """
    Compute implied volatility by bisection.
    """
    # Initial bounds for sigma
    sigma_low = 1e-8
    sigma_high = 5.0
    sigma_mid = (sigma_low + sigma_high) / 2.0
    for _ in range(max_iter):
        price = bs_call_price(S, K, T, r, sigma_mid)
        if abs(price - C_market) < tol:
            return sigma_mid
        if price > C_market:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid
        sigma_mid = (sigma_low + sigma_high) / 2.0
    return sigma_mid

# ----------------------- Refined Resampling (Equations 67–73, Page 11) -----------------------
def refined_resample(V_tilde, weights):
    V_tilde = np.array(V_tilde)
    weights = np.array(weights)
    sort_indices = np.argsort(V_tilde)
    V_sorted = V_tilde[sort_indices]
    W_sorted = weights[sort_indices]
    cdf = np.cumsum(W_sorted)
    N = len(V_tilde)
    u = np.random.uniform(0, 1, N)
    V_refined = np.zeros(N)
    for i in range(N):
        idx = np.searchsorted(cdf, u[i])
        if idx == 0:
            V_refined[i] = V_sorted[0]
        else:
            cdf_lower = cdf[idx - 1]
            cdf_upper = cdf[idx]
            fraction = (u[i] - cdf_lower) / (cdf_upper - cdf_lower) if cdf_upper > cdf_lower else 0
            V_refined[i] = V_sorted[idx - 1] + fraction * (V_sorted[idx] - V_sorted[idx - 1])
    return V_refined

# ----------------------- Simulate Bates Model Under Risk-Neutral Measure -----------------------
def simulate_bates_rn(
    S0,         # initial stock price
    v0,         # initial variance
    r,          # risk-free rate (risk-neutral drift)
    kappa,      # mean-reversion speed for v
    theta,      # long-run variance
    sigma_v,    # volatility of volatility
    rho,        # correlation between Brownian motions
    lambda_j,   # jump intensity (per year)
    mu_J,       # mean jump (in log–space)
    sigma_J,    # jump volatility (in log–space)
    T,          # time horizon in years
    Nsteps,     # number of time steps
    seed=None,
    use_log_euler=True
):
    """
    Simulate Bates model paths under the risk-neutral measure.
    
    The risk-neutral dynamics for S are:
      dS/S = r dt + sqrt(v) dW_S + (e^Z - 1) dN,
    and in log–Euler discretisation the update becomes:
      logS(t+Δt) = logS(t) + [r - λ_j (E[e^Z]-1) - 0.5*v(t)]Δt + sqrt(v(t)Δt)*Z + J,
    where J = Z if a jump occurs (with probability λ_j Δt), else 0.
    """
    if seed is not None:
        np.random.seed(seed)
        
    dt = T / Nsteps
    t = np.linspace(0, T, Nsteps+1)
    
    S = np.zeros(Nsteps+1)
    v = np.zeros(Nsteps+1)
    logS = np.zeros(Nsteps+1)
    
    S[0] = S0
    v[0] = v0
    logS[0] = math.log(S0)
    
    # Precompute jump adjustment: E[e^Z] = exp(mu_J + 0.5*sigma_J^2)
    jump_adjust = np.exp(mu_J + 0.5 * sigma_J**2)
    
    # Generate independent normal increments for price and variance
    Z1 = np.random.normal(0, 1, Nsteps)
    Z2 = np.random.normal(0, 1, Nsteps)
    
    for i in range(Nsteps):
        # Correlated increments for variance:
        z1 = Z1[i]
        z2 = Z2[i]
        dW_v = rho * z1 + math.sqrt(1 - rho**2) * z2
        
        # Variance update (Euler–Maruyama)
        v_old = v[i]
        v_next = v_old + kappa*(theta - v_old)*dt + sigma_v * math.sqrt(max(v_old, 0)) * math.sqrt(dt) * dW_v
        v_next = max(v_next, 0.0)
        v[i+1] = v_next
        
        # Jump: with probability λ_j * dt, a jump occurs.
        if np.random.uniform(0, 1) < lambda_j * dt:
            jump = np.random.normal(mu_J, sigma_J)
        else:
            jump = 0.0
        
        if use_log_euler:
            logS_old = logS[i]
            # Drift term: risk-neutral drift adjusted for jumps: r - λ_j (E[e^Z]-1)
            drift = (r - lambda_j*(jump_adjust - 1) - 0.5*v_old)*dt
            diffusion = math.sqrt(max(v_old, 0)*dt)*z1
            logS_next = logS_old + drift + diffusion + jump
            logS[i+1] = logS_next
            S[i+1] = math.exp(logS_next)
        else:
            # Direct Euler (not recommended for log-prices)
            S_old = S[i]
            diffusion = S_old*math.sqrt(max(v_old, 0)*dt)*z1
            jump_component = (math.exp(jump)-1)*S_old
            S[i+1] = S_old + r*S_old*dt + diffusion + jump_component
            
    return t, S, v

# ----------------------- Monte Carlo Option Pricing Under Bates Model -----------------------
def price_options_bates(
    S0, v0, r, kappa, theta, sigma_v, rho, lambda_j, mu_J, sigma_J,
    T, Nsteps, n_paths, strikes, seed=None
):
    """
    Simulate Bates paths under the risk-neutral measure and price European call options.
    Returns a dictionary with strike prices, option prices, and implied volatilities.
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / Nsteps
    payoffs = {K: [] for K in strikes}
    
    # Simulate n_paths paths
    for path in range(n_paths):
        t, S_path, _ = simulate_bates_rn(
            S0, v0, r, kappa, theta, sigma_v, rho,
            lambda_j, mu_J, sigma_J, T, Nsteps, seed=None, use_log_euler=True
        )
        S_T = S_path[-1]
        for K in strikes:
            payoff = max(S_T - K, 0)
            payoffs[K].append(payoff)
    
    # Discount and average the payoffs to get option prices
    option_prices = {}
    for K in strikes:
        price = np.exp(-r*T) * np.mean(payoffs[K])
        option_prices[K] = price
        
    # Compute implied volatilities for each strike using the BS formula
    implied_vols = {}
    for K in strikes:
        C_market = option_prices[K]
        iv = implied_vol_call(C_market, S0, K, T, r)
        implied_vols[K] = iv
        
    return {"option_prices": option_prices, "implied_vols": implied_vols}

# ----------------------- Black-Scholes Functions (see above) -----------------------
def bs_call_price(S, K, T, r, sigma):
    if T <= 0:
        return max(S-K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def implied_vol_call(C_market, S, K, T, r, tol=1e-6, max_iter=100):
    sigma_low = 1e-8
    sigma_high = 5.0
    sigma_mid = (sigma_low + sigma_high) / 2.0
    for _ in range(max_iter):
        price = bs_call_price(S, K, T, r, sigma_mid)
        if abs(price - C_market) < tol:
            return sigma_mid
        if price > C_market:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid
        sigma_mid = (sigma_low + sigma_high) / 2.0
    return sigma_mid

# ----------------------- Example Usage -----------------------
if __name__ == "__main__":
    # Assume these calibrated model parameters (from your calibration procedure):
    # Under the physical measure, but for risk-neutral simulation we use r instead of mu.
    S0 = 100.0
    v0 = 0.05
    r = 0.05             # risk-free rate
    kappa = 1.2
    theta = 0.04
    sigma_v = 0.011
    rho = -0.48
    lambda_j = 0.85      # jump intensity (per year)
    mu_J = -0.9          # jump mean (log-space)
    sigma_J = 0.2        # jump volatility (log-space)
    
    # Simulation settings
    T = 1.0
    Nsteps = 252
    n_paths = 10000     # number of Monte Carlo paths for option pricing
    
    # Define a range of strikes (for example, moneyness around S0)
    strikes = np.linspace(80, 120, 21)
    
    # Price options using the risk-neutral Bates simulation
    results = price_options_bates(S0, v0, r, kappa, theta, sigma_v, rho,
                                  lambda_j, mu_J, sigma_J, T, Nsteps, n_paths, strikes, seed=42)
    
    option_prices = results["option_prices"]
    implied_vols = results["implied_vols"]
    
    # Print the option prices and implied volatilities
    print("Strike\tOption Price\tImplied Volatility")
    for K in strikes:
        print(f"{K:.2f}\t{option_prices[K]:.4f}\t\t{implied_vols[K]:.4f}")
    
    # Plot the volatility smile
    plt.figure(figsize=(8,5))
    plt.plot(strikes, [implied_vols[K] for K in strikes], marker='o', label="Implied Volatility")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.title("Implied Volatility Smile under Bates Model")
    plt.legend()
    plt.show()
