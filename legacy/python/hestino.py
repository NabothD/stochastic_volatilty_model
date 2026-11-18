import numpy as np
import math
import matplotlib.pyplot as plt

def generate_heston_paths_stochastic_rate(
    S0,      # Initial stock price
    T,       # Total time in years
    r0,      # Initial short rate
    alpha,   # Mean-reversion speed for the rate
    rbar,    # Long-term mean rate
    sigma_r, # Volatility of the short rate
    kappa,   # Mean-reversion speed for variance
    theta,   # Long-term variance
    v0,      # Initial variance
    rho,     # Correlation between dW_s and dW_v
    xi,      # Volatility of volatility
    steps,   # Number of time steps
    Npaths   # Number of simulated paths
):
    """
    Generates stock price paths under the Heston model with a stochastic short rate
    following a simple Vasicek process:
        dr(t) = alpha * (rbar - r(t)) dt + sigma_r * dW_r(t).
    """

    dt = T / steps
    
    # Arrays to store the simulated paths
    prices = np.zeros(steps)
    variances = np.zeros(steps)
    rates = np.zeros((Npaths, steps))
    
    # Initialize current values (for all paths)
    S_t = np.full(Npaths, S0)
    v_t = np.full(Npaths, v0)
    r_t = np.full(Npaths, r0)
    
    for t in range(steps):
        # Draw correlated Brownian increments for price & vol
        # (2D correlated increments for stock & variance)
        # plus a separate increment for the short rate (assume uncorrelated for simplicity).
        W = np.random.multivariate_normal(
                mean=[0, 0],
                cov=[[1, rho],
                     [rho, 1]],
                size=Npaths
            ) * np.sqrt(dt)
        
        # Uncorrelated increment for the short rate
        Z_r = np.random.normal(0, 1, Npaths) * np.sqrt(dt)
        
        # 1) Update the stock price
        #    Drift is r_t - 0.5*v_t, diffusion is sqrt(v_t)*dW_s
        S_t = S_t * np.exp((r_t - 0.5 * v_t) * dt + np.sqrt(v_t) * W[:, 0])
        
        # 2) Update the variance (Heston volatility process)
        #    v_{t+dt} = v_t + kappa*(theta - v_t)*dt + xi*sqrt(v_t)*dW_v
        v_t = np.abs(v_t + kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * W[:, 1])
        
        # 3) Update the short rate (Vasicek process)
        #    r_{t+dt} = r_t + alpha*(rbar - r_t)*dt + sigma_r*dW_r
        r_t = r_t + alpha * (rbar - r_t) * dt + sigma_r * Z_r
        
        # Store the new values
        prices[t] = S_t
        variances[t] = v_t
        rates[:, t] = r_t
    
    return prices, variances


# Example parameters
S0       = 100.0
T        = 1.0
steps    = 252
Npaths   = 1

# Short rate (Vasicek) parameters
r0       = 0.1     # initial short rate
alpha    = 0.5     # mean-reversion speed
rbar     = 0.05    # long-term mean rate
sigma_r  = 0.02    # volatility of the short rate

# Heston parameters
kappa    = 1.0
theta    = 0.05
v0       = 0.05
xi       = 0.05
rho      = -0.5

# Generate the paths
prices, variances = generate_heston_paths_stochastic_rate(
    S0, T, r0, alpha, rbar, sigma_r,
    kappa, theta, v0, rho, xi,
    steps, Npaths
)

# 3) Save to CSV
data = np.column_stack((prices, np.sqrt(variances)))
np.savetxt("my_heston.csv", data, delimiter=",", header="price,variance", comments="")

# Plot stock price paths
plt.figure(figsize=(8,5))
plt.plot(prices.T)
plt.title('Heston Model with Stochastic Rate: Stock Price Paths')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.grid(True)
plt.show()

# Plot volatility paths
plt.figure(figsize=(8,5))
plt.plot(np.sqrt(variances).T)
plt.axhline(np.sqrt(theta), color='k', linestyle='--', label=r'$\sqrt{\theta}$')
plt.title('Heston Model: Stochastic Volatility Paths')
plt.xlabel('Time Steps')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.show()

