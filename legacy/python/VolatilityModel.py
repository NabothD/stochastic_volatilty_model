import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql
import chaospy as cp
from scipy.optimize import minimize
import yfinance as yf

# Fetch historical data for the S&P 500
ticker = "^GSPC"  # Ticker for S&P 500
data = yf.download(ticker, start="2010-01-01", end="2023-12-31")  # Adjust dates as needed

# Calculate log returns
data['Log Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
data.dropna(inplace=True)

# Compute historical volatility
historical_volatility = data['Log Returns'].std() * np.sqrt(252)  # Annualized volatility

print(data.head())
print(f"Historical Volatility: {historical_volatility:.2%}")


# Continue with the rest of your workflow (QuantLib, Chaospy, etc.)


# Define a calibration function
def calibrate_heston_params(log_returns):
    def heston_loss(params):
        kappa, theta, sigma, rho, v0 = params
        if not (0 < v0 and 0 < sigma and -1 <= rho <= 1):
            return np.inf  # Reject invalid parameters

        dt = 1 / 252  # Daily steps
        simulated_volatility = []
        variance = v0

        for ret in log_returns:
            dw1 = np.random.normal(0, np.sqrt(dt))
            dw2 = rho * dw1 + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt))
            variance = max(0, variance + kappa * (theta - variance) * dt + sigma * np.sqrt(variance) * dw2)
            simulated_volatility.append(np.sqrt(variance))

        simulated_volatility = np.array(simulated_volatility)
        return np.mean((simulated_volatility - np.abs(log_returns))**2)

    # Initial guesses for parameters
    initial_guess = [1.0, historical_volatility**2, 0.2, 0.0, historical_volatility**2]
    bounds = [(0.1, 5), (0, 0.1), (0.01, 1), (-0.99, 0.99), (0.001, 0.1)]

    # Minimize the loss function
    result = minimize(heston_loss, initial_guess, bounds=bounds)
    return result.x  # Optimized parameters

# Calibrate parameters
log_returns = data['Log Returns'].values
kappa, theta, sigma, rho, v0 = calibrate_heston_params(log_returns)


# Define Heston process
risk_free_rate = 0.01  # Example risk-free rate
dividend_yield = 0.0

# Fetch spot price as a scalar float
spot_price = data['Adj Close'].iloc[-1]  # Extract last adjusted close price

# If the column is multi-indexed or contains a single value, ensure it's cast to float
spot_price = float(spot_price)
print(f"Spot Price: {spot_price}")  # Confirm it's now a float

# Define Heston Process with corrected spot price
process = ql.HestonProcess(
    ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), risk_free_rate, ql.Actual365Fixed())),
    ql.YieldTermStructureHandle(ql.FlatForward(0, ql.NullCalendar(), dividend_yield, ql.Actual365Fixed())),
    ql.QuoteHandle(ql.SimpleQuote(spot_price)),  # Ensure spot_price is numeric
    v0, kappa, theta, sigma, rho
)

# Monte Carlo simulation of volatility
time_steps = 252  # 1 year
num_paths = 10000
dt = 1 / 252
volatility_paths = []

for _ in range(num_paths):
    variance = v0
    path = []

    for _ in range(time_steps):
        dw1 = np.random.normal(0, np.sqrt(dt))
        dw2 = rho * dw1 + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt))
        variance = max(0, variance + kappa * (theta - variance) * dt + sigma * np.sqrt(variance) * dw2)
        path.append(np.sqrt(variance))

    volatility_paths.append(path)

volatility_paths = np.array(volatility_paths)


# Ensure the bounds are valid
kappa_dist = cp.Uniform(max(0, 0.9 * kappa), 1.1 * kappa)  # kappa > 0
theta_dist = cp.Uniform(max(0, 0.9 * theta), 1.1 * theta)  # theta > 0
sigma_dist = cp.Uniform(max(0, 0.9 * sigma), 1.1 * sigma)  # sigma > 0
rho_dist = cp.Uniform(max(-1, 0.9 * rho), min(1, 1.1 * rho))  # -1 <= rho <= 1
v0_dist = cp.Uniform(max(0, 0.9 * v0), 1.1 * v0)  # v0 > 0

# Joint distribution
joint_dist = cp.J(kappa_dist, theta_dist, sigma_dist, rho_dist, v0_dist)



order = 2
orthogonal_polynomials = cp.expansion.stieltjes(order, joint_dist)


# # Generate PCE
# order = 2
# orthogonal_polynomials = cp.orth_ttr(order, joint_dist)
nodes, weights = cp.generate_quadrature(order+1, joint_dist, rule="gaussian")

# Simulate volatility at quadrature nodes
volatility_samples = []
for params in nodes.T:
    kappa, theta, sigma, rho, v0 = params
    variance = v0
    path = []

    for _ in range(time_steps):
        dw1 = np.random.normal(0, np.sqrt(dt))
        dw2 = rho * dw1 + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt))
        variance = max(0, variance + kappa * (theta - variance) * dt + sigma * np.sqrt(variance) * dw2)
        path.append(np.sqrt(variance))

    volatility_samples.append(np.mean(path))  # Use mean volatility for PCE

volatility_samples = np.array(volatility_samples)
pce_model = cp.fit_quadrature(orthogonal_polynomials, nodes, weights, volatility_samples)


# Quantify moments
pce_mean = cp.E(pce_model, joint_dist)
pce_variance = cp.Var(pce_model, joint_dist)

# Compare with Monte Carlo
mc_mean = np.mean(volatility_paths[:, -1])  # Mean volatility at the last time step
mc_variance = np.var(volatility_paths[:, -1])

print(f"Monte Carlo Mean Volatility: {mc_mean}")
print(f"Monte Carlo Variance: {mc_variance}")
print(f"PCE Mean Volatility: {pce_mean}")
print(f"PCE Variance: {pce_variance}")

# Plot results
plt.hist(volatility_paths[:, -1], bins=50, alpha=0.5, label="Monte Carlo")
plt.axvline(mc_mean, color="blue", linestyle="dashed", label="MC Mean")
plt.axvline(pce_mean, color="red", linestyle="dashed", label="PCE Mean")
plt.legend()
plt.title("Volatility Distribution")
plt.xlabel("Volatility")
plt.ylabel("Frequency")
plt.show()
