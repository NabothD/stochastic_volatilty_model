import QuantLib as ql
import numpy as np

# Parameters for the Heston model
spot_price = 100.0    # Initial stock price (S0)
risk_free_rate = 0.04 # Risk-free rate (mu)
v0 = 0.04             # Initial variance
kappa = 1.1          # Mean reversion rate
theta = 0.04          # Long-run variance
sigma = 0.01           # Volatility of variance
rho = -0.5            # Correlation between Brownian motions
maturity = 1.0/12       # Time to maturity (T)

# Setup evaluation date and calendar
settlement_date = ql.Date(1, 1, 2024)
ql.Settings.instance().evaluationDate = settlement_date

# Yield term structures
day_count = ql.Actual360()
rf_curve = ql.YieldTermStructureHandle(
    ql.FlatForward(settlement_date, risk_free_rate, day_count)
)
dividend_yield = ql.YieldTermStructureHandle(
    ql.FlatForward(settlement_date, 0.0, day_count)
)

# Spot handle
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))

# Define the Heston process
heston_process = ql.HestonProcess(
    rf_curve, dividend_yield, spot_handle, v0, kappa, theta, sigma, rho
)

# Manual simulation using HestonProcess.evolve
num_paths = 10000
time_steps = 360
dt = maturity / time_steps
np.random.seed(42)  # For reproducibility

# Initialize arrays
prices = np.zeros((num_paths, time_steps + 1))
prices[:, 0] = spot_price  # Initial price

# Simulate paths
for i in range(num_paths):
    variance = v0
    price = spot_price
    for t in range(1, time_steps + 1):
        # Generate correlated random numbers
        dw1 = np.random.normal(0, np.sqrt(dt))
        dw2 = rho * dw1 + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt))

        # Evolve variance and price
        variance = max(0, variance + kappa * (theta - variance) * dt + sigma * np.sqrt(variance) * dw2)
        price = price * np.exp((risk_free_rate - 0.5 * variance) * dt + np.sqrt(variance) * dw1)

        prices[i, t] = price

# Extract terminal prices
terminal_prices = prices[:, -1]

# Calculate mean and variance
quantlib_mean = np.mean(terminal_prices)
quantlib_variance = np.var(terminal_prices)

# Analytical solution for mean and variance
def heston_analytical_mean_variance(S0, mu, v0, kappa, theta, T):
    mean = S0 * np.exp(mu * T)
    exp_neg_kappa_t = np.exp(-kappa * T)
    variance = (
        S0**2 * np.exp(2 * mu * T) *
        ((1 - exp_neg_kappa_t) / kappa * theta + v0 * exp_neg_kappa_t)
        - (mean**2)
    )
    return mean, variance

analytical_mean, analytical_variance = heston_analytical_mean_variance(
    spot_price, risk_free_rate, v0, kappa, theta, maturity
)

# Output results
print(f"Analytical Mean: {analytical_mean:.6f}")
print(f"QuantLib Simulated Mean: {quantlib_mean:.6f}")
print(f"Analytical Variance: {analytical_variance:.6f}")
print(f"QuantLib Simulated Variance: {quantlib_variance:.6f}")

# Compare results
mean_diff = abs(analytical_mean - quantlib_mean)
variance_diff = abs(analytical_variance - quantlib_variance)
print(f"Difference in Mean: {mean_diff:.6f}")
print(f"Difference in Variance: {variance_diff:.6f}")
