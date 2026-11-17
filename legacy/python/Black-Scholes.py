
import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """Calculate European call option price using the Black-Scholes formula."""
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Calculate European put option price using the Black-Scholes formula."""
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def main():
    # Randomly selected stock from S&P 500: Boeing Company (BA)
    # Parameters (assumed for demonstration purposes)
    S = 500.00       # Current stock price in dollars
    K = 510.00       # Strike price in dollars
    T = 0.1          # Time to maturity in years (6 months)
    r = 0.043         # Annual risk-free interest rate (5%)
    sigma = 0.20     # Annual volatility (30%)

    # Calculate European option prices
    european_call = black_scholes_call(S, K, T, r, sigma)
    european_put = black_scholes_put(S, K, T, r, sigma)

    # For non-dividend-paying stocks, the American call option price equals the European call option price
    american_call = european_call

    # Since American put options can be exercised early, their price is typically higher than European puts
    # For simplicity, we'll acknowledge this without complex calculations
    american_put_lower_bound = european_put  # American put is at least as valuable as European put

    # Output the results
    print(f"European Call Option Price: ${european_call:.2f}")
    print(f"European Put Option Price: ${european_put:.2f}")
    print(f"American Call Option Price: ${american_call:.2f}")
    print(f"American Put Option Price: At least ${american_put_lower_bound:.2f}")

if __name__ == "__main__":
    main()
