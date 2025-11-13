import numpy as np
import pandas as pd
import yfinance as yf
import QuantLib as ql
import emcee
import chaospy as cp
import matplotlib.pyplot as plt

#############################################
# 1. Data Retrieval from Yahoo Finance
#############################################
# Get S&P 500 data (e.g., last year)
ticker = "^GSPC"
data = yf.download(ticker, start="2023-01-01", end="2024-01-01")
prices = data['Close'].dropna()

# Compute simple returns to get a rough sense of volatility (just as illustration)
returns = prices.pct_change().dropna()
annual_vol = np.sqrt(252) * prices.pct_change().std(axis=0)


print("Estimated annualized volatility from historical returns:", annual_vol)

# In practice, for calibration you need option market prices (from another data source).
# Here, we will simulate synthetic market option prices to illustrate the approach.

#############################################
# 2. Synthetic Market Option Data Generation
#############################################

# Let's assume we have a set of strikes and maturities and a "true" Heston parameter set
# True parameters (example, not from real data)
true_params = {
    "kappa": 3.0,
    "theta": 0.04,
    "sigma": 0.9,
    "rho": -0.7,
    "v0": 0.04
}
r = 0.01  # risk-free rate
day_count = ql.Actual365Fixed()
# Provide the specific market for UnitedStates
calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
prices = data['Close'].dropna()  # prices is now a Series of closing prices
spot = float(prices.iloc[-1])    # now spot is guaranteed to be a float


# Define a function to create a Heston process and model
def heston_model(params, spot, r):
    v0 = params["v0"]
    kappa = params["kappa"]
    theta = params["theta"]
    sigma = params["sigma"]
    rho = params["rho"]
    
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, r, day_count))
    dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, 0.0, day_count))
    heston_process = ql.HestonProcess(flat_ts, dividend_ts, spot_handle, v0, kappa, theta, sigma, rho)
    model = ql.HestonModel(heston_process)
    return model

# Example of strikes and maturities
maturities = [30, 90, 180]   # days to maturity
strikes = [0.9*spot, spot, 1.1*spot]

# Generate synthetic market prices using "true_params"
def generate_market_data(true_params):
    model = heston_model(true_params, spot, r)
    engine = ql.AnalyticHestonEngine(model)
    market_prices = {}
    
    for T in maturities:
        expiry = ql.Date.todaysDate() + T
        for K in strikes:
            payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
            exercise = ql.EuropeanExercise(expiry)
            option = ql.VanillaOption(payoff, exercise)
            option.setPricingEngine(engine)
            price = option.NPV()
            market_prices[(T, K)] = price
    return market_prices

market_data = generate_market_data(true_params)

#############################################
# 3. Define the Likelihood and Prior
#############################################

# We'll define log_prior and log_likelihood. We'll assume relatively uninformative priors.
def log_prior(params):
    kappa, theta, sigma, rho, v0 = params
    if kappa > 0 and theta > 0 and sigma > 0 and -1 < rho < 1 and v0 > 0:
        # Simple uniform or weakly informative log-prior:
        return 0.0
    return -np.inf

def model_prices_from_params(params, spot, r):
    # Convert array to dict
    p = {"kappa": params[0], "theta": params[1], "sigma": params[2], "rho": params[3], "v0": params[4]}
    model = heston_model(p, spot, r)
    engine = ql.AnalyticHestonEngine(model)
    
    prices_dict = {}
    for (T, K), mp in market_data.items():
        expiry = ql.Date.todaysDate() + T
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
        exercise = ql.EuropeanExercise(expiry)
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(engine)
        prices_dict[(T, K)] = option.NPV()
    return prices_dict

def log_likelihood(params):
    mp = model_prices_from_params(params, spot, r)
    # Gaussian errors
    # Assume some measurement noise std, say sigma_error = 0.1
    sigma_error = 0.5
    residuals = []
    for key, market_price in market_data.items():
        model_price = mp[key]
        residuals.append(market_price - model_price)
    residuals = np.array(residuals)
    ll = -0.5 * np.sum((residuals/sigma_error)**2) - len(residuals)*np.log(sigma_error*np.sqrt(2*np.pi))
    return ll

def log_posterior(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params)
    return lp + ll

#############################################
# 4. Run MCMC using emcee
#############################################
ndim = 5  # kappa, theta, sigma, rho, v0
nwalkers = 50
initial_guess = [true_params["kappa"], true_params["theta"], true_params["sigma"], true_params["rho"], true_params["v0"]]
p0 = [initial_guess + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
nsteps = 1000
sampler.run_mcmc(p0, nsteps, progress=True)

# Extract samples
samples = sampler.get_chain(discard=500, thin=10, flat=True)
print("Mean posterior parameters:", np.mean(samples, axis=0))

#############################################
# 5. Uncertainty Quantification with Polynomial Chaos
#############################################

# We'll create a polynomial chaos expansion over the posterior distributions.
# First, we approximate the posterior distribution of parameters by their empirical distribution (sample from MCMC).

# Let's say we want to do UQ on option prices at a particular strike and maturity.
# Chaospy requires a known distribution. We can approximate each parameter's posterior marginal distribution by a kernel density or assume normal approximation.

kappa_samples = samples[:,0]
theta_samples = samples[:,1]
sigma_samples = samples[:,2]
rho_samples = samples[:,3]
v0_samples = samples[:,4]

# For simplicity, let's fit normal distributions to each marginal posterior:
kappa_dist = cp.Normal(np.mean(kappa_samples), np.std(kappa_samples))
theta_dist = cp.Normal(np.mean(theta_samples), np.std(theta_samples))
sigma_dist = cp.Normal(np.mean(sigma_samples), np.std(sigma_samples))
rho_dist = cp.Normal(np.mean(rho_samples), np.std(rho_samples))
v0_dist = cp.Normal(np.mean(v0_samples), np.std(v0_samples))

joint_dist = cp.J(kappa_dist, theta_dist, sigma_dist, rho_dist, v0_dist)

# Define a model wrapper for PCE evaluation: given parameters -> option price
def heston_option_price(theta):
    # theta: [kappa, theta, sigma, rho, v0]
    # Let's pick a single (T, K) for demonstration:
    T_test = maturities[1]
    K_test = strikes[1]
    p = {"kappa": theta[0], "theta": theta[1], "sigma": theta[2], "rho": theta[3], "v0": theta[4]}
    model = heston_model(p, spot, r)
    engine = ql.AnalyticHestonEngine(model)
    expiry = ql.Date.todaysDate() + T_test
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K_test)
    exercise = ql.EuropeanExercise(expiry)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)
    return option.NPV()

# Create polynomial basis and perform PCE:
order = 3
polynomial_expansion = cp.generate_expansion(order, joint_dist)
nodes, weights = cp.generate_quadrature(order, joint_dist, rule="G")
vals = np.array([heston_option_price(node) for node in nodes.T])  # Evaluate model at quadrature nodes
pce_approx = cp.fit_quadrature(polynomial_expansion, nodes, weights, vals)

# Now we have a PCE approximation of the option price as a function of parameters.
# Compute mean and variance of the option price under the posterior distribution:
mean_price = cp.E(pce_approx, joint_dist)
var_price = cp.Var(pce_approx, joint_dist)

print("PCE-based mean of option price:", mean_price)
print("PCE-based variance of option price:", var_price)

# We can also compute Sobol indices for sensitivity:
S = cp.Sens_m(pce_approx, joint_dist)
print("First-order Sobol indices:", S)
