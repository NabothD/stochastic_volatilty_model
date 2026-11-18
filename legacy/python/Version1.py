import QuantLib as ql
import chaospy as cp
import numpy as np

# Define the Heston model (same as before)
def heston_model(spot, v0, kappa, theta, sigma, rho, risk_free_rate, dividend_yield):
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    risk_free_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(risk_free_rate)), ql.Actual365Fixed())
    )
    dividend_curve = ql.YieldTermStructureHandle(
        ql.FlatForward(0, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(dividend_yield)), ql.Actual365Fixed())
    )
    process = ql.HestonProcess(risk_free_curve, dividend_curve, spot_handle, v0, kappa, theta, sigma, rho)
    return ql.HestonModel(process)

# Price an option using the Heston model
def price_option(model, option_type, strike, maturity_date):
    engine = ql.AnalyticHestonEngine(model)
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(option_type, strike),
        ql.EuropeanExercise(maturity_date)
    )
    option.setPricingEngine(engine)
    return option.NPV()

# Parameter distributions
kappa_dist = cp.Uniform(1.0, 2.0)
theta_dist = cp.Uniform(0.01, 0.05)
sigma_dist = cp.Uniform(0.1, 0.3)
rho_dist = cp.Uniform(-0.9, -0.3)
joint_dist = cp.J(kappa_dist, theta_dist, sigma_dist, rho_dist)

# Monte Carlo simulation
samples = 5000
uncertain_params = joint_dist.sample(samples)

spot_price = 100.0
strike_price = 100.0
v0 = 0.04
risk_free_rate = 0.05
dividend_yield = 0.0
maturity_date = ql.Date(1, 1, 2025)

prices_mc = []

for params in uncertain_params.T:
    kappa, theta, sigma, rho = params
    model = heston_model(spot_price, v0, kappa, theta, sigma, rho, risk_free_rate, dividend_yield)
    prices_mc.append(price_option(model, ql.Option.Call, strike_price, maturity_date))

# Polynomial Chaos Expansion
expansion_order = 2  # Polynomial order
orthogonal_polynomials = cp.orth_ttr(expansion_order, joint_dist)
nodes, weights = cp.generate_quadrature(expansion_order, joint_dist, rule="gaussian")
prices_pce = []

# Evaluate model at quadrature nodes
for params in nodes.T:
    kappa, theta, sigma, rho = params
    model = heston_model(spot_price, v0, kappa, theta, sigma, rho, risk_free_rate, dividend_yield)
    prices_pce.append(price_option(model, ql.Option.Call, strike_price, maturity_date))

# Fit PCE
prices_pce = np.array(prices_pce)
pce_model = cp.fit_quadrature(orthogonal_polynomials, nodes, weights, prices_pce)

# Validation
# Generate new samples
validation_samples = 1000
validation_params = joint_dist.sample(validation_samples)

# Monte Carlo validation
validation_prices_mc = []
for params in validation_params.T:
    kappa, theta, sigma, rho = params
    model = heston_model(spot_price, v0, kappa, theta, sigma, rho, risk_free_rate, dividend_yield)
    validation_prices_mc.append(price_option(model, ql.Option.Call, strike_price, maturity_date))

# PCE validation
validation_prices_pce = cp.call(pce_model, validation_params)

# Compare mean and variance
mc_mean = np.mean(validation_prices_mc)
mc_var = np.var(validation_prices_mc)
pce_mean = np.mean(validation_prices_pce)
pce_var = np.var(validation_prices_pce)

print(f"Monte Carlo Mean: {mc_mean:.2f}, Variance: {mc_var:.2f}")
print(f"PCE Mean: {pce_mean:.2f}, Variance: {pce_var:.2f}")

# Visualization
import matplotlib.pyplot as plt

plt.hist(validation_prices_mc, bins=50, alpha=0.5, label="Monte Carlo")
plt.hist(validation_prices_pce, bins=50, alpha=0.5, label="PCE")
plt.legend()
plt.title("Option Price Distributions: Monte Carlo vs. PCE")
plt.xlabel("Option Price")
plt.ylabel("Frequency")
plt.show()
