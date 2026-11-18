import numpy as np
import pandas as pd
import yfinance as yf
import QuantLib as ql
from scipy.optimize import minimize
from datetime import datetime

#############################################
# 1. Get Spot Price Data (S&P 500)
#############################################
ticker = "SPY"
data = yf.download(ticker, period="1y", interval="1d")

prices = data['Close'].dropna()
spot = float(prices.iloc[-1])  # current spot price

#############################################
# 2. Get Option Chain Data from yfinance
#############################################
snp = yf.Ticker(ticker)
expirations = snp.options  # list of expiration dates as strings YYYY-MM-DD
if not expirations:
    raise ValueError("No options data found for the S&P 500 (^GSPC).")

chosen_expiry_str = expirations[10]  # take the first available expiration
option_chain = snp.option_chain(chosen_expiry_str)
calls = option_chain.calls.dropna()

# Filter for a reasonable set of strikes around spot
strikes = calls['strike']
atm_strike = spot
# Choose strikes close to the ATM strike for calibration:
selected_strikes = strikes[(strikes > 0.9*atm_strike) & (strikes < 1.1*atm_strike)]
selected_calls = calls[calls['strike'].isin(selected_strikes)]

if selected_calls.empty:
    raise ValueError("No suitable call options found near the ATM strike.")

# Use 'lastPrice' as market price. Consider midpoint of bid/ask for more realism.
market_data = {}
for idx, row in selected_calls.iterrows():
    K = row['strike']
    market_price = row['lastPrice']
    # Convert chosen_expiry_str to QuantLib Date
    expiry_dt = datetime.strptime(chosen_expiry_str, '%Y-%m-%d').date()
    expiry_ql = ql.Date(expiry_dt.day, expiry_dt.month, expiry_dt.year)
    valuation_date = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = valuation_date
    day_count = ql.Actual365Fixed()
    T_days = day_count.dayCount(valuation_date, expiry_ql)
    market_data[(T_days, K)] = market_price

if not market_data:
    raise ValueError("No market data points collected for calibration.")

#############################################
# 3. Set up the Heston Model Pricing
#############################################
r = 0.01  # flat interest rate assumption
dividend_rate = 0.0
calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
day_count = ql.Actual365Fixed()

def heston_model(params, spot, r):
    kappa, theta, sigma, rho, v0 = params
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, r, day_count))
    dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, dividend_rate, day_count))
    heston_process = ql.HestonProcess(flat_ts, dividend_ts, spot_handle, v0, kappa, theta, sigma, rho)
    model = ql.HestonModel(heston_process)
    engine = ql.AnalyticHestonEngine(model)
    return model, engine

def heston_price(params, T_days, K, engine):
    # Price a European call option
    expiry = ql.Date.todaysDate() + int(T_days)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    exercise = ql.EuropeanExercise(expiry)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)
    return option.NPV()

#############################################
# 4. Define the Objective Function
#############################################
def objective(params):
    # params = [kappa, theta, sigma, rho, v0]
    # Check parameter domain:
    kappa, theta, sigma, rho, v0 = params
    if kappa <= 0 or theta <= 0 or sigma <= 0 or v0 <= 0 or rho <= -1 or rho >= 1:
        return 1e10  # large penalty for invalid domain

    model, engine = heston_model(params, spot, r)
    sse = 0.0
    for (T_days, K), market_price in market_data.items():
        model_price = heston_price(params, T_days, K, engine)
        sse += (market_price - model_price)**2
    return sse

#############################################
# 5. Calibration via Optimization
#############################################
# Initial guess (rough)
initial_guess = [1.0, 0.02, 0.2, 0.0, 0.02]
# Bounds to ensure parameters stay in valid region
# kappa > 0, theta > 0, sigma > 0, -1 < rho < 1, v0 > 0
bounds = [(1e-6, 10), (1e-6, 0.5), (1e-6, 5), (-0.999, 0.999), (1e-6, 1)]

result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B', options={'maxiter':500, 'disp':False})
calibrated_params = result.x

print("Calibrated Parameters:")
print("kappa:", calibrated_params[0])
print("theta:", calibrated_params[1])
print("sigma:", calibrated_params[2])
print("rho:  ", calibrated_params[3])
print("v0:   ", calibrated_params[4])


