import yfinance as yf
import pandas as pd

ticker = "SPY"
stk = yf.Ticker(ticker)
expiries = stk.options                               # list of expiration dates :contentReference[oaicite:0]{index=0}

# pick a subset of expiries for speed
expiries = expiries[:5]

# build a DataFrame of (expiry, strike, mid_price)
opt_data = []


for exp in expiries:
    chain = stk.option_chain(exp)
    calls = chain.calls[['strike','bid','ask']].copy()   # ← explicit copy
    calls['mid']    = (calls['bid'] + calls['ask'])/2.0
    calls['expiry'] = exp
    opt_data.append(calls[['expiry','strike','mid']])
opt_df = pd.concat(opt_data, ignore_index=True)


from scipy.stats import norm

def bs_call_price(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def implied_volatility(C_mkt, S, K, T, r, initial=0.2, tol=1e-6):
    sigma = initial
    for _ in range(50):
        price = bs_call_price(S, K, T, r, sigma)
        vega  = S * norm.pdf((np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))) * np.sqrt(T)
        diff = price - C_mkt
        if abs(diff) < tol:
            return sigma
        sigma -= diff/vega
    return sigma  # fallback


import numpy as np

# parameters
S0 = prices[-1]
r = 0.02

# unique sorted strikes and expiries
strikes  = np.unique(opt_df.strike)
expiries = pd.to_datetime(opt_df.expiry).unique()
Tgrid    = np.array([(ed - pd.Timestamp.today()).days/252 for ed in expiries])

# surface array
IV = np.zeros((len(Tgrid), len(strikes)))

for i, exp in enumerate(expiries):
    T = Tgrid[i]
    sub = opt_df[opt_df.expiry == exp]
    for j, K in enumerate(strikes):
        row = sub[sub.strike == K]
        if not row.empty:
            C_mkt = row.mid.values[0]
            IV[i,j] = implied_volatility(C_mkt, S0, K, T, r)
        else:
            IV[i,j] = np.nan

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

K_grid, T_grid = np.meshgrid(strikes, Tgrid)

fig = plt.figure(figsize=(10,7))
ax  = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(K_grid, T_grid, IV, cmap='viridis', edgecolor='none')
ax.set_xlabel('Strike')
ax.set_ylabel('Time to Expiry (yr)')
ax.set_zlabel('Implied Volatility')
fig.colorbar(surf, shrink=0.5, label='IV')
plt.title('Implied Volatility Surface')  
plt.show()
