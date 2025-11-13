import yfinance as yf
import pandas as pd
import numpy as np



ticker = "^GSPC"
data = yf.download(ticker, start='2007-01-01', end='2015-12-31', progress=False, auto_adjust=False)

if data.empty:
    raise ValueError("Failed to download data. Check ticker or internet connection.")

# Compute log returns
data['LogReturn'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
data = data.dropna()

# Save
data[['LogReturn']].to_csv("sp500_returns.csv")
print("Saved daily log returns to 'sp500_returns.csv'")
