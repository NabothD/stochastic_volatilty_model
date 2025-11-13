import yfinance as yf
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Download S&P 500 data for 2008
ticker = "^OEX"
start_date = "2019-06-11"
end_date = "2021-06-11"
data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)

# Check data
if data.empty:
    raise ValueError("No data fetched. Check ticker symbol or internet connection.")

# Use Adjusted Close price
prices = data['Adj Close']

# Compute daily log returns
log_returns = np.log(prices / prices.shift(1))

# Realized volatility: rolling average of absolute returns × sqrt(252) to annualize
realized_vol = log_returns.abs().rolling(window=20).mean() * np.sqrt(252)

# Combine price and volatility, then clean
df = pd.concat([prices, realized_vol], axis=1)
df.columns = ["Price", "ProxyVol"]
df.dropna(inplace=True)

# Reset index for clean Excel export
df.reset_index(drop=True, inplace=True)

# Save to Excel
df.to_excel("sp500_historical.xlsx", index=False)
print("✅ S&P 500 data with realized volatility saved to 'sp500_historical.xlsx'")

# Compute daily log returns
log_returns = (prices / prices.shift(1))

# Drop the first NaN value
log_returns = log_returns.dropna()

# Plot the daily log returns
plt.figure(figsize=(12, 6))
plt.plot(log_returns.index, log_returns.values, label='Returns', linewidth=2)
plt.xlabel('Date', fontsize=20)
plt.ylabel('Returns', fontsize=20)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(prices.index, prices, label='Price', linewidth=2)
plt.xlabel('Date', fontsize=20)
plt.ylabel('Price', fontsize=20)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
plt.show()


# import yfinance as yf
# import pandas as pd

# # Download S&P 500 and VIX data for 2012–2014
# tickers    = ["^GSPC", "^VIX"]
# start_date = "2018-01-01"
# end_date   = "2022-12-31"

# data = yf.download(tickers, start=start_date, end=end_date,
#                    progress=False, auto_adjust=False)

# # Sanity check
# if data.empty or data['Adj Close'].isnull().all().all():
#     raise ValueError("No data fetched. Check ticker symbols or internet connection.")

# # Extract Adjusted Close series
# sp500 = data['Adj Close']['^GSPC']
# vix   = data['Adj Close']['^VIX']

# # Combine into one DataFrame
# df = pd.concat([sp500, vix], axis=1)
# df.columns = ["Price", "ProxyVol"]
# df.dropna(inplace=True)

# # (Optional) keep the date index or reset it:
# # df.reset_index(drop=True, inplace=True)

# # Save to Excel
# output_file = "sp500_historical.xlsx"
# df.to_excel(output_file, index=False)
# print(f"✅ Saved S&P 500 and VIX data to '{output_file}'")
