import QuantLib as ql

def heston_quantlib(S0, K, T, r, kappa, theta, sigma, rho, v0, option_type='call'):
    # --- QuantLib setup ---
    calendar      = ql.NullCalendar()
    todays_date   = ql.Date().todaysDate()
    maturity_date = todays_date + int(T * 365)
    day_count     = ql.Actual365Fixed()

    spot_handle      = ql.QuoteHandle(ql.SimpleQuote(S0))
    rate_handle      = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, r, day_count))
    dividend_handle  = ql.YieldTermStructureHandle(ql.FlatForward(0, calendar, 0.0, day_count))

    heston_process = ql.HestonProcess(rate_handle,
                                      dividend_handle,
                                      spot_handle,
                                      v0, kappa, theta, sigma, rho)

    payoff   = ql.PlainVanillaPayoff(
                    ql.Option.Call if option_type=='call' else ql.Option.Put,
                    K)
    exercise = ql.EuropeanExercise(maturity_date)
    option   = ql.VanillaOption(payoff, exercise)

    engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process))
    option.setPricingEngine(engine)

    return option.NPV()


# --- Market & model inputs ---
S0 = 100     # spot price
K  = 110        # strike
T  = 1/12        # time to expiry in years
r  = 0.1      # risk-free rate



# ---- Single (scalar) Heston parameters ----
kappa = 1.1      # mean reversion speed
theta = 0.04   # long-run variance
sigma = 0.01  # vol-of-vol
rho   = -0.5     # correlation
v0    = 0.04     # initial variance

# --- Price the option once ---
price = heston_quantlib(S0, K, T, r, kappa, theta, sigma, rho, v0)
print(f"Heston Model Price: {price:.4f}")
