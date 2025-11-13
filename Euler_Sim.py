import numpy as np
import math
import matplotlib.pyplot as plt

def simulate_heston_euler(
    S0,        # initial stock price
    v0,        # initial variance
    mu,        # drift of S
    kappa,     # mean-reversion speed for v
    theta,     # long-run variance
    sigma_v,   # vol of variance (vol of vol)
    rho,       # correlation between Brownian increments dW1, dW2
    T,         # total time in years
    Nsteps,    # number of steps
    seed=500,
    use_log_euler=True
):
    """
    Simulate Heston paths using Euler–Maruyama for v(t) and 
    either direct Euler or log-Euler for S(t).

    Returns:
        t: array of times, length (Nsteps+1)
        S: array of prices, length (Nsteps+1)
        v: array of variances, length (Nsteps+1)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / Nsteps
    t = np.linspace(0, T, Nsteps + 1)

    S = np.zeros(Nsteps + 1)
    v = np.zeros(Nsteps + 1)

    S[0] = S0
    v[0] = v0

    # Generate random draws for each step
    Z1 = np.random.normal(0, 1, Nsteps)
    Z2 = np.random.normal(0, 1, Nsteps)

    for i in range(Nsteps):
        # Correlated increment for the second Wiener process
        z1 = Z1[i]
        z2 = Z2[i]
        dW2 = rho * z1 + math.sqrt(1 - rho**2)*z2

        # Euler update for variance
        v_old = v[i]
        v_next = (v_old
                  + kappa*(theta - v_old)*dt
                  + sigma_v*math.sqrt(max(v_old,0))*math.sqrt(dt)*dW2)
        # Enforce non-negativity
        v_next = max(v_next, 0.0)

        # Update price
        S_old = S[i]
        if use_log_euler:
            # Log-Euler approach
            logS_old = math.log(S_old)
            logS_next = (logS_old
                         + (mu - 0.5*v_old)*dt
                         + math.sqrt(max(v_old,0))*math.sqrt(dt)*z1)
            S_next = math.exp(logS_next)
        else:
            # Direct Euler approach (less common for Heston but simpler code)
            S_next = (S_old
                      + mu*S_old*dt
                      + S_old*math.sqrt(max(v_old,0))*math.sqrt(dt)*z1)
            # Could become negative if step is large or volatility is big

        S[i+1] = S_next
        v[i+1] = v_next

    return t, S, v

def simulate_and_save_euler(filename="my_heston_euler.csv"):
    """
    Example function that simulates with fixed parameters using Euler,
    then saves (time, price, variance) to CSV for later reuse.
    """
    # 1) Choose your parameters
    S0      = 100.0
    v0      = 0.05
    mu      = 0.1
    kappa   = 1
    theta   = 0.05
    sigma_v = 0.05
    rho     = -0.7
    T       = 1.0
    Nsteps  = 252

    # 2) Simulate using Euler
    t, S, v = simulate_heston_euler(
        S0, v0, mu, kappa, theta, sigma_v, rho, T, Nsteps, seed=42, use_log_euler=True
    )

    # 3) Save to CSV
    data = np.column_stack((t, S, v))
    np.savetxt(filename, data, delimiter=",", header="time,price,variance", comments="")
    print(f"Euler-simulated data saved to {filename}")

def load_prices_from_csv(filename="my_heston_euler.csv"):
    """
    Load previously simulated data from CSV, return arrays of time, price, and variance.
    """
    data = np.loadtxt(filename, delimiter=",", skiprows=1)  # skip the header
    t = data[:, 0]
    prices = data[:, 1]
    variances = data[:, 2]
    return t, prices, variances

def plot_price_variance(t, S, v):
    """
    Plot the price path and variance over time, plus optional sqrt(v) for volatility.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), sharex=True)
    ax1.plot(t, S, label="Price")
    ax1.set_ylabel("Price")
    ax1.legend(loc="best")

    ax2.plot(t, v, label="Variance", color="red")
    # ax2.plot(t, np.sqrt(v), label="Vol = sqrt(v)", color="orange", linestyle="--")
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("Variance / Vol")
    ax2.legend(loc="best")

    plt.suptitle("Heston Simulation (Euler Scheme): Price & Variance")
    plt.show()

if __name__ == "__main__":
    # ----------------------
    # 1) Simulate & save
    # ----------------------
    simulate_and_save_euler("my_heston_euler.csv")

    # ----------------------
    # 2) Load & plot
    # ----------------------
    t, prices, variances = load_prices_from_csv("my_heston_euler.csv")
    print("Loaded paths from CSV:")
    print(" times:", t[:5], "...")
    print(" prices:", prices[:5], "...")
    print(" variances:", variances[:5], "...")

    # 3) Plot
    plot_price_variance(t, prices, variances)

