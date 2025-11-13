import numpy as np
import math
import matplotlib.pyplot as plt

def simulate_heston_QES(
    S0, v0,               # initial price & initial variance
    mu, kappa, theta, sigma_v, rho,
    T, Nsteps,
    seed=None
):
    """
    Simulate Heston paths using Andersen's Quadratic-Exponential (QE) scheme
    for variance, and a log-Euler step for the price with correlation.
    
    Returns:
        t:      array of times of length (Nsteps+1)
        S:      array of simulated prices of length (Nsteps+1)
        v:      array of simulated variances of length (Nsteps+1)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / Nsteps
    t = np.linspace(0, T, Nsteps + 1)

    # Arrays to store the simulated paths
    S = np.zeros(Nsteps + 1)
    v = np.zeros(Nsteps + 1)

    # Initialize
    S[0] = S0
    v[0] = v0

    # Pre-generate randoms for each step
    Z_1 = np.random.normal(0, 1, Nsteps)
    Z_2 = np.random.normal(0, 1, Nsteps)

    for i in range(Nsteps):
        # The second Brownian increment is correlated:
        z1 = Z_1[i]
        z2 = Z_2[i]
        dW2 = rho * z1 + math.sqrt(1 - rho**2) * z2

        v_old = v[i]

        # ---------- QES update for v(t+dt) ----------
        # Andersen's "Case 1" if (4 * m_t * phi < 1). 
        # For brevity, we keep an illustrative form:

        # Mean of the next variance (if no randomness)
        m_t = v_old * math.exp(-kappa*dt) + theta*(1 - math.exp(-kappa*dt))
        # phi
        phi = (sigma_v**2 * (1 - math.exp(-kappa*dt))) / (4*kappa)

        # QES "Case 1" check
        draw_u = np.random.rand()
        if (4 * m_t * phi) > 1e-12 and (4 * m_t * phi) < 1.0:
            A = math.sqrt(4 * m_t * phi)
            # This formula ensures positivity:
            v_new = (A / (1 + (1 - A)/(1 + A) * draw_u))**2
        else:
            # Fallback, e.g. Euler step
            dW_v = math.sqrt(dt) * z1
            v_new = (v_old 
                     + kappa * (theta - v_old) * dt
                     + sigma_v * math.sqrt(max(v_old, 0)) * dW_v)
            v_new = max(v_new, 0.0)

        v[i+1] = v_new

        # ---------- Update Price S(t+dt) ----------
        # Simple log-Euler using v_old as local variance:
        dW1 = math.sqrt(dt)*z1
        logS_old = math.log(S[i])
        logS_new = (logS_old
                    + (mu - 0.5 * v_old)*dt
                    + math.sqrt(max(v_old, 0.0))* dW1)
        S[i+1] = math.exp(logS_new)

    return t, S, v


def simulate_and_save_QES(filename="my_heston_QES.csv"):
    """
    Example function that simulates with fixed parameters,
    then saves (time, price, variance) to CSV for later reuse.
    """
    # 1) Choose your parameters
    S0      = 100.0
    v0      = 0.04
    mu      = 0.05
    kappa   = 1
    theta   = 0.04
    sigma_v = 0.6
    rho     = -0.5
    T       = 1.0
    Nsteps  = 252

    # 2) Simulate via QE scheme
    t, S, v = simulate_heston_QES(S0, v0, mu, kappa, theta, sigma_v, rho, T, Nsteps, seed=42)

    # 3) Save to CSV
    data = np.column_stack((t, S, v))
    # Writes a header row and no '#' comment character
    np.savetxt(filename, data, delimiter=",", header="time,price,variance", comments="")
    print(f"Simulated data saved to {filename}")


def load_prices_from_csv(filename="my_heston_QES.csv"):
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
    Plot the price path and variance (or volatility) over time.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), sharex=True)
    ax1.plot(t, S, label="Price")
    ax1.set_ylabel("Price")
    ax1.legend(loc="best")

    ax2.plot(t, np.sqrt(v), label="Vol (sqrt of variance)", color="orange")
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("Volatility")
    ax2.legend(loc="best")

    plt.suptitle("Heston Simulation (QE Scheme): Price & Volatility")
    plt.show()


if __name__ == "__main__":
    # ----------------------
    # 1) Simulate & save
    # ----------------------
    simulate_and_save_QES("my_heston_QES.csv")

    # ----------------------
    # 2) Load & plot
    # ----------------------
    t, prices, variances = load_prices_from_csv("my_heston_QES.csv")
    print("Loaded paths from CSV:")
    print(" times:", t[:5], "...")
    print(" prices:", prices[:5], "...")
    print(" variances:", variances[:5], "...")

    # 3) Plot
    plot_price_variance(t, prices, variances)
