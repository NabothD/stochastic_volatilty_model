import numpy as np
import math
import matplotlib.pyplot as plt

def simulate_bates_euler(
    S0,        # initial stock price
    v0,        # initial variance
    mu,        # drift of S
    kappa,     # mean-reversion speed for v
    theta,     # long-run variance
    sigma_v,   # vol of variance (vol of vol)
    rho,       # correlation between Brownian increments dW1, dW2
    lambda_jump,  # jump intensity (annualized)
    mu_J,         # mean of log jump size
    sigma_J,      # std dev of log jump size
    T,         # total time in years
    Nsteps,    # number of steps
    seed=10000,
    use_log_euler=True
):
    """
    Simulate Bates model paths using Euler–Maruyama for variance and log-Euler for S(t) with jumps.
    
    The Bates model adds jumps to the Heston model:
        dS(t) = mu S(t) dt + sqrt(v(t)) S(t) dW_S(t) + (e^{Z(t)} - 1) S(t) dq(t)
    where dq(t) is a Poisson process with intensity lambda_jump and 
    Z(t) ~ N(mu_J, sigma_J^2).
    
    In the log-Euler scheme the update is:
        log S_{t+Δt} = log S_t + [mu - lambda_jump*(exp(mu_J+0.5*sigma_J^2)-1) - 0.5*v(t)]*Δt
                         + sqrt(v(t)Δt)*z1 + J*Z,
    where J ~ Bernoulli(lambda_jump * Δt) and Z ~ N(mu_J, sigma_J^2) when J=1.
    
    Returns:
        t: array of times (length Nsteps+1)
        S: array of prices (length Nsteps+1)
        v: array of variances (length Nsteps+1)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / Nsteps
    t = np.linspace(0, T, Nsteps + 1)

    S = np.zeros(Nsteps + 1)
    v = np.zeros(Nsteps + 1)

    S[0] = S0
    v[0] = v0

    # Generate independent standard normals for price and second Brownian motion
    Z1 = np.random.normal(0, 1, Nsteps)
    Z2 = np.random.normal(0, 1, Nsteps)

    for i in range(Nsteps):
        # Get correlated increments:
        z1 = Z1[i]
        z2 = Z2[i]
        dW2 = rho * z1 + math.sqrt(1 - rho**2) * z2

        # Update variance v(t) using Euler-Maruyama
        v_old = v[i]
        v_next = v_old + kappa * (theta - v_old) * dt + sigma_v * math.sqrt(max(v_old, 0)) * math.sqrt(dt) * dW2
        # Enforce non-negativity
        v_next = max(v_next, 0.0)
        v[i+1] = v_next

        # Simulate jump indicator:
        # The probability of a jump in dt is lambda_jump * dt.
        if np.random.rand() < lambda_jump * dt:
            J = 1
            Z_jump = np.random.normal(mu_J, sigma_J)
        else:
            J = 0
            Z_jump = 0.0

        # Update price using log-Euler scheme:
        if use_log_euler:
            logS_old = math.log(S[i])
            # The compensator is subtracted to keep the process a martingale.
            compensator = lambda_jump * (math.exp(mu_J + 0.5 * sigma_J**2) - 1)
            drift = mu - compensator - 0.5 * v_old
            logS_next = logS_old + drift * dt + math.sqrt(max(v_old, 0)) * math.sqrt(dt) * z1 + J * Z_jump
            S[i+1] = math.exp(logS_next)
        else:
            # Direct Euler update (less common for Bates because of potential negativity)
            # Note: When a jump occurs, multiply by jump factor
            jump_factor = math.exp(Z_jump) if J == 1 else 1.0
            S[i+1] = S[i] + mu * S[i] * dt + S[i] * math.sqrt(max(v_old, 0)) * math.sqrt(dt) * z1
            S[i+1] *= jump_factor

    return t, S, v

def simulate_and_save_bates(filename="my_bates_euler.csv"):
    """
    Simulate using the Bates model with jumps and save (time, price, variance) to CSV.
    """
    # Parameters (you can adjust these to generate more jumps)
    S0      = 100.0
    v0      = 0.04
    mu      = 0.2
    kappa   = 1
    theta   = 0.04
    sigma_v = 0.01
    rho     = 0.4
    lambda_jump = 1    # Increase lambda_jump to have more frequent jumps
    mu_J    = -0.8         # Mean jump (in log-scale)
    sigma_J = 0.2          # Std dev of jump sizes (in log-scale)
    T       = 1.0
    Nsteps  = 252

    t, S, v = simulate_bates_euler(S0, v0, mu, kappa, theta, sigma_v, rho,
                                   lambda_jump, mu_J, sigma_J, T, Nsteps, seed=42,
                                   use_log_euler=True)

    data = np.column_stack((t, S, v))
    np.savetxt(filename, data, delimiter=",", header="time,price,variance", comments="")
    print(f"Bates-simulated data saved to {filename}")

def load_prices_from_csv(filename):
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    t = data[:, 0]
    prices = data[:, 1]
    variances = data[:, 2]
    return t, prices, variances

def plot_price_variance(t, S, v):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), sharex=True)
    ax1.plot(t, S, label="Price")
    ax1.set_ylabel("Price")
    ax1.legend(loc="best")

    ax2.plot(t, v, label="Variance", color="red")
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("Variance")
    ax2.legend(loc="best")

    plt.suptitle("Bates Simulation (Euler Scheme): Price & Variance")
    plt.show()

if __name__ == "__main__":
    # Simulate and save Bates data
    simulate_and_save_bates("my_bates_euler.csv")

    # Load and plot the simulated Bates data
    t, prices, variances = load_prices_from_csv("my_bates_euler.csv")
    print("Loaded Bates simulation:")
    print(" times:", t[:5], "...")
    print(" prices:", prices[:5], "...")
    print(" variances:", variances[:5], "...")
    plot_price_variance(t, prices, variances)
