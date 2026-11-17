import numpy as np
import math
import matplotlib.pyplot as plt

def simulate_heston_bates(
    S0,         # initial stock price
    v0,         # initial variance
    mu,         # drift of S
    kappa,      # mean-reversion speed for v
    theta,      # long-run variance
    sigma_v,    # volatility of variance ("vol of vol")
    rho,        # correlation between Brownian motions for S and v
    lambda_j,   # jump intensity (per year)
    mu_J,       # mean jump size (in log–space)
    sigma_J,    # jump size volatility (in log–space)
    T,          # total time in years
    Nsteps,     # number of steps
    seed=None,
    use_log_euler=True
):
    """
    Simulate Bates (Heston with jumps) paths using Euler–Maruyama for v(t) and a log–Euler
    method for S(t) that incorporates jumps.
    
    The price dynamics are:
      dS(t) = μ S(t) dt + √(v(t)) S(t) dW_S(t) + (e^(Z(t)) - 1) S(t) dQ(t)
    In the log–Euler discretisation the update becomes:
      logS(t+Δt) = logS(t) + [μ - λ_j (exp(μ_J + 0.5σ_J²) - 1) - 0.5 v(t)]Δt + √(v(t)Δt)·Z₁ + J,
    where J = Z if a jump occurs with probability λ_j·Δt, and 0 otherwise.
    
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
    logS = np.zeros(Nsteps + 1)
    logS[0] = math.log(S0)
    
    # Precompute jump adjustment term: E[e^Z] = exp(mu_J + 0.5*sigma_J^2)
    jump_adjust = np.exp(mu_J + 0.5 * sigma_J**2) - 1.0
    
    # Generate random draws for the Brownian increments
    Z1 = np.random.normal(0, 1, Nsteps)
    Z2 = np.random.normal(0, 1, Nsteps)
    
    for i in range(Nsteps):
        # Correlated Brownian increments for variance
        z1 = Z1[i]
        z2 = Z2[i]
        dW2 = rho * z1 + math.sqrt(1 - rho**2) * z2
        
        # Euler update for variance (same as in your Heston simulation)
        v_old = v[i]
        v_next = v_old + kappa * (theta - v_old) * dt + sigma_v * math.sqrt(max(v_old, 0)) * math.sqrt(dt) * dW2
        v_next = max(v_next, 0.0)
        v[i+1] = v_next
        
        # Determine if a jump occurs in this step:
        # The probability of at least one jump in dt is approximately lambda_j * dt.
        if np.random.uniform(0, 1) < lambda_j * dt:
            # A jump occurs; sample jump size from Normal(mu_J, sigma_J)
            jump = np.random.normal(mu_J, sigma_J)
        else:
            jump = 0.0
        
        if use_log_euler:
            # Log-Euler update for log-price
            logS_old = logS[i]
            # Drift adjusted for jumps: subtract λ_j*(E[e^Z]-1)*dt
            drift = (mu - lambda_j * jump_adjust - 0.5 * v_old) * dt
            diffusion = math.sqrt(max(v_old,0) * dt) * z1
            logS_next = logS_old + drift + diffusion + jump  # add jump term in log–space
            logS[i+1] = logS_next
            S[i+1] = math.exp(logS_next)
        else:
            # Direct Euler approach (less common for log–price simulation)
            S_old = S[i]
            diffusion = S_old * math.sqrt(max(v_old, 0) * dt) * z1
            jump_component = (math.exp(jump) - 1) * S_old  # multiplicative jump component
            S_next = S_old + mu * S_old * dt + diffusion + jump_component
            S[i+1] = S_next

    return t, S, v

def simulate_and_save_bates(filename="my_heston_bates.csv"):
    """
    Simulate Bates model data and save (time, price, variance) to CSV.
    """
    # Set parameters
    S0      = 100.0
    v0      = 0.05
    mu      = 0.1
    kappa   = 1
    theta   = 0.05
    sigma_v = 0.01
    rho     = -0.5
    lambda_j = 0.2       # jump intensity per year (e.g., on average one jump per year)
    mu_J    = -0.8       # mean jump (in log-price)
    sigma_J = 0.2        # jump size volatility
    T       = 1.0
    Nsteps  = 252

    t, S, v = simulate_heston_bates(S0, v0, mu, kappa, theta, sigma_v, rho,
                                    lambda_j, mu_J, sigma_J, T, Nsteps, seed=42,
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

    plt.suptitle("Bates Model Simulation: Price & Variance")
    plt.show()

if __name__ == "__main__":
    # Simulate Bates (Heston with jumps) data and save to CSV
    simulate_and_save_bates("my_heston_bates.csv")
    
    # Load and plot
    t, prices, variances = load_prices_from_csv("my_heston_bates.csv")
    print("First few rows of simulated Bates data:")
    print("Time:", t[:5])
    print("Price:", prices[:5])
    print("Variance:", variances[:5])
    
    plot_price_variance(t, prices, variances)
