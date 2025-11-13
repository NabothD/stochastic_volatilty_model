from weakref import ref
import numpy as np
from scipy.stats import norm, invgamma, bernoulli
import random
import matplotlib.pyplot as plt
import seaborn as sns
import math
import yfinance as yf

# ----------------------- Refined Resampling (Equations 67–73, Page 11) -----------------------
def refined_resample(V_tilde, W_tilde):


    # 1) Sort the particles and their weights in ascending order by V
    sort_indices = np.argsort(V_tilde)
    V_sorted = np.array(V_tilde)[sort_indices]
    W_sorted = np.array(W_tilde)[sort_indices]
    N = len(V_sorted)

    # Edge cases
    if N == 1:
        return V_sorted.copy()  # trivial single-particle case


    partial_sum = np.zeros(N+1)
    for k in range(N):
        partial_sum[k+1] = partial_sum[k] + W_sorted[k]


    cdf_left = np.zeros(N-1)
    cdf_right = np.zeros(N-1)
    slope = np.zeros(N-1)

    # Interval j=0: from V_sorted[0] to V_sorted[1]
    cdf_left[0] = 0.0
    if N > 1:
        cdf_right[0] = W_sorted[0] + 0.5 * W_sorted[1]
        denom = (V_sorted[1] - V_sorted[0])
        slope[0] = (cdf_right[0] - cdf_left[0]) / denom if denom > 1e-15 else 0.0

    # Intervals j=1..N-2
    for j in range(1, N-2):
        cdf_left[j] = cdf_right[j-1]
        # increment in cdf is 0.5 * W_sorted[j] + 0.5 * W_sorted[j+1]
        inc = 0.5 * W_sorted[j] + 0.5 * W_sorted[j+1]
        cdf_right[j] = cdf_left[j] + inc
        denom = (V_sorted[j+1] - V_sorted[j])
        slope[j] = (cdf_right[j] - cdf_left[j]) / denom if denom > 1e-15 else 0.0

    # Last interval j=N-2: from V_sorted[N-2] to V_sorted[N-1]
    if N > 1:
        j = N-2
        cdf_left[j] = 0.0 if j == 0 else cdf_right[j-1]
        # increment in cdf is 0.5 * W_sorted[j] + W_sorted[j+1], if we follow eq. (72) strictly
        # but we only have W_sorted up to index N-1 => W_sorted[N-1] is the last
        # j+1 = N-1, so the increment is 0.5 * W_sorted[N-2] + W_sorted[N-1].
        inc = 0.5 * W_sorted[j] + W_sorted[j+1]
        # Ensure we don't exceed 1.0
        cdf_right[j] = min(cdf_left[j] + inc, 1.0)
        denom = (V_sorted[j+1] - V_sorted[j])
        slope[j] = (cdf_right[j] - cdf_left[j]) / denom if denom > 1e-15 else 0.0

    # Now we have piecewise intervals: [V_sorted[j], V_sorted[j+1]], cdf from cdf_left[j] to cdf_right[j].
    # We'll invert that piecewise function for uniform random draws.

    # 4) Inverse transform sampling
    u = np.random.rand(N)  # draw N uniform random numbers in [0,1]
    V_refined = np.zeros(N)

    for i in range(N):
        # find the interval j where cdf_left[j] <= u[i] < cdf_right[j]
        # We'll do a simple linear search for clarity. For large N, consider np.searchsorted.
        ui = u[i]
        # Special cases: if ui <= cdf_left[0], we clamp to V_sorted[0].
        # If ui >= cdf_right[N-2], we clamp to V_sorted[N-1].
        if ui <= cdf_left[0]:
            V_refined[i] = V_sorted[0]
            continue
        # find j in [0..N-2]
        j_found = None
        for j in range(N-1):
            if ui < cdf_right[j] or j == (N-2):
                j_found = j
                break
        
        if j_found is None:
            # fallback if not found
            V_refined[i] = V_sorted[-1]
            continue
        
        # local interpolation
        # cdf_left[j] + slope[j]*(v - V_sorted[j]) = u[i]
        # => v = V_sorted[j] + (u[i] - cdf_left[j]) / slope[j]
        if abs(slope[j_found]) < 1e-15:
            # degenerate slope => all points are at V_sorted[j_found]
            V_refined[i] = V_sorted[j_found]
        else:
            frac = (ui - cdf_left[j_found]) / slope[j_found]
            V_refined[i] = V_sorted[j_found] + frac

        # clamp if fraction tries to go beyond the interval
        if V_refined[i] < V_sorted[j_found]:
            V_refined[i] = V_sorted[j_found]
        elif V_refined[i] > V_sorted[j_found+1]:
            V_refined[i] = V_sorted[j_found+1]

    return V_refined

# ----------------------- Heston with Merton Jumps Estimation -----------------------
def heston_with_merton_jumps_estimation(
    n_samples, delta_t, maturity, prices, n_particles, 
    # Heston initial guesses:
    mu_0, kappa_0, theta_0, sigma_0, rho_0,
    # Heston prior parameters:
    mu_eta_0, tau_eta_0, mu_beta_0, lambda_beta_0,
    a_sigma_0, b_sigma_0, mu_psi_0, tau_psi_0, a_omega_0, b_omega_0,
    # Jump parameters & priors:
    lambda_jump_0,       # initial jump probability (or threshold fraction)
    mu_j0, sigma_j0      # prior for jump-size distribution: Normal(mu_j0, sigma_j0)
):
    n = len(prices) - 1  # number of time intervals
    # Precompute the unadjusted return series R_raw = S(kΔt)/S((k-1)Δt)
    R_raw = prices[1:] / prices[:-1]  # shape (n,)
    
    # Initialize storage for MCMC samples
    mu_samples     = np.zeros(n_samples)
    kappa_samples  = np.zeros(n_samples)
    theta_samples  = np.zeros(n_samples)
    sigma_samples  = np.zeros(n_samples)
    rho_samples    = np.zeros(n_samples)
    lambda_samples = np.zeros(n_samples)  # overall jump intensity estimates
    mu_j_samples   = np.zeros(n_samples)  # jump size mean samples
    sig_j_samples  = np.zeros(n_samples)  # jump size std samples

    # Prior hyperparameters (as given)
    eta_prior_mean = mu_eta_0
    eta_prior_precision = tau_eta_0
    lambda_beta_0 = np.array(lambda_beta_0)  # should be 2x2 matrix
    mu_beta_0 = np.array(mu_beta_0)           # shape (2,)
    sigma_prior_a = a_sigma_0
    sigma_prior_b = b_sigma_0
    psi_prior_mean = mu_psi_0
    psi_prior_precision = tau_psi_0
    omega_prior_a = a_omega_0
    omega_prior_b = b_omega_0

    # Set current parameter guesses (initial values)
    mu    = mu_0
    kappa = kappa_0
    theta = theta_0
    sigma = sigma_0
    rho   = rho_0
    lam_j = lambda_jump_0
    mu_j  = mu_j0
    sig_j = sigma_j0

    # To store volatility estimates from each MCMC iteration
    volatility_estimates_all = np.zeros((n_samples, n))
    
    # Initialize volatility particles at time 0 (set equal to theta as per Equation (60))
    V = np.full(n_particles, theta)
    
    for i in range(n_samples):
        # Initialize jump indicator and jump size arrays for the current MCMC iteration.
        J = np.zeros(n_particles, dtype=int)
        Z = np.zeros(n_particles)
        
        # Time-series arrays for volatility, jump size and jump probability at each time step
        v_est = np.zeros(n)       # v(kΔt) estimates (Equation (74))
        z_est = np.zeros(n)       # average jump size Z(kΔt) per time step (Equation (81))
        lam_t_series = np.ones(n)  # jump probability λ(kΔt) per time step (Equation (79))
    
        # ------------------- Particle Filtering Loop over time steps -------------------
        for k in range(n):
            # Step 1: For each particle, generate jump proposals
            # J̃_j(kΔt) ~ Bernoulli(lam_j)  [Equation (75)]
            J_candidates = bernoulli.rvs(lambda_jump_0, size=n_particles)
            # J_candidates = np.zeros(n_particles)
            # Z̃_j(kΔt) ~ Normal(μ_j, σ_j) for jump particles [Equation (76)]
            Z_candidates = np.random.normal(mu_j, sig_j, size=n_particles)
            
            # Generate independent standard normals for the Euler propagation
            Eps = np.random.normal(0, 1, size=n_particles)
            
            V_candidates = np.zeros(n_particles)
            weights = np.zeros(n_particles)
            sum_w = 0.0
            
            # Step 2: For each particle j, propagate volatility using Euler–Maruyama discretisation
            for j in range(n_particles):
                J_candidates[j] = bernoulli.rvs(lam_j)
                eps_j = np.random.normal(0, 1)
                # Compute residual z_j as per Equation (62)
                z_j = (R_raw[k] - mu * delta_t - 1.0) / math.sqrt(V[j] * delta_t)
                # Compute w_j using Equation (63)
                w_j = rho * z_j + math.sqrt(1.0 - rho**2) * eps_j
                # Propagate volatility using Equation (64)
                old_vol = V[j]
                new_vol = old_vol + kappa * (theta - old_vol) * delta_t + sigma * math.sqrt(delta_t * old_vol) * w_j
                # new_vol = max(new_vol, 1e-12)          #############################################
                V_candidates[j] = new_vol
            
            # Step 3: Compute weights using Equation (77)
                if J_candidates[j] == 0:
                    weights[j] = (1.0 / math.sqrt(2 * math.pi * V_candidates[j] * delta_t)) * \
                                 math.exp(-0.5 * ((R_raw[k] - mu * delta_t - 1.0)**2) / (V_candidates[j] * delta_t))
                else:  # if J_candidates[j] == 1
                    weights[j] = (1.0 / (math.exp(Z_candidates[j]) * math.sqrt(2 * math.pi * V_candidates[j] * delta_t))) * \
                                 math.exp(-0.5 * ((R_raw[k] - math.exp(Z_candidates[j]) * (mu * delta_t + 1.0))**2) / (math.exp(2 * Z_candidates[j]) * V_candidates[j] * delta_t))
                sum_w += weights[j]


            if sum_w == 0:
                weights = np.full(n_particles, 1.0/n_particles)
            else:
                weights = np.array(weights) / sum_w
                


            # **** Correction: Compute jump probability at this time step BEFORE resampling ****
            
            
            # Step 4: Resample volatility particles using refined resampling (Equations (67)-(73))
            V = refined_resample(V_candidates, weights)
            Z= refined_resample(Z_candidates, weights)
            
            # Estimate volatility at time k as the mean of refined particles (Equation (74))
            lam_t_series[k] = np.sum(J_candidates * weights)

           
            v_est[k] = np.mean(V)

            

            # Step 5: Compute average jump size at time k (Equation (81))
            z_est[k] = np.mean(Z)
        volatility_estimates_all[i, :] = v_est
        # ------------------- End of Particle Filtering Loop -------------------
        # Step 6: Adjust observed returns to neutralize jumps using Equation (84)
        for k in range(n): 
            jump_factor = 1.0 - lam_t_series[k] * (1.0 - math.exp(-z_est[k]))
            R_raw[k] = R_raw[k] * jump_factor

        
        # Step 7: Update overall jump parameters
        # Equation (80): λ_i = (1/n) Σ λ(kΔt)
        lam_j = np.sum(lam_t_series)*delta_t
        # Overall average jump size: Z_i = (1/N) Σ Z(kΔt)
        Z_i = np.mean(z_est)
        # Equations (82)-(83): Weighted mean and std for jump sizes
        sum_lambda = np.sum(lam_t_series)
        
        if sum_lambda == 0:
            mu_j_new = mu_j
            sig_j_new = sig_j
        else:
            sum_lamZ = np.sum(z_est * lam_t_series)
            mu_j_new = sum_lamZ / sum_lambda
            sum_sqr = 0.0
            for k in range(n):
                diff = (z_est[k] - mu_j_new)
                sum_sqr += lam_t_series[k] * diff * diff
            var_j = sum_sqr / (((n - 1) / n) * sum_lambda)
            var_j = max(var_j, 1e-15)
            sig_j_new = math.sqrt(var_j)
        mu_j  = mu_j_new
        sig_j = sig_j_new

        # ------------------- Bayesian Updates -------------------
        # Step 2a: Update drift parameter μ using Bayesian regression (Equations (13)-(23))
        x_s = (1.0 / math.sqrt(delta_t)) / np.sqrt(v_est)
        y_s = (1.0 / math.sqrt(delta_t)) * (R_raw / np.sqrt(v_est))
        ols_est = np.dot(x_s, y_s) / np.dot(x_s, x_s)
        Tau_eta = np.dot(x_s, x_s) + eta_prior_precision
        eta_posterior_mean = (eta_prior_precision * eta_prior_mean + np.dot(x_s, x_s) * ols_est) / Tau_eta
        eta_sample = np.random.normal(eta_posterior_mean, 1.0 / math.sqrt(Tau_eta))
        mu = (eta_sample - 1.0) / delta_t
        mu = np.clip(mu, -10.0, 10.0)
        mu_samples[i] = mu

        # Step 2b: Update (κ, θ) via volatility regression (Equations (24)-(44))
        y_list, x1_list, x2_list = [], [], []
        for k in range(1, n):
            denom = math.sqrt(delta_t * v_est[k-1])
            if denom < 1e-15:
                raise ValueError(f"Volatility too close to zero at step {k-1} in regression.")
            y_list.append(v_est[k] / denom)
            x1_list.append(1.0 / denom)
            x2_list.append(v_est[k-1] / denom)
        y_vec = np.array(y_list)
        x1_vec = np.array(x1_list)
        x2_vec = np.array(x2_list)
        X_mat = np.column_stack((x1_vec, x2_vec))
        
        beta_hat_ols = np.linalg.pinv(X_mat.T @ X_mat) @ (X_mat.T @ y_vec)
        Lambda_beta = X_mat.T @ X_mat + lambda_beta_0
        rhs = lambda_beta_0 @ mu_beta_0 + (X_mat.T @ X_mat) @ beta_hat_ols
        mu_beta = np.linalg.pinv(Lambda_beta) @ rhs
        cov_beta = sigma**2 * np.linalg.pinv(Lambda_beta)
        beta_draw = np.random.multivariate_normal(mean=mu_beta, cov=cov_beta)
        kappa = (1.0 - beta_draw[1]) / delta_t
        theta = beta_draw[0] / (kappa * delta_t)
        kappa = np.clip(kappa, 1e-6, 1e5)
        theta = np.clip(theta, 1e-6, 1e5)
        kappa_samples[i] = kappa
        theta_samples[i] = theta

        # Step 2c: Update σ² by sampling from its inverse-gamma posterior (Equations (42)-(44))
        sigma_b = sigma_prior_b + 0.5 * (np.dot(y_vec, y_vec) + mu_beta_0 @ lambda_beta_0 @ mu_beta_0 - mu_beta.T @ Lambda_beta @ mu_beta)
        sigma_a = sigma_prior_a + n / 2.0
        sigma_sq = invgamma.rvs(a=sigma_a, scale=sigma_b)
        sigma = math.sqrt(sigma_sq)
        sigma_samples[i] = sigma

        # Step 2d: Update ρ using residuals from price and volatility equations (Equations (45)-(59))
        e1_rho = np.zeros(n)
        e2_rho = np.zeros(n)
        for t in range(n):
            if t == 0:
                e1_rho[t] = 0.0
            else:
                denom1 = math.sqrt(delta_t * max(v_est[t-1], 1e-12))
                numerator1 = R_raw[t] - (1.0 + mu * delta_t)
                e1_rho[t] = numerator1 / denom1
            if t == 0:
                e2_rho[t] = 0.0
            else:
                dv = v_est[t] - v_est[t-1]
                drift = kappa * (theta - v_est[t-1]) * delta_t
                denom2 = math.sqrt(delta_t * max(v_est[t-1], 1e-12))
                e2_rho[t] = (dv - drift) / denom2
        e_rho = np.column_stack((e1_rho, e2_rho))
        A_rho = e_rho.T @ e_rho

        A11 = A_rho[0, 0]
        A12 = A_rho[0, 1]
        A22 = A_rho[1, 1]
        a_omega = omega_prior_a + n / 2.0
        b_omega = omega_prior_b + 0.5 * (A22 - (A12**2) / max(A11, 1e-12))

        omega_draw = invgamma.rvs(a=a_omega, scale=b_omega)

        tau_psi = A11 + psi_prior_precision
        mu_psi = (A12 + psi_prior_mean * psi_prior_precision) / tau_psi

        psi_draw = norm.rvs(loc=mu_psi, scale=math.sqrt(omega_draw / tau_psi))

        new_rho = psi_draw / math.sqrt(psi_draw**2 + omega_draw)
        rho = np.clip(new_rho, -0.9999, 0.9999)
        rho_samples[i] = rho

        # Store current jump parameter samples (no further update)
        lambda_samples[i] = lam_j
        mu_j_samples[i] = mu_j
        sig_j_samples[i] = sig_j


    # ------------------- Final Aggregation -------------------
    mu_hat = np.mean(mu_samples)
    kappa_hat = np.mean(kappa_samples)
    theta_hat = np.mean(theta_samples)
    sigma_hat = np.mean(sigma_samples)
    rho_hat = np.mean(rho_samples)
    
    return {
        'mu': mu_hat,
        'kappa': kappa_hat,
        'theta': theta_hat,
        'sigma': sigma_hat,
        'rho': rho_hat,
        'lambda': np.mean(lambda_samples),
        'mu_j': np.mean(mu_j_samples),
        'sig_j': np.mean(sig_j_samples),
        'mu_samples': mu_samples,
        'kappa_samples': kappa_samples,
        'theta_samples': theta_samples,
        'sigma_samples': sigma_samples,
        'rho_samples': rho_samples,
        'lambda_samples': lambda_samples,
        'mu_j_samples': mu_j_samples,
        'sig_j_samples': sig_j_samples,
        'volatility_estimates_all': volatility_estimates_all
    }

# ----------------------- Plotting and Data Loading -----------------------
def plot_parameter_distributions(true_values, parameter_samples, parameter_names):
    n_params = len(parameter_names)
    fig, axes = plt.subplots(1, n_params, figsize=(15, 5), sharey=False)
    for i, param in enumerate(parameter_names):
        sns.kdeplot(parameter_samples[param], ax=axes[i], label='Estimate distribution', color='blue')
        if param in true_values:
            axes[i].axvline(true_values[param], color='red', linestyle='--', label='True value')
        axes[i].set_title(f"Parameter: {param}")
        axes[i].set_xlabel("Estimate")
        axes[i].set_ylabel("Empirical PDF")
        axes[i].legend()
    plt.show()

def load_prices_from_csv(filename):
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    t = data[:, 0]
    prices = data[:, 1]
    variances = data[:, 2]
    return t, prices, variances

# ----------------------- Main Example Usage -----------------------
if __name__ == "__main__":
    T = 1.0        # 1 year
    Nsteps = 252   # trading days in a year
    t, prices, v_path = load_prices_from_csv("prices_with_jumps.csv")
    maturity = T
    true_values = {'mu': 0.1, 'kappa': 1, 'theta': 0.05, 'sigma': 0.01, 'rho': -0.5, 'lambda': 1, 'mu_j': -0.8, 'sig_j': 0.2}

    # Plot synthetic paths
    t_grid = np.linspace(0, T, Nsteps + 1)
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(t_grid, prices, label="Synthetic Price")
    ax[0].set_ylabel("Price")
    ax[0].legend()
    ax[1].plot(t_grid, np.sqrt(v_path), label="Synthetic Vol (sqrt(v))")
    ax[1].set_ylabel("Volatility")
    ax[1].set_xlabel("Time (years)")
    ax[1].legend()
    plt.suptitle("Heston Simulation with Known (True) Parameters")
    plt.show()


    # ticker = "^GSPC"  # Apple as an example
    # start_date = "2018-01-01"
    # end_date   = "2022-07-01"
    # data = yf.download(ticker, start=start_date, end=end_date)

    # # print(data)
    # # print(data.columns)
    # # Drop any missing rows just in case:
    # data = data.dropna(subset=[("Close", "^GSPC")])

    # # 2) Extract the 'Adj Close' prices as a NumPy array
    # prices = data[("Close", "^GSPC")].values
    
    # # 3) Decide on a time step.  For example, if you treat each row as one
    # #    trading day and want "1 year" = 252 trading days, then:
    # delta_t = 1.0 / 252.0
    # # The total number of price points:
    # n_steps = len(prices)
    # # The implied "maturity" in years is then:
    # maturity = n_steps * delta_t

    # # 2) Compute daily returns (log or pct). For realized volatility,
    # #    you can use log returns:
    # data["LogReturn"] = np.log(data["Close"] / data["Close"].shift(1))

    # # 3) Rolling standard deviation over a 21-day window (approx 1 month).
    # #    Multiplying by sqrt(252) annualizes the daily volatility.
    # window = 21
    # data[f"RealizedVol_{window}"] = (
    #     data["LogReturn"].rolling(window).std() * np.sqrt(252)
    # )

    # # Now df["RealizedVol_21"] is your "real" (historical) volatility measure 
    # # over a 21-day rolling window.

    # # 4) Plot it
    # plt.figure(figsize=(10,5))
    # plt.plot(data.index, data[f"RealizedVol_{window}"], label=f"{window}-day Realized Vol")
    # plt.title(f"S&P500 {window}-Day Realized Vol (annualized)")
    # plt.xlabel("Date")
    # plt.ylabel("Volatility")
    # plt.legend()
    # plt.show()


    # Set parameters for estimation
    n_samples = 100
    n_particles = 100

    # Initial parameter values
    mu_0 = 0.08
    kappa_0 = 1.2
    theta_0 = 0.04
    sigma_0 = 0.011
    rho_0 = -0.48

    # Priors
    mu_eta_0 = 1.000125
    tau_eta_0 = 1.0/math.sqrt(0.001**2)
    mu_beta_0 = [35e-7, 0.998]        # <--- shape (2,)
    lambda_beta_0 = [[10, 0], [0, 5]] # shape (2,2)
    a_sigma_0 = 149
    b_sigma_0 = 0.026

    
    mu_psi_0 = -0.45
    tau_psi_0 = 1.0/math.sqrt(0.25**2)
    a_omega_0 = 1.03
    b_omega_0 = 0.05

    #initial values of the jump parameters
    lambda_jump_0 = 0.85
    mu_j0 = -0.9
    sigma_j0 = 0.3

    delta_t = 1.0 / len(prices)

    # Run the estimation procedure
    results = heston_with_merton_jumps_estimation(
        n_samples, delta_t, maturity, prices,
        n_particles, mu_0, kappa_0, theta_0, sigma_0, rho_0, 
        mu_eta_0, tau_eta_0, mu_beta_0, lambda_beta_0, a_sigma_0, b_sigma_0, 
        mu_psi_0, tau_psi_0, a_omega_0, b_omega_0, lambda_jump_0, mu_j0, sigma_j0
    )
    
    print("Estimated Parameters:")
    print(f"mu: {results['mu']}")
    print(f"kappa: {results['kappa']}")
    print(f"theta: {results['theta']}")
    print(f"sigma: {results['sigma']}")
    print(f"rho: {results['rho']}")
    print(f"lambda: {results['lambda']}")
    print(f"mu_j: {results['mu_j']}")
    print(f"sig_j: {results['sig_j']}")

    vols = results['volatility_estimates_all']
    final_volatility_path = vols[-1, :]
    plt.figure(figsize=(8, 5))
    plt.plot(final_volatility_path, label='Estimated Volatility (final iteration)')
    plt.plot(v_path, label="Synthetic Volatility (sqrt(v))")
    plt.xlabel('Time Step')
    plt.ylabel('Volatility')
    plt.legend()
    plt.title('Particle-Filtered Volatility (Final MCMC Iteration)')
    plt.show()

    avg_volatility_path = vols.mean(axis=0)
    plt.figure(figsize=(8, 5))
    plt.plot(avg_volatility_path, label='Mean Volatility (over MCMC)')
    plt.plot(v_path, label="Synthetic Volatility (sqrt(v))")
    plt.xlabel('Time Step')
    plt.ylabel('Volatility')
    plt.legend()
    plt.title('Mean Particle-Filtered Volatility Estimate Over MCMC')
    plt.show()

    # Collect parameter samples for plotting distributions
    parameter_samples = {
        'mu': results.get('mu_samples', []),
        'kappa': results.get('kappa_samples', []),
        'theta': results.get('theta_samples', []),
        'sigma': results.get('sigma_samples', []),
        'rho': results.get('rho_samples', []),
        'lambda': results.get('lambda_samples', []),
        'mu_j': results.get('mu_j_samples', []),
        'sig_j': results.get('sig_j_samples', [])
    }
    parameter_names = ['mu', 'kappa', 'theta', 'sigma', 'rho', 'lambda', 'mu_j', 'sig_j']
    plot_parameter_distributions(true_values, parameter_samples, parameter_names)
