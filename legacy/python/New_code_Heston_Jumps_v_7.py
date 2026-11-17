import numpy as np
from scipy.stats import norm, invgamma, bernoulli
import random
import matplotlib.pyplot as plt
import seaborn as sns
import math
import yfinance as yf
import pandas as pd
import scipy.stats as st


# ----------------------- Vectorized Particle Propagation -----------------------
def propagate_particles(V, R_val, mu, delta_t, kappa, theta, sigma, rho, J_candidates, Z_candidates):
    """
    Vectorized propagation of particles and computation of weights.
    Parameters:
      V           : Current volatility particles (shape: (n_particles,))
      R_val       : Adjusted return at the current time step (scalar)
      mu, delta_t, kappa, theta, sigma, rho : model parameters (scalars)
      J_candidates: Jump indicators for each particle (array, shape: (n_particles,))
      Z_candidates: Candidate jump sizes for each particle (array, shape: (n_particles,))
    Returns:
      new_V     : Propagated volatility particles (n_particles,)
      weights   : Corresponding weights (n_particles,)
    """
    n_particles = V.shape[0]
    eps = np.random.normal(0, 1, size=n_particles)
    # Compute standardized residual for each particle
    z = (R_val - mu * delta_t - 1.0) / np.sqrt(V * delta_t)
    w = rho * z + np.sqrt(1 - rho**2) * eps
    new_V = V + kappa * (theta - V) * delta_t + sigma * np.sqrt(V * delta_t) * w

    # Compute weights for non-jump particles:
    denom_nonjump = np.sqrt(2 * math.pi * new_V * delta_t)
    exponent_nonjump = -0.5 * ((R_val - mu * delta_t - 1.0)**2) / (new_V * delta_t)
    weights = np.where(J_candidates == 0,
                       1.0 / denom_nonjump * np.exp(exponent_nonjump),
                       0.0)
    
    # For jump particles:
    exp_Z = np.exp(Z_candidates)
    denom_jump = exp_Z * np.sqrt(2 * math.pi * new_V * delta_t)
    exponent_jump = -0.5 * ((R_val - exp_Z * (mu * delta_t + 1.0))**2) / (exp_Z**2 * new_V * delta_t)
    # Update weights for particles with a jump
    weights = np.where(J_candidates == 1,
                       1.0 / denom_jump * np.exp(exponent_jump),
                       weights)
    
    # Normalize weights
    sum_w = np.sum(weights)
    if sum_w == 0:
        weights = np.full(n_particles, 1.0/n_particles)
    else:
        weights /= sum_w
    return new_V, weights

# ----------------------- Refined Resampling (Equations 67–73) -----------------------
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
    # Unadjusted return series
    R_raw = prices[1:] / prices[:-1]
    
    # Storage for MCMC samples
    mu_samples     = np.zeros(n_samples)
    kappa_samples  = np.zeros(n_samples)
    theta_samples  = np.zeros(n_samples)
    sigma_samples  = np.zeros(n_samples)
    rho_samples    = np.zeros(n_samples)
    lambda_samples = np.zeros(n_samples)
    mu_j_samples   = np.zeros(n_samples)
    sig_j_samples  = np.zeros(n_samples)

    # Prior hyperparameters
    eta_prior_mean = mu_eta_0
    eta_prior_precision = tau_eta_0
    lambda_beta_0 = np.array(lambda_beta_0)
    mu_beta_0 = np.array(mu_beta_0)
    sigma_prior_a = a_sigma_0
    sigma_prior_b = b_sigma_0
    psi_prior_mean = mu_psi_0
    psi_prior_precision = tau_psi_0
    omega_prior_a = a_omega_0
    omega_prior_b = b_omega_0

    # Set current parameter guesses
    mu    = mu_0
    kappa = kappa_0
    theta = theta_0
    sigma = sigma_0
    rho   = rho_0
    lam_j = lambda_jump_0
    mu_j  = mu_j0
    sig_j = sigma_j0

    # Storage for volatility estimates and jump info
    volatility_estimates_all = np.zeros((n_samples, n))
    lam_t_series_all = np.zeros((n_samples, n))
    Z_series_all = np.zeros((n_samples, n))

    # Initialize volatility particles at time 0
    V = np.full(n_particles, theta)
    # Start with a copy of raw returns for iterative jump neutralization
    R_adj = np.copy(R_raw)
    
    for i in range(n_samples):
        # Arrays for current MCMC iteration
        v_est = np.zeros(n)
        z_est = np.zeros(n)
        lam_t_series = np.zeros(n)
        
        # ------------------- Particle Filtering Loop -------------------
        for k in range(n):
            # Generate jump indicators and candidate jump sizes for all particles
            J_candidates = bernoulli.rvs(lambda_jump_0, size=n_particles)
            Z_candidates = np.random.normal(mu_j, sig_j, size=n_particles)
            
            # Use the vectorized propagation function
            V_candidates, weights = propagate_particles(V, R_adj[k], mu, delta_t, kappa, theta, sigma, rho, J_candidates, Z_candidates)
            
            # Resample volatility and jump candidates using refined resampling
            V = refined_resample(V_candidates, weights)
            Z_refined = refined_resample(Z_candidates, weights)
            
            # Record estimated volatility and jump size for current time step
            v_est[k] = np.mean(V)
            z_est[k] = np.mean(Z_refined)
            
            lam_t_series[k] = np.sum(J_candidates * weights)
        
        volatility_estimates_all[i, :] = v_est
        lam_t_series_all[i, :] = lam_t_series
        Z_series_all[i, :] = z_est
        
        # ------------------- Adjust Returns to Neutralize Jumps -------------------
        for k in range(n): 
            jump_factor = 1.0 - lam_t_series[k] * (1.0 - math.exp(-z_est[k]))
            R_adj[k] = R_raw[k] * jump_factor

        
        # Step 7: Update overall jump parameters
        # Equation (80): λ_i = (1/n) Σ λ(kΔt)
        lam_j = np.sum(lam_t_series)
        print(lam_j)
    
        # Overall average jump size: Z_i = (1/N) Σ Z(kΔt)
        Z_i = np.mean(z_est)
        # Equations (82)-(83): Weighted mean and std for jump sizes
        sum_lambda = np.sum(lam_t_series)
        if sum_lambda == 0:
            print("This happened")
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
        'volatility_estimates_all': volatility_estimates_all,
        'Returns': R_adj
        
    }

# ----------------------- Plotting and Data Loading -----------------------
def plot_parameter_distributions(true_values, parameter_samples, parameter_names):
    n_params = len(parameter_names)
    fig, axes = plt.subplots(1, n_params, figsize=(10, 5), sharey=False)
    for i, param in enumerate(parameter_names):
        sns.kdeplot(parameter_samples[param], ax=axes[i], label='Estimate distribution', color='blue')
        if param in true_values:
            axes[i].axvline(true_values[param], color='red', linestyle='--', label='True value')
        # axes[i].set_title(f"Parameter: {param}")
        axes[i].set_xlabel("Estimate", fontsize=22)
        axes[i].set_ylabel("Empirical PDF", fontsize=22)
        axes[i].legend(fontsize=22)
    plt.show()
def load_prices_from_csv(filename):
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    t = data[:, 0]
    prices = data[:, 1]
    variances = data[:, 2]
    return t, prices, variances



def fit_best_distribution(data, distributions=None):
    """
    Fit a list of candidate distributions to data and return the best by AIC and BIC.
    
    Parameters:
    -----------
    data : array-like
        1D array of samples (e.g., posterior draws for a parameter).
    distributions : list of str, optional
        List of scipy.stats distribution names to consider. Default includes:
        ['norm', 'lognorm', 'gamma', 'beta', 'invgamma', 'expon']
    
    Returns:
    --------
    sorted_results : list of tuples
        Each tuple is (dist_name, info_dict) sorted by increasing AIC.
        info_dict contains:
          - 'aic': Akaike Information Criterion
          - 'bic': Bayesian Information Criterion
          - 'params': fitted distribution parameters
    """
    data = np.asarray(data)
    best = {}
    if distributions is None:
        distributions = ['norm', 'lognorm', 'gamma', 'beta', 'invgamma', 'expon']
        
    for dist_name in distributions:
        dist = getattr(st, dist_name, None)
        if dist is None:
            continue
        try:
            params = dist.fit(data)
            # Log-likelihood
            logpdf = dist.logpdf(data, *params)
            ll = np.sum(logpdf)
            k = len(params)
            # AIC and BIC
            aic = 2*k - 2*ll
            bic = k*np.log(len(data)) - 2*ll
            best[dist_name] = {'aic': aic, 'bic': bic, 'params': params}
        except Exception:
            continue
    
    # Sort by AIC
    sorted_results = sorted(best.items(), key=lambda kv: kv[1]['aic'])
    return sorted_results



import chaospy as cp


def scipy_to_chaospy(dist_name, params):
    """
    Map a fitted scipy.stats distribution to a chaospy distribution.
    
    dist_name: name of the scipy.stats distribution
    params: tuple returned by dist.fit(data)
    """
    if dist_name == 'norm':
        loc, scale = params
        return cp.Normal(loc, scale)
    elif dist_name == 'lognorm':
        s, loc, scale = params
        # SciPy lognorm: X ~ lognorm(s, loc, scale) => ln(X-loc) ~ N(ln(scale), s^2)
        mu_log = np.log(scale)
        sigma_log = s
        return cp.LogNormal(mu_log, sigma_log)
    elif dist_name == 'gamma':
        a, loc, scale = params
        # Assume loc ~ 0; otherwise need a shift
        return cp.Gamma(a, scale)
    elif dist_name == 'beta':
        a, b, loc, scale = params
        # SciPy beta: support [loc, loc+scale]
        lower = loc
        upper = loc + scale
        return cp.Beta(a, b, lower=lower, upper=upper)
    elif dist_name == 'invgamma':
        a, loc, scale = params
        # SciPy invgamma: support > loc; assume loc ~ 0
        return cp.InverseGamma(a, scale)
    elif dist_name == 'expon':
        loc, scale = params
        # SciPy expon: support [loc, ∞); assume loc ~ 0
        return cp.Exponential(scale)
    else:
        raise ValueError(f"No mapping for distribution '{dist_name}'")


def plot_best_fit_distribution(param_name, samples, fit_info, ax):
    """
    Plot histogram of samples and overlay the fitted distribution PDF.
    
    param_name: str, name of the parameter
    samples: 1D numpy array of posterior samples
    fit_info: tuple(dist_name, info_dict) as returned from fit_best_distribution
    ax: matplotlib Axes to plot on
    """
    dist_name, info = fit_info
    params = info['params']
    dist = getattr(st, dist_name)
    
    # Histogram
    ax.hist(samples, bins=30, density=True, alpha=0.6)
    
    # PDF curve
    x = np.linspace(np.min(samples), np.max(samples), 200)
    pdf_vals = dist.pdf(x, *params)
    ax.plot(x, pdf_vals, linewidth=2, label=f'{dist_name} fit')
    
    ax.set_title(f"{param_name}: best fit = {dist_name}")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Density")
    ax.legend()



def fit_without_extremes(data, lower_pct=1, upper_pct=99, distributions=None):
    """
    Trim data to exclude extreme tails and fit candidate distributions.
    
    Parameters:
    -----------
    data : array-like
        1D array of samples.
    lower_pct, upper_pct : float
        Percentile bounds to trim the data (e.g., 1 and 99).
    distributions : list of str, optional
        Candidate SciPy distribution names.
    """
    # Compute percentile bounds
    low, high = np.percentile(data, [lower_pct, upper_pct])
    trimmed = data[(data >= low) & (data <= high)]
    
    # Fit distributions on trimmed data
    fit_results = fit_best_distribution(trimmed, distributions=distributions)
    return trimmed, fit_results


def summarize(samples):
    mean       = np.mean(samples)
    std        = np.std(samples, ddof=1)              # sample standard deviation
    lower, upper = np.percentile(samples, [2.5, 97.5])  # 2.5th and 97.5th percentiles
    return mean, std, lower, upper

# ----------------------- Main Example Usage -----------------------
if __name__ == "__main__":
    print("Started")
    T = 3.0        # 1 year
    Nsteps = 3*252  # trading days in a year
    # t, prices, v_path = load_prices_from_csv("prices_with_jumps.csv")
    # prices, v_path = load_prices_from_csv2("my_heston.csv")
    t, prices, v_path = load_prices_from_csv("prices_with_jumps.csv")
    maturity = T
    true_values = {'mu': 0.1, 'kappa': 1.1, 'theta': 0.04, 'sigma': 0.01, 'rho': -0.5, 'lambda': 3, 'mu_j': -0.3, 'sig_j': 0.2}

    t_grid = np.linspace(0, T, Nsteps + 1)
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(t_grid, prices, label="Synthetic Price")
    ax[0].set_ylabel("Price")
    ax[0].legend()
    ax[1].plot(t_grid, np.sqrt(v_path), label="Synthetic Vol (sqrt(v))")
    ax[1].set_ylabel("Volatility")
    ax[1].set_xlabel("Time (years)")
    ax[1].legend()
    plt.suptitle("Heston Simulation with True Parameters")
    plt.show()

    n_samples = 10
    n_particles = 10

    mu_0 = 0.08
    kappa_0 = 1.3
    theta_0 = 0.04
    sigma_0 = 0.011
    rho_0 = -0.6
    delta_t = T / len(prices)
    mu_eta_0 = 1.00125
    tau_eta_0 = 1.0/math.sqrt(0.001**2)
    mu_beta_0 = [35e-6, 0.998]        # <--- shape (2,)
    lambda_beta_0 = [[10 , 0], [0, 5]] # shape (2,2)
    a_sigma_0 = 149
    b_sigma_0 = 0.026
    mu_psi_0 = -0.45
    tau_psi_0 = 1.0/math.sqrt(0.25**2)
    a_omega_0 = 1.03
    b_omega_0 = 0.05

    lambda_jump_0 = 0.16
    mu_j0 = -0.33
    sigma_j0 = 0.2
    def ensure_array(x):
        if isinstance(x, tuple):
            x = x[0]  # unwrap
        return np.array(x)

    

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

    Returns = results['Returns']
    plt.figure(figsize=(8, 6))
    plt.plot(Returns, label='Returns', linewidth=2)           # <-- Set line width here
    plt.xlabel('Time / days', fontsize=20)                   # <-- Set x-label font size
    plt.ylabel('Returns', fontsize=20)                        # <-- Set y-label font size
    plt.legend(fontsize=20)                                   # <-- Set legend font size
    plt.tick_params(axis='both', which='major', labelsize=20) # <-- Set tick label font size
    plt.show()

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



    # Assuming parameter_samples is a dictionary of lists or arrays
    parameter_df = pd.DataFrame(parameter_samples)
    vol_df = pd.DataFrame(final_volatility_path)
    df2 = pd.DataFrame({
        'Real': v_path[: len(final_volatility_path)],
        'Final': final_volatility_path,
        'Averege' : avg_volatility_path
    })


    # Save as CSV instead of Excel (lighter and MATLAB can read it easily)
    parameter_df.to_csv("parameter_samples_tr.csv", index=False)
    vol_df.to_csv("final_vol_samples_tr.csv", index=False)
    df2.to_csv("vol_samples_Bates_syn_tr.csv", index=False)
    parameter_names = ['mu', 'kappa', 'theta', 'sigma', 'rho', 'lambda', 'mu_j', 'sig_j']
    plot_parameter_distributions(true_values, parameter_samples, parameter_names)

    for name in ['mu','kappa','theta','sigma','rho']:
        samples = np.array(results[f'{name}_samples'])
        m, s, lo, hi = summarize(samples)
        print(f"{name:6s}: mean={m:.4f},  sd={s:.4f},  95% CI=[{lo:.4f}, {hi:.4f}]")
        print("Skew:", st.skew(samples), "Kurtosis:", st.kurtosis(samples))



    candidates = ['norm', 'lognorm', 'gamma', 'invgamma']
    for param in ['mu','kappa','theta','sigma','rho', 'mu_j','lambda','sig_j']:
        samples = np.array(results[f'{param}_samples'])
        trimmed, fit_results = fit_without_extremes(samples, lower_pct=2.5, upper_pct=97.5, distributions=candidates)
        best_name, best_info = fit_results[0]
        best_params = best_info['params']
        # Print mapping info
        cp_dist = scipy_to_chaospy(best_name, best_params)
        
        print(f"\nParameter: {param}")
        print(f" Best fit: {best_name}")
        print(f" SciPy params: {best_info}")
        print(f" Chaospy dist: {cp_dist}")


        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(samples, bins=30, density=True, alpha=0.6, label='Samples')
        x = np.linspace(samples.min(), samples.max(), 200)
        pdf_vals = st.__dict__[best_name].pdf(x, *best_params)
        ax.plot(x, pdf_vals, 'r-', lw=2, label=f'{best_name} PDF')
        ax.set_title(f"{param}: {best_name} fit")
        ax.set_xlabel(param)
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        plt.show()

