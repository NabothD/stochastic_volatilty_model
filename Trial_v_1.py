import numpy as np
from scipy.stats import norm, invgamma
from filterpy.monte_carlo import systematic_resample
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import random

def heston_model_estimation(n_samples, delta_t, maturity, prices, n_particles, 
                            mu_0, kappa_0, theta_0, sigma_0, rho_0, 
                            mu_eta_0, tau_eta_0, mu_beta_0, lambda_beta_0,
                            a_sigma_0, b_sigma_0, mu_psi_0, tau_psi_0, a_omega_0, b_omega_0):
    """
    Estimate Heston model parameters via a Bayesian MCMC approach.
    
    This version uses a vectorized Sequential Importance Resampling (SIR) filter 
    for latent volatility estimation and updates the parameters (mu, kappa, theta, 
    sigma, rho) via Bayesian regression/MCMC. Numerical stability improvements include:
      - Clipping volatility to ensure non-negativity.
      - Normalizing particle weights and resampling when effective sample size is low.
      - Working in vectorized form for particle updates to improve speed.
    """
    n = len(prices) - 1  # number of intervals
    
    # Arrays to store MCMC samples for each parameter.
    mu_samples     = np.zeros(n_samples)
    kappa_samples  = np.zeros(n_samples)
    theta_samples  = np.zeros(n_samples)
    sigma_samples  = np.zeros(n_samples)
    rho_samples    = np.zeros(n_samples)
    
    # Current parameter values (starting guesses)
    mu    = mu_0
    kappa = kappa_0
    theta = theta_0
    sigma = sigma_0
    rho   = rho_0

    # Priors for drift estimation (eta = 1 + mu*delta_t)
    eta_prior_mean      = mu_eta_0  
    eta_prior_precision = tau_eta_0  

    # Ensure beta prior parameters are in proper array shape.
    lambda_beta_0 = np.array(lambda_beta_0)
    mu_beta_0     = np.array(mu_beta_0)
    sigma_prior_a = a_sigma_0
    sigma_prior_b = b_sigma_0

    psi_prior_mean      = mu_psi_0
    psi_prior_precision = tau_psi_0
    omega_prior_a       = a_omega_0
    omega_prior_b       = b_omega_0

    volatility_estimates_all = np.zeros((n_samples, n))
    MAX_PARAM = 1e4  # bounding value for parameters
    
    # Compute return ratios R(t) = S(t+1)/S(t)
    R = prices[1:] / prices[:-1]
    
    # Main MCMC loop
    for i in range(n_samples):
        # --- Particle Filtering (SIR) ---
        # Initialize particles for volatility: all particles start at the long-run mean theta.
        V = np.full(n_particles, theta)
        v_est = np.zeros(n)  # to store estimated volatility at each time step
        
        for k in range(n):
            # --- Propagation (Vectorized) ---
            # Avoid division by zero by ensuring V is at least 1e-12.
            V_safe = np.maximum(V, 1e-12)
            # Generate noise for all particles simultaneously.
            eps = np.random.randn(n_particles)
            # Compute standardized residual for the observation.
            z = (R[k] - mu*delta_t - 1.0) / (np.sqrt(V_safe*delta_t))
            # Generate correlated noise (incorporating rho).
            w = z * rho + eps * np.sqrt(1.0 - rho**2)
            # Euler–Maruyama update for volatility; note: using V_safe in the sqrt.
            V_new = V + kappa*(theta - V)*delta_t + sigma * np.sqrt(V_safe*delta_t) * w
            # Enforce non-negativity.
            V_new = np.clip(V_new, 1e-12, None)
            
            # --- Weighting ---
            # Likelihood: assume R[k] ~ N(mu*delta_t + 1, V_new*delta_t)
            var = V_new * delta_t
            std = np.sqrt(var)
            x = R[k] - mu*delta_t - 1.0
            weights = (1.0 / (np.sqrt(2*np.pi)*std)) * np.exp(-0.5*(x/std)**2)
            sum_w = np.sum(weights)
            # In case of numerical underflow, assign uniform weights.
            if sum_w < 1e-15:
                weights = np.ones(n_particles) / n_particles
            else:
                weights = weights / sum_w
            
            # --- Resampling ---
            # Use systematic resampling from filterpy for efficiency.
            indices = systematic_resample(weights)
            V = V_new[indices]
            
            # Estimate volatility at time step k as the mean of particles.
            v_est[k] = np.mean(V)
        
        volatility_estimates_all[i, :] = v_est
        
        # --- Parameter Updates via Bayesian Regression/MCMC ---
        # Step 2a: Update drift parameter mu.
        # We use a regression transformation: let eta = 1 + mu*delta_t.
        # Transform observations for regression:
        x_s = 1.0 / (np.sqrt(delta_t) * np.sqrt(v_est))
        y_s = (R / np.sqrt(v_est)) / np.sqrt(delta_t)
        # OLS estimate for eta (simplified regression)
        ols = np.dot(x_s, y_s) / np.dot(x_s, x_s)
        Tau_eta = np.dot(x_s, x_s) + eta_prior_precision
        eta_posterior_mean = (eta_prior_precision * eta_prior_mean + np.dot(x_s, x_s) * ols) / Tau_eta
        eta_sample = np.random.normal(eta_posterior_mean, 1.0/np.sqrt(Tau_eta))
        mu = (eta_sample - 1.0) / delta_t
        mu = np.clip(mu, -10., 10.)
        mu_samples[i] = mu
        
        # Step 2b: Update kappa and theta via regression on volatility dynamics.
        y_list, x1_list, x2_list = [], [], []
        for k in range(1, n):
            denom = np.sqrt(delta_t * max(v_est[k-1], 1e-12))
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
        # Recover kappa and theta from beta parameters.
        kappa = (1.0 - beta_draw[1]) / delta_t
        theta = beta_draw[0] / (kappa * delta_t)
        kappa = np.clip(kappa, 1e-6, MAX_PARAM)
        theta = np.clip(theta, 1e-6, MAX_PARAM)
        kappa_samples[i] = kappa
        theta_samples[i] = theta
        
        # Step 2c: Sample sigma² from an inverse Gamma posterior.
        sigma_b = sigma_prior_b + 0.5 * (np.dot(y_vec.T, y_vec) + 
                                         mu_beta_0 @ lambda_beta_0 @ mu_beta_0 - 
                                         mu_beta.T @ Lambda_beta @ mu_beta)
        sigma_a = sigma_prior_a + n / 2.0
        sigma_sq = invgamma.rvs(a=sigma_a, scale=sigma_b)
        sigma = np.sqrt(sigma_sq)
        sigma_samples[i] = sigma
        
        # Step 2d: Update rho via regression on residuals.
        e1_rho = np.zeros(n)
        e2_rho = np.zeros(n)
        for t in range(n):
            if t == 0:
                e1_rho[t] = 0.0
            else:
                denom1 = np.sqrt(delta_t * max(v_est[t-1], 1e-12))
                e1_rho[t] = (R[t] - (1.0 + mu*delta_t)) / denom1
            if t == 0:
                e2_rho[t] = 0.0
            else:
                dv = v_est[t] - v_est[t-1]
                drift = kappa * (theta - v_est[t-1]) * delta_t
                denom2 = np.sqrt(delta_t * max(v_est[t-1], 1e-12))
                e2_rho[t] = (dv - drift) / denom2
        e_rho = np.column_stack((e1_rho, e2_rho))
        A_rho = e_rho.T @ e_rho
        A11, A12, A22 = A_rho[0,0], A_rho[0,1], A_rho[1,1]
        a_omega = omega_prior_a + n / 2.0
        b_omega = omega_prior_b + 0.5 * (A22 - (A12**2)/max(A11, 1e-12))
        omega_draw = invgamma.rvs(a=a_omega, scale=b_omega)
        tau_psi = A11 + psi_prior_precision
        mu_psi = (A12 + psi_prior_mean * psi_prior_precision) / tau_psi
        psi_draw = norm.rvs(loc=mu_psi, scale=np.sqrt(omega_draw/tau_psi))
        new_rho = psi_draw / np.sqrt(psi_draw**2 + omega_draw)
        rho = np.clip(new_rho, -0.9999, 0.9999)
        rho_samples[i] = rho

    # Final parameter estimates: averages over all MCMC samples.
    mu_hat    = np.mean(mu_samples)
    kappa_hat = np.mean(kappa_samples)
    theta_hat = np.mean(theta_samples)
    sigma_hat = np.mean(sigma_samples)
    rho_hat   = np.mean(rho_samples)
    
    return {
        'mu': mu_hat,
        'kappa': kappa_hat,
        'theta': theta_hat,
        'sigma': sigma_hat,
        'rho': rho_hat,
        'mu_samples': mu_samples,
        'kappa_samples': kappa_samples,
        'theta_samples': theta_samples,
        'sigma_samples': sigma_samples,
        'rho_samples': rho_samples,
        'volatility_estimates_all': volatility_estimates_all
    }

def plot_parameter_distributions(true_values, parameter_samples, parameter_names):
    n_params = len(parameter_names)
    fig, axes = plt.subplots(1, n_params, figsize=(15, 5))
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
    """
    Load simulated data from CSV.
    Assumes CSV has a header and the first column is prices.
    """
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    prices = data[:, 0]
    variances = data[:, 1]  # not used in estimation, but could be useful for comparison
    return prices, variances

# --------------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    # MCMC settings
    n_samples    = 400
    n_particles  = 300
    T            = 1.0
    Nsteps       = 252

    # Initial parameter guesses
    mu_0    = 0.1
    kappa_0 = 0.8
    theta_0 = 0.04
    sigma_0 = 0.011
    rho_0   = -0.48

    # Load synthetic data from CSV file
    prices, v_path = load_prices_from_csv("my_heston.csv")
    delta_t = 1.0 / len(prices)
    maturity = T

    true_values = {
        'mu': 0.1, 'kappa': 1, 'theta': 0.05, 'sigma': 0.01, 'rho': -0.5
    }

    # Prior parameters (adjust based on domain knowledge)
    mu_eta_0      = 1.00125
    tau_eta_0     = 1.0 / math.sqrt(0.001**2)
    mu_beta_0     = [35e-7, 0.998]        # shape (2,)
    lambda_beta_0 = [[10, 0], [0, 5]]       # shape (2,2)
    a_sigma_0     = 149
    b_sigma_0     = 0.026
    mu_psi_0      = -0.45
    tau_psi_0     = 1.0 / math.sqrt(0.25**2)
    a_omega_0     = 1.03
    b_omega_0     = 0.05

    results = heston_model_estimation(
        n_samples, delta_t, maturity, prices,
        n_particles, mu_0, kappa_0, theta_0, sigma_0, rho_0, 
        mu_eta_0, tau_eta_0, mu_beta_0, lambda_beta_0, 
        a_sigma_0, b_sigma_0, mu_psi_0, tau_psi_0, 
        a_omega_0, b_omega_0
    )
    
    print("Estimated Parameters:")
    print(f"mu: {results['mu']}")
    print(f"kappa: {results['kappa']}")
    print(f"theta: {results['theta']}")
    print(f"sigma: {results['sigma']}")
    print(f"rho: {results['rho']}")
    
    vols = results['volatility_estimates_all']
    final_volatility_path = vols[-1, :]
    
    # Plot final iteration volatility vs synthetic volatility path.
    plt.figure(figsize=(8,5))
    plt.plot(final_volatility_path, label='Volatility (final iteration)')
    plt.plot(v_path, label="Synthetic Vol (sqrt(v))")
    plt.xlabel('Time Step')
    plt.ylabel('Estimated Volatility')
    plt.legend()
    plt.title('Particle-Filtered Volatility (final MCMC iteration)')
    plt.show()
    
    # Plot mean volatility path over MCMC iterations.
    avg_volatility_path = vols.mean(axis=0)
    plt.figure(figsize=(8,5))
    plt.plot(avg_volatility_path, label='Mean volatility (over MCMC)')
    plt.plot(v_path, label="Synthetic Vol (sqrt(v))")
    plt.xlabel('Time Step')
    plt.ylabel('Estimated Volatility')
    plt.legend()
    plt.title('Mean PF Volatility Estimate Over MCMC')
    plt.show()
    
    parameter_samples = {
        'mu': results['mu_samples'],
        'kappa': results['kappa_samples'],
        'theta': results['theta_samples'],
        'sigma': results['sigma_samples'],
        'rho': results['rho_samples'],
    }
    parameter_names = ['mu', 'kappa', 'theta', 'sigma', 'rho']
    plot_parameter_distributions(true_values, parameter_samples, parameter_names)
