import numpy as np
from scipy.stats import norm, invgamma

import matplotlib.pyplot as plt
import seaborn as sns

# Particle Filtering Heston Model Estimation
# -----------------------------------------------------------------------------------------------------------------------------------------
def estimate_heston_model(S, dt, n_samples, N_particles, priors):
    """
    Estimate the Heston model parameters and volatility process using particle filtering.

    Parameters:
        S: array-like, observed asset prices.
        dt: float, time step.
        n_samples: int, number of outer sampling loops.
        N_particles: int, number of particles for particle filtering.
        priors: dict, prior distribution parameters.

    Returns:
        dict: Estimated parameters and volatility process.
    """
    n = len(S) - 1
    R = np.log(S[1:] / S[:-1])  # Equation (12): Returns from prices

    # Initialize parameter estimates
    mu, kappa, theta, sigma, rho = priors['mu'], priors['kappa'], priors['theta'], priors['sigma'], priors['rho']
    mu_estimates, kappa_estimates, theta_estimates, sigma_estimates, rho_estimates = [], [], [], [], []

    # Particle filter for volatility estimation
    V_particles = np.full(N_particles, theta)  # Initialize particles for volatility (Equation 60)

    for i in range(n_samples):
        # Step 1: Particle Filtering
        for k in range(1, n):
            # Generate new particles
            epsilon_v = np.random.normal(size=N_particles)
            weights = np.zeros(N_particles)

            for j in range(N_particles):
                V_prev = V_particles[j]
                V_proposed = V_prev + kappa * (theta - V_prev) * dt + sigma * np.sqrt(V_prev * dt) * epsilon_v[j]
                
                # Calculate particle weights (Equation 65)
                likelihood = norm.pdf(R[k], loc=mu * dt, scale=np.sqrt(V_proposed * dt))
                weights[j] = likelihood

                # Update particle
                V_particles[j] = V_proposed

            # Normalize weights
            weights /= np.sum(weights)

            # Resample particles based on weights
            indices = np.random.choice(range(N_particles), size=N_particles, p=weights)
            V_particles = V_particles[indices]

        # Estimate volatility at each step
        V_estimated = np.mean(V_particles)

        # Step 2: Parameter Estimation
        # Update mu (Equations 13–23)
        eta_prior = priors['eta']
        tau_prior = priors['tau']

        x_s = np.sqrt(dt / V_estimated) * R
        y_s = 1 / np.sqrt(dt * V_estimated)

        eta_posterior = (tau_prior * eta_prior + np.sum(y_s * x_s)) / (tau_prior + np.sum(y_s**2))
        mu_posterior = eta_posterior / dt

        # Update kappa, theta, and sigma (Equations 25–44)
        a_kappa = priors['a_kappa']
        b_kappa = priors['b_kappa']
        a_theta = priors['a_theta']
        b_theta = priors['b_theta']
        a_sigma = priors['a_sigma']
        b_sigma = priors['b_sigma']

        kappa_posterior_shape = a_kappa + 0.5 * len(V_particles)
        kappa_posterior_rate = b_kappa + 0.5 * np.sum((V_particles - theta)**2 / V_particles)
        kappa_posterior = invgamma.rvs(kappa_posterior_shape, scale=kappa_posterior_rate)

        theta_posterior_shape = a_theta + 0.5 * len(V_particles)
        theta_posterior_rate = b_theta + 0.5 * np.sum((V_particles - kappa)**2)
        theta_posterior = invgamma.rvs(theta_posterior_shape, scale=theta_posterior_rate)

        sigma_posterior_shape = a_sigma + 0.5 * len(V_particles)
        sigma_posterior_rate = b_sigma + 0.5 * np.sum((V_particles)**2)
        sigma_posterior = invgamma.rvs(sigma_posterior_shape, scale=sigma_posterior_rate)

        # Update rho (Equations 45–59)
        rho_prior_mean = priors['rho_prior_mean']
        rho_prior_variance = priors['rho_prior_variance']
        rho_likelihood_mean = np.sum(R * V_particles) / np.sum(V_particles)
        rho_likelihood_variance = 1 / np.sum(V_particles)

        rho_posterior_mean = (rho_prior_variance * rho_likelihood_mean + rho_likelihood_variance * rho_prior_mean) / \
                             (rho_prior_variance + rho_likelihood_variance)
        rho_posterior_variance = 1 / (1 / rho_prior_variance + 1 / rho_likelihood_variance)
        rho_posterior = np.random.normal(rho_posterior_mean, np.sqrt(rho_posterior_variance))

        # Store estimates
        mu_estimates.append(mu_posterior)
        kappa_estimates.append(kappa_posterior)
        theta_estimates.append(theta_posterior)
        sigma_estimates.append(sigma_posterior)
        rho_estimates.append(rho_posterior)

    # Compute final estimates (mean of samples)
    mu_final = np.mean(mu_estimates)
    kappa_final = np.mean(kappa_estimates)
    theta_final = np.mean(theta_estimates)
    sigma_final = np.mean(sigma_estimates)
    rho_final = np.mean(rho_estimates)

    return {
        'mu': mu_final,
        'kappa': kappa_final,
        'theta': theta_final,
        'sigma': sigma_final,
        'rho': rho_final,
        'volatility': V_estimated
    }
# -------------------------------------------------------------------------------------------------------------------



def estimate_heston_model_with_jumps(S, dt, n_samples, N_particles, priors):
    """
    Estimate the Heston model with jumps using particle filtering.

    Parameters:
        S: array-like, observed asset prices.
        dt: float, time step.
        n_samples: int, number of outer sampling loops.
        N_particles: int, number of particles for particle filtering.
        priors: dict, prior distribution parameters for Heston and jump components.

    Returns:
        dict: Estimated parameters and volatility process with jumps.
    """
    n = len(S) - 1
    R = np.zeros(n)
    for k in range(1, n):
        R[k] = S[k] / S[k-1]  # Log returns, Equation (12)
    

    # Initialize parameter estimates
    mu, kappa, theta, sigma, rho = priors['mu'], priors['a_kappa'], priors['a_theta'], priors['a_sigma'], priors['rho_prior_mean']
    lambda_jump, mu_jump, sigma_jump = priors['lambda_jump'], priors['mu_jump'], priors['sigma_jump']

    mu_estimates, kappa_estimates, theta_estimates = [], [], []
    sigma_estimates, rho_estimates, lambda_estimates = [], [], []
    mu_jump_estimates, sigma_jump_estimates = [], []

    # Initialize particles
    V_particles = np.full(N_particles, theta)  # Volatility particles
    J_particles = np.zeros(N_particles)        # Jump particles

    for i in range(n_samples):
        for k in range(1, n):
            epsilon_v = np.random.normal(size=N_particles)
            epsilon_jump = np.random.normal(size=N_particles)
            weights = np.zeros(N_particles)

            for j in range(N_particles):
                V_prev = V_particles[j]
                J_prev = J_particles[j]

                # Generate jump occurrences and sizes, Eq. (75, 76)
                if np.random.rand() < lambda_jump * dt:
                    J_new = mu_jump + sigma_jump * epsilon_jump[j]
                else:
                    J_new = 

                # Volatility update, Eq. (64)
                V_proposed = V_prev + kappa * (theta - V_prev) * dt + sigma * np.sqrt(V_prev * dt) * epsilon_v[j]

                # Particle weight, Eq. (77)
                likelihood = norm.pdf(R[k], loc=mu * dt + J_new, scale=np.sqrt(V_proposed * dt))
                weights[j] = likelihood

                # Update particles
                V_particles[j] = V_proposed
                J_particles[j] = J_new

            # Normalize weights to avoid numerical underflow
            weights = np.exp(weights - np.max(weights))
            weights /= np.sum(weights)

            # Resample particles
            indices = np.random.choice(range(N_particles), size=N_particles, p=weights)
            V_particles = V_particles[indices]
            J_particles = J_particles[indices]

        # Estimate volatility and jump components
        V_estimated = np.mean(V_particles)
        J_estimated = np.mean(J_particles)

        # Bayesian updates for parameters
        x_s = np.sqrt(dt / V_estimated) * (R[k] - J_estimated)
        y_s = 1 / np.sqrt(dt * V_estimated)

        eta_posterior = (priors['tau'] * priors['eta'] + np.sum(y_s * x_s)) / (priors['tau'] + np.sum(y_s**2))
        mu_posterior = eta_posterior / dt

        # Update kappa, theta, and sigma (inverse gamma posterior updates)
        kappa_posterior = invgamma.rvs(
            priors['a_kappa'] + 0.5 * N_particles,
            scale=priors['b_kappa'] + 0.5 * np.sum((V_particles - theta)**2 / V_particles)
        )

        theta_posterior = invgamma.rvs(
            priors['a_theta'] + 0.5 * N_particles,
            scale=priors['b_theta'] + 0.5 * np.sum((V_particles - kappa)**2)
        )

        sigma_posterior = invgamma.rvs(
            priors['a_sigma'] + 0.5 * N_particles,
            scale=priors['b_sigma'] + 0.5 * np.sum((V_particles)**2)
        )

        rho_posterior_mean = (
            priors['rho_prior_variance'] * np.sum(R[k] * V_particles) / np.sum(V_particles) +
            priors['rho_prior_mean'] / priors['rho_prior_variance']
        ) / (1 / priors['rho_prior_variance'] + 1 / np.sum(V_particles))

        rho_posterior_variance = 1 / (1 / priors['rho_prior_variance'] + 1 / np.sum(V_particles))
        rho_posterior = np.random.normal(rho_posterior_mean, np.sqrt(rho_posterior_variance))

        # Jump parameters
        lambda_posterior = invgamma.rvs(
            priors['lambda_jump'] + np.sum(J_particles > 0),
            scale=priors['lambda_jump'] + n * dt
        )

        mu_jump_posterior = invgamma.rvs(
            priors['mu_jump'] + 0.5 * np.sum(J_particles > 0),
            scale=priors['mu_jump'] + 0.5 * np.sum((J_particles - mu_jump)**2)
        )

        sigma_jump_posterior = invgamma.rvs(
            priors['sigma_jump'] + 0.5 * np.sum(J_particles > 0),
            scale=priors['sigma_jump'] + 0.5 * np.sum((J_particles - mu_jump)**2)
        )


         # Store estimates
        mu_estimates.append(mu_posterior)
        kappa_estimates.append(kappa_posterior)
        theta_estimates.append(theta_posterior)
        sigma_estimates.append(sigma_posterior)
        rho_estimates.append(rho_posterior)
        lambda_estimates.append(lambda_posterior)
        mu_jump_estimates.append(mu_jump_posterior)
        sigma_jump_estimates.append(sigma_jump_posterior)

    # Compute final estimates (mean of samples)
    mu_final = np.mean(mu_estimates)
    kappa_final = np.mean(kappa_estimates)
    theta_final = np.mean(theta_estimates)
    sigma_final = np.mean(sigma_estimates)
    rho_final = np.mean(rho_estimates)
    lambda_final = np.mean(lambda_estimates)
    mu_jump_final = np.mean(mu_jump_estimates)
    sigma_jump_final = np.mean(sigma_jump_estimates)

    return {
        'mu': mu_final,
        'kappa': kappa_final,
        'theta': theta_final,
        'sigma': sigma_final,
        'rho': rho_final,
        'lambda_jump': lambda_final,
        'mu_jump': mu_jump_final,
        'sigma_jump': sigma_jump_final,
        'mu_samples': mu_estimates,
        'kappa_samples': kappa_estimates,
        'theta_samples': theta_estimates,
        'sigma_samples': sigma_estimates,
        'rho_samples': rho_estimates,
        'lambda_samples': lambda_estimates,
        'mu_jump_samples': mu_jump_estimates,
        'sigma_jump_samples': sigma_jump_estimates,
        'volatility': V_estimated
    }


def plot_parameter_distributions(true_values, parameter_samples, parameter_names):
    """
    Plot empirical PDFs of parameter estimates with true values.

    Parameters:
        true_values: dict
            Dictionary of true parameter values, e.g., {'mu': -0.4, 'kappa': 1.2, ...}.
        parameter_samples: dict
            Dictionary of parameter samples, e.g., {'mu': [...], 'kappa': [...], ...}.
        parameter_names: list
            List of parameter names, e.g., ['mu', 'kappa', 'theta', 'sigma', 'rho'].
    """
    n_params = len(parameter_names)
    fig, axes = plt.subplots(1, n_params, figsize=(15, 5), sharey=True)

    for i, param in enumerate(parameter_names):
        # Plot KDE for parameter samples
        sns.kdeplot(parameter_samples[param], ax=axes[i], label='Estimate distribution', color='blue')
        
        # Add true value as a vertical line
        if param in true_values:
            axes[i].axvline(true_values[param], color='red', linestyle='--', label='True value')
        
        # Titles and labels
        axes[i].set_title(f"Parameter: {param}")
        axes[i].set_xlabel("Estimate")
        axes[i].set_ylabel("Empirical PDF")
        axes[i].legend()

    # plt.tight_layout()
    plt.show()



    

# Example usage for Heston with jumps
priors = {
    'mu': 0.988, 'a_kappa': 1.03, 'b_kappa': 0.05, 'a_theta': 0.05, 'b_theta': 0.08, 'a_sigma': 149, 'b_sigma': 0.025, 'rho_prior_mean': -0.45, 'rho_prior_variance':0.3,
    'lambda_jump': 0.15, 'mu_jump': 1.00125, 'sigma_jump': 0.001,
    'eta': 1.0, 'tau': 1.0, 'beta': [0.5, 2.0]  # Broader prior
}

# priors = {
#     'mu_jump': 1.00125,  # μ₀ᵛ
#     'sigma_jump': 0.001,  # σ₀ᵛ
#     'Lambda_0': [[10, 0], [0, 5]],  # Λ₀
#     'mu_0': [35e-6, 0.988],  # μ₀ (two-dimensional)
#     'a_sigma': 149,  # a₀ᵛ
#     'b_sigma': 0.025,  # b₀ᵛ
#     'mu_rho': -0.45,  # μ₀ᵠ
#     'sigma_rho': 0.3,  # σ₀ᵠ
#     'a_kappa': 1.03,  # a₀ᵡ
#     'b_kappa': 0.05,  # b₀ᵡ
#     'lambda_jump': 0.15,  # λᵗʰ
#     'mu_rho_jump': -0.96,  # μ₀ʲ
#     'sigma_rho_jump': 0.3  # σ₀ʲ
# }

S = np.array([100, 102, 101, 104, 103])  # Example prices
dt = 1 / (365*3)  # Time step (daily)
n_samples = 200  # Number of outer iterations
N_particles = 400  # Number of particles

results = estimate_heston_model_with_jumps(S, dt, n_samples, N_particles, priors)
# print("Estimated Parameters with Jumps:", results)

# Extract parameter samples from the results or directly use the lists within the function
mu_estimates = results.get('mu_samples', [])
kappa_estimates = results.get('kappa_samples', [])
theta_estimates = results.get('theta_samples', [])
sigma_estimates = results.get('sigma_samples', [])
rho_estimates = results.get('rho_samples', [])
lambda_estimates = results.get('lambda_samples', [])
mu_jump_estimates = results.get('mu_jump_samples', [])
sigma_jump_estimates = results.get('sigma_jump_samples', [])






# Example usage
# Define true parameter values (these should match the ones used for simulation, if known)
true_values = {
    'mu': -0.4, 'kappa': 1.2, 'theta': 0.05, 'sigma': 0.2, 'rho': -0.6,
    'lambda_jump': 0.1, 'mu_jump': 0.2, 'sigma_jump': 0.15
}





# Gather parameter samples from the estimation process
parameter_samples = {
    'mu': mu_estimates,
    'kappa': kappa_estimates,
    'theta': theta_estimates,
    'sigma': sigma_estimates,
    'rho': rho_estimates,
    'lambda_jump': lambda_estimates,
    'mu_jump': mu_jump_estimates,
    'sigma_jump': sigma_jump_estimates
}

# Specify parameter names to plot (adjust based on Heston or Heston-with-jumps)
parameter_names = ['mu', 'kappa', 'theta', 'sigma', 'rho', 'lambda_jump', 'mu_jump', 'sigma_jump']

# Plot the distributions

# Debugging: Check variability of the samples
for param_name, samples in parameter_samples.items():
    print(f"{param_name}: min={np.min(samples)}, max={np.max(samples)}, mean={np.mean(samples)}, std={np.std(samples)}")

plot_parameter_distributions(true_values, parameter_samples, parameter_names)