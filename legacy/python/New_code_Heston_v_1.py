import numpy as np
from scipy.stats import norm, invgamma
from filterpy.monte_carlo import systematic_resample
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas as pd



class ParticleFilter:
    def __init__(self, n_particles, initial_value, process_model, observation_model, process_noise, observation_noise):
        self.n_particles = n_particles
        self.particles = np.full(n_particles, initial_value)  # shape (n_particles,)
        self.weights = np.ones(n_particles) / n_particles
        self.process_model = process_model
        self.observation_model = observation_model
        self.process_noise = process_noise
        self.observation_noise = observation_noise

    def predict(self):
        noise = np.random.normal(0, self.process_noise, self.n_particles)
        self.particles = self.process_model(self.particles) + noise

    def update(self, observation):
        # Evaluate likelihood of each particle given the new observation
        likelihoods = norm.pdf(
            observation, 
            loc=self.observation_model(self.particles),
            scale=self.observation_noise
        )
        self.weights *= likelihoods
        self.weights += 1e-300  # Avoid divide-by-zero
        self.weights /= np.sum(self.weights)

    def resample(self):
        # Normalize weights (again, just to be safe)
        self.weights = np.clip(self.weights, 1e-300, None)
        self.weights /= np.sum(self.weights)

        # Resample particles based on weights
        indices = systematic_resample(self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.n_particles)

    def estimate(self):
        """
        Return (mean, variance) of the current particle set.
        """
        return np.mean(self.particles), np.var(self.particles)


def heston_model_estimation(
    n_samples, delta_t, maturity, prices, n_particles, 
    mu_0, kappa_0, theta_0, sigma_0, rho_0, 
    mu_eta_0, tau_eta_0, mu_beta_0, lambda_beta_0,
    a_sigma_0, b_sigma_0, mu_psi_0, tau_psi_0, a_omega_0, b_omega_0
):
    """
    Perform parameter estimation for the Heston model using Algorithm 1.
    """
    n = len(prices) - 1  # number of intervals
    # Ratios R(t) = S(t+1)/S(t)
    R = prices[1:] / prices[:-1]

    # Initialize arrays to store parameter estimates
    mu_samples = np.zeros(n_samples)
    kappa_samples = np.zeros(n_samples)
    theta_samples = np.zeros(n_samples)
    sigma_samples = np.zeros(n_samples)
    rho_samples = np.zeros(n_samples)

    # Current parameter guesses
    mu = mu_0
    kappa = kappa_0
    theta = theta_0
    sigma = sigma_0
    rho = rho_0

    # Define your prior distribution parameters
    eta_prior_mean = mu_eta_0                 # scalar
    eta_prior_precision = tau_eta_0           # scalar

    # <-- Make beta_prior_mean a 1D array of shape (2,), not (2,1)
    beta_prior_mean = np.array(mu_beta_0)     # shape = (2,)

    beta_prior_precision = np.array(lambda_beta_0)  # shape = (2,2)
    sigma_prior_a = a_sigma_0
    sigma_prior_b = b_sigma_0

    psi_prior_mean = mu_psi_0
    psi_prior_precision = tau_psi_0
    omega_prior_a = a_omega_0
    omega_prior_b = b_omega_0
    volatility_estimates_all = np.zeros((n_samples, n))
    # Initialize Particle Filter for volatility
    process_model = lambda v: v + kappa * (theta - v) * delta_t  # mean-reverting
    observation_model = lambda v: np.sqrt(np.clip(v, 1e-8, None))

    pf = ParticleFilter(n_particles, theta_0, process_model, observation_model, sigma_0, sigma_0)

    MAX_PARAM = 1e2  # or choose a smaller bounding if you prefer


    for i in range(n_samples):
        #----------------------------------------------------------
        # Step 1: Particle Filtering to estimate volatility path
        #----------------------------------------------------------
        volatility_estimates = np.zeros(n)
        for t in range(n):
            pf.predict()
            # clamp PF's internal volatility so it doesn't explode:
            pf.particles = np.clip(pf.particles, 1e-10, 1e6)
            
            pf.update(R[t])
            pf.resample()
            est_mean, _ = pf.estimate()
            volatility_estimates[t] = max(est_mean, 1e-10)  # clamp to avoid zero or negative
            volatility_estimates_all[i, :] = volatility_estimates
        # Make sure volatility_estimates is shape (n,)
        # (That is already the case from np.zeros(n).)

        #----------------------------------------------------------
        # Step 2a: Estimate mu (drift) using eqns (13)–(23)
        #     We'll treat "eta = 1 + mu * dt", etc.
        #----------------------------------------------------------
        # Summation term: sum(R / volatility_estimates) 
        # But watch out for shape mismatch: R is (n,), volatility_estimates is (n,).
        # So R / volatility_estimates is (n,), then sum is scalar.
        sum_term = np.sum(R / volatility_estimates)

        # Posterior mean for eta
        #   = ( tau_eta_0 * mu_eta_0 + sum(R/volatility) ) / ( tau_eta_0 + n )

        eta_posterior_mean = (
            eta_prior_precision * eta_prior_mean + sum_term
        ) / (eta_prior_precision + n)
        eta_posterior_var = 1.0/(eta_prior_precision + n)
        eta_sample = norm.rvs(loc=eta_posterior_mean, scale=np.sqrt(eta_posterior_var))
        mu = (eta_sample - 1.0)/delta_t

        # Convert from eta to mu
        # if we assume eta = 1 + mu*dt  => mu = (eta - 1)/dt
        mu = np.clip(mu, -10., 10.)
        mu_samples[i] = mu

        #----------------------------------------------------------
        # Step 2b: Estimate (kappa, theta) from eqns (24)–(44)
        #----------------------------------------------------------
        # Build design matrix X: shape (n,2)
        #   X = [ 1, volatility_estimate ] 
        #   R ~ X beta
        # Make sure volatility_estimates is shape (n,)
        X = np.column_stack((np.ones(n), volatility_estimates))
        # shape(X) = (n,2)
        
        # X^T X => shape(2,2), X^T R => shape(2,)
        XtX = X.T @ X
        Xty = X.T @ R

        # Posterior precision
        posterior_precision = beta_prior_precision + XtX
        # Posterior mean
        #   temp = beta_prior_precision @ beta_prior_mean + X^T R
        temp = (beta_prior_precision @ beta_prior_mean) + Xty
        posterior_mean = np.linalg.inv(posterior_precision) @ temp  # shape (2,)

        # Sample each component of beta from a *univariate* Normal with same stdev
        # (This is an approximation — real code might need a 2D normal.)
        # beta_draw = norm.rvs(loc=posterior_mean, scale=np.sqrt(1.0/sigma))

        posterior_cov = sigma * np.linalg.inv(posterior_precision)
        beta_draw = np.random.multivariate_normal(mean=posterior_mean, cov=posterior_cov)

        # beta_samples shape => (2,)

        # Suppose the relationship is:
        #    kappa = (1 - beta[1]) / dt,  theta = beta[0] / (kappa*dt)
        # (But confirm the exact formula with your references.)
        kappa = (1.0 - beta_draw[1]) / delta_t
        theta = beta_draw[0]/(kappa*delta_t)

        kappa = np.clip(kappa, 1e-6, MAX_PARAM)
        theta = np.clip(theta, 1e-6, MAX_PARAM)
        kappa_samples[i] = kappa
        theta_samples[i] = theta

        #----------------------------------------------------------
        # Step 2c: Sample sigma^2 from an Inverse Gamma
        #----------------------------------------------------------
        residuals = R - X @ beta_draw  # shape (n,)
        sum_sq = np.sum(residuals**2)
        # Posterior scale
        sigma_b = sigma_prior_b + 0.5 * sum_sq
        # Posterior shape
        sigma_a = sigma_prior_a + n / 2.0
        # Draw from IG:
        if (not np.isfinite(sigma_b)) or (sigma_b <= 0):
            sigma_b = 1e3
        sigma_sq = invgamma.rvs(a=sigma_a, scale=sigma_b)
        sigma = np.sqrt(sigma_sq)
        sigma_samples[i] = sigma

        #----------------------------------------------------------
        # Step 2d: Estimate rho from eqns (45)–(59)
        # Here it's simplified. We do a single-sample approach:
        #----------------------------------------------------------
        # For demonstration, just do a small random walk around old rho
        # or do a simple Normal prior for psi. 


        e1_rho = np.zeros(n)
        e2_rho = np.zeros(n)

        for t in range(n):
            # eq. (45):
            #   e1^rho(kΔt) = [R(kΔt) - (1 + μ·Δt)] / sqrt(Δt·v((k-1)Δt))
            #   Here v((k-1)Δt) is volatility_estimates[t-1], but for t=0 we skip.
            if t == 0:
                # no prior v((−1)Δt)
                e1_rho[t] = 0.0
            else:
                denom1 = np.sqrt(delta_t * np.clip(volatility_estimates[t-1],1e-12,None))
                numerator1 = R[t] - (1.0 + mu*delta_t)
                e1_rho[t] = numerator1 / denom1

            # eq. (46):
            #   e2^rho(kΔt) = [ v(kΔt) - v((k-1)Δt)  - κ(θ - v((k-1)Δt))Δt ]
            #                 / sqrt(Δt·v((k-1)Δt))
            if t == 0:
                e2_rho[t] = 0.0
            else:
                dv = volatility_estimates[t] - volatility_estimates[t-1]
                drift = kappa*(theta - volatility_estimates[t-1])*delta_t
                denom2 = np.sqrt(delta_t * np.clip(volatility_estimates[t-1],1e-12,None))
                e2_rho[t] = (dv - drift) / denom2

        # 2) Form the matrix e^rho = [e1_rho, e2_rho], shape (n,2)
        e_rho = np.column_stack((e1_rho, e2_rho))

        # 3) A^rho = (e^rho)' e^rho => shape (2,2)
        A_rho = e_rho.T @ e_rho
        A11 = A_rho[0,0]
        A12 = A_rho[0,1]
        A22 = A_rho[1,1]

        # 4) Draw omega from Inverse Gamma (Eqs. 56-57)
        #    a^omega = a_omega_0 + n/2
        #    b^omega = b_omega_0 + 0.5*( A22 - A12^2 / A11 )
        a_omega = omega_prior_a + n/2.
        b_omega = omega_prior_b + 0.5*( A22 - (A12**2)/A11 )
        omega_draw = invgamma.rvs(a=a_omega, scale=b_omega)

        # 5) Draw psi from Normal (Eqs. 54-55)
        #    tau^psi = A11 + tau_psi_0
        #    mu^psi  = [ A12 + mu_psi_0 * tau_psi_0 ] / tau^psi
        tau_psi = A11 + psi_prior_precision
        mu_psi  = (A12 + psi_prior_mean * psi_prior_precision)/tau_psi
        psi_draw = norm.rvs(loc=mu_psi, scale=np.sqrt(omega_draw/tau_psi))

        # 6) Finally, rho = psi / sqrt(psi^2 + omega)
        new_rho = psi_draw / np.sqrt(psi_draw**2 + omega_draw)
        # clip if desired
        rho = np.clip(new_rho, -0.9999, 0.9999)

        rho_samples[i] = rho





        # psi_posterior_mean = (
        #     psi_prior_mean * psi_prior_precision
        #     + np.dot(R, volatility_estimates)
        # ) / (psi_prior_precision + n)
        # psi_posterior_var = 1.0 / (psi_prior_precision + n)
        # psi_sample = norm.rvs(loc=psi_posterior_mean, scale=np.sqrt(psi_posterior_var))

        # # Then set rho = psi / sqrt(psi^2 + b_omega_0) as a placeholder
        # rho = psi_sample / np.sqrt(psi_sample**2 + omega_prior_b)
        # rho = np.clip(rho, -0.9999, 0.9999)
        # rho_samples[i] = rho

        # Debug prints if needed
        #print(f"beta_samples.shape={beta_samples.shape}, kappa={kappa}, theta={theta}, rho={rho}")

    #--------------------------------------------
    # Final estimates: average the MCMC draws
    #--------------------------------------------
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
        'mu_samples': mu_samples,
        'kappa_samples': kappa_samples,
        'theta_samples': theta_samples,
        'sigma_samples': sigma_samples,
        'rho_samples': rho_samples,
        'volatility_estimates_all': volatility_estimates_all
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
    fig, axes = plt.subplots(1, n_params, figsize=(15, 5), sharey=False)

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

# ----------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    # df_spx = yf.download("^GSPC", period="2y", interval="1d")
    # df_vix = yf.download("^VIX",  period="2y", interval="1d")


    # # 3) Combine them into one DataFrame to line up by date
    # df = pd.DataFrame({
    #     "SPX_Close": [df_spx["Close"]],
    #     "VIX_Close": [df_vix["Close"]]
    # })


    # print(df.head())

    # # 4) Plot them to get a quick visual comparison
    # fig, ax1 = plt.subplots(figsize=(10,5))

    # # Plot S&P 500 on the left axis
    # color1 = 'tab:blue'
    # ax1.set_xlabel("Date")
    # ax1.set_ylabel("S&P 500 (Close)", color=color1)
    # ax1.plot(df.index, df["SPX_Close"], color=color1, label='S&P 500')
    # ax1.tick_params(axis='y', labelcolor=color1)

    # # Plot VIX on the right axis
    # ax2 = ax1.twinx()
    # color2 = 'tab:red'
    # ax2.set_ylabel("VIX (Close)", color=color2)
    # ax2.plot(df.index, df["VIX_Close"], color=color2, label='VIX')
    # ax2.tick_params(axis='y', labelcolor=color2)

    # plt.title("S&P 500 vs. VIX (Daily Close)")
    # fig.tight_layout()
    # plt.show()





    # prices = np.array(df_spx)
    

    # Set your parameters
    n_samples = 200
    
    maturity = 1
    delta_t = maturity/n_samples
    n_particles = 300

    # Initial parameter values
    mu_0 = 0.1
    kappa_0 = 1
    theta_0 = 0.05
    sigma_0 = 0.01
    rho_0 = -0.5

    # Priors
    mu_eta_0 = 1.00125
    tau_eta_0 = 0.001
    mu_beta_0 = [35e-6, 0.988]        # <--- shape (2,)
    lambda_beta_0 = [[10, 0], [0, 5]] # shape (2,2)
    a_sigma_0 = 10
    b_sigma_0 = 0.025
    mu_psi_0 = -0.45
    tau_psi_0 = 0.3
    a_omega_0 = 1.03
    b_omega_0 = 0.05

    # Generate synthetic prices
    n_steps = int(maturity / delta_t)
    # prices = np.array([100, 102, 101, 104, 103]) 
    # prices = np.zeros(n_steps)
    # prices[0] = 100
    # np.random.seed(42)
    # for t in range(1, n_steps):
    #     dt = delta_t
    #     dW = np.random.normal(0, np.sqrt(dt))
    #     prices[t] = prices[t - 1] * np.exp((mu_0 - 0.5 * sigma_0**2)*dt + sigma_0*dW)


    
    
    np.random.seed(42)
    prices = np.zeros(n_steps)
    prices[0] = 100

    for t in range(1, n_steps):
        # Draw a random increment ~ Normal(0, sqrt(dt))
        dW = np.random.normal(0, np.sqrt(delta_t))
        # Update price using the GBM discrete approximation
        prices[t] = prices[t - 1] * np.exp(
            (mu_0 - 0.5 * sigma_0**2) * delta_t + sigma_0 * dW
        )

    # Run the estimation
    results = heston_model_estimation(
        n_samples, delta_t, maturity, prices,
        n_particles, mu_0, kappa_0, theta_0, sigma_0, rho_0, 
        mu_eta_0, tau_eta_0, mu_beta_0, lambda_beta_0, a_sigma_0, b_sigma_0, 
        mu_psi_0, tau_psi_0, a_omega_0, b_omega_0
    )
    # vols = results['volatility_estimates_all']  # shape (n_samples, n)
    # avg_volatility_path = vols.mean(axis=0)  # shape (n,)
    # plt.figure(figsize=(8,5))
    # plt.plot(avg_volatility_path, label='Volatility (mean over all iterations)')
    # plt.xlabel('Time Step')
    # plt.ylabel('Estimated Vol')
    # plt.legend()
    # plt.title('Mean PF Volatility Estimate Over MCMC')
    # plt.show()

    
    vols = results['volatility_estimates_all']  # shape (n_samples, n)
    final_volatility_path = vols[-1, :]         # last iteration
    plt.figure(figsize=(8,5))
    plt.plot(final_volatility_path, label='Volatility (final iteration)')
    plt.xlabel('Time Step')
    plt.ylabel('Estimated Vol')
    plt.legend()
    plt.title('Particle-Filtered Volatility (final MCMC iteration)')
    plt.show()



    mu_estimates = results.get('mu_samples', []),
    kappa_estimates = results.get('kappa_samples', []),
    theta_estimates = results.get('theta_samples', []),
    sigma_estimates = results.get('sigma_samples', []),
    rho_estimates = results.get('rho_samples', [])
    


    parameter_samples = {
        'mu': mu_estimates,
        'kappa': kappa_estimates,
        'theta': theta_estimates,
        'sigma': sigma_estimates,
        'rho': rho_estimates,
    }

    true_values = {
    'mu': -0.4, 'kappa': 1.2, 'theta': 0.05, 'sigma': 0.2, 'rho': -0.6
    }
    # Specify parameter names to plot (adjust based on Heston or Heston-with-jumps)
    parameter_names = ['mu', 'kappa', 'theta', 'sigma', 'rho']

    # Plot the distributions
    plot_parameter_distributions(true_values, parameter_samples, parameter_names)
    # Print results
    # print("Estimated Parameters:")
    # for k, v in results.items():
    #     print(f"{k}: {v:.4f}")






