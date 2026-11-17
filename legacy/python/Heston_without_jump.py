import numpy as np
from scipy.stats import norm, invgamma
from filterpy.monte_carlo import systematic_resample

class ParticleFilter:
    def __init__(self, n_particles, initial_value, process_model, observation_model, process_noise, observation_noise):
        self.n_particles = n_particles
        self.particles = np.full(n_particles, initial_value)
        self.weights = np.ones(n_particles) / n_particles
        self.process_model = process_model
        self.observation_model = observation_model
        self.process_noise = process_noise
        self.observation_noise = observation_noise

    def predict(self):
        noise = np.random.normal(0, self.process_noise, self.n_particles)
        self.particles = self.process_model(self.particles) + noise

    def update(self, observation):
        likelihoods = norm.pdf(observation, self.observation_model(self.particles), self.observation_noise)
        self.weights *= likelihoods
        self.weights += 1e-300  # Avoid divide by zero
        self.weights /= np.sum(self.weights)

    def resample(self):
        # Ensure weights are normalized
        self.weights = np.clip(self.weights, 1e-300, None)
        self.weights /= np.sum(self.weights)
        # Resample particles based on weights
        indices = systematic_resample(self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.n_particles)


    def estimate(self):
        return np.mean(self.particles), np.var(self.particles)

# Define the function for the Heston model estimation

def heston_model_estimation(n_samples, delta_t, maturity, prices, n_particles, 
                            mu_0, kappa_0, theta_0, sigma_0, rho_0, 
                            mu_eta_0, tau_eta_0, mu_beta_0, lambda_beta_0,
                            a_sigma_0, b_sigma_0, mu_psi_0, tau_psi_0, a_omega_0, b_omega_0):
    """
    Perform parameter estimation for the Heston model using Algorithm 1.
    Inputs:
    - n_samples: Number of MCMC samples
    - delta_t: Time step size
    - maturity: Maturity of the model
    - prices: List or array of prices [S(0), S(1), ..., S(n)]
    - n_particles: Number of particles for particle filtering
    - Initial values for parameters: mu_0, kappa_0, theta_0, sigma_0, rho_0
    - Prior distribution parameters for eta, beta, sigma^2, psi, and omega
    """
    n = len(prices) - 1  # Number of time steps

    # Initialize arrays to store parameter estimates
    mu_samples = np.zeros(n_samples)
    kappa_samples = np.zeros(n_samples)
    theta_samples = np.zeros(n_samples)
    sigma_samples = np.zeros(n_samples)
    rho_samples = np.zeros(n_samples)

    # Compute ratios R(t) (Equation 12)
    R = prices[1:] / prices[:-1]

    # Initial guesses for parameters
    mu = mu_0
    kappa = kappa_0
    theta = theta_0
    sigma = sigma_0
    rho = rho_0

    # Define prior distributions
    eta_prior_mean = mu_eta_0
    eta_prior_precision = tau_eta_0
    beta_prior_mean = np.array(mu_beta_0).reshape(-1, 1)  # Reshape to 2x1
    beta_prior_precision = np.array(lambda_beta_0)
    sigma_prior_a = a_sigma_0
    sigma_prior_b = b_sigma_0
    psi_prior_mean = mu_psi_0
    psi_prior_precision = tau_psi_0
    omega_prior_a = a_omega_0
    omega_prior_b = b_omega_0

    # Initialize Particle Filter
    process_model = lambda v: v + kappa * (theta - v) * delta_t  # Mean-reverting process
    observation_model = lambda v: np.sqrt(np.clip(v, 1e-8, None))

    pf = ParticleFilter(n_particles, theta_0, process_model, observation_model, sigma_0, sigma_0)

    for i in range(n_samples):
        # Step 1: Particle Filtering for Volatility Estimation
        
        # Compute volatility_estimates for n-1 steps to match R
        volatility_estimates = np.zeros(n)

        for t in range(n):
            pf.predict()
            pf.update(R[t])
            pf.resample()
            estimate, _ = pf.estimate()
            volatility_estimates[t] = estimate

        # Reshape volatility_estimates as a column vector
        volatility_estimates = volatility_estimates.reshape(-1, 1)

       


        # Step 2: Estimate mu (Equations 13–23)
        eta_samples = norm.rvs(
            loc=(eta_prior_precision * eta_prior_mean + np.sum(R / volatility_estimates)) / (eta_prior_precision + len(R)),
            scale=np.sqrt(1 / (eta_prior_precision + len(R)))
        )
        mu = (eta_samples - 1) / delta_t
        mu_samples[i] = mu

        X = np.column_stack([
            np.ones(n),            # for intercept (beta_0)
            volatility_estimates   # for beta_1
        ])
        XtX = X.T @ X
        Xty = X.T @ R



        # # Compute posterior mean for beta
        # XtX = np.clip(volatility_estimates.T @ volatility_estimates, 1e-8, 1e8)
        #  # Compute Xty with aligned dimensions
        # Xty = np.clip(volatility_estimates.T @ R.reshape(-1, 1), 1e-8, 1e8)
        posterior_precision = beta_prior_precision + XtX
        temp = (beta_prior_precision @ beta_prior_mean) + (Xty)
        posterior_mean = np.linalg.inv(posterior_precision) @ temp



        # posterior_precision = beta_prior_precision + XtX
        # posterior_mean = np.linalg.pinv(posterior_precision) @ \
        #                 (beta_prior_precision @ beta_prior_mean + Xty)

        beta_samples = norm.rvs(
            loc=posterior_mean,          # shape (2,)
            scale=np.sqrt(1 / sigma)     # just a single std dev
        )


        
        kappa = (1 - beta_samples[1]) / delta_t
        theta = beta_samples[0] / (kappa * delta_t)
        print(kappa.shape)
        kappa_samples[i] = kappa
        theta_samples[i] = theta

        # Sample sigma squared
        residuals = R - X @ beta_samples
        sigma_prior_b = sigma_prior_b + 0.5 * np.sum(residuals**2)
        sigma_sq = invgamma.rvs(a=sigma_prior_a + n/2, scale=sigma_prior_b)
        sigma = np.sqrt(sigma_sq)
        sigma_samples[i] = sigma

        # Step 4: Estimate rho (Equations 45–48)


        # 1) Compute e1^rho(kΔt), e2^rho(kΔt)
        e1_rho = np.zeros(n)
        e2_rho = np.zeros(n)
        for k in range(n):
            # eq. (45):
            #   e1^rho = ( R(kΔt) - [ μ*Δt + ... ] ) / sqrt( Δt * v((k-1)Δt ) )
            #   but since R(kΔt) = S[k]/S[k-1], you might want to adapt that form.
            #   We treat "R(kΔt) - 1" as the increment minus 1, etc.
            denom = np.sqrt(delta_t* np.clip(volatility_estimates[k],1e-8,None))
            e1_rho[k] = (R[k] - (1 + mu*delta_t)) / denom

            # eq. (46):
            #   e2^rho = [ v(kΔt) - v((k-1)Δt) - κ [θ - v((k-1)Δt)]Δt ] 
            #            / sqrt( Δt * v((k-1)Δt ) )
            #   We approximate v((k-1)Δt) by vol_est[k-1] for k>0.  For k=0 you
            #   have to be careful, but we only do k=1..n-1 effectively.
            if k == 0:
                e2_rho[k] = 0.0  # or skip
            else:
                dv = volatility_estimates[k] - volatility_estimates[k-1]
                drift = kappa*(theta - volatility_estimates[k-1])*delta_t
                e2_rho[k] = (dv - drift)/np.sqrt(delta_t* np.clip(volatility_estimates[k-1],1e-8,None))

        # 2) Build A^rho = e'^rho e^rho
        #    e^rho is n×2: [ e1_rho, e2_rho ]
        e_rho = np.column_stack((e1_rho, e2_rho))
        A_rho = e_rho.T @ e_rho  # 2×2
        A11 = A_rho[0,0]
        A12 = A_rho[0,1]
        A22 = A_rho[1,1]

        # 3) Draw omega from its IG posterior (Eqs. 56-57)
        omega_prior_a = omega_prior_a + n/2.
        omega_prior_b = omega_prior_b + 0.5*(A22 - A12**2/A11)
        omega_draw = invgamma.rvs(a=omega_prior_a, scale=omega_prior_b)

        #    Draw psi from Normal posterior (Eqs. 54-55)
        tau_psi = A11 + psi_prior_precision
        mu_psi = (A12 + psi_prior_mean*psi_prior_precision)/tau_psi
        psi_draw = norm.rvs( loc=mu_psi, scale=np.sqrt(omega_draw/tau_psi) )

        # 4) Recover rho
        rho = psi_draw / np.sqrt(psi_draw**2 + omega_draw)
        rho_samples[i] = rho






        psi_samples = norm.rvs(
            loc=(
                psi_prior_mean * psi_prior_precision
                + np.dot(R.T, volatility_estimates)
            ) / (psi_prior_precision + len(R)),
            scale=np.sqrt(1 / (psi_prior_precision + len(R)))
        )
        rho = psi_samples / np.sqrt(psi_samples**2 + omega_prior_b)
        rho_samples[i] = rho
        print("beta_samples.shape =", beta_samples.shape)
        print("kappa =", kappa, "shape =", np.shape(kappa))

        print("psi_samples =", psi_samples, "shape =", np.shape(psi_samples))
        print("rho =", rho, "shape =", np.shape(rho))

    # Final estimates (mean of samples)
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
        'rho': rho_hat
    }

# Define initial parameters from Table 1
n_samples = 100  # Number of MCMC samples
delta_t = 1/252  # Daily frequency (252 trading days in a year)
maturity = 3  # 3 years maturity
n_particles = 100  # Number of particles for particle filtering

# Initial parameter values
mu_0 = 0.1
kappa_0 = 1
theta_0 = 0.05
sigma_0 = 0.01
rho_0 = -0.5

# Prior distribution parameters
mu_eta_0 = 1.00125
tau_eta_0 = 0.001
mu_beta_0 = [35e-6, 0.988]
lambda_beta_0 = [[10, 0], [0, 5]]
a_sigma_0 = 149
b_sigma_0 = 0.025
mu_psi_0 = -0.45
tau_psi_0 = 0.3
a_omega_0 = 1.03
b_omega_0 = 0.05

# Generate synthetic price data (e.g., geometric Brownian motion)
n_steps = int(maturity / delta_t)
prices = np.zeros(n_steps)
prices[0] = 100  # Initial price
np.random.seed(42)  # Seed for reproducibility
for t in range(1, n_steps):
    dt = delta_t
    dW = np.random.normal(0, np.sqrt(dt))
    prices[t] = prices[t-1] * np.exp((mu_0 - 0.5 * sigma_0**2) * dt + sigma_0 * dW)

# Call the Heston model estimation function
results = heston_model_estimation(
    n_samples=n_samples, delta_t=delta_t, maturity=maturity, prices=prices,
    n_particles=n_particles, mu_0=mu_0, kappa_0=kappa_0, theta_0=theta_0,
    sigma_0=sigma_0, rho_0=rho_0, mu_eta_0=mu_eta_0, tau_eta_0=tau_eta_0,
    mu_beta_0=mu_beta_0, lambda_beta_0=lambda_beta_0, a_sigma_0=a_sigma_0,
    b_sigma_0=b_sigma_0, mu_psi_0=mu_psi_0, tau_psi_0=tau_psi_0, a_omega_0=a_omega_0,
    b_omega_0=b_omega_0
)

# Output results
print("Estimated Parameters:")
print(f"mu: {results['mu']}")
print(f"kappa: {results['kappa']}")
print(f"theta: {results['theta']}")
print(f"sigma: {results['sigma']}")
print(f"rho: {results['rho']}")








# import numpy as np
# from scipy.stats import norm, invgamma

# # Define the function for the Heston model estimation

# def heston_model_estimation(n_samples, delta_t, maturity, prices, n_particles, 
#                             mu_0, kappa_0, theta_0, sigma_0, rho_0, 
#                             mu_eta_0, tau_eta_0, mu_beta_0, lambda_beta_0,
#                             a_sigma_0, b_sigma_0, mu_psi_0, tau_psi_0, a_omega_0, b_omega_0):
#     """
#     Perform parameter estimation for the Heston model using Algorithm 1.
#     Inputs:
#     - n_samples: Number of MCMC samples
#     - delta_t: Time step size
#     - maturity: Maturity of the model
#     - prices: List or array of prices [S(0), S(1), ..., S(n)]
#     - n_particles: Number of particles for particle filtering
#     - Initial values for parameters: mu_0, kappa_0, theta_0, sigma_0, rho_0
#     - Prior distribution parameters for eta, beta, sigma^2, psi, and omega
#     """
#     n = len(prices) - 1  # Number of time steps
    
#     # Initialize arrays to store parameter estimates
#     mu_samples = np.zeros(n_samples)
#     kappa_samples = np.zeros(n_samples)
#     theta_samples = np.zeros(n_samples)
#     sigma_samples = np.zeros(n_samples)
#     rho_samples = np.zeros(n_samples)
    
#     # Compute ratios R(t) (Equation 12)
#     R = prices[1:] / prices[:-1]
    
#     # Initial guesses for parameters
#     mu = mu_0
#     kappa = kappa_0
#     theta = theta_0
#     sigma = sigma_0
#     rho = rho_0
    
#     # Define prior distributions
#     eta_prior_mean = mu_eta_0
#     eta_prior_precision = tau_eta_0
#     beta_prior_mean = np.array(mu_beta_0)
#     beta_prior_precision = np.array(lambda_beta_0)
#     sigma_prior_a = a_sigma_0
#     sigma_prior_b = b_sigma_0
#     psi_prior_mean = mu_psi_0
#     psi_prior_precision = tau_psi_0
#     omega_prior_a = a_omega_0
#     omega_prior_b = b_omega_0

#     # Initialize volatility array
#     volatilities = np.zeros((n, n_particles))  # Particle filtering storage

#     for i in range(n_samples):
#         # Step 1: Particle Filtering for Volatility Estimation
#         for t in range(1, n):
#             # Generate particles for volatilities (Equation 64)
#             for j in range(n_particles):
#                 if t == 1:
#                     volatilities[t, j] = theta  # Initialize particles to theta
#                 else:
#                     w_j = (R[t-1] - mu * delta_t - 1) / np.sqrt(delta_t * max(volatilities[t-1, j], 1e-8))
#                     epsilon_j = np.random.normal(0, 1)
#                     volatilities[t, j] = volatilities[t-1, j] + kappa * (theta - volatilities[t-1, j]) * delta_t \
#                                        + sigma * np.sqrt(delta_t * max(volatilities[t-1, j], 1e-8)) * (rho * w_j + np.sqrt(1 - rho**2) * epsilon_j)

#         # Average over particles for volatility estimate
#         volatility_estimates = np.clip(np.mean(volatilities, axis=1), 1e-8, None)

#         # Step 2: Estimate mu (Equations 13–23)
#         eta_samples = norm.rvs(
#             loc=(eta_prior_precision * eta_prior_mean + np.sum(R / volatility_estimates)) / (eta_prior_precision + len(R)),
#             scale=np.sqrt(1 / (eta_prior_precision + len(R)))
#         )
#         mu = (eta_samples - 1) / delta_t
#         mu_samples[i] = mu

#         # Step 3: Estimate kappa, theta, sigma (Equations 24–44)
#         # Reshape volatility_estimates and R as column vectors
#         volatility_estimates = volatility_estimates.reshape(-1, 1)  # Column vector
#         R = R.reshape(-1, 1)  # Column vector

#         # Ensure beta_prior_precision is properly shaped
#         beta_prior_precision = np.array(beta_prior_precision)  # Convert to NumPy array if not already
#         if beta_prior_precision.shape != (2, 2):
#             raise ValueError("beta_prior_precision must be a 2x2 matrix.")

#         # Calculate beta_samples with proper matrix dimensions
#         loc = np.linalg.inv(beta_prior_precision + volatility_estimates.T @ volatility_estimates) @ \
#             (beta_prior_precision @ np.array(beta_prior_mean).reshape(-1, 1) + volatility_estimates.T @ R)
#         beta_samples = norm.rvs(
#             loc=loc.flatten(),  # Flatten to pass as mean to norm.rvs
#             scale=np.sqrt(1 / sigma)
#         )


        
#         kappa = (1 - beta_samples[1]) / delta_t
#         theta = beta_samples[0] / (kappa * delta_t)
#         kappa_samples[i] = kappa
#         theta_samples[i] = theta

#         # Sample sigma squared
#         scale = max(b_sigma_0 + 0.5 * np.sum((R - beta_samples[0])**2), 1e-8)
#         sigma_sq = invgamma.rvs(a=sigma_prior_a + n / 2, scale=scale)
#         sigma = np.sqrt(sigma_sq)
#         sigma_samples[i] = sigma

#         # Step 4: Estimate rho (Equations 45–48)
#         psi_samples = norm.rvs(
#             loc=(psi_prior_mean * psi_prior_precision + np.dot(R.T, volatility_estimates)) / (psi_prior_precision + len(R)),
#             scale=np.sqrt(1 / (psi_prior_precision + len(R)))
#         )
#         rho = psi_samples / np.sqrt(psi_samples**2 + omega_prior_b)
#         rho_samples[i] = rho

#     # Final estimates (mean of samples)
#     mu_hat = np.mean(mu_samples)
#     kappa_hat = np.mean(kappa_samples)
#     theta_hat = np.mean(theta_samples)
#     sigma_hat = np.mean(sigma_samples)
#     rho_hat = np.mean(rho_samples)

#     return {
#         'mu': mu_hat,
#         'kappa': kappa_hat,
#         'theta': theta_hat,
#         'sigma': sigma_hat,
#         'rho': rho_hat,
#         'volatility_estimates': volatility_estimates
#     }

# # Define initial parameters from Table 1
# n_samples = 500  # Number of MCMC samples
# delta_t = 1/252  # Daily frequency (252 trading days in a year)
# maturity = 3  # 3 years maturity
# n_particles = 100  # Number of particles for particle filtering

# # Initial parameter values
# mu_0 = 0.5
# kappa_0 = 1
# theta_0 = 0.05
# sigma_0 = 0.01
# rho_0 = -0.5

# # Prior distribution parameters
# mu_eta_0 = 1.00125
# tau_eta_0 = 0.001
# mu_beta_0 = [35e-6, 0.988]
# lambda_beta_0 = [[10, 0], [0, 5]]
# a_sigma_0 = 1.2
# b_sigma_0 = 0.025
# mu_psi_0 = -0.45
# tau_psi_0 = 0.3
# a_omega_0 = 1.03
# b_omega_0 = 0.05

# # Generate synthetic price data (e.g., geometric Brownian motion)
# n_steps = int(maturity / delta_t)
# prices = np.zeros(n_steps)
# prices[0] = 100  # Initial price
# np.random.seed(42)  # Seed for reproducibility
# for t in range(1, n_steps):
#     dt = delta_t
#     dW = np.random.normal(0, np.sqrt(dt))
#     prices[t] = prices[t-1] * np.exp((mu_0 - 0.5 * sigma_0**2) * dt + sigma_0 * dW)

# # Call the Heston model estimation function
# results = heston_model_estimation(
#     n_samples=n_samples, delta_t=delta_t, maturity=maturity, prices=prices,
#     n_particles=n_particles, mu_0=mu_0, kappa_0=kappa_0, theta_0=theta_0,
#     sigma_0=sigma_0, rho_0=rho_0, mu_eta_0=mu_eta_0, tau_eta_0=tau_eta_0,
#     mu_beta_0=mu_beta_0, lambda_beta_0=lambda_beta_0, a_sigma_0=a_sigma_0,
#     b_sigma_0=b_sigma_0, mu_psi_0=mu_psi_0, tau_psi_0=tau_psi_0, a_omega_0=a_omega_0,
#     b_omega_0=b_omega_0
# )

# # Output results
# print("Estimated Parameters:")
# print(f"mu: {results['mu']}")
# print(f"kappa: {results['kappa']}")
# print(f"theta: {results['theta']}")
# print(f"sigma: {results['sigma']}")
# print(f"rho: {results['rho']}")
