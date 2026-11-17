import numpy as np
from scipy.stats import norm, invgamma
from filterpy.monte_carlo import systematic_resample
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas as pd
import math
import random


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
    lambda_beta_0 = np.array(lambda_beta_0)
    mu_beta_0 = np.array(mu_beta_0)
    sigma_prior_a = a_sigma_0
    sigma_prior_b = b_sigma_0
    psi_prior_mean = mu_psi_0
    psi_prior_precision = tau_psi_0
    omega_prior_a = a_omega_0
    omega_prior_b = b_omega_0
    volatility_estimates_all = np.zeros((n_samples, n))
    
    MAX_PARAM = 1e2  # or choose a smaller bounding if you prefer


    # Ratios R(t) = S(t+1)/S(t)
    R = prices[1:] / prices[:-1]

    V = [theta]*n_particles   

    for i in range(n_samples):


        # Particle storage: V[j] will be the current volatility value for particle j
            # Initialize all particles V_j(0) = θ
        
        
        # We'll store the final volatility estimates here
        v_est = [0.0]*n

        # Main time loop
        for k in range(n):
            # STEP 1: For each particle, propose a new volatility candidate
            V_candidates = [0.0]*n_particles
            weights      = [0.0]*n_particles
            
            for j in range(n_particles):
                # (61) draw eps ~ N(0,1)
                eps = np.random.normal(0,1)
                
                # (62) residual z_j(kΔt) using previous V[j] and R[k]
                #     Typically: z_j = ( R(kΔt) - μΔt - 1 ) / sqrt( V_j((k-1)Δt)*( (k-1)Δt ) ), etc.
                z = (R[k] - mu*delta_t - 1.0) / math.sqrt( V[j]*delta_t )
                
                # (63) w_j(kΔt) = z_j ρ + ε_j sqrt(1 - ρ²)
                w_ = z*rho + eps*math.sqrt(1.0 - rho**2)

                # (64) new candidate
                # V_j^tilde(kΔt) = V_j((k-1)Δt) + κ(θ - V_j((k-1)Δt)) Δt + σ_v sqrt(Δt V_j((k-1)Δt)) * w_
                old_vol = V[j]
                new_vol = old_vol + kappa*(theta - old_vol)*delta_t + sigma*math.sqrt(delta_t*old_vol)*w_
                # Keep it nonnegative
                new_vol = max(new_vol, 1e-12)

                V_candidates[j] = new_vol

            # STEP 2: Compute weights from likelihood (65) and normalize (66)
            #  W_j(kΔt) = (1 / sqrt(2π V_j(kΔt) dt)) * exp( - ( (R(kΔt) - μdt -1)^2 ) / [2 V_j(kΔt) dt ] )
            sum_w = 0.0
            for j in range(n_particles):
                vol_j = V_candidates[j]
                # PDF for normal with variance = vol_j * dt
                # If your model's normalizing constant is different, adapt accordingly.
                var = vol_j * delta_t
                std = math.sqrt(var)
                x   = (R[k] - mu*delta_t - 1.0)
                pdf = (1.0/(math.sqrt(2.0*math.pi)*std)) * math.exp(-0.5*(x/std)**2)
                weights[j] = pdf
                sum_w     += pdf

            # Avoid divide-by-zero
            if sum_w < 1e-15:  
                # fallback, e.g. reset uniform weights
                weights = [1.0/n_particles]*n_particles
            else:
                # normalize
                weights = [w_/sum_w for w_ in weights]

            # STEP 3: Pair up particles and weights, then RESAMPLE (67)–(73)
            # A simple approach: 
            #   1) sort (V_candidates) in ascending order
            #   2) build a piecewise‐linear CDF
            #   3) sample from that CDF with inverse transform

            # 3A. Sort by ascending volatility
            pairs = sorted(zip(V_candidates, weights), key=lambda p: p[0])
            sorted_vols  = [p[0] for p in pairs]
            sorted_w     = [p[1] for p in pairs]

            # 3B. Build the cumulative distribution for the “continuous” approach
            #     The text shows a piecewise expression, but we can approximate
            #     by taking midpoints or “equal jumps” in between. For simplicity,
            #     do a standard discrete CDF:
            cdf = []
            running_sum = 0.0
            for j in range(n_particles):
                running_sum += sorted_w[j]
                cdf.append(running_sum)

            # 3C. Draw new “refined” particles via inverse‐transform sampling
            #     i.e., for each of N uniform draws U in [0,1], find the vol
            #     whose cdf bracket contains U. 
            new_particles = []
            for _ in range(n_particles):
                u = random.random()
                # find index j s.t. cdf[j-1] < u <= cdf[j]
                # or use a manual loop if we want to avoid libraries:
                idx = 0
                while idx < n_particles and cdf[idx] < u:
                    idx += 1
                # to avoid out-of-bounds
                idx = min(idx, n_particles-1)
                new_particles.append(sorted_vols[idx])

            # Overwrite V with the newly refined particles
            V = new_particles

            # STEP 4: Estimate volatility at this step by the mean of refined particles (74)
            v_est[k] = sum(V)/n_particles

        volatility_estimates_all[i, :] = v_est







#------------------------------------------------------------------------------------------------------------------------------------------------------

        # Make sure volatility_estimates is shape (n,)
        # (That is already the case from np.zeros(n).)

        #----------------------------------------------------------
        # Step 2a: Estimate mu (drift) using eqns (13)–(23)
        #     We'll treat "eta = 1 + mu * dt", etc.
        #----------------------------------------------------------
        # Summation term: sum(R / volatility_estimates) 
        # But watch out for shape mismatch: R is (n,), volatility_estimates is (n,).
        # So R / volatility_estimates is (n,), then sum is scalar.

        
        # Build design matrix X: shape (n,2)
        #   X = [ 1, volatility_estimate ] 
        #   R ~ X beta
        # Make sure volatility_estimates is shape (n,)
        

        x_s = (1.0/np.sqrt(delta_t))*1.0/np.sqrt(v_est)
        y_s = (1.0/np.sqrt(delta_t))*(R/np.sqrt(v_est))

        ols =  np.dot(x_s.T, y_s)/(np.dot(x_s.T, x_s))


        Tau_eta = np.dot(x_s.T, x_s) + eta_prior_precision

        
        # sum_term = np.sum(R / v_est)

        # Posterior mean for eta
        #   = ( tau_eta_0 * mu_eta_0 + sum(R/volatility) ) / ( tau_eta_0 + n )

        eta_posterior_mean = (eta_prior_precision * eta_prior_mean + np.dot(x_s.T, x_s)*ols) / (Tau_eta)
        
        eta_sample = np.random.normal(eta_posterior_mean, 1.0/np.sqrt(Tau_eta))
        mu = (eta_sample - 1.0)/delta_t

        # Convert from eta to mu
        # if we assume eta = 1 + mu*dt  => mu = (eta - 1)/dt
        mu = np.clip(mu, -10., 10.)
        mu_samples[i] = mu

#----------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Step 2b: Estimate (kappa, theta) from eqns (24)–(44)
        y_list, x1_list, x2_list = [], [], []

        for k in range(1, n):
            # Denominator for transformations: sqrt(∆t * v((k-1)*∆t))
            denom = np.sqrt(delta_t * v_est[k-1])
            if denom < 1e-15:
                raise ValueError(f"Volatility too close to zero at step {k-1}, cannot form regressors.")

            # yᵛₖ = v(k∆t) / sqrt(∆t * v((k-1)∆t))
            y_list.append(v_est[k] / denom)

            # x₁ₖ = 1 / sqrt(∆t * v((k-1)∆t))
            x1_list.append(1.0 / denom)

            # x₂ₖ = v((k-1)∆t) / sqrt(∆t * v((k-1)∆t)) = sqrt(v((k-1)∆t) / ∆t)
            x2_list.append(v_est[k-1] / denom)

        # Convert to NumPy arrays

        beta_1 = kappa*theta*delta_t
        beta_2 = 1-kappa*delta_t

        Beta =np.array([beta_1,beta_2])
        y_vec = np.array(y_list)                 # shape (n-1,)


        x1_vec = np.array(x1_list)                # shape (n-1,)
        x2_vec = np.array(x2_list)                # shape (n-1,)

        # Xᵛ = [ x1   x2 ], shape (n-1, 2)
        X_mat = np.column_stack((x1_vec, x2_vec))
        # y_vec = X_mat@Beta.T 

        # ---------------------------------------------------------------
        # 2. OLS estimate:  β̂ = (Xᵛᵀ Xᵛ)⁻¹ Xᵛᵀ yᵛ  (Eq. (38))
        # ---------------------------------------------------------------
        # Use pseudo-inverse to guard against ill-conditioned Xᵛᵀ Xᵛ
        beta_hat_ols = np.linalg.pinv(np.dot(X_mat.T ,X_mat)) @ (X_mat.T @ y_vec)
  
        # ---------------------------------------------------------------
        # 3. Bayesian update (Eqns (36)–(39)) given prior β₀, Λ₀
        #    Posterior precision: Λ^β = (Xᵛ)ᵀ Xᵛ + Λ₀
        #    Posterior mean: µ^β = (Λ^β)⁻¹ [ Λ₀ β₀ + (Xᵛ)ᵀ yᵛ ]
        # ---------------------------------------------------------------
        
        Lambda_beta = np.dot(X_mat.T, X_mat) + lambda_beta_0
        rhs = lambda_beta_0 @ mu_beta_0 + (X_mat.T @ X_mat) @ beta_hat_ols
        mu_beta = np.linalg.pinv(Lambda_beta) @ rhs

        # ---------------------------------------------------------------
        # (Optional) If you have or estimate σᵛ², you can sample from
        # the posterior:
        #   β ∼ Normal( mu_beta,  σᵛ² * (Λ^β)⁻¹ ).
        # ---------------------------------------------------------------
        cov_beta = sigma**2 * np.linalg.pinv(Lambda_beta)

        # Draw one sample from the posterior distribution:
        beta_draw = np.random.multivariate_normal(mean=mu_beta, cov=cov_beta)

        #----------------------------------------------------------
        X = np.column_stack((np.ones(n), v_est))
        
        
        kappa = (1.0 - beta_draw[1]) / delta_t
        theta = beta_draw[0]/(kappa*delta_t)

        kappa = np.clip(kappa, 1e-6, MAX_PARAM)
        theta = np.clip(theta, 1e-6, MAX_PARAM)
        kappa_samples[i] = kappa
        theta_samples[i] = theta

        #----------------------------------------------------------
        # Step 2c: Sample sigma^2 from an Inverse Gamma
        #----------------------------------------------------------
        # Posterior scale
        sigma_b = sigma_prior_b + 0.5 * (y_vec.T @ y_vec + mu_beta_0 @ lambda_beta_0 @ mu_beta_0  
                                         - mu_beta.T @ Lambda_beta @ mu_beta )
        # Posterior shape
        sigma_a = sigma_prior_a + n / 2.0
        # Draw from IG:
        sigma_sq = invgamma.rvs(a=sigma_a, scale=sigma_b)
        sigma = np.sqrt(sigma_sq)
        sigma_samples[i] = sigma

        #----------------------------------------------------------
        # Step 2d: Estimate rho from eqns (45)–(59)
        e1_rho = np.zeros(n)
        e2_rho = np.zeros(n)

        for t in range(n):
            # eq. (45):
            if t == 0:
                # no prior v((−1)Δt)
                e1_rho[t] = 0.0
            else:
                denom1 = np.sqrt(delta_t * np.clip(v_est[t-1],1e-12,None))
                numerator1 = R[t] - (1.0 + mu*delta_t)
                e1_rho[t] = numerator1 / denom1

            # eq. (46):
            if t == 0:
                e2_rho[t] = 0.0
            else:
                dv = v_est[t] - v_est[t-1]
                drift = kappa*(theta - v_est[t-1])*delta_t
                denom2 = np.sqrt(delta_t * np.clip(v_est[t-1],1e-12,None))
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




def load_prices_from_csv(filename="my_heston_euler.csv"):
    """
    Load previously simulated data from CSV, return arrays of time, price, and variance.
    """
    data = np.loadtxt(filename, delimiter=",", skiprows=1)  # skip the header
    t = data[:, 0]
    prices = data[:, 1]
    variances = data[:, 2]
    return t, prices, variances



# ----------------------------------------------------------------
# Example usage
if __name__ == "__main__":

    # Set your parameters
    n_samples = 100
    n_particles = 100

    # Initial parameter values
    T       = 1.0
    Nsteps  = 252

    mu_0 = 0.2
    kappa_0 =2
    theta_0 = 0.03
    sigma_0 = 0.015
    rho_0 = -0.38

    t, prices, v_path = load_prices_from_csv("my_heston_euler.csv")

    delta_t = 1.0/len(prices)

    T = 1.0

    maturity = T

    true_values = {
    'mu': 0.1, 'kappa': 1, 'theta': 0.05, 'sigma': 0.01, 'rho': -0.5
    }

    # Plot the synthetic path
    t_grid = np.linspace(0, T, Nsteps+1)
    fig, ax = plt.subplots(2, 1, figsize=(9,6), sharex=True)

    ax[0].plot(t_grid, prices, label="Synthetic Price")
    ax[0].set_ylabel("Price")
    ax[0].legend()

    ax[1].plot(t_grid, np.sqrt(v_path), label="Synthetic Vol (sqrt(v))")
    ax[1].set_ylabel("Volatility")
    ax[1].set_xlabel("Time (years)")
    ax[1].legend()

    plt.suptitle("Heston Simulation with Known (True) Parameters")
    plt.show()

    # Priors
    mu_eta_0 = 1.00125
    tau_eta_0 = 1.0/math.sqrt(0.001**2)
    mu_beta_0 = [35e-7, 0.99]        # <--- shape (2,)
    lambda_beta_0 = [[10, 0], [0, 5]] # shape (2,2)
    a_sigma_0 = 149
    b_sigma_0 = 0.025
    mu_psi_0 = -0.45
    tau_psi_0 = 1.0/math.sqrt(0.3**2)
    a_omega_0 = 1.03
    b_omega_0 = 0.05


    results = heston_model_estimation(
        n_samples, delta_t, maturity, prices,
        n_particles, mu_0, kappa_0, theta_0, sigma_0, rho_0, 
        mu_eta_0, tau_eta_0, mu_beta_0, lambda_beta_0, a_sigma_0, b_sigma_0, 
        mu_psi_0, tau_psi_0, a_omega_0, b_omega_0
    )
    # vols = results['volatility_estimates_all']  # shape (n_samples, n)
    

    print("Estimated Parameters:")
    print(f"mu: {results['mu']}")
    print(f"kappa: {results['kappa']}")
    print(f"theta: {results['theta']}")
    print(f"sigma: {results['sigma']}")
    print(f"rho: {results['rho']}")

    
    vols = results['volatility_estimates_all']  # shape (n_samples, n)
    final_volatility_path = vols[-1, :]         # last iteration
    plt.figure(figsize=(8,5))
    plt.plot(final_volatility_path, label='Volatility (final iteration)')   # Estimated volatility 
    plt.plot(v_path, label="Synthetic Vol (sqrt(v))")
    # plt.plot(data.index, data[f"RealizedVol_{window}"], label=f"{window}-day Realized Vol") #real volatility
    plt.xlabel('Time Step')
    plt.ylabel('Estimated Vol')
    plt.legend()
    plt.title('Particle-Filtered Volatility (final MCMC iteration)')
    plt.show()


    avg_volatility_path = vols.mean(axis=0)  # shape (n,)
    plt.figure(figsize=(8,5))
    plt.plot(avg_volatility_path, label='Volatility (mean over all iterations)')
    plt.plot(v_path, label="Synthetic Vol (sqrt(v))")
    plt.xlabel('Time Step')
    plt.ylabel('Estimated Vol')
    plt.legend()
    plt.title('Mean PF Volatility Estimate Over MCMC')
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

    
    # Specify parameter names to plot (adjust based on Heston or Heston-with-jumps)
    parameter_names = ['mu', 'kappa', 'theta', 'sigma', 'rho']

    # Plot the distributions
    plot_parameter_distributions(true_values, parameter_samples, parameter_names)







