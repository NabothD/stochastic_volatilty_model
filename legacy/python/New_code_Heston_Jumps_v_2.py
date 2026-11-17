import numpy as np
from filterpy.monte_carlo import systematic_resample
from scipy.stats import norm, invgamma, bernoulli
import random
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas as pd
import math


def heston_with_merton_jumps_estimation(
    n_samples, delta_t, maturity, prices, n_particles, 
    # -- Heston initial guesses --
    mu_0, kappa_0, theta_0, sigma_0, rho_0,
    # -- Heston priors (same as your original code) --
    mu_eta_0, tau_eta_0, mu_beta_0, lambda_beta_0,
    a_sigma_0, b_sigma_0, mu_psi_0, tau_psi_0, a_omega_0, b_omega_0,
    # -- Jump parameters & priors --
    lambda_jump_0,       # initial guess for the "probability" or "fraction" of jumps
    mu_j0, sigma_j0,     # prior guess for jump size distribution, e.g. Normal(mu_j0, sigma_j0)  # hyperparameters if you want to do Bayesian updates of jump size
):
    """
    Illustrative Heston-with-Merton-jumps estimator.

    :param lambda_jump_0: initial "threshold" or fraction of particles that declare a jump
    :param mu_j0, sigma_j0: initial guess of jump size distribution N(mu_j0, sigma_j0)
    :param mu_j_prior_0, sigma_j_prior_0: hyperparameters for Bayesian update of (mu_j, sigma_j)

    NOTE: This code extends the particle filter in a minimal way to illustrate
          how to incorporate jump draws and weighting.  You will need to adapt
          it carefully to your final model/paper equations.
    """
    # number of price intervals:
    n = len(prices) - 1  
    # simple ratio array
    R_raw = prices[1:] / prices[:-1]

    # Storage for final draws
    mu_samples     = np.zeros(n_samples)
    kappa_samples  = np.zeros(n_samples)
    theta_samples  = np.zeros(n_samples)
    sigma_samples  = np.zeros(n_samples)
    rho_samples    = np.zeros(n_samples)
    lambda_samples = np.zeros(n_samples)  # store overall jump ratio
    mu_j_samples   = np.zeros(n_samples)  # store jump-size mean
    sig_j_samples  = np.zeros(n_samples)  # store jump-size std

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

    # Current parameter guesses
    mu    = mu_0
    kappa = kappa_0
    theta = theta_0
    sigma = sigma_0
    rho   = rho_0
    # Jump parameters
    lam_j = lambda_jump_0
    mu_j  = mu_j0
    sig_j = sigma_j0

    # Precompute un‐jumped returns R_raw, but we will adjust each time step
    # according to eq. (84).

    MAX_PARAM = 1e4
    volatility_estimates_all = np.zeros((n_samples, n))

    for i_iter in range(n_samples):

        # Initialize particles for volatility:
        V = np.full(n_particles, theta)    # shape (N,)
        # Initialize particles for J and Z as well
        # (Though we will resample them each time step)
        J = np.zeros(n_particles, dtype=int)  
        Z = np.zeros(n_particles)          

        # We'll keep a time‐series estimate of volatility and jump size each step
        v_est = np.zeros(n)
        z_est = np.zeros(n)
        lam_t_series = np.zeros(n)
        

        # -------------- MAIN PARTICLE FILTER TIME LOOP --------------
        for k in range(n):
            # Step 1: propose new (J_j, Z_j) for each particle
            # Bernoulli( lam_j ) from eq. (75)
            # Z_j from Normal( mu_j, sig_j ) if J_j=1
            J_candidates = bernoulli.rvs(lam_j, size=n_particles)
            Z_candidates = np.where(
                (J_candidates == 1),
                np.random.normal(mu_j, sig_j, size=n_particles),
                0.0
            )

            # Equation (84) "neutralization" factor:
            #  R_star(kΔt) = R_raw[k] * [1 - λ(kΔt)(1 - exp(-Z_j)) ]
            #  but here λ(kΔt) ~ lam_j for illustration.  Each particle j
            #  might use J_j & Z_j, but the eq. in the paper uses
            #  the *weighted average* λ(kΔt) and Z(kΔt). For a purely
            #  “particle-level” approach, we do:
            R_star = R_raw
            # Step 2: Propose new volatility from eq. (61)-(64).
            # We also need the “z_j” for eq. (62).  We'll do:
            #   z_j = ( R_star(kΔt) - μΔt - 1 ) / sqrt( V_j((k-1)Δt)*Δt )
            V_candidates = np.zeros(n_particles)
            for j in range(n_particles):
                eps = np.random.normal(0,1)
                old_vol = V[j]
                # eq. (62)
                z_j = (R_star[j] - (1.0 + mu*delta_t)) / math.sqrt(max(old_vol*delta_t,1e-15))
                # eq. (63)
                w_j = z_j*rho + eps * math.sqrt(1.0 - rho**2)
                # eq. (64)
                new_vol = old_vol + kappa*(theta - old_vol)*delta_t + sigma*math.sqrt(delta_t*old_vol)*w_j
                new_vol = max(new_vol, 1e-12)
                V_candidates[j] = new_vol

            # Step 3: Compute weights from eq. (77).
            #   if J_j=0 => normal pdf with R_star = R(kΔt)
            #   if J_j=1 => separate pdf with the jump factor
            #   Actually eq. (77) in the paper shows two pieces, but let's do:
            weights = np.zeros(n_particles)
            for j in range(n_particles):
                var = V_candidates[j] * delta_t
                std = math.sqrt(var)
                x   = R_star[j] - (1.0 + mu*delta_t)
                # Normal pdf:
                pdf_val = (1.0/(math.sqrt(2.0*math.pi)*std)) * np.exp(-0.5*(x/std)**2)
                weights[j] = pdf_val

            sum_w = np.sum(weights)
            if sum_w < 1e-15:
                weights[:] = 1.0/n_particles
            else:
                weights /= sum_w

            # Step 4: Resample (V, J, Z) from the refined distribution
            #    same approach as in the no-jumps code
            #    for simplicity, use a systematic or stratified approach
            cdf = np.cumsum(weights)
            new_V = np.zeros(n_particles)
            new_J = np.zeros(n_particles, dtype=int)
            new_Z = np.zeros(n_particles)
            for jj in range(n_particles):
                u = random.random()
                idx = np.searchsorted(cdf, u)
                new_V[jj] = V_candidates[idx]
                new_J[jj] = J_candidates[idx]
                new_Z[jj] = Z_candidates[idx]

            # Overwrite
            V = new_V
            J = new_J
            Z = new_Z

            # Step 5: Estimate volatility & jump size at time k
            v_est[k] = np.mean(V)
            z_est[k] = np.mean(Z)  # average jump among refined particles

            # Step 6: Probability of jump at time k from eq. (79):
            #   λ(kΔt) = sum_j [ J_j W_j ]
            lam_t_series[k] = np.sum(J)*1.0/n_particles  # if we used uniform W_j after resampling
            # or you can do a pre-resampling average with the old weights.

        for k in range(n):
            jump_factor = (1.0 - np.exp(-z_est[k]))
            R_star[k] = R_raw[k] * (1.0 - lam_j * jump_factor)

        volatility_estimates_all[i_iter, :] = v_est
        
        # After we finish all time steps:
        # eq. (80):  λ_i = (1/n) * sum_{k=1..n} λ(kΔt)
        lam_j = np.mean(lam_t_series)

        # eq. (81):  Z(kΔt) = (1/N) sum_{j=1..N} Z_j(kΔt) among refined
        # but we've been storing z_est for each k.  If you want overall:
        Z_i = np.mean(z_est)  # average jump across all time steps

        # eqs. (82)-(83): Weighted mean / std dev of jumps if you want
        # a quick approximation:
        sum_lambda = np.sum(lam_t_series)
        if sum_lambda < 1e-15:
            # means effectively no jumps occurred
            mu_j_new = mu_j   # or fallback
            sig_j_new = sig_j
        else:
            sum_lamZ = np.sum(z_est * lam_t_series)
            mu_j_new = sum_lamZ / sum_lambda

            # (83) says:
            #   sigma_j = sqrt(  [ sum_{k=1}^n lambda(kΔt)* ( Z(kΔt) - mu_j )^2 ] * ((n-1)/n) / sum_lambda )
            sum_sqr = 0.0
            for k in range(n):
                diff = (z_est[k] - mu_j_new)
                sum_sqr += lam_t_series[k]*diff*diff
            var_j = sum_sqr * ((n-1)/n) / sum_lambda
            if var_j < 0.0:
                var_j = 1e-15
            sig_j_new = math.sqrt(var_j)

        # Overwrite
        mu_j  = mu_j_new
        sig_j = sig_j_new

        #----------------------------------------------------------
        # Step 2a: Estimate mu (drift) using eqns (13)–(23)
        #     We'll treat "eta = 1 + mu * dt", etc.
        #----------------------------------------------------------
        # Summation term: sum(R / volatility_estimates) 
        # But watch out for shape mismatch: R is (n,), volatility_estimates is (n,).
        # So R / volatility_estimates is (n,), then sum is scalar.


        x_s = (1.0/np.sqrt(delta_t))*1.0/v_est
        y_s = (1.0/np.sqrt(delta_t))*(R_star/v_est)

        ols = 1.0/(np.dot(x_s, x_s.T)) * np.dot(x_s, y_s.T)


        Tau_eta = np.dot(x_s, x_s.T) + tau_eta_0

        
        # sum_term = np.sum(R / v_est)

        # Posterior mean for eta
        #   = ( tau_eta_0 * mu_eta_0 + sum(R/volatility) ) / ( tau_eta_0 + n )

        eta_posterior_mean = (
            eta_prior_precision * eta_prior_mean + Tau_eta
        ) / (eta_prior_precision + Tau_eta*ols)
        eta_posterior_var = (Tau_eta + eta_prior_precision)
        eta_sample = norm.rvs(loc=eta_posterior_mean, scale=1.0/np.sqrt(eta_posterior_var))
        mu = (eta_sample - 1.0)/delta_t

        # Convert from eta to mu
        # if we assume eta = 1 + mu*dt  => mu = (eta - 1)/dt
        mu = np.clip(mu, -10., 10.)
        mu_samples[i_iter] = mu

        #----------------------------------------------------------
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
        y_vec = np.array(y_list)                  # shape (n-1,)
        x1_vec = np.array(x1_list)                # shape (n-1,)
        x2_vec = np.array(x2_list)                # shape (n-1,)

        # Xᵛ = [ x1   x2 ], shape (n-1, 2)
        X_mat = np.column_stack((x1_vec, x2_vec))


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
        lambda_beta_0 = np.array(lambda_beta_0)
        mu_beta_0 = np.array(mu_beta_0)
        Lambda_beta = np.dot(X_mat.T, X_mat) + lambda_beta_0
        rhs = lambda_beta_0 @ mu_beta_0 + (X_mat.T @ X_mat) @ beta_hat_ols
        mu_beta = np.linalg.pinv(Lambda_beta) @ rhs

        # ---------------------------------------------------------------
        # (Optional) If you have or estimate σᵛ², you can sample from
        # the posterior:
        #   β ∼ Normal( mu_beta,  σᵛ² * (Λ^β)⁻¹ ).
        # For demonstration we pick a dummy σᵛ² = 1e-4:
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
        kappa_samples[i_iter] = kappa
        theta_samples[i_iter] = theta
        lambda_samples[i_iter] = lam_j
        mu_j_samples[i_iter] = mu_j
        sig_j_samples[i_iter] = sig_j

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
        sigma_samples[i_iter] = sigma

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
                denom1 = np.sqrt(delta_t * np.clip(v_est[t-1],1e-12,None))
                numerator1 = R_star[t] - (1.0 + mu*delta_t)
                e1_rho[t] = numerator1 / denom1

            # eq. (46):
            #   e2^rho(kΔt) = [ v(kΔt) - v((k-1)Δt)  - κ(θ - v((k-1)Δt))Δt ]
            #                 / sqrt(Δt·v((k-1)Δt))
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

        rho_samples[i_iter] = rho


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
        'lambda':  np.mean(lambda_samples),
        'mu_j':    np.mean(mu_j_samples),
        'sig_j': np.mean(sig_j_samples),
        'mu_samples': mu_samples,
        'kappa_samples': kappa_samples,
        'theta_samples': theta_samples,
        'sigma_samples': sigma_samples,
        'rho_samples': rho_samples,
        'lambda_samples': lambda_samples,
        'mu_j_samples':   mu_j_samples,
        'sig_j_samples':  sig_j_samples,
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






# ----------------------------------------------------------------
# Example usage
if __name__ == "__main__":



    ticker = "^GSPC"  # Apple as an example
    start_date = "2018-01-01"
    end_date   = "2022-07-01"
    data = yf.download(ticker, start=start_date, end=end_date)

    # print(data)
    # print(data.columns)
    # Drop any missing rows just in case:
    data = data.dropna(subset=[("Close", "^GSPC")])

    # 2) Extract the 'Adj Close' prices as a NumPy array
    prices = data[("Close", "^GSPC")].values
    
    # 3) Decide on a time step.  For example, if you treat each row as one
    #    trading day and want "1 year" = 252 trading days, then:
    delta_t = 1.0 / 252.0
    # The total number of price points:
    n_steps = len(prices)
    # The implied "maturity" in years is then:
    maturity = n_steps * delta_t

    # 2) Compute daily returns (log or pct). For realized volatility,
    #    you can use log returns:
    data["LogReturn"] = np.log(data["Close"] / data["Close"].shift(1))

    # 3) Rolling standard deviation over a 21-day window (approx 1 month).
    #    Multiplying by sqrt(252) annualizes the daily volatility.
    window = 21
    data[f"RealizedVol_{window}"] = (
        data["LogReturn"].rolling(window).std() * np.sqrt(252)
    )

    # Now df["RealizedVol_21"] is your "real" (historical) volatility measure 
    # over a 21-day rolling window.

    # 4) Plot it
    plt.figure(figsize=(10,5))
    plt.plot(data.index, data[f"RealizedVol_{window}"], label=f"{window}-day Realized Vol")
    plt.title(f"S&P500 {window}-Day Realized Vol (annualized)")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.show()






    # Set your parameters
    n_samples = 100


    n_particles = 200

    # Initial parameter values
    mu_0 = 0.46
    kappa_0 = 1.17
    theta_0 = 0.06
    sigma_0 = 0.006
    rho_0 = -0.4

    # Priors
    mu_eta_0 = 1.00125
    tau_eta_0 = 0.001
    mu_beta_0 = [35e-6, 0.988]        # <--- shape (2,)
    lambda_beta_0 = [[10, 0], [0, 5]] # shape (2,2)
    a_sigma_0 = 149
    b_sigma_0 = 0.025
    mu_psi_0 = -0.45
    tau_psi_0 = 0.3
    a_omega_0 = 1.03
    b_omega_0 = 0.05
    lambda_jump_0 = 0.15
    mu_j0 = -0.96
    sigma_j0 = 0.3

    # # Generate synthetic prices
    # delta_t = maturity/n_samples
    # n_steps = int(maturity / delta_t)

    # np.random.seed(42)
    # prices = np.zeros(n_steps)
    # prices[0] = 100

    # for t in range(1, n_steps):
    #     # Draw a random increment ~ Normal(0, sqrt(dt))
    #     dW = np.random.normal(0, np.sqrt(delta_t))
    #     # Update price using the GBM discrete approximation
    #     prices[t] = prices[t - 1] * np.exp(
    #         (mu_0 - 0.5 * sigma_0**2) * delta_t + sigma_0 * dW
    #     )

    # Run the estimation
    results = heston_with_merton_jumps_estimation(
        n_samples, delta_t, maturity, prices,
        n_particles, mu_0, kappa_0, theta_0, sigma_0, rho_0, 
        mu_eta_0, tau_eta_0, mu_beta_0, lambda_beta_0, a_sigma_0, b_sigma_0, 
        mu_psi_0, tau_psi_0, a_omega_0, b_omega_0, lambda_jump_0, mu_j0, sigma_j0
    )
    # vols = results['volatility_estimates_all']  # shape (n_samples, n)
    
    

    print("Estimated Parameters:")
    print(f"mu: {results['mu']}")
    print(f"kappa: {results['kappa']}")
    print(f"theta: {results['theta']}")
    print(f"sigma: {results['sigma']}")
    print(f"rho: {results['rho']}")
    print(f"lambda: {results['lambda']}")
    print(f"mu_j: {results['mu_j']}")
    print(f"sig_j: {results['sig_j']}")

    
    vols = results['volatility_estimates_all']  # shape (n_samples, n)
    final_volatility_path = vols[-1, :]         # last iteration
    plt.figure(figsize=(8,5))
    plt.plot(final_volatility_path, label='Volatility (final iteration)')   # Estimated volatility 
    # plt.plot(data.index, data[f"RealizedVol_{window}"], label=f"{window}-day Realized Vol") #real volatility
    plt.xlabel('Time Step')
    plt.ylabel('Estimated Vol')
    plt.legend()
    plt.title('Particle-Filtered Volatility (final MCMC iteration)')
    plt.show()


    avg_volatility_path = vols.mean(axis=0)  # shape (n,)
    plt.figure(figsize=(8,5))
    plt.plot(avg_volatility_path, label='Volatility (mean over all iterations)')
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
    lambda_estimates = results.get('lambda_samples', [])
    mu_j_estimates = results.get('mu_j_samples', [])
    sig_j_estimates = results.get('sig_j_samples', [])



    parameter_samples = {
        'mu': mu_estimates,
        'kappa': kappa_estimates,
        'theta': theta_estimates,
        'sigma': sigma_estimates,
        'rho': rho_estimates,
        'lambda': lambda_estimates,
        'mu_j': mu_j_estimates,
        'sig_j': sig_j_estimates
    }

    true_values = {
    'mu': 0.44, 'kappa': 1.17, 'theta': 0.06, 'sigma': 0.006, 'rho': -0.41, 'lambda' : 1, 'mu_j': -0.8, 'sig_j': 0.2
    }
    # Specify parameter names to plot (adjust based on Heston or Heston-with-jumps)
    parameter_names = ['mu', 'kappa', 'theta', 'sigma', 'rho', 'lambda', 'mu_j', 'sig_j']

    # Plot the distributions
    plot_parameter_distributions(true_values, parameter_samples, parameter_names)
    # Print results
    # print("Estimated Parameters:")
    # for k, v in results.items():
    #     print(f"{k}: {v:.4f}")






