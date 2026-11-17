import numpy as np
from numpy.random import default_rng
from scipy.stats import norm, invgamma, beta as beta_dist
from filterpy.monte_carlo import systematic_resample

###############################################################################
#                 Jump-augmented Particle Filter  (Algorithm 2)
###############################################################################
class JumpParticleFilter:
    """
    Implements Equations (75)–(77) at each PF step:
      - (75)  J_j(kΔt) ~ Bernoulli( λ * Δt )
      - (76)  Z_j(kΔt) ~ Normal( mu^I, sigma^I ) if J_j=1, else 0
      - (77)  modifies the likelihood: 
              if J_j=0 => the usual no-jump pdf
              if J_j=1 => the jump pdf with an extra exponential factor
    Then we store v_j, J_j, Z_j for each particle j.
    """
    def __init__(self,
                 n_particles,
                 v_init,           # initial volatility guess
                 lambda_jump,      # jump intensity param (for eq.(75))
                 mu_jump,          # mu^I
                 sigma_jump,       # sigma^I
                 kappa, theta, sigma, rho, mu,  # Heston params used in vol update
                 dt,
                 rng=None):
        self.n_particles = n_particles
        self.dt = dt
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho   = rho
        self.mu    = mu
        self.lambda_jump = lambda_jump
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump

        self.rng = rng if rng else default_rng()

        # Each particle tracks v_j, plus J_j, Z_j for the current step.
        self.v_particles = np.full(n_particles, v_init, dtype=float)
        self.J_particles = np.zeros(n_particles, dtype=int)
        self.Z_particles = np.zeros(n_particles, dtype=float)

        # Weights
        self.weights = np.ones(n_particles)/n_particles

    def predict(self):
        """
        For k -> k+1 step:
          1) eq.(75): draw J_j from Bernoulli(lambda_jump * dt)
          2) eq.(76): if J_j=1, draw Z_j from Normal(mu_jump, sigma_jump)
          3) Heston update for volatility:
             v_j((k)Δt) = v_j((k-1)Δt) + kappa(...)*Δt + sigma sqrt(...) * e_j
             with correlation for e_j if you want, or just normal. 
             We'll do a simple approach for demonstration.
        """
        dt = self.dt
        new_v = np.zeros(self.n_particles, dtype=float)
        new_J = np.zeros(self.n_particles, dtype=int)
        new_Z = np.zeros(self.n_particles, dtype=float)

        # For correlation, we do eV_j ~ N(0,1), eS_j ~ N(0,1), then combine with rho:
        eV = self.rng.normal(0,1,self.n_particles)
        eS = self.rng.normal(0,1,self.n_particles)
        # but we only need eV for volatility update. eS is used in eq.(77)? 
        # Actually eq.(77) is about returns, so we'll incorporate that in .update().

        for j in range(self.n_particles):
            # eq.(75)
            p_jump = self.lambda_jump * dt
            # clamp p_jump in [0,1]
            if p_jump<0: p_jump=0
            if p_jump>1: p_jump=1
            if self.rng.random() < p_jump:
                new_J[j] = 1
                # eq.(76)
                new_Z[j] = self.rng.normal(self.mu_jump, self.sigma_jump)
            else:
                new_J[j] = 0
                new_Z[j] = 0.0

            old_v = self.v_particles[j]
            # Heston vol update (like eq.(77) for v_j):
            #   v_{k} = v_{k-1} + kappa*(theta - v_{k-1})*dt + sigma*sqrt(v_{k-1}*dt)* eV_j
            # We'll just do the standard form:
            # ensure positivity
            incr = self.kappa*(self.theta - old_v)*dt + self.sigma*np.sqrt(max(old_v,1e-12)*dt)*eV[j]
            new_v[j] = old_v + incr
            if new_v[j]<1e-12:
                new_v[j]=1e-12

        self.v_particles = new_v
        self.J_particles = new_J
        self.Z_particles = new_Z

    def update(self, R_obs):
        """
        eq.(77): the likelihood depends on whether J_j=0 or 1.
         - If J_j=0 => normal pdf( R_obs; mean=1 + mu*dt, var=v_j * dt )
         - If J_j=1 => normal pdf( R_obs; mean=exp(Z_j)*(1 + mu*dt), var= (exp(Z_j)-1)*v_j*dt ) 
           or however your eq.(77) precisely states the factor. 
        """
        dt = self.dt
        # clamp vol
        v_eff = np.clip(self.v_particles, 1e-12, None)
        w = np.zeros(self.n_particles, dtype=float)

        for j in range(self.n_particles):
            if self.J_particles[j] == 0:
                # no jump
                mean_return = 1.0 + self.mu*dt
                var_return  = v_eff[j]*dt
            else:
                # jump
                # eq.(77) suggests a factor exp(Z_j)
                mean_return = (1.0 + self.mu*dt)*np.exp(self.Z_particles[j])
                # variance might be (exp(Z_j)-1)*v_j*dt
                var_return  = np.abs(np.expm1(self.Z_particles[j])) * v_eff[j]*dt

            resid = R_obs - mean_return
            # normal pdf
            denom = np.sqrt(2*np.pi*var_return)
            w[j] = np.exp(-0.5*(resid**2)/var_return)/denom

        self.weights *= w
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)

    def resample(self):
        self.weights = np.clip(self.weights, 1e-300, None)
        self.weights /= np.sum(self.weights)
        idx = systematic_resample(self.weights)
        self.v_particles = self.v_particles[idx]
        self.J_particles = self.J_particles[idx]
        self.Z_particles = self.Z_particles[idx]
        self.weights.fill(1.0/self.n_particles)

    def estimate_vol(self):
        """(78) Weighted average of volatility. 
        In practice, eq.(78) talks about Z_j, but you can do same for v_j."""
        return np.sum(self.v_particles*self.weights)

    def estimate_jump_fraction(self):
        """(79) Weighted fraction of jumps. 
        eq.(79): lambda(kΔt) = sum_j[ J_j * w_j ]."""
        return np.sum(self.J_particles*self.weights)

    def estimate_jump_size(self):
        """(78) Weighted average of Z_j. 
           eq.(78) might talk about sum_j[Z_j w_j], or only among j s.t. J_j=1. 
           For simplicity, we do a direct weighted average."""
        return np.sum(self.Z_particles*self.weights)

###############################################################################
#         The main MCMC procedure for Heston + Jumps  (Algorithm 2)
###############################################################################
def heston_with_jumps_estimation(
    n_samples, delta_t, maturity, prices, n_particles,
    # initial guesses
    mu_0, kappa_0, theta_0, sigma_0, rho_0,
    lambda_0, muI_0, sigmaI_0,
    # priors for mu
    mu_eta_0, tau_eta_0,
    # priors for (kappa,theta), etc. -> same as your Table 1
    mu_beta_0, lambda_beta_0,
    a_sigma_0, b_sigma_0,
    mu_psi_0, tau_psi_0,
    a_omega_0, b_omega_0,
    # new priors for jump parameters eq.(80),(82),(83)
    # let's define Beta prior for lambda => alpha_lambda0, beta_lambda0
    alpha_lambda0=1.0, beta_lambda0=1.0,
    # Normal-InverseGamma for (mu^I, sigma^I):
    #    mu^I | sigma^I ~ Normal(muI0_0, (tauI0_0 * sigma^I)^{-1})
    #    sigma^I^2 ~ InvGamma(aI0, bI0)
    muI0_0=-0.50,  tauI0_0=1.0,
    aI0=2.0,       bI0=0.5
):
    rng = default_rng()

    n = len(prices)-1
    R = prices[1:] / prices[:-1]  # eq.(12)

    # Arrays to store chain
    mu_chain    = np.zeros(n_samples)
    kappa_chain = np.zeros(n_samples)
    theta_chain = np.zeros(n_samples)
    sigma_chain = np.zeros(n_samples)
    rho_chain   = np.zeros(n_samples)

    lambda_chain= np.zeros(n_samples)
    muI_chain   = np.zeros(n_samples)
    sigI_chain  = np.zeros(n_samples)

    # Current params
    mu_curr    = mu_0
    kappa_curr = kappa_0
    theta_curr = theta_0
    sigma_curr = sigma_0
    rho_curr   = rho_0
    lambda_curr= lambda_0
    muI_curr   = muI_0
    sigI_curr  = sigmaI_0

    for i in range(n_samples):
        #------------------------------------------------------------------
        # Step 1: Particle Filtering with jumps (Eqs.75–77)
        #------------------------------------------------------------------
        pf = JumpParticleFilter(n_particles=n_particles,
                                v_init=theta_curr,
                                lambda_jump=lambda_curr,
                                mu_jump=muI_curr,
                                sigma_jump=sigI_curr,
                                kappa=kappa_curr,
                                theta=theta_curr,
                                sigma=sigma_curr,
                                rho=rho_curr,
                                mu=mu_curr,
                                dt=delta_t,
                                rng=rng)

        vol_estimates = np.zeros(n)
        lambda_est    = np.zeros(n)
        Z_est         = np.zeros(n)

        for k_idx in range(n):
            pf.predict()
            pf.update(R[k_idx])
            pf.resample()

            # Weighted avg volatility
            vol_estimates[k_idx] = pf.estimate_vol()
            # Weighted fraction of jumps
            lambda_est[k_idx]    = pf.estimate_jump_fraction()
            # Weighted average of Z
            Z_est[k_idx]         = pf.estimate_jump_size()

        # eq.(78)–(79): we have per-step arrays of λ(kΔt) and Z(kΔt).
        # eq.(80),(82),(83) will use aggregates or sums from them.

        #------------------------------------------------------------------
        # Step 2: "Neutralize" R -> R_corrected by eq.(84)
        #         R_c(k) = R(k) / [ 1 - lambda_est(k)*(1 - exp(-Z_est(k))) ]
        #------------------------------------------------------------------
        R_corrected = np.zeros(n)
        for k_idx in range(n):
            denom = 1.0 - lambda_est[k_idx]*(1.0 - np.exp(-Z_est[k_idx]))
            if denom < 1e-12:
                denom=1e-12
            R_corrected[k_idx] = R[k_idx]/denom

        #----------------------------------------------------------
        # Step 3a: Sample mu from eq.(13)–(23),
        #          with "eta = 1 + mu dt"
        #          using the corrected R_c
        #----------------------------------------------------------
        sum_term = np.sum(R_corrected / vol_estimates)
        n_f      = len(R_corrected)
        eta_post_mean = (tau_eta_0*mu_eta_0 + sum_term)/(tau_eta_0 + n_f)
        eta_post_var  = 1.0/(tau_eta_0 + n_f)
        eta_draw = rng.normal(loc=eta_post_mean, scale=np.sqrt(eta_post_var))
        mu_curr  = (eta_draw - 1.0)/delta_t

        #----------------------------------------------------------
        # Step 3b: (kappa, theta) from eq.(24)–(44) using R_corrected
        #----------------------------------------------------------
        X = np.column_stack( (np.ones(n_f), vol_estimates) )
        XtX = X.T @ X
        Xty = X.T @ R_corrected
        # Posterior
        prior_prec = np.array(lambda_beta_0)
        post_prec  = prior_prec + XtX
        temp = (prior_prec @ np.array(mu_beta_0)) + Xty
        post_mean = np.linalg.inv(post_prec) @ temp
        # For demonstration, sample from a "diag" normal approx:
        # Or do a 2D normal properly. We'll do 2D:
        post_cov = np.linalg.inv(post_prec)
        beta_draw = rng.multivariate_normal(mean=post_mean, cov=post_cov)
        # eq.(40) => kappa = (1 - beta[1])/dt, theta= beta[0]/(kappa*dt)
        b0, b1 = beta_draw
        kappa_curr = max(1e-6, (1.0 - b1)/delta_t)
        theta_curr = max(1e-6, b0/(kappa_curr*delta_t))

        #----------------------------------------------------------
        # Step 3c: sigma^2 from eq.(42)–(44)
        #----------------------------------------------------------
        residuals = R_corrected - X@beta_draw
        s2 = np.sum(residuals**2)
        shape_post = a_sigma_0 + n_f/2
        scale_post = b_sigma_0 + 0.5*s2
        sigma2_draw = invgamma.rvs(a=shape_post, scale=scale_post)
        sigma_curr  = np.sqrt(sigma2_draw)

        #----------------------------------------------------------
        # Step 3d: rho from eq.(45)–(59) (placeholder)
        #          We'll do a random walk or you can do the full bivariate approach
        #----------------------------------------------------------
        rho_curr = rho_curr + 0.01*rng.standard_normal()
        rho_curr = np.clip(rho_curr, -0.9999, 0.9999)

        #----------------------------------------------------------
        # Step 3e: Now sample (lambda, mu^I, sigma^I) from eq.(80),(82),(83).
        #          We'll do a standard set of conjugate updates:
        #
        #   eq.(80):  lambda ~ Beta( alpha_lambda0 + Σ_k J(k),  beta_lambda0 + n-Σ_k J(k) )
        #             But your eq.(79) has λ(k) as a fraction. We interpret that as:
        #             "Total # of jumps" = sum( J(k) ), or we do sum(lambda_est).
        #
        #   eq.(82):  mu^I ~ Normal( ..., ... ) 
        #   eq.(83):  sigma^I^2 ~ InvGamma(...)
        #
        # For demonstration, we treat "effective count of jumps" = sum( lambda_est(k)*n_particles ) or
        #   sum(lambda_est(k)) * N? We'll do sum(lambda_est(k)) *some factor. It's approximate.
        #----------------------------------------------------------
        # Weighted total jumps across time:
        total_lambda = np.sum(lambda_est)  # sum of eq.(79) over k
        # # of time steps is n. Let "effective" = total_lambda * n_particles (or just do partial).
        alpha_post = alpha_lambda0 + total_lambda*n_particles
        beta_post  = beta_lambda0  + (n - total_lambda)*n_particles
        # draw new lambda
        # clamp
        if alpha_post<=0: alpha_post=1e-3
        if beta_post <=0: beta_post =1e-3
        lambda_curr = beta_dist.rvs(a=alpha_post, b=beta_post, random_state=rng)
        # eq.(82),(83) => we treat {Z_est(k)} as "data" for mu^I, sigma^I
        #  Suppose sumZ = Σ (Z_est(k)*n_particles*?), sumZ^2 likewise. We'll do a naive approach.
        Z_array = Z_est  # shape(n,)
        # keep only those k s.t. λ_est(k)>some threshold if you prefer. We'll keep them all.
        # sample sigma^I^2 from IG
        sum_z2 = np.sum(Z_array**2)* (n_particles)  # approximate
        aI_post = aI0 + 0.5*n  # # of obs ~ n
        bI_post = bI0 + 0.5*sum_z2
        sigI2_draw = invgamma.rvs(a=aI_post, scale=bI_post, random_state=rng)
        sigI_curr = np.sqrt(sigI2_draw)
        # now mu^I from Normal( muI0, (tauI0 * sigma^I)^{-1} ) posterior
        # typically posterior mean = [ (muI0_0*tauI0_0) + sum(Z_est)/some_variance ] / [ tauI0_0 + #obs ]
        sumZ = np.sum(Z_array)*n_particles
        tauI_post = tauI0_0 + n
        muI_post_mean = (tauI0_0*muI0_0 + sumZ)/(tauI_post)
        muI_post_var  = sigI2_draw/tauI_post  # scaled by sigma^I^2 if that’s eq.(82)
        muI_curr = rng.normal(loc=muI_post_mean, scale=np.sqrt(muI_post_var))

        #------------------------------------------------------------------
        # Save draws
        #------------------------------------------------------------------
        mu_chain[i]     = mu_curr
        kappa_chain[i]  = kappa_curr
        theta_chain[i]  = theta_curr
        sigma_chain[i]  = sigma_curr
        rho_chain[i]    = rho_curr
        lambda_chain[i] = lambda_curr
        muI_chain[i]    = muI_curr
        sigI_chain[i]   = sigI_curr

    # final estimates as means:
    results = {
        'mu':     np.mean(mu_chain),
        'kappa':  np.mean(kappa_chain),
        'theta':  np.mean(theta_chain),
        'sigma':  np.mean(sigma_chain),
        'rho':    np.mean(rho_chain),
        'lambda': np.mean(lambda_chain),
        'muI':    np.mean(muI_chain),
        'sigI':   np.mean(sigI_chain)
    }
    return results

###############################################################################
#                  Example Usage / Testing
###############################################################################
if __name__=="__main__":
    # Synthetic data
    np.random.seed(42)
    n_samples   = 30
    delta_t     = 1/252
    maturity    = 1
    n_steps     = int(maturity/delta_t)
    prices      = np.zeros(n_steps+1)
    prices[0]   = 100

    mu_true     = 0.05
    sigma_true  = 0.3
    for k in range(1,n_steps+1):
        dW = np.random.normal(0, np.sqrt(delta_t))
        prices[k] = prices[k-1]*np.exp((mu_true-0.5*sigma_true**2)*delta_t + sigma_true*dW)

    out = heston_with_jumps_estimation(
        n_samples=n_samples, delta_t=delta_t, maturity=maturity,
        prices=prices,
        n_particles=200,
        mu_0=0.05, kappa_0=2.0, theta_0=0.04, sigma_0=0.3, rho_0=-0.4,
        lambda_0=0.2, muI_0=-1.0, sigmaI_0=0.4,
        mu_eta_0=1.00125, tau_eta_0=0.001,
        mu_beta_0=[35e-6,0.988], lambda_beta_0=[[10,0],[0,5]],
        a_sigma_0=149, b_sigma_0=0.025,
        mu_psi_0=-0.45, tau_psi_0=0.3,
        a_omega_0=1.03, b_omega_0=0.05,
        # new jump priors
        alpha_lambda0=1.0, beta_lambda0=1.0, # a Beta(1,1) ~ uniform
        muI0_0=-1.0, tauI0_0=1.0,
        aI0=2.0, bI0=0.5
    )

    print("==== Final MCMC Averages ====")
    for k,v in out.items():
        print(f"{k}: {v:.4f}")
