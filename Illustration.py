import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm

# --- Set up the problem ---
# True parameter value (unknown to the estimator)
true_theta = 1.0

# Likelihood: assume observations come from N(theta, likelihood_sigma^2)
likelihood_sigma = 0.5

# Prior: initially, we believe theta is around prior_mu with uncertainty prior_sigma
prior_mu = 0.0
prior_sigma = 1.0

# Generate synthetic data from the true distribution
np.random.seed(42)
data = np.random.normal(loc=true_theta, scale=likelihood_sigma, size=10)

# Define a range for theta values (for plotting densities)
theta_vals = np.linspace(-1, 3, 400)

# Compute the prior density (this remains fixed)
prior_pdf = norm.pdf(theta_vals, loc=prior_mu, scale=prior_sigma)

# --- Set up the animation plot ---
fig, ax = plt.subplots(figsize=(8,6))
line_prior, = ax.plot(theta_vals, prior_pdf, label="Prior", color="blue", lw=2)
line_posterior, = ax.plot([], [], label="Posterior", color="red", lw=2)
line_likelihood, = ax.plot([], [], label="Likelihood (current data)", color="green", lw=2)
title_text = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center", fontsize=12)

ax.set_xlabel(r'$\theta$')
ax.set_ylabel('Density')
ax.set_title("Bayesian Updating: Prior to Posterior")
ax.legend()

# --- Functions to update the posterior sequentially ---
# We use the conjugate update formulas for a Normal likelihood with known variance:
#   Posterior precision: 1/σ_p^2 = 1/σ_prior^2 + n/σ_lik^2
#   Posterior mean: μ_p = (μ_prior/σ_prior^2 + (sum(x_i))/σ_lik^2) / (1/σ_prior^2 + n/σ_lik^2)
cumulative_sum = 0.0
n_points = 0

def init():
    line_posterior.set_data([], [])
    line_likelihood.set_data([], [])
    title_text.set_text("")
    return line_posterior, line_likelihood, title_text

def update(frame):
    global cumulative_sum, n_points
    n_points += 1
    # Get the new data point
    x = data[frame]
    cumulative_sum += x
    
    # Compute posterior parameters
    prec_prior = 1.0 / (prior_sigma ** 2)
    prec_lik = n_points / (likelihood_sigma ** 2)
    post_prec = prec_prior + prec_lik
    post_sigma = np.sqrt(1.0 / post_prec)
    post_mu = (prior_mu * prec_prior + cumulative_sum / (likelihood_sigma ** 2)) / post_prec
    
    # Compute the updated posterior density (red curve)
    post_pdf = norm.pdf(theta_vals, loc=post_mu, scale=post_sigma)
    
    # Compute the likelihood for the current data point (green curve)
    lik_pdf = norm.pdf(x, loc=theta_vals, scale=likelihood_sigma)
    
    # Update the plot lines
    line_posterior.set_data(theta_vals, post_pdf)
    line_likelihood.set_data(theta_vals, lik_pdf)
    title_text.set_text(f"After {n_points} data points: x = {x:.2f}, Posterior: μ = {post_mu:.2f}, σ = {post_sigma:.2f}")
    return line_posterior, line_likelihood, title_text

# Create the animation: each frame uses one new data point.
anim = FuncAnimation(fig, update, frames=len(data), init_func=init, interval=1500, blit=True, repeat=False)

plt.show()
