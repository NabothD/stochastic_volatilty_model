import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------
# 1. Setup: define some "particle" weights
# ----------------------------------------
# Suppose we have 8 particles with these (normalized) weights:
weights = np.array([0.05, 0.15, 0.10, 0.20, 0.05, 0.25, 0.10, 0.10])
weights /= weights.sum()  # ensure they sum to 1

N = len(weights)
particles = np.arange(1, N+1)  # indices 1..N (just for plotting)

# ----------------------------------------
# 2. Compute the step (empirical) CDF
# ----------------------------------------
# This is the usual ECDF: at each particle i, the CDF jumps by weights[i].
ecdf = np.cumsum(weights)

# For plotting a step function, we'll use "step" in matplotlib
# but we want to replicate the typical "right" or "post" style step.
ecdf_x = np.concatenate(([particles[0]], particles))
ecdf_y = np.concatenate(([0], ecdf))

# ----------------------------------------
# 3. Compute a "connected" or "linear" CDF
# ----------------------------------------
# In the screenshot, the formula basically linearly connects
# the jumps between consecutive particles.
# We'll create a piecewise linear function that goes from
# (particle[i], ecdf[i-1]) to (particle[i], ecdf[i]) etc.

# We'll do it manually by sampling a few points between each particle index.
connected_x = []
connected_y = []
for i in range(N):
    if i == 0:
        # start at x=particles[0], y=0
        connected_x.append(particles[i])
        connected_y.append(0)
    else:
        # connect from (particles[i-1], ecdf[i-1]) to (particles[i], ecdf[i])
        connected_x.append(particles[i])
        connected_y.append(ecdf[i])

# Now we have a piecewise-linear version. Alternatively, you might
# define a function and do an even finer interpolation.

# ----------------------------------------
# 4. Generate some uniform draws for resampling illustration
# ----------------------------------------
# We'll show how random u in [0,1] maps onto the connected CDF
num_draws = 5
u_values = np.random.rand(num_draws)
u_values.sort()  # sort them just for a neat plot

# We'll find each u's corresponding "particle" by inverting the connected CDF.
# Since our "connected" version is piecewise linear, we can do a simple search:
resampled_particles = []
for u in u_values:
    # Find the first place where ecdf >= u
    idx = np.searchsorted(ecdf, u)
    # Particle index is 'idx' (0-based), but we stored them 1-based
    # We do a simple approach with the step function. 
    # If you want a purely linear approach, you'd interpolate in the small segment.
    resampled_particles.append(particles[idx])

# ----------------------------------------
# 5. Plot everything
# ----------------------------------------
plt.figure(figsize=(8, 5))

# Plot the step CDF
plt.step(ecdf_x, ecdf_y, where='post', label='Empirical CDF (step)', color='purple')

# Plot the connected CDF
plt.plot(connected_x, connected_y, label='Connected CDF (linear)', color='blue', linestyle='--', marker='o')

# Plot uniform draws and how they map to new particles
# We'll show them on the same vertical axis (probability),
# placing them at x=0 or negative just for illustration.
for i, u in enumerate(u_values):
    plt.plot([0, resampled_particles[i]], [u, u], 'r--', alpha=0.7)  # horizontal line
    plt.plot([resampled_particles[i], resampled_particles[i]], [0, u], 'r--', alpha=0.7)  # vertical line
    plt.scatter([0], [u], color='red', zorder=3)  # the uniform sample
    plt.scatter([resampled_particles[i]], [u], color='red', zorder=3)  # the mapped point

# Aesthetic details
plt.xlabel('Particle index (or possible states)')
plt.ylabel('Probability')
plt.title('Step vs. Connected CDF with Resampling Illustration')
plt.legend()
plt.grid(True)
plt.xlim(left=-0.5, right=N+1.5)
plt.ylim(0, 1.05)
plt.show()
