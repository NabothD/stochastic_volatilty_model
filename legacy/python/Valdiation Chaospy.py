import chaospy as cp
import numpy as np

# Define input distribution
dist = cp.Normal(0, 1)  # Standard normal distribution

# Define test function
def test_func(x):
    return x**2 + x

# Generate orthogonal polynomial basis for PCE
order = 2
poly_basis = cp.expansion.stieltjes(order, dist)  # Updated function for orthogonal basis

# Generate quadrature points and weights
nodes, weights = cp.generate_quadrature(order + 1, dist, rule="gaussian")

# Evaluate the test function at quadrature nodes
function_evals = test_func(nodes).flatten()  # Flatten ensures the shape matches the expected input

# Ensure nodes, weights, and function_evals align
nodes = np.atleast_2d(nodes)  # Ensure nodes are a 2D array

# Perform PCE
expansion = cp.fit_quadrature(poly_basis, nodes, weights, function_evals)

# Monte Carlo simulation
mc_samples = 10000  # Number of Monte Carlo samples
mc_points = dist.sample(mc_samples)  # Draw samples from the distribution
mc_evals = test_func(mc_points)  # Evaluate the test function on samples

# Analytical solution for moments
analytical_mean = 1  # E[X^2 + X] for standard normal
analytical_variance = 3  # Var[X^2 + X] for standard normal

# Validate moments using PCE
pce_mean = cp.E(expansion, dist)
pce_variance = cp.Var(expansion, dist)

# Validate moments using Monte Carlo
mc_mean = np.mean(mc_evals)
mc_variance = np.var(mc_evals)

# Print results
print("==== Analytical Results ====")
print("Analytical Mean:", analytical_mean)
print("Analytical Variance:", analytical_variance)

print("\n==== PCE Results ====")
print("PCE Mean:", pce_mean)
print("PCE Variance:", pce_variance)

print("\n==== Monte Carlo Results ====")
print("Monte Carlo Mean:", mc_mean)
print("Monte Carlo Variance:", mc_variance)
