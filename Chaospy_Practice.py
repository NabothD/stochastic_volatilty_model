import chaospy as cp
import numpy as np
import matplotlib.pyplot as plt
import tkinter

#lets use runga kutta to solve

def model(x, u0, c0, c1, c2):
    def c(x):
        if x < 0.5:
            return c0
        elif 0.5 <= x < 0.7:
            return c1
        else:
            return c2

    N = len(x)
    u = np.zeros(N)

    u[0] = u0
    for n in range(N - 1):
        dx = x[n + 1] - x[n]
        K1 = -dx * u[n] * c(x[n])
        K2 = -dx * u[n] + K1 / 2 * c(x[n] + dx / 2)
        u[n + 1] = u[n] + K1 + K2

    return u

# For polynomial Chaos
c0 = cp.Normal(0.5, 0.15)
c1 = cp.Uniform(0.5, 2.5)
c2 = cp.Uniform(0.03, 0.07)

distribution = cp.J(c0, c1, c2)

u0 = 1  
nodes, weights = cp.generate_quadrature(order=3, dist=distribution, rule="Gaussian")
# Define the x array and evaluate the model at sample points
x = np.linspace(0, 1, 101)
samples = [model(x, u0, node[0], node[1], node[2])
           for node in nodes.T]

# Generate orthogonal polynomials for polynomial chaos expansion
polynomials = cp.orth_ttr(order=3, dist=distribution)

model_approx = cp.fit_quadrature(
    polynomials, nodes, weights, samples
)

# Calculate statistical information about the model response
mean = cp.E(model_approx, distribution)
deviation = cp.Std(model_approx, distribution)


# Use the Agg backend to avoid GUI requirements
plt.switch_backend('Agg')

# Assuming `x`, `mean`, and `deviation` are already defined
# Create the plot as before
plt.plot(x, mean, label='Mean', color='blue')
plt.fill_between(x, mean - deviation, mean + deviation, color='lightblue', alpha=0.5, label='Mean ± Deviation')

# Add labels and legend
plt.xlabel('x')
plt.ylabel('u')
plt.title('Plot of u against x with Mean and Deviation')
plt.legend()
plt.show()

# Save the plot to a file (e.g., 'plot.png')
plt.savefig('plot.png', format='png')

print("Plot saved as 'plot.png'")


