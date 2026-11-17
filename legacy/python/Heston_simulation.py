import numpy as np
import math
import matplotlib.pyplot as plt

def generate_heston_paths(S, T, r, kappa, theta, v_0, rho, xi, 
                          steps, Npaths):
    dt = T/steps
    size = (Npaths, steps)
    prices = np.zeros(steps)
    sigs = np.zeros(steps)
    S_t = S
    v_t = v_0
    for t in range(steps):
        WT = np.random.multivariate_normal(np.array([0,0]), 
                                           cov = np.array([[1,rho],
                                                          [rho,1]]), 
                                           size=Npaths) * np.sqrt(dt) 
        
        S_t = S_t*(np.exp( (r- 0.5*v_t)*dt+ np.sqrt(v_t) *WT[:,0] ) ) 
        v_t = np.abs(v_t + kappa*(theta-v_t)*dt + xi*np.sqrt(v_t)*WT[:,1])
        prices[t] = S_t
        sigs[t] = v_t

    
    return prices, sigs



kappa =1.1
theta = 0.04
v_0 =  0.04
sigma = 0.01
r = 0.1
S = 100
paths =1
steps = 252
T = 1
rho = -0.5


prices,sigs = generate_heston_paths(S, T, r, kappa, theta,
                                    v_0, rho, sigma, steps, paths)        
    
plt.figure(figsize=(7,6))
plt.plot(prices.T)
plt.title('Heston Price Paths Simulation')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.show()

plt.figure(figsize=(7,6))
plt.plot(np.sqrt(sigs).T)
plt.axhline(np.sqrt(theta), color='black', label=r'$\sqrt{\theta}$')
plt.title('Heston Stochastic Vol Simulation')
plt.xlabel('Time Steps')
plt.ylabel('Volatility')
plt.legend(fontsize=15)
plt.show()

# 3) Save to CSV
data = np.column_stack((prices, sigs))
np.savetxt("my_heston.csv", data, delimiter=",", header="price,variance", comments="")