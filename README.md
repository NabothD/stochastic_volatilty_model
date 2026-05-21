# Stochastic Volatility Model

A comprehensive implementation of stochastic volatility models including Black-Scholes, Heston, and Bates models with Monte Carlo simulations and uncertainty quantification.

## Overview

This project implements various financial option pricing models with focus on:
- **Black-Scholes Model**: Classic option pricing framework
- **Heston Model**: Stochastic volatility model with mean reversion
- **Bates Model**: Heston model extended with jump processes
- **Bayesian Approach**: Parameter estimation using Bayesian methods
- **Uncertainty Quantification**: Polynomial Chaos Expansion (PCE) via Chaospy

## Project Structure

```
src/
├── models/              # Core pricing models
│   ├── black_scholes.py
│   ├── heston.py
│   ├── bates.py
│   └── euler_simulation.py
├── calibration/         # Parameter estimation and calibration
│   ├── traditional_calibration.py
│   ├── bayesian_approach.py
│   └── particle_filter.py
├── uncertainty_quantification/  # Chaospy and PCE implementations
│   ├── chaospy_heston.py
│   ├── chaospy_bates.py
│   └── chaospy_validation.py
└── utils/               # Utility functions
    ├── data_generator.py
    └── plotting.py

data/                   # Data files and results
├── historical/
└── results/

notebooks/              # Jupyter notebooks for analysis
├── model_illustration.ipynb
└── real_data_analysis.ipynb

matlab/                 # Original MATLAB implementations
└── *.m files

docs/                   # Documentation and figures
└── figures/
```

## Installation

### Requirements
- Python 3.7+
- NumPy
- SciPy
- Pandas
- Matplotlib
- QuantLib (optional, for validation)
- Chaospy (for uncertainty quantification)

### Setup
```bash
pip install -r requirements.txt
```

## Usage

### Basic Black-Scholes Option Pricing
```python
from src.models.black_scholes import black_scholes_call

price = black_scholes_call(S=100, K=100, r=0.05, sigma=0.2, T=1.0)
```

### Heston Model Simulation
```python
from src.models.heston import HestonSimulation

sim = HestonSimulation(S0=100, kappa=2.0, theta=0.05, sigma=0.3, rho=-0.7, r=0.05)
prices = sim.euler_scheme(T=1.0, num_steps=252, num_paths=10000)
```

### Bayesian Parameter Estimation
```python
from src.calibration.bayesian_approach import BayesianCalibration

calibrator = BayesianCalibration()
parameters = calibrator.estimate(option_data, market_prices)
```

## Model Details

### Black-Scholes
- Constant volatility assumption
- Closed-form solution for European options
- Baseline for comparison

### Heston Model
- Time-varying volatility with mean reversion
- Parameters: κ (mean reversion speed), θ (long-term variance), σ (vol of vol), ρ (correlation)
- Monte Carlo simulation using Euler scheme

### Bates Model
- Heston model + jump processes
- Additional jump parameters: λ (jump intensity), μ_J (jump mean), σ_J (jump std)
- Better capture of tail risks

## Uncertainty Quantification

Using Polynomial Chaos Expansion (PCE) via Chaospy to:
- Propagate parameter uncertainty through models
- Generate confidence intervals on option prices
- Perform global sensitivity analysis

## References

- Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility"
- Bates, D. S. (1996). "Jumps and Stochastic Volatility"
- Glasserman, P. (2004). "Monte Carlo Methods in Financial Engineering"

## License

This project is provided as-is for educational and research purposes.

## Author

NabothD
