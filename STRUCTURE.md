# Directory Structure Guide

## Project Organization

This document describes the recommended directory structure for the stochastic volatility model project.

### Root Level Files
- `README.md` - Project overview and quick start guide
- `requirements.txt` - Python package dependencies
- `STRUCTURE.md` - This file, documenting project layout
- `.gitignore` - Git ignore rules

### src/
Main source code directory containing all modules.

#### src/models/
Core pricing models and simulation methods.
- `black_scholes.py` - Black-Scholes option pricing with closed-form solutions
- `heston.py` - Heston stochastic volatility model
- `bates.py` - Bates model combining Heston with jump processes
- `euler_simulation.py` - Euler scheme and other numerical integration methods

#### src/calibration/
Parameter estimation and model calibration utilities.
- `traditional_calibration.py` - Optimization-based parameter calibration
- `bayesian_approach.py` - Bayesian inference for parameter estimation
- `particle_filter.py` - Particle filtering methods for state estimation

#### src/uncertainty_quantification/
Uncertainty quantification using Polynomial Chaos Expansion.
- `chaospy_heston.py` - Chaospy implementation for Heston model
- `chaospy_bates.py` - Chaospy implementation for Bates model
- `chaospy_validation.py` - Validation routines for PCE results

#### src/utils/
Utility functions and helpers.
- `data_generator.py` - Generate and load market data
- `plotting.py` - Visualization and charting utilities

### data/
Data files organized by type and purpose.

#### data/historical/
Historical market data and calibration data.

#### data/results/
Output files from simulations and model runs.

### notebooks/
Jupyter notebooks for analysis, visualization, and demonstration.
- `model_illustration.ipynb` - Illustration of basic model behavior
- `real_data_analysis.ipynb` - Analysis using real market data

### matlab/
Legacy MATLAB implementations (for reference).
- `.m files` - Original MATLAB code

### docs/
Documentation and generated figures.
- `figures/` - Model output visualizations

## File Naming Conventions

- **Python modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case`
- **Data files**: `descriptive_name.csv` or `.xlsx`
- **Notebooks**: `descriptive_name.ipynb`
- **Figures**: `model_name_description.png`

## Version Control

- Only source code and documentation are version controlled
- Data files are tracked but not committed (see `.gitignore`)
- Outputs should be regenerated from scripts
- Original MATLAB code kept for reference only
