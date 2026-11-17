"""svuq – uncertainty quantification for stochastic-volatility models.

Bayesian (particle filter + Gibbs) estimation of Heston and Bates parameters
from asset prices, semi-analytic option pricing, and propagation of the
parameter posterior into option prices (Monte Carlo and polynomial chaos).
"""

from .models import BatesParams, HestonParams, SimulatedPath, simulate
from .priors import Priors
from .particle_filter import run_filter
from .estimation import PosteriorResult, estimate
from .pricing import (bates_call, black_scholes_call, heston_call, implied_volatility,
                      intrinsic_value)

__all__ = [
    "HestonParams", "BatesParams", "SimulatedPath", "simulate", "Priors", "run_filter",
    "estimate", "PosteriorResult", "heston_call", "bates_call", "black_scholes_call",
    "implied_volatility", "intrinsic_value",
]
__version__ = "1.0.0"
