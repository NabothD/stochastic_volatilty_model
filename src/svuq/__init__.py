"""svuq: uncertainty quantification for stochastic-volatility models.

Bayesian (particle filter + Gibbs) estimation of Heston and Bates
parameters from asset prices, vectorised option pricing, propagation of
the posterior into option prices, and online volatility forecasting.
"""

from . import plotting
from .estimation import Estimator, GibbsSampler, PosteriorResult, SamplerConfig
from .filtering import FilterResult, ParticleFilter
from .forecasting import Nowcast, VolatilityForecaster
from .models import BatesParams, HestonParams, SimulatedPath, make_params, simulate
from .pricing import (
                      FourierPricer,
                      bates_call,
                      black_scholes_call,
                      heston_call,
                      implied_volatility,
                      intrinsic_value,
)
from .priors import Priors
from .uq import OptionSpec, PCESurrogate, PosteriorPricer, PriceDistribution, posterior_draws

__all__ = [
    "BatesParams", "Estimator", "FilterResult", "FourierPricer", "GibbsSampler",
    "HestonParams", "Nowcast", "OptionSpec", "ParticleFilter", "PCESurrogate",
    "PosteriorPricer", "PosteriorResult", "PriceDistribution", "Priors", "SamplerConfig",
    "SimulatedPath", "VolatilityForecaster", "bates_call", "black_scholes_call",
    "heston_call", "implied_volatility", "intrinsic_value", "make_params",
    "plotting", "posterior_draws", "simulate",
]
__version__ = "2.0.0"
