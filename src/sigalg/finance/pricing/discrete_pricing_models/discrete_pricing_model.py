"""Marker class for discrete pricing models."""

from ....processes.base.stochastic_process import StochasticProcess


class DiscretePricingModel(StochasticProcess):
    """Marker class for discrete pricing models."""

    _repr_name = "DiscretePricingModel"
    _str_name = "Discrete pricing model"
