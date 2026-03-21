"""Later."""

from __future__ import annotations

from ....processes.base.stochastic_process import StochasticProcess


class GeometricPricingModel(StochasticProcess):
    """Abstract base class for geometric pricing models."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enum_mode: str | None = None
