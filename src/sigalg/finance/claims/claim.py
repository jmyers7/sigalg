"""Later."""

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Real

from sigalg.core.random_objects.random_variable import RandomVariable
from sigalg.processes.base.stochastic_process import StochasticProcess


class Claim(ABC):
    """Abstract base class for various types of contingent claims."""

    @property
    @abstractmethod
    def payoff(self) -> RandomVariable:
        """Return the payoff of the claim as a random variable."""
        pass

    @abstractmethod
    def replicating_portfolio(
        self, **kwargs
    ) -> tuple[StochasticProcess, StochasticProcess, StochasticProcess, Real]:
        """Return the replicating portfolio for the claim."""
        pass
