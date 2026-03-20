"""Later."""

from __future__ import annotations

from abc import ABC, abstractmethod

from sigalg.core.random_objects.random_variable import RandomVariable


class Claim(ABC):
    """Abstract base class for various types of contingent claims."""

    def __init__(self, is_path_independent: bool):
        if not isinstance(is_path_independent, bool):
            raise TypeError("is_path_independent must be a boolean.")
        self.is_path_independent = is_path_independent

    @property
    @abstractmethod
    def payoff(self) -> RandomVariable:
        """Return the payoff of the claim as a random variable."""
        pass
