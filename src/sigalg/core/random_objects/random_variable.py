"""Class for modeling random variables.

Classes
-------
RandomVariable
    A class representing a random variable, which is a 1-dimensional random vector.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

from .random_vector import RandomVector

if TYPE_CHECKING:
    from ..base.event import Event
    from ..base.sample_space import SampleSpace
    from .random_vector import RandomVector


# TODO: Update docstrings
class RandomVariable(RandomVector):
    """A class representing a random variable, which is a 1-dimensional random vector."""

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        domain: SampleSpace | None = None,
        name: Hashable | None = "X",
    ) -> None:
        super().__init__(domain=domain, name=name)

    def from_randint(
        self,
        low: int,
        high: int,
        random_state: int | None = None,
    ) -> RandomVariable:
        """Generate a random variable with integer outputs uniformly sampled from the range [low, high).

        Parameters
        ----------
        low : int
            The lower bound (inclusive) of the random integers.
        high : int
            The upper bound (exclusive) of the random integers.
        random_state : int | None, default=None
            An optional seed for the random number generator to ensure reproducibility. If `None`, the random number generator is not seeded.

        Returns
        -------
        self : RandomVariable
            A random variable with integer outputs uniformly sampled from the range [low, high).
        """
        return super().from_randint(
            low=low, high=high, dim=1, random_state=random_state
        )

    def from_randnorm(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        random_state: int | None = None,
    ) -> RandomVariable:
        """Generate a random variable with outputs sampled from a normal distribution.

        Parameters
        ----------
        loc : float, default=0.0
            The mean (center) of the normal distribution.
        scale : float, default=1.0
            The standard deviation (spread or width) of the normal distribution.
        random_state : int | None, default=None
            An optional seed for the random number generator to ensure reproducibility. If `None`, the random number generator is not seeded.

        Returns
        -------
        self : RandomVariable
            A random variable with outputs sampled from a normal distribution.
        """
        return super().from_randnorm(
            loc=loc, scale=scale, dim=1, random_state=random_state
        )

    # --------------------- factory methods --------------------- #

    # TODO: Update docstrings
    @classmethod
    def indicator_of(cls, event: Event) -> RandomVariable:
        """Create the indicator random variable of a given event.

        Parameters
        ----------
        event : Event
            The event for which the indicator random variable is to be created.

        Returns
        -------
        indicator_rv : RandomVariable
            The indicator random variable of the given event.
        """
        name = f"I_{event.name}" if event.name is not None else "indicator"

        outputs = {
            outcome: 1 if outcome in event else 0 for outcome in event.sample_space
        }
        return cls(domain=event.sample_space, name=name).from_dict(outputs)

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Get the string representation of the random variable.

        Returns
        -------
        repr_str : str
            The string representation of the random variable.
        """
        if self.dimension == 1:
            data = self.data.to_frame()
            data.columns = [self.name] if self.name is not None else ["value"]
        else:
            data = self.data
        if self.name is None:
            return f"Random variable:\n{data}"
        else:
            return f"Random variable '{self.name}':\n{data}"
