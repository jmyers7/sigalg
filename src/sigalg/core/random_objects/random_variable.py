"""Random variable module."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

from .random_vector import RandomVector

if TYPE_CHECKING:
    from ..base.event import Event
    from ..base.index import Index
    from ..base.sample_space import SampleSpace
    from .random_vector import RandomVector


class RandomVariable(RandomVector):
    """A class representing a random variable, which is a 1-dimensional random vector."""

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        domain: SampleSpace | None = None,
        index: Index | None = None,
        name: Hashable | None = "X",
    ) -> None:
        super().__init__(domain=domain, vector_index=index, name=name)

    # --------------------- factory methods --------------------- #

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
