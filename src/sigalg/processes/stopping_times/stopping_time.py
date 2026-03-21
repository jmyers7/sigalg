from __future__ import annotations

from collections.abc import Hashable, Mapping
from math import inf

from ...core.random_objects.random_variable import RandomVariable
from ...core.sigma_algebras.filtration import Filtration


class StoppingTime(RandomVariable):
    """Pass."""

    def __init__(
        self,
        filtration: Filtration,
        name: Hashable | None = "tau",
    ) -> None:
        if not isinstance(filtration, Filtration):
            raise TypeError("filtration must be an instance of Filtration")

        self.filtration = filtration
        self.time = filtration.time

        super().__init__(domain=filtration.sample_space, name=name)

    # --------------------- constructors --------------------- #

    def from_dict(self, outputs: Mapping[Hashable, Hashable]) -> StoppingTime:
        """Pass."""
        if not set(outputs.values()) - {inf} <= set(self.time.data):
            raise ValueError(
                "The range of the stopping time must be in the time index of the stochastic process."
            )

        self = super().from_dict(outputs=outputs)

        for t, event in self.sigma_algebra.atom_id_to_event.items():
            if t == inf:
                check_alg = self.filtration.finest
            else:
                check_alg = self.filtration[t]
            if event not in check_alg:
                raise TypeError(
                    "One of the level sets of the stopping time is not measurable wrt "
                    "the appropriate sigma algebra in the filtration"
                )

        return self

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Get the string representation of the stopping time.

        Returns
        -------
        repr_str : str
            The string representation of the stopping time.
        """
        if self.dimension == 1:
            data = self.data.to_frame()
            data.columns = [self.name] if self.name is not None else ["value"]
        else:
            data = self.data
        if self.name is None:
            return f"Stopping time:\n{data}"
        else:
            return f"Stopping time '{self.name}':\n{data}"
