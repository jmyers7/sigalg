from collections.abc import Hashable, Mapping  # noqa: D100

from ..base.sample_space import SampleSpace
from .random_vector import RandomVector


class RandomVariable(RandomVector):
    """A class representing a random variable, which is a 1-dimensional random vector."""

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        outputs: Mapping[Hashable, Hashable],
        domain: SampleSpace,
        name: Hashable | None = "X",
    ) -> None:
        super().__init__(outputs=outputs, domain=domain, name=name)
        if self.dimension != 1:
            raise ValueError("RandomVariable must be 1-dimensional.")

    # --------------------- Representation --------------------- #

    def __repr__(self) -> str:
        """Get the string representation of the random variable.

        Returns
        -------
        repr_str : str
            The string representation of the random variable.
        """
        if self.dimension == 1:
            data = self.data.to_frame()
            data.columns = [self.name]
        else:
            data = self.data
        return f"Random variable '{self.name}':\n{data}"
