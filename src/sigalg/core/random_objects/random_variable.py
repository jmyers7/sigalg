from collections.abc import Hashable, Mapping  # noqa: D100

from ..base.sample_space import SampleSpace
from .random_vector import RandomVector


class RandomVariable(RandomVector):
    """A class representing a random variable, which is a 1-dimensional random vector."""

    def __init__(
        self,
        outputs: Mapping[Hashable, Hashable],
        domain: SampleSpace,
        name: Hashable | None = "X",
    ) -> None:
        super().__init__(outputs=outputs, domain=domain, name=name)
        if self.dimension != 1:
            raise ValueError("RandomVariable must be 1-dimensional.")
