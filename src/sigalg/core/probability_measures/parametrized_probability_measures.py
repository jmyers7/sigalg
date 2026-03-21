from __future__ import annotations

from collections.abc import Callable, Hashable
from numbers import Real
from typing import TYPE_CHECKING

from ..base.sample_space import SampleSpace

if TYPE_CHECKING:
    from .probability_measure import ProbabilityMeasure


class ParametrizedProbabilityMeasures:
    """Pass."""

    def __init__(
        self,
        sample_space: SampleSpace,
        parametrization: Callable[[tuple], dict[Hashable, Real]],
        name: Hashable | None = "P",
    ) -> None:
        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be an instance of SampleSpace.")
        if not callable(parametrization):
            raise TypeError("parametrization must be a callable function.")
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("name must be a hashable object or None.")

        self.sample_space = sample_space
        self.parametrization = parametrization
        self.name = name

    def __call__(self, theta: tuple) -> ProbabilityMeasure:
        """Pass."""
        from .probability_measure import ProbabilityMeasure

        outputs = self.parametrization(theta)
        if self.name is not None:
            name = f"{self.name}_{theta}" if self.name is not None else None

        probability_measure = ProbabilityMeasure(
            sample_space=self.sample_space, name=name
        ).from_dict(outputs)

        return probability_measure
