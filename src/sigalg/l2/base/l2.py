from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.base.probability_space import ProbabilitySpace
    from ...core.base.sample_space import SampleSpace
    from ...core.probability_measures.probability_measure import ProbabilityMeasure
    from ...core.random_objects.random_variable import RandomVariable
    from ...core.sigma_algebras.sigma_algebra import SigmaAlgebra


class L2:

    # --------------------- constructor --------------------- #

    def __init__(self, probability_space: ProbabilitySpace, name: str = "H") -> None:
        self._validate_parameters(probability_space=probability_space, name=name)
        self._probability_space = probability_space
        self._sample_space = probability_space.sample_space
        self._sigma_algebra = probability_space.sigma_algebra
        self._probability_measure = probability_space.probability_measure
        self._name = name

    # --------------------- properties --------------------- #

    @property
    def probability_space(self) -> ProbabilitySpace:
        return self._probability_space

    @property
    def sample_space(self) -> SampleSpace:
        return self._sample_space

    @property
    def sigma_algebra(self) -> SigmaAlgebra:
        return self._sigma_algebra

    @sigma_algebra.setter
    def sigma_algebra(self, sigma_algebra: SigmaAlgebra) -> None:
        self._probability_space.sigma_algebra = sigma_algebra
        self._sigma_algebra = sigma_algebra

    @property
    def probability_measure(self) -> ProbabilityMeasure:
        return self._probability_measure

    @probability_measure.setter
    def probability_measure(self, probability_measure: ProbabilityMeasure) -> None:
        self._probability_space.probability_measure = probability_measure
        self._probability_measure = probability_measure

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._validate_parameters(probability_space=self._probability_space, name=name)
        self._name = name

    # --------------------- Hilbert space methods --------------------- #

    def inner(self, X: RandomVariable, Y: RandomVariable) -> Real:
        from ..projections.expectations import expectation

        return expectation(X * Y)

    def norm(self, X: RandomVariable) -> Real:
        from ..projections.expectations import expectation

        return expectation(X * X) ** 0.5

    def distance(self, X: RandomVariable, Y: RandomVariable) -> Real:
        return self.inner((X - Y), (X - Y)) ** 0.5

    # --------------------- validation methods --------------------- #

    def _validate_parameters(
        self, probability_space: ProbabilitySpace, name: str
    ) -> None:
        from ...core.base.probability_space import ProbabilitySpace

        if not isinstance(probability_space, ProbabilitySpace):
            raise TypeError("probability_space must be a ProbabilitySpace instance.")
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
