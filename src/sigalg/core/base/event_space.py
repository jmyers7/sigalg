from typing import TYPE_CHECKING

from ..sigma_algebras import SigmaAlgebraMethods
from .sample_space import SampleSpaceMethods

if TYPE_CHECKING:
    from ..probability_measures import ProbabilityMeasure
    from ..sigma_algebras import SigmaAlgebra
    from .probability_space import ProbabilitySpace
    from .sample_space import SampleSpace


class EventSpace(SampleSpaceMethods, SigmaAlgebraMethods):

    # --------------------- constructor --------------------- #

    def __init__(
        self, sample_space: SampleSpace, sigma_algebra: SigmaAlgebra | None = None
    ):
        from ..sigma_algebras import SigmaAlgebra

        self._validate_parameters(sample_space, sigma_algebra)
        self._sample_space = sample_space
        if sigma_algebra is None:
            sigma_algebra = SigmaAlgebra.power_set(sample_space)
        self._sigma_algebra = sigma_algebra

    # --------------------- properties --------------------- #

    @property
    def sample_space(self) -> SampleSpace:
        return self._sample_space

    @property
    def sigma_algebra(self) -> SigmaAlgebra:
        return self._sigma_algebra

    @sigma_algebra.setter
    def sigma_algebra(self, sigma_algebra) -> None:
        self._validate_parameters(self.sample_space, sigma_algebra)
        self._sigma_algebra = sigma_algebra

    # --------------------- conversion methods --------------------- #

    def make_probability_space(
        self,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> ProbabilitySpace:
        from .probability_space import ProbabilitySpace

        return ProbabilitySpace(
            sample_space=self.sample_space,
            sigma_algebra=self.sigma_algebra,
            probability_measure=probability_measure,
        )

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return (
            f"EventSpace(sample_space={self.sample_space.name}, "
            f"sigma_algebra={self.sigma_algebra.name})"
        )

    def __str__(self) -> str:
        header = f"Event space ({self.sample_space.name}, {self.sigma_algebra.name})"
        separator = "=" * len(header)
        return (
            header
            + "\n"
            + separator
            + "\n\n* "
            + repr(self.sample_space)
            + "\n\n* "
            + repr(self.sigma_algebra)
        )

    # --------------------- equality --------------------- #

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EventSpace):
            return False
        return (
            self.sample_space == other.sample_space
            and self.sigma_algebra == other.sigma_algebra
        )

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(sample_space: SampleSpace, sigma_algebra: SigmaAlgebra):
        from ..sigma_algebras import SigmaAlgebra
        from .sample_space import SampleSpace

        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")
        if sigma_algebra is not None and not isinstance(sigma_algebra, SigmaAlgebra):
            raise TypeError("sigma_algebra must be a SigmaAlgebra instance.")
        if sigma_algebra is not None and sigma_algebra.sample_space != sample_space:
            raise ValueError(
                "sigma_algebra's sample_space must match the provided sample_space."
            )
