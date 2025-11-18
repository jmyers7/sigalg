from typing import TYPE_CHECKING

from ..spaces import ProbabilitySpace, ProbabilitySpaceMethods
from .featurized_sample_space import FeaturizedSampleSpace, FeaturizedSampleSpaceMethods

if TYPE_CHECKING:
    from ..probability_measures import ProbabilityMeasure
    from ..sigma_algebras import SigmaAlgebra
    from ..spaces import SampleSpace


class FeaturizedProbabilitySpace(ProbabilitySpaceMethods, FeaturizedSampleSpaceMethods):

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        probability_space: ProbabilitySpace,
        featurized_sample_space: FeaturizedSampleSpace,
    ):
        self._probability_space = probability_space
        self._sample_space = probability_space.sample_space
        self._sigma_algebra = probability_space.sigma_algebra
        self._probability_measure = probability_space.probability_measure
        self._featurized_sample_space = featurized_sample_space

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

    @property
    def probability_measure(self) -> ProbabilityMeasure:
        return self._probability_measure

    @property
    def featurized_sample_space(self) -> FeaturizedSampleSpaceMethods:
        return self._featurized_sample_space
