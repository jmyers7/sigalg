from collections.abc import Hashable
from typing import TYPE_CHECKING

import pandas as pd

from ..spaces import ProbabilitySpace, ProbabilitySpaceMethods
from .featurized_sample_space import FeaturizedSampleSpace, FeaturizedSampleSpaceMethods

if TYPE_CHECKING:
    from ..probability_measures import ProbabilityMeasure
    from ..random_objects import RandomVariable
    from ..sigma_algebras import SigmaAlgebra
    from ..spaces import SampleSpace


class FeaturizedProbabilitySpace(ProbabilitySpaceMethods, FeaturizedSampleSpaceMethods):

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        probability_space: ProbabilitySpace,
        fss: FeaturizedSampleSpace,
        name: str = None,
    ):
        self._validate_parameters(probability_space, fss)
        self._probability_space = probability_space
        self._sample_space = probability_space.sample_space
        self._sigma_algebra = probability_space.sigma_algebra
        self._probability_measure = probability_space.probability_measure
        self._featurized_sample_space = fss
        self._values = fss.values

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

    @property
    def values(self) -> pd.DataFrame:
        return self.featurized_sample_space.values

    # --------------------- data access methods --------------------- #

    def get_feature_rv(self, feature_index: Hashable) -> RandomVariable:
        from ..random_objects import RandomVariable

        values = self.featurized_sample_space.values[feature_index]
        name = values.name
        return RandomVariable.from_values(
            values=values, probability_space=self.probability_space, name=name
        )

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        header = (
            f"Featurized probability space ("
            f"{self.sample_space.name}, "
            f"{self.sigma_algebra.name}, "
            f"{self.probability_measure.name}, "
            f"{self.featurized_sample_space.name})"
        )
        separator = "=" * len(header)
        return (
            f"{header}\n"
            f"{separator}\n\n"
            f"* Sample space {self.sample_space.name}:\n"
            f"{self.sample_space.values.to_list()}\n\n"
            f"* Sigma algebra {self.sigma_algebra.name}:\n"
            f"{self.sigma_algebra.values.to_frame()}\n\n"
            f"* Probability measure {self.probability_measure.name}:\n"
            f"{self.probability_measure.values.to_frame()}\n\n"
            f"* Feature embedding {self.featurized_sample_space.name}:\n"
            f"{self.featurized_sample_space.values}"
        )

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        probability_space: ProbabilitySpace,
        fss: FeaturizedSampleSpace,
    ) -> None:
        if not isinstance(probability_space, ProbabilitySpace):
            raise TypeError("probability_space must be a ProbabilitySpace instance.")
        if not isinstance(fss, FeaturizedSampleSpace):
            raise TypeError(
                "featurized_sample_space must be a FeaturizedSampleSpace instance."
            )
        if probability_space.sample_space != fss.sample_space:
            raise ValueError(
                "The sample_space of probability_space and featurized_sample_space must be the same."
            )
