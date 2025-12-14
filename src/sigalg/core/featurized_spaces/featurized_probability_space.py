from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

from ..base.sample_space import SampleSpaceMethods
from ..probability_measures.probability_measure import ProbabilityMeasureMethods
from ..sigma_algebras.sigma_algebra import SigmaAlgebraMethods
from .feature_embedding import FeatureEmbeddingMethods

if TYPE_CHECKING:
    from ..base.probability_space import ProbabilitySpace
    from ..base.sample_space import SampleSpace
    from ..probability_measures import ProbabilityMeasure
    from ..random_objects import RandomVariable
    from ..sigma_algebras import SigmaAlgebra
    from .feature_embedding import FeatureEmbedding


class FeaturizedProbabilitySpace(
    FeatureEmbeddingMethods,
    SampleSpaceMethods,
    SigmaAlgebraMethods,
    ProbabilityMeasureMethods,
):

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        sample_space: SampleSpace,
        feature_embedding: FeatureEmbedding,
        sigma_algebra: SigmaAlgebra | None = None,
        probability_measure: ProbabilityMeasure | None = None,
    ):
        from ..probability_measures import ProbabilityMeasure
        from ..sigma_algebras import SigmaAlgebra

        self._validate_parameters(
            sample_space,
            feature_embedding,
            sigma_algebra,
            probability_measure,
        )

        self.sample_space = sample_space
        self._feature_embedding = feature_embedding
        if sigma_algebra is None:
            sigma_algebra = SigmaAlgebra.power_set(sample_space)
        self._sigma_algebra = sigma_algebra
        if probability_measure is None:
            probability_measure = ProbabilityMeasure.uniform(sample_space)
        self._probability_measure = probability_measure

    # --------------------- properties --------------------- #

    @property
    def sigma_algebra(self) -> SigmaAlgebra:
        return self._sigma_algebra

    @sigma_algebra.setter
    def sigma_algebra(self, sigma_algebra: SigmaAlgebra) -> None:
        self._validate_parameters(
            self.sample_space,
            self.feature_embedding,
            sigma_algebra,
            self.probability_measure,
        )
        self._sigma_algebra = sigma_algebra
        self.probability_space.sigma_algebra = sigma_algebra

    @property
    def probability_measure(self) -> ProbabilityMeasure:
        return self._probability_measure

    @probability_measure.setter
    def probability_measure(self, probability_measure: ProbabilityMeasure) -> None:
        self._validate_parameters(
            self.sample_space,
            self.feature_embedding,
            self.sigma_algebra,
            probability_measure,
        )
        self._probability_measure = probability_measure
        self.probability_space.probability_measure = probability_measure

    @property
    def feature_embedding(self) -> FeatureEmbedding:
        return self._feature_embedding

    @feature_embedding.setter
    def feature_embedding(self, feature_embedding: FeatureEmbedding) -> None:
        self._validate_parameters(
            self.sample_space,
            feature_embedding,
            self.sigma_algebra,
            self.probability_measure,
        )
        self._feature_embedding = feature_embedding

    @property
    def probability_space(self) -> ProbabilitySpace:
        from ..base import ProbabilitySpace

        if not hasattr(self, "_probability_space"):
            self._probability_space = ProbabilitySpace(
                sample_space=self.sample_space,
                sigma_algebra=self.sigma_algebra,
                probability_measure=self.probability_measure,
            )
        return self._probability_space

    # --------------------- data access methods --------------------- #

    def get_feature_rv(self, feature_index: Hashable) -> RandomVariable:
        from ..random_objects import RandomVariable

        values = self.feature_embedding.values[feature_index]
        name = values.name
        rv = RandomVariable(values=values, name=name)
        rv.add_probability_measure_to_domain(self.probability_measure)
        return rv

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return (
            f"FeaturizedProbabilitySpace("
            f"sample_space={self.sample_space.name}, "
            f"sigma_algebra={self.sigma_algebra.name}, "
            f"probability_measure={self.probability_measure.name}, "
            f"feature_embedding={self.feature_embedding.name})"
        )

    def __str__(self) -> str:
        header = (
            f"Featurized probability space ("
            f"{self.sample_space.name}, "
            f"{self.sigma_algebra.name}, "
            f"{self.probability_measure.name}, "
            f"{self.feature_embedding.name})"
        )
        separator = "=" * len(header)
        return (
            header
            + "\n"
            + separator
            + "\n\n* "
            + repr(self.sample_space)
            + "\n\n* "
            + repr(self.sigma_algebra)
            + "\n\n* "
            + repr(self.probability_measure)
            + "\n\n* "
            + repr(self.feature_embedding)
        )

    # --------------------- equality --------------------- #

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeaturizedProbabilitySpace):
            return False
        return (
            self.sample_space == other.sample_space
            and self.sigma_algebra == other.sigma_algebra
            and self.probability_measure == other.probability_measure
            and self.feature_embedding == other.feature_embedding
        )

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        sample_space: SampleSpace,
        feature_embedding: FeatureEmbedding,
        sigma_algebra: SigmaAlgebra | None = None,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> None:
        from ..base import SampleSpace
        from ..probability_measures import ProbabilityMeasure
        from ..sigma_algebras import SigmaAlgebra
        from .feature_embedding import FeatureEmbedding

        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")
        if not isinstance(feature_embedding, FeatureEmbedding):
            raise TypeError("feature_embedding must be a FeatureEmbedding instance.")
        if sigma_algebra is not None and not isinstance(sigma_algebra, SigmaAlgebra):
            raise TypeError("sigma_algebra must be a SigmaAlgebra instance.")
        if sigma_algebra is not None and sigma_algebra.sample_space != sample_space:
            raise ValueError("sigma_algebra must be defined on the given sample_space.")
        if probability_measure is not None and not isinstance(
            probability_measure, ProbabilityMeasure
        ):
            raise TypeError(
                "probability_measure must be a ProbabilityMeasure instance."
            )
        if (
            probability_measure is not None
            and probability_measure.sample_space != sample_space
        ):
            raise ValueError(
                "probability_measure must be defined on the given sample_space."
            )
        if not feature_embedding.values.index.equals(sample_space.values):
            raise ValueError(
                "feature_embedding must be defined on the given sample_space."
            )
