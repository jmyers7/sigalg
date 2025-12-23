"""Featurized probability spaces combining probability spaces with feature embeddings.

This module provides the `FeaturizedProbabilitySpace` class, which represents a
featurized probability space `(Omega, F, P, X)` where `(Omega, F, P)` is a probability
space and `X` is a feature embedding `X: Omega -> S` (i.e., a random vector).

Classes
-------
FeaturizedProbabilitySpace
    Represents a featurized probability space `(Omega, F, P, X)`.

Examples
--------
>>> from sigalg.core import FeaturizedProbabilitySpace, SampleSpace, RandomVector
>>> from sigalg.core import RandomVariable, ProbabilityMeasure
>>> Omega = SampleSpace(["s0", "s1"])
>>> X = RandomVector(outputs={"s0": (1, 2), "s1": (3, 4)}, domain=Omega, name="X")
>>> probs = {"s0": 0.5, "s1": 0.5}
>>> probability_measure = ProbabilityMeasure(sample_space=Omega, probabilities=probs)
>>> fps = FeaturizedProbabilitySpace(
...     sample_space=Omega, feature_embedding=X, probability_measure=probability_measure
... )
>>> fps.P("s0")
0.5
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..base.sample_space import SampleSpaceMethods
from ..probability_measures.probability_measure import ProbabilityMeasureMethods
from ..sigma_algebras.sigma_algebra import SigmaAlgebraMethods

if TYPE_CHECKING:
    from ..base.probability_space import ProbabilitySpace
    from ..base.sample_space import SampleSpace
    from ..probability_measures import ProbabilityMeasure
    from ..random_objects.random_vector import RandomVector
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra


class FeaturizedProbabilitySpace(
    SampleSpaceMethods,
    SigmaAlgebraMethods,
    ProbabilityMeasureMethods,
):
    """A featurized probability space combining probabilistic and feature structures.

    A `FeaturizedProbabilitySpace` represents the quadruple `(Omega, F, P, X)` where
    `(Omega, F, P)` is a probability space and `X: Omega -> S` is a feature embedding
    (i.e., a random vector).

    The class has attributes `sample_space`, `sigma_algebra`, `probability_measure`, and
    `feature_embedding`, and inherits methods from `SampleSpaceMethods`,
    `SigmaAlgebraMethods`, and `ProbabilityMeasureMethods`. This allows methods from the first three components to be called directly on the featurized probability space.

    Parameters
    ----------
    sample_space : SampleSpace
        The sample space `Omega` containing all possible outcomes.
    feature_embedding : RandomVector
        The feature embedding function `X: Omega -> S`.
    sigma_algebra : SigmaAlgebra, optional
        Sigma-algebra `F` defining measurable events. If `None`, a power set
        sigma-algebra is created.
    probability_measure : ProbabilityMeasure, optional
        Probability measure `P` assigning probabilities to outcomes. If `None`,
        a uniform probability measure is created.

    Raises
    ------
    TypeError
        If `sample_space` is not a `SampleSpace`, `feature_embedding` is not a
        `RandomVector`, `sigma_algebra` is not a `SigmaAlgebra`, or
        `probability_measure` is not a `ProbabilityMeasure`.
    ValueError
        If `sigma_algebra` or `probability_measure` have different sample spaces
        than the provided `sample_space`, or if `feature_embedding` is not defined
        on `sample_space`.

    Examples
    --------
    >>> from sigalg.core import FeaturizedProbabilitySpace, SampleSpace
    >>> from sigalg.core import FeatureEmbedding, RandomVariable, ProbabilityMeasure
    >>> Omega = SampleSpace(["s0", "s1", "s2"])
    >>> X = RandomVariable(outputs={"s0": 1, "s1": 3, "s2": 5}, domain=Omega, name="X")
    >>> Y = RandomVariable(outputs={"s0": 2, "s1": 4, "s2": 6}, domain=Omega, name="Y")
    >>> embedding = FeatureEmbedding(random_variables=[X, Y])
    >>> probs = {"s0": 0.5, "s1": 0.3, "s2": 0.2}
    >>> measure = ProbabilityMeasure(sample_space=Omega, probabilities=probs)
    >>> fps = FeaturizedProbabilitySpace(
    ...     sample_space=Omega,
    ...     feature_embedding=embedding,
    ...     probability_measure=measure
    ... )
    >>> fps.P("s0")
    0.5
    >>> features = fps.get_sample_features("s0")
    >>> features.values.tolist()
    [1, 2]
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        sample_space: SampleSpace,
        feature_embedding: RandomVector,
        sigma_algebra: SigmaAlgebra | None = None,
        probability_measure: ProbabilityMeasure | None = None,
    ):
        from ..probability_measures import ProbabilityMeasure
        from ..sigma_algebras import SigmaAlgebra

        self._validate_parameters(
            sample_space=sample_space,
            feature_embedding=feature_embedding,
            sigma_algebra=sigma_algebra,
            probability_measure=probability_measure,
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
        """Get the sigma-algebra defining measurable events.

        Returns
        -------
        sigma_algebra : SigmaAlgebra
            The sigma-algebra `F` of this featurized probability space.
        """
        return self._sigma_algebra

    @sigma_algebra.setter
    def sigma_algebra(self, sigma_algebra: SigmaAlgebra) -> None:
        """Set the sigma-algebra defining measurable events.

        Parameters
        ----------
        sigma_algebra : SigmaAlgebra
            New sigma-algebra `F`. Must have the same sample space as this
            featurized probability space.

        Raises
        ------
        TypeError
            If `sigma_algebra` is not a `SigmaAlgebra` instance.
        ValueError
            If `sigma_algebra`'s sample space does not match this featurized
            probability space's sample space.
        """
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
        """Get the probability measure assigning probabilities to events.

        Returns
        -------
        probability_measure : ProbabilityMeasure
            The probability measure `P` of this featurized probability space.
        """
        return self._probability_measure

    @probability_measure.setter
    def probability_measure(self, probability_measure: ProbabilityMeasure) -> None:
        """Set the probability measure assigning probabilities to events.

        Parameters
        ----------
        probability_measure : ProbabilityMeasure
            New probability measure `P`. Must have the same sample space as this
            featurized probability space.

        Raises
        ------
        TypeError
            If `probability_measure` is not a `ProbabilityMeasure` instance.
        ValueError
            If `probability_measure`'s sample space does not match this featurized
            probability space's sample space.
        """
        self._validate_parameters(
            self.sample_space,
            self.feature_embedding,
            self.sigma_algebra,
            probability_measure,
        )
        self._probability_measure = probability_measure
        self.probability_space.probability_measure = probability_measure

    @property
    def feature_embedding(self) -> RandomVector:
        """Get the feature embedding function.

        Returns
        -------
        feature_embedding : FeatureEmbedding
            The feature embedding function `X` of this featurized probability space.
        """
        return self._feature_embedding

    @feature_embedding.setter
    def feature_embedding(self, feature_embedding: RandomVector) -> None:
        """Set the feature embedding function.

        Parameters
        ----------
        feature_embedding : FeatureEmbedding
            New feature embedding function `X`. Must be defined on the same
            sample space as this featurized probability space.

        Raises
        ------
        TypeError
            If `feature_embedding` is not a `FeatureEmbedding` instance.
        ValueError
            If `feature_embedding` is not defined on this featurized probability
            space's sample space.
        """
        self._validate_parameters(
            self.sample_space,
            feature_embedding,
            self.sigma_algebra,
            self.probability_measure,
        )
        self._feature_embedding = feature_embedding

    @property
    def probability_space(self) -> ProbabilitySpace:
        """Get the underlying probability space `(Omega, F, P)`.

        Returns the probability space component `(Omega, F, P)` without the feature
        embedding.

        Returns
        -------
        probability_space : ProbabilitySpace
            The underlying probability space `(Omega, F, P)`.
        """
        from ..base import ProbabilitySpace

        if not hasattr(self, "_probability_space"):
            self._probability_space = ProbabilitySpace(
                sample_space=self.sample_space,
                sigma_algebra=self.sigma_algebra,
                probability_measure=self.probability_measure,
            )
        return self._probability_space

    @property
    def random_vector(self) -> RandomVector:
        """Get the feature embedding as a random vector.

        Returns the feature embedding function `X: Omega -> S` as a `RandomVector`.

        Returns
        -------
        random_vector : RandomVector
            The feature embedding function as a `RandomVector`.
        """
        return self.feature_embedding

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Get a developer-focused string representation.

        Returns
        -------
        repr_str : str
            String representation showing component names.
        """
        return (
            f"FeaturizedProbabilitySpace("
            f"sample_space={self.sample_space.name}, "
            f"sigma_algebra={self.sigma_algebra.name}, "
            f"probability_measure={self.probability_measure.name}, "
            f"feature_embedding={self.feature_embedding.name})"
        )

    def __str__(self) -> str:
        """Get a user-friendly string representation.

        Returns a detailed multi-line representation showing all components
        of the featurized probability space `(Omega, F, P, X)`.

        Returns
        -------
        str_repr : str
            Detailed description including all four components.
        """
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
        """Test equality with another featurized probability space.

        Two featurized probability spaces are equal if they have the same sample space,
        sigma-algebra, probability measure, and feature embedding function.

        Parameters
        ----------
        other : object
            Object to compare against.

        Returns
        -------
        is_equal : bool
            `True` if `other` is a `FeaturizedProbabilitySpace` with identical
            components, `False` otherwise.
        """
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
        feature_embedding: RandomVector,
        sigma_algebra: SigmaAlgebra | None = None,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> None:
        """Validate parameters for creating a featurized probability space.

        Ensures all components are compatible and form a valid featurized
        probability space `(Omega, F, P, X)`.

        Parameters
        ----------
        sample_space : SampleSpace
            Sample space `Omega`.
        feature_embedding : FeatureEmbedding
            Feature embedding function `X` mapping `Omega` to feature space.
        sigma_algebra : SigmaAlgebra or None, optional
            Sigma-algebra `F` on `Omega`. Must be defined on `sample_space`.
        probability_measure : ProbabilityMeasure or None, optional
            Probability measure `P` on `(Omega, F)`. Must be defined on `sample_space`.

        Raises
        ------
        TypeError
            If any parameter has incorrect type.
        ValueError
            If components are incompatible or feature embedding is not defined
            on the sample space.
        """
        from ..base.sample_space import SampleSpace
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from ..random_objects.random_vector import RandomVector
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")
        if not isinstance(feature_embedding, RandomVector):
            raise TypeError("feature_embedding must be a RandomVector instance.")
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
        if not feature_embedding.data.index.equals(sample_space.data):
            raise ValueError(
                "feature_embedding must be defined on the given sample_space."
            )
