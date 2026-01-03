"""Probability space module.

This module provides the ProbabilitySpace class, which models a probability space `(Omega, F, P)` consisting of a sample space `Omega`, sigma-algebra `F`, and probability measure `P`.

Classes
-------
ProbabilitySpace
    Represents a probability space `(Omega, F, P)`.

Examples
--------
>>> from sigalg.core import ProbabilitySpace, SampleSpace
>>> prob_space = ProbabilitySpace.from_dict(probabilities={"H": 0.5, "T": 0.5})
>>> prob_space.sample_space  # doctest: +NORMALIZE_WHITESPACE
Sample space 'Omega':
['H', 'T']
>>> prob_space.P("H")
0.5
"""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np

from ..probability_measures.probability_measure import ProbabilityMeasureMethods
from ..sigma_algebras.sigma_algebra import SigmaAlgebraMethods
from .sample_space import SampleSpaceMethods

if TYPE_CHECKING:
    from ..probability_measures import ProbabilityMeasure
    from ..sigma_algebras import SigmaAlgebra
    from .sample_space import SampleSpace


class ProbabilitySpace(
    SampleSpaceMethods, SigmaAlgebraMethods, ProbabilityMeasureMethods
):
    """A class representing a probability space.

    A probability space `(Omega, F, P)` consists of a sample space `Omega` containing all possible outcomes, a sigma-algebra `F` defining measurable events, and a probability measure `P` assigning probabilities to events.

    `ProbabilitySpace` has attributes `sample_space`, `sigma_algebra`, and `probability_measure` that access the underlying components. It also inherits methods from `SampleSpaceMethods`, `SigmaAlgebraMethods`, and `ProbabilityMeasureMethods`, allowing direct access to their functionalities directly on the `ProbabilitySpace` instance.

    Parameters
    ----------
    sample_space : SampleSpace
        The sample space containing all possible outcomes.
    sigma_algebra : SigmaAlgebra, optional
        Sigma-algebra defining measurable events. If `None`, a power set
        sigma-algebra is created.
    probability_measure : ProbabilityMeasure, optional
        Probability measure assigning probabilities to outcomes. If `None`,
        a uniform probability measure is created.

    Raises
    ------
    TypeError
        If `sample_space` is not a `SampleSpace`, `sigma_algebra` is not a
        `SigmaAlgebra`, or `probability_measure` is not a `ProbabilityMeasure`.
    ValueError
        If `sigma_algebra` or `probability_measure` have different sample spaces
        than the provided `sample_space`.

    Examples
    --------
    >>> from sigalg.core import ProbabilitySpace, SampleSpace
    >>> Omega = SampleSpace.generate_sequence(size=3, prefix="s")
    >>> # Create with uniform probability
    >>> prob_space = ProbabilitySpace(sample_space=Omega)
    >>> prob_space.probability_measure # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'P':
            probability
    sample
    s_0        0.333333
    s_1        0.333333
    s_2        0.333333
    >>> # Create with custom probabilities
    >>> prob_space = ProbabilitySpace.from_dict(
    ...     sample_space=Omega,
    ...     probabilities={"s_0": 0.5, "s_1": 0.3, "s_2": 0.2}
    ... )
    >>> prob_space.P("s_0")
    0.5
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        sample_space: SampleSpace,
        sigma_algebra: SigmaAlgebra | None = None,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> None:
        from ..probability_measures import ProbabilityMeasure
        from ..sigma_algebras import SigmaAlgebra

        self._validate_parameters(sample_space, sigma_algebra, probability_measure)
        self.sample_space = sample_space
        if sigma_algebra is None:
            sigma_algebra = SigmaAlgebra.power_set(sample_space)
        self._sigma_algebra = sigma_algebra
        if probability_measure is None:
            probability_measure = ProbabilityMeasure.uniform(sample_space)
        self._probability_measure = probability_measure

    # --------------------- properties --------------------- #

    @property
    def probability_measure(self) -> ProbabilityMeasure:
        """Get the probability measure assigning probabilities to events.

        Returns
        -------
        probability_measure : ProbabilityMeasure
            The probability measure of this probability space.
        """
        return self._probability_measure

    @probability_measure.setter
    def probability_measure(self, probability_measure: ProbabilityMeasure) -> None:
        """Set the probability measure assigning probabilities to events.

        Parameters
        ----------
        probability_measure : ProbabilityMeasure
            New probability measure. Must have the same sample space as this
            probability space.

        Raises
        ------
        TypeError
            If `probability_measure` is not a `ProbabilityMeasure` instance.
        ValueError
            If `probability_measure`'s sample space does not match this probability
            space's sample space.
        """
        self._validate_parameters(
            self.sample_space, self.sigma_algebra, probability_measure
        )
        self._probability_measure = probability_measure

    @property
    def sigma_algebra(self) -> SigmaAlgebra:
        """Get the sigma-algebra defining measurable events.

        Returns
        -------
        sigma_algebra : SigmaAlgebra
            The sigma-algebra of this probability space.
        """
        return self._sigma_algebra

    @sigma_algebra.setter
    def sigma_algebra(self, sigma_algebra: SigmaAlgebra) -> None:
        """Set the sigma-algebra defining measurable events.

        Parameters
        ----------
        sigma_algebra : SigmaAlgebra
            New sigma-algebra. Must have the same sample space as this
            probability space.

        Raises
        ------
        TypeError
            If `sigma_algebra` is not a `SigmaAlgebra` instance.
        ValueError
            If `sigma_algebra`'s sample space does not match this probability
            space's sample space.
        """
        self._validate_parameters(
            self.sample_space, sigma_algebra, self.probability_measure
        )
        self._sigma_algebra = sigma_algebra

    # --------------------- factory methods --------------------- #

    @classmethod
    def from_dict(
        cls,
        probabilities: dict[Hashable, Real],
        sample_space: SampleSpace | None = None,
        sigma_algebra: SigmaAlgebra | None = None,
    ) -> ProbabilitySpace:
        """Create a probability space from a dictionary of probabilities.

        Convenience factory method that creates a probability measure from a
        dictionary mapping outcomes to probabilities, then constructs a
        probability space.

        Parameters
        ----------
        probabilities : dict[Hashable, Real]
            Dictionary mapping sample point indices to their probabilities.
            Probabilities must be non-negative and sum to 1.
        sample_space : SampleSpace | None, default=None
            The sample space containing all possible outcomes. If `None`, a sample
            space is created from the keys of the `probabilities` dictionary.
        sigma_algebra : SigmaAlgebra | None, default=None
            Sigma-algebra defining measurable events. If `None`, a power set
            sigma-algebra is created.

        Returns
        -------
        probability_space : ProbabilitySpace
            A new probability space with the specified probabilities.

        Examples
        --------
        >>> from sigalg.core import ProbabilitySpace
        >>> prob_space = ProbabilitySpace.from_dict(
        ...     probabilities={"H": 0.6, "T": 0.4}
        ... )
        >>> prob_space.sample_space # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega':
        ['H', 'T']
        >>> prob_space.P("H")
        0.6
        """
        from ..probability_measures import ProbabilityMeasure

        probability_measure = ProbabilityMeasure(sample_space=sample_space).from_dict(
            probabilities
        )
        return cls(
            sample_space=probability_measure.sample_space,
            sigma_algebra=sigma_algebra,
            probability_measure=probability_measure,
        )

    # --------------------- methods --------------------- #

    def get_event_as_probability_space(
        self, indices: list[Hashable]
    ) -> ProbabilitySpace:
        """Create a conditional probability space given an event.

        Given a probability space `(Omega, F, P)` and an event `A`, this method creates a new probability space `(A, F_A, P_A)` where `F_A` is the sigma-algebra restricted to `A` and `P_A` is the conditional probability measure on `A`.

        Parameters
        ----------
        indices : list[Hashable]
            `list[Hashable]` of sample point indices defining the conditioning event.

        Returns
        -------
        probability_space : ProbabilitySpace
            A new probability space representing the conditional distribution.

        Raises
        ------
        ValueError
            If the event has zero probability.

        Examples
        --------
        >>> from sigalg.core import ProbabilitySpace, SampleSpace
        >>> Omega = SampleSpace.generate_sequence(size=3, prefix="s")
        >>> prob_space = ProbabilitySpace.from_dict(
        ...     sample_space=Omega,
        ...     probabilities={"s_0": 0.5, "s_1": 0.3, "s_2": 0.2}
        ... )
        >>> cond_space = prob_space.get_event_as_probability_space(["s_0", "s_1"])
        >>> bool(abs(cond_space.P("s_0") - 0.625) < 1e-10)
        True
        """
        from ..probability_measures import ProbabilityMeasure
        from ..sigma_algebras import SigmaAlgebra

        event = self.get_event(indices)

        event_probability = self.probability_measure(event)
        if event_probability < 1e-10:
            raise ValueError(
                "Cannot create ProbabilitySpace for event with zero probability."
            )

        event_sample_space = event.to_sample_space()

        conditional_probabilities = {
            idx: self.probability_measure(idx) / event_probability for idx in event
        }

        event_probability_measure = ProbabilityMeasure(
            sample_space=event_sample_space
        ).from_dict(conditional_probabilities)

        event_atom_ids = {
            idx: self.sigma_algebra.sample_id_to_atom_id[idx] for idx in event
        }
        event_sigma_algebra = SigmaAlgebra(sample_space=event_sample_space).from_dict(
            sample_id_to_atom_id=event_atom_ids
        )

        return ProbabilitySpace(
            sample_space=event_sample_space,
            sigma_algebra=event_sigma_algebra,
            probability_measure=event_probability_measure,
        )

    def sample(self, size: int = 1, random_state: int | None = None) -> list[Hashable]:
        """Generate random samples from this probability space.

        Samples outcomes according to the probability measure, returning a
        `list[Hashable]` of sample point indices.

        Parameters
        ----------
        size : int, default=1
            Number of samples to generate. Must be positive.
        random_state : int, optional
            Random seed for reproducibility. If `None`, results are not reproducible.

        Returns
        -------
        sample : list[Hashable]
            `list[Hashable]` of sampled outcomes from the sample space.

        Raises
        ------
        ValueError
            If `size` is not a positive integer.

        Examples
        --------
        >>> from sigalg.core import ProbabilitySpace, SampleSpace
        >>> Omega = SampleSpace().from_list(["H", "T"])
        >>> prob_space = ProbabilitySpace(sample_space=Omega)
        >>> samples = prob_space.sample(size=10, random_state=42)
        >>> len(samples)
        10
        """
        if not isinstance(size, int) or size < 1:
            raise ValueError("size must be a positive integer.")
        if random_state is not None:
            np.random.seed(random_state)

        outcomes = list(self.sample_space)
        probabilities = [self.P(outcome) for outcome in outcomes]
        samples = np.random.choice(outcomes, size=size, p=probabilities)

        return [
            outcomes[outcomes.index(s)] if hasattr(outcomes, "index") else s
            for s in samples
        ]

    # --------------------- equality --------------------- #

    def __eq__(self, other: object) -> bool:
        """Check equality with another probability space.

        Two probability spaces are equal if they have the same sample space,
        sigma-algebra, and probability measure.

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the other object is a `ProbabilitySpace` with identical
            components, `False` otherwise.
        """
        if not isinstance(other, ProbabilitySpace):
            return False
        return (
            self.sample_space == other.sample_space
            and self.sigma_algebra == other.sigma_algebra
            and self.probability_measure == other.probability_measure
        )

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a concise string representation of the probability space.

        Returns
        -------
        repr_str : str
            A string representation showing the probability space's component names.
        """
        return (
            f"ProbabilitySpace("
            f"sample_space={self.sample_space.name}, "
            f"sigma_algebra={self.sigma_algebra.name}, "
            f"probability_measure={self.probability_measure.name})"
        )

    def __str__(self) -> str:
        """Return a detailed string representation of the probability space.

        Returns
        -------
        repr_str : str
            A formatted string showing the probability space header and detailed
            representations of its components.
        """
        header = (
            f"Probability space ("
            f"{self.sample_space.name}, "
            f"{self.sigma_algebra.name}, "
            f"{self.probability_measure.name})"
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
        )

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        sample_space: SampleSpace,
        sigma_algebra: SigmaAlgebra | None,
        probability_measure: ProbabilityMeasure | None,
    ) -> None:
        """Validate probability space construction parameters.

        Parameters
        ----------
        sample_space : SampleSpace
            The sample space to validate.
        sigma_algebra : SigmaAlgebra or None
            The sigma-algebra to validate.
        probability_measure : ProbabilityMeasure or None
            The probability measure to validate.

        Raises
        ------
        TypeError
            If `sample_space` is not a `SampleSpace` instance, `sigma_algebra` is not
            a `SigmaAlgebra` instance (when provided), or `probability_measure` is
            not a `ProbabilityMeasure` instance (when provided).
        ValueError
            If `sigma_algebra` or `probability_measure` have sample spaces that do
            not match the provided `sample_space`.
        """
        from ..probability_measures import ProbabilityMeasure
        from ..sigma_algebras import SigmaAlgebra
        from .sample_space import SampleSpace

        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")
        if sigma_algebra is not None and not isinstance(sigma_algebra, SigmaAlgebra):
            raise TypeError("sigma_algebra must be a SigmaAlgebra instance.")
        if probability_measure is not None and not isinstance(
            probability_measure, ProbabilityMeasure
        ):
            raise TypeError(
                "probability_measure must be a ProbabilityMeasure instance."
            )
        if sigma_algebra is not None and sigma_algebra.sample_space != sample_space:
            raise ValueError("sigma_algebra must be defined on the given sample_space.")
        if (
            probability_measure is not None
            and probability_measure.sample_space != sample_space
        ):
            raise ValueError(
                "probability_measure must be defined on the given sample_space."
            )
