"""Event space module.

This module provides the `EventSpace` class, which models a measurable space `(Omega, F)` consisting of a sample space `Omega` and a sigma-algebra `F`.

Classes
-------
EventSpace
    Represents an event space `(Omega, F)`.

Examples
--------
>>> from sigalg.core import EventSpace, SampleSpace
>>> Omega = SampleSpace.generate_sequence(size=3)
>>> # Create event space with default power set sigma-algebra
>>> event_space = EventSpace(sample_space=Omega)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..sigma_algebras.sigma_algebra import SigmaAlgebraMethods
from .sample_space import SampleSpaceMethods

if TYPE_CHECKING:
    from ..probability_measures import ProbabilityMeasure
    from ..sigma_algebras import SigmaAlgebra
    from .probability_space import ProbabilitySpace
    from .sample_space import SampleSpace


class EventSpace(SampleSpaceMethods, SigmaAlgebraMethods):
    """A class representing a measurable space.

    An event space `(Omega, F)` consists of a sample space `Omega` and a sigma-algebra `F` that defines which subsets of the sample space are measurable events.

    `EventSpace` has attributes `sample_space` and `sigma_algebra` that access the underlying components. It also inherits methods from `SampleSpaceMethods` and `SigmaAlgebraMethods`, allowing direct access to their functionalities directly on the `EventSpace` instance.

    Parameters
    ----------
    sample_space : SampleSpace
        The underlying sample space containing all possible outcomes.
    sigma_algebra : SigmaAlgebra, optional
        Sigma-algebra defining measurable events. If `None`, a power set
        sigma-algebra is created, making all subsets measurable.

    Raises
    ------
    TypeError
        If `sample_space` is not a `SampleSpace` instance or `sigma_algebra`
        is not a `SigmaAlgebra` instance.
    ValueError
        If `sigma_algebra`'s sample space does not match the provided `sample_space`.

    Examples
    --------
    >>> from sigalg.core import EventSpace, SampleSpace, SigmaAlgebra
    >>> Omega = SampleSpace.generate_sequence(size=3)
    >>> # Create with default power set sigma-algebra
    >>> event_space = EventSpace(sample_space=Omega)
    >>> # Create with custom sigma-algebra
    >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
    ...     sample_id_to_atom_id={"omega_0": 0, "omega_1": 0, "omega_2": 1},
    ... )
    >>> event_space = EventSpace(
    ...     sample_space=Omega,
    ...     sigma_algebra=F
    ... )
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self, sample_space: SampleSpace, sigma_algebra: SigmaAlgebra | None = None
    ):
        from ..sigma_algebras import SigmaAlgebra

        self._validate_parameters(sample_space, sigma_algebra)
        self.sample_space = sample_space
        if sigma_algebra is None:
            sigma_algebra = SigmaAlgebra.power_set(sample_space)
        self._sigma_algebra = sigma_algebra

    # --------------------- properties --------------------- #

    @property
    def sigma_algebra(self) -> SigmaAlgebra:
        """Get the sigma-algebra defining measurable events.

        Returns
        -------
        sigma_algebra : SigmaAlgebra
            The sigma-algebra of this event space.
        """
        return self._sigma_algebra

    @sigma_algebra.setter
    def sigma_algebra(self, sigma_algebra) -> None:
        """Set the sigma-algebra defining measurable events.

        Parameters
        ----------
        sigma_algebra : SigmaAlgebra
            New sigma-algebra. Must have the same sample space as this event space.

        Raises
        ------
        TypeError
            If `sigma_algebra` is not a `SigmaAlgebra` instance.
        ValueError
            If `sigma_algebra`'s sample space does not match this event space's sample space.
        """
        self._validate_parameters(self.sample_space, sigma_algebra)
        self._sigma_algebra = sigma_algebra

    # --------------------- conversion methods --------------------- #

    def make_probability_space(
        self,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> ProbabilitySpace:
        """Convert this event space to a probability space.

        Creates a `ProbabilitySpace` by adding a probability measure to this
        event space. If no probability measure is provided, a uniform
        probability measure is created.

        Parameters
        ----------
        probability_measure : ProbabilityMeasure, optional
            Probability measure to use. If `None`, a uniform probability
            measure is created.

        Returns
        -------
        probability_space : ProbabilitySpace
            A probability space with this event space's sample space and
            sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import EventSpace, SampleSpace
        >>> Omega = SampleSpace.generate_sequence(size=3, prefix="s")
        >>> event_space = EventSpace(sample_space=Omega)
        >>> prob_space = event_space.make_probability_space()
        >>> prob_space.P("s_0")
        0.333...
        """
        from .probability_space import ProbabilitySpace

        return ProbabilitySpace(
            sample_space=self.sample_space,
            sigma_algebra=self.sigma_algebra,
            probability_measure=probability_measure,
        )

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a concise string representation of the event space.

        Returns
        -------
        repr_str : str
            A string representation showing the event space's sample space
            and sigma-algebra names.
        """
        return (
            f"EventSpace(sample_space={self.sample_space.name}, "
            f"sigma_algebra={self.sigma_algebra.name})"
        )

    def __str__(self) -> str:
        """Return a detailed string representation of the event space.

        Returns
        -------
        repr_str : str
            A formatted string showing the event space header and detailed
            representations of its components.
        """
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
        """Check equality with another event space.

        Two event spaces are equal if they have the same sample space and
        sigma-algebra.

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        are_equal : bool
            True if the other object is an `EventSpace` with identical `sample_space`
            and `sigma_algebra`, `False` otherwise.
        """
        if not isinstance(other, EventSpace):
            return False
        return (
            self.sample_space == other.sample_space
            and self.sigma_algebra == other.sigma_algebra
        )

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(sample_space: SampleSpace, sigma_algebra: SigmaAlgebra):
        """Validate event space construction parameters.

        Parameters
        ----------
        sample_space : SampleSpace
            The sample space to validate.
        sigma_algebra : SigmaAlgebra or None
            The sigma-algebra to validate.

        Raises
        ------
        TypeError
            If `sample_space` is not a `SampleSpace` instance or `sigma_algebra`
            is not a `SigmaAlgebra` instance (when provided).
        ValueError
            If `sigma_algebra`'s sample space does not match the provided
            `sample_space`.
        """
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
