"""A class representing an event space."""

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
    r"""A class representing a sample space.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    sample_space : SampleSpace
        The sample space containing all possible outcomes.
    sig_alg : SigmaAlgebra | None, default=None
        Sigma-algebra defining measurable events. If `None`, a power set
        sigma-algebra is created.

    Raises
    ------
    TypeError
        If `sample_space` is not a `SampleSpace` instance or `sig_alg`
        is not a `SigmaAlgebra` instance.
    ValueError
        If the sample space of `sig_alg` does not match the provided `sample_space`.

    Examples
    --------
    >>> from sigalg.core import EventSpace, SampleSpace, SigmaAlgebra
    >>> Omega = SampleSpace().from_sequence(size=3)
    >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
    ...     {
    ...         0: 0,
    ...         1: 0,
    ...         2: 1,
    ...     }
    ... )
    >>> event_space = EventSpace(sample_space=Omega, sig_alg=F)
    >>> print(event_space) # doctest: +NORMALIZE_WHITESPACE
    Event space (Omega, F)
    ======================
    <BLANKLINE>
    * Sample space 'Omega':
    [0, 1, 2]
    <BLANKLINE>
    * Sigma algebra 'F':
            atom ID
    sample
    0             0
    1             0
    2             1

    Notes
    -----
    An *event space* is a pair $(\Omega, \mathcal{F})$ consisting of a sample space $\Omega$ and a $\sigma$-algebra $\mathcal{F}$ on $\Omega$. In general measure theory, this is just called a *measurable space*, but the terminology used here is intended to reflect the probabilistic context.

    See also the [notebook](https://johnmyers-phd.com/sigalg/dictionary/){target="_blank"} on the docs website.
    """

    # --------------------- constructor --------------------- #

    def __init__(self, sample_space: SampleSpace, sig_alg: SigmaAlgebra | None = None):
        from ..sigma_algebras import SigmaAlgebra

        self._validate_parameters(sample_space, sig_alg)
        self.sample_space = sample_space
        if sig_alg is None:
            sig_alg = SigmaAlgebra.power_set(sample_space)
        self._sig_alg = sig_alg

    # --------------------- properties --------------------- #

    @property
    def sig_alg(self) -> SigmaAlgebra:
        """Get the sigma-algebra defining measurable events.

        Returns
        -------
        sig_alg : SigmaAlgebra
            The sigma-algebra of this event space.

        Examples
        --------
        >>> from sigalg.core import EventSpace, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     }
        ... )
        >>> event_space = EventSpace(sample_space=Omega, sig_alg=F)
        >>> print(event_space.sig_alg) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                atom ID
        sample
        0             0
        1             0
        2             1
        """
        return self._sig_alg

    @sig_alg.setter
    def sig_alg(self, sig_alg: SigmaAlgebra) -> None:
        """Set the sigma-algebra defining measurable events.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            New sigma-algebra. Must have the same sample space as this event space.

        Raises
        ------
        TypeError
            If `sig_alg` is not a `SigmaAlgebra` instance.
        ValueError
            If `sig_alg`'s sample space does not match this event space's sample space.
        """
        self._validate_parameters(self.sample_space, sig_alg)
        self._sig_alg = sig_alg

    # --------------------- conversion methods --------------------- #

    def make_probability_space(
        self,
        prob_measure: ProbabilityMeasure | None = None,
    ) -> ProbabilitySpace:
        """Convert this event space to a probability space by adding a probability measure.

        Parameters
        ----------
        prob_measure : ProbabilityMeasure | None, default=None
            Probability measure to use. If `None`, a uniform probability
            measure is created.

        Returns
        -------
        probability_space : ProbabilitySpace
            A probability space with this event space's sample space and
            sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import EventSpace, ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     }
        ... )
        >>> event_space = EventSpace(sample_space=Omega, sig_alg=F)
        >>> # Create a probability space with a uniform probability measure
        >>> prob_space = event_space.make_probability_space()
        >>> print(prob_space) # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, F, P)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
        [0, 1, 2]
        <BLANKLINE>
        * Sigma algebra 'F':
                atom ID
        sample
        0             0
        1             0
        2             1
        <BLANKLINE>
        * Probability measure 'P':
                probability
        sample
        0               0.333333
        1               0.333333
        2               0.333333
        >>> # Create a probability space with a custom probability measure
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict(
        ...     {
        ...         0: 0.2,
        ...         1: 0.5,
        ...         2: 0.3,
        ...     }
        ... )
        >>> prob_space = event_space.make_probability_space(prob_measure=P)
        >>> print(prob_space) # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, F, P)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
        [0, 1, 2]
        <BLANKLINE>
        * Sigma algebra 'F':
                atom ID
        sample
        0             0
        1             0
        2             1
        <BLANKLINE>
        * Probability measure 'P':
                probability
        sample
        0               0.2
        1               0.5
        2               0.3
        """
        from .probability_space import ProbabilitySpace

        return ProbabilitySpace(
            sample_space=self.sample_space,
            sig_alg=self.sig_alg,
            prob_measure=prob_measure,
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
            f"sig_alg={self.sig_alg.name})"
        )

    def __str__(self) -> str:
        """Return a detailed string representation of the event space.

        Returns
        -------
        repr_str : str
            A formatted string showing the event space header and detailed
            representations of its components.
        """
        header = f"Event space ({self.sample_space.name}, {self.sig_alg.name})"
        separator = "=" * len(header)
        return (
            header
            + "\n"
            + separator
            + "\n\n* "
            + repr(self.sample_space)
            + "\n\n* "
            + repr(self.sig_alg)
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
            and `sig_alg`, `False` otherwise.
        """
        if not isinstance(other, EventSpace):
            return False
        return self.sample_space == other.sample_space and self.sig_alg == other.sig_alg

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(sample_space: SampleSpace, sig_alg: SigmaAlgebra):
        """Validate event space construction parameters.

        Parameters
        ----------
        sample_space : SampleSpace
            The sample space to validate.
        sig_alg : SigmaAlgebra or None
            The sigma-algebra to validate.

        Raises
        ------
        TypeError
            If `sample_space` is not a `SampleSpace` instance or `sig_alg`
            is not a `SigmaAlgebra` instance (when provided).
        ValueError
            If `sig_alg`'s sample space does not match the provided
            `sample_space`.
        """
        from ..sigma_algebras import SigmaAlgebra
        from .sample_space import SampleSpace

        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")
        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be a SigmaAlgebra instance.")
        if sig_alg is not None and sig_alg.sample_space != sample_space:
            raise ValueError(
                "sig_alg's sample_space must match the provided sample_space."
            )
