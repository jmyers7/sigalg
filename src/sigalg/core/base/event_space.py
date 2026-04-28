"""A class representing an event space."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING

from ..sigma_algebras.sigma_algebra import SigmaAlgebraMethods

if TYPE_CHECKING:
    from ..probability_measures import ProbabilityMeasure
    from ..sigma_algebras import SigmaAlgebra
    from .probability_space import ProbabilitySpace
    from .sample_space import SampleSpace


class EventSpace(SigmaAlgebraMethods):
    r"""A class representing a sample space.

    See the Notes section below for the mathematical details.

    If both `sample_space` and `sig_alg` are provided during initialization, the `sample_space` of the provided `sig_alg` must match the provided `sample_space`. If only one of them is provided, the other will be automatically created to be compatible with it (i.e. if only `sample_space` is given, a power set sigma-algebra will be created on that sample space; if only `sig_alg` is given, the sample space will be taken from the sigma-algebra). If neither is provided, both will be initialized to `None` and can be set later.

    Parameters
    ----------
    sample_space : SampleSpace | None, default=None
        The sample space of the event space.
    sig_alg : SigmaAlgebra | None, default=None
        The sigma-algebra of the event space.

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

    def __init__(
        self,
        sample_space: SampleSpace | None = None,
        sig_alg: SigmaAlgebra | None = None,
    ):
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        self._validate_parameters(sample_space, sig_alg)

        if sample_space is None and sig_alg is not None:
            sample_space = sig_alg.sample_space
        if sig_alg is None and sample_space is not None:
            sig_alg = SigmaAlgebra.power_set(sample_space)
        self._sig_alg = sig_alg
        self._sample_space = sample_space

    def from_dict(
        self, sample_id_to_atom_id: Mapping[Hashable, Hashable]
    ) -> EventSpace:
        """Create an event space from a dictionary mapping sample IDs to atom IDs to construct the sigma-algebra.

        If a `sample_space` was not provided during initialization, it will be created from the keys of the provided mapping. If it was provided, the keys of the mapping must match the sample space, and the sigma-algebra will have its `sample_space` attribute set to the provided `sample_space`.

        Parameters
        ----------
        sample_id_to_atom_id : Mapping[Hashable, Hashable]
            A mapping from sample IDs to atom IDs, which will be used to construct the sigma-algebra.

        Returns
        -------
        self : EventSpace
            The current `EventSpace` instance.

        Examples
        --------
        >>> from sigalg.core import EventSpace
        >>> event_space = EventSpace().from_dict({0: 0, 1: 1, 2: 1})
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
        1             1
        2             1
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        self._sig_alg = SigmaAlgebra(sample_space=self.sample_space).from_dict(
            sample_id_to_atom_id
        )
        if self.sample_space is None:
            self._sample_space = self.sig_alg.sample_space
        return self

    # --------------------- properties --------------------- #

    @property
    def sample_space(self) -> SampleSpace:
        """Get the sample space of the event space.

        Returns
        -------
        sample_space : SampleSpace
            The sample space of the event space.

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
        >>> print(event_space.sample_space) # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega':
        [0, 1, 2]
        """
        return self._sample_space

    @sample_space.setter
    def sample_space(self, sample_space: SampleSpace) -> None:
        """Set the sample space of the event space.

        Setting a new sample space will set the sigma-algebra to the power-set sigma-algebra.

        Parameters
        ----------
        sample_space : SampleSpace
            The new sample space to set.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .sample_space import SampleSpace

        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")
        self._sample_space = sample_space
        self._sig_alg = SigmaAlgebra.power_set(sample_space)

    @property
    def sig_alg(self) -> SigmaAlgebra:
        """Get the sigma-algebra of the event space.

        Returns
        -------
        sig_alg : SigmaAlgebra
            The sigma-algebra of the event space.

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
        """Set the sigma-algebra of the event space.

        Setting a new sigma-algebra will set the sample space to the sample space of the new sigma-algebra if the sample space was not set during initialization. If the sample space was set during initialization, it must match the sample space of the new sigma-algebra.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The new sigma-algebra to set.
        """
        self._validate_parameters(self.sample_space, sig_alg)
        self._sig_alg = sig_alg
        if self.sample_space is None:
            self._sample_space = sig_alg.sample_space

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
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
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

    # --------------------- data access methods --------------------- #

    def __iter__(self):
        """Allow unpacking of event space components.

        Enables syntax like: `Omega, F = event_space`, where `Omega` is the sample space of the event space and `F` is its sigma-algebra.

        Yields
        ------
        sample_space : SampleSpace
            The sample space.
        sig_alg : SigmaAlgebra
            The sigma-algebra.

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
        >>> Omega1, F1 = event_space
        >>> print(Omega1) # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega':
        [0, 1, 2]
        >>> print(F1) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                atom ID
        sample
        0             0
        1             0
        2             1
        """
        yield self.sample_space
        yield self.sig_alg

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
    def _validate_parameters(
        sample_space: SampleSpace | None, sig_alg: SigmaAlgebra | None
    ):
        """Validate event space construction parameters.

        Parameters
        ----------
        sample_space : SampleSpace | None
            The sample space to validate.
        sig_alg : SigmaAlgebra or None | None
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

        if sample_space is not None and not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance, if given.")
        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be a SigmaAlgebra instance, if given.")

        if (
            sample_space is not None
            and sig_alg is not None
            and sig_alg.sample_space != sample_space
        ):
            raise ValueError(
                "sig_alg's sample_space must match the provided sample_space."
            )
