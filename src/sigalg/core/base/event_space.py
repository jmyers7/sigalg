"""A class representing an event space."""

from __future__ import annotations

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
    Define a sample space and sigma-algebra.
    >>> from sigalg.core import EventSpace, SampleSpace, SigmaAlgebra
    >>> Omega = SampleSpace.from_sequence(size=3)
    >>> F = SigmaAlgebra(
    ...     sample_space=Omega,
    ...     mapping={
    ...         0: 0,
    ...         1: 0,
    ...         2: 1,
    ...     },
    ... )

    Create an event space and print it.

    >>> event_space = EventSpace(sample_space=Omega, sig_alg=F)
    >>> print(event_space)  # doctest: +NORMALIZE_WHITESPACE
    Event space (Omega, F)
    ======================
    <BLANKLINE>
    * Sample space 'Omega':
     sample
          0
          1
          2
    <BLANKLINE>
    * Sigma algebra 'F':
            atom_ID
    sample
    0             0
    1             0
    2             1

    Notes
    -----
    An *event space* is a pair $(\Omega, \mathcal{F})$ consisting of a sample space $\Omega$ and a $\sigma$-algebra $\mathcal{F}$ on $\Omega$. In general measure theory, this is just called a *measurable space*, but the terminology used here is intended to reflect the probabilistic context.
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        sample_space: SampleSpace | None = None,
        sig_alg: SigmaAlgebra | None = None,
    ):
        self._validate_parameters(sample_space, sig_alg)
        self._sample_space, self._sig_alg = self._generate_components(
            sample_space, sig_alg
        )

    def _generate_components(self, sample_space, sig_alg):
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        parameter_cases = (sample_space is not None, sig_alg is not None)
        if parameter_cases == (1, 0):
            sig_alg = SigmaAlgebra.power_set(sample_space)
        if parameter_cases == (0, 1):
            sample_space = sig_alg._sample_space

        return sample_space, sig_alg

    # --------------------- properties --------------------- #

    @property
    def sample_space(self) -> SampleSpace | None:
        """Get the sample space of the event space.

        The `sample_space` parameter is settable. If the event space is not empty, the new sample space must contain the same number of sample points as the current sample space, and the sigma-algebra will be updated to be defined on the new sample space with the same atom structure as before. If the event space is empty, then setting the sample space will set the sigma-algebra to be the power set sigma-algebra on the new sample space.

        Returns
        -------
        sample_space : SampleSpace | None
            The sample space of the event space.

        Examples
        --------
        Define a sample space and sigma-algebra.

        >>> from sigalg.core import EventSpace, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...     },
        ... )

        Instantiate an event space and print it.

        >>> event_space = EventSpace(Omega, F)
        >>> print(event_space)  # doctest: +NORMALIZE_WHITESPACE
        Event space (Omega, F)
        ======================
        <BLANKLINE>
        * Sample space 'Omega':
         sample
              0
              1
              2
              3
        <BLANKLINE>
        * Sigma algebra 'F':
                atom_ID
        sample
        0             0
        1             1
        2             2
        3             2

        Set the sample space of the event space to a new sample space. Print to check.

        >>> S = SampleSpace(["a", "b", "c", "d"], name="S")
        >>> event_space.sample_space = S
        >>> print(event_space)  # doctest: +NORMALIZE_WHITESPACE
        Event space (S, F)
        ==================
        <BLANKLINE>
        * Sample space 'S':
        sample
            a
            b
            c
            d
        <BLANKLINE>
        * Sigma algebra 'F':
                atom_ID
        sample
        a             0
        b             1
        c             2
        d             2

        Create an empty `EventSpace` instance and set its sample space.

        >>> empty_event_space = EventSpace()
        >>> empty_event_space.sample_space = S

        Print the event space and note the sigma-algebra is the power-set sigma-algebra by default.

        >>> print(empty_event_space)  # doctest: +NORMALIZE_WHITESPACE
        Event space (S, power_set)
        ==========================
        <BLANKLINE>
        * Sample space 'S':
        sample
            a
            b
            c
            d
        <BLANKLINE>
        * Sigma algebra 'power_set':
            atom_ID
        sample
        a            a
        b            b
        c            c
        d            d
        """
        return self._sample_space

    @sample_space.setter
    def sample_space(self, sample_space: SampleSpace) -> None:
        """Set the sample space of the event space.

        If the event space is not empty, the new sample space must contain the same number of sample points as the current sample space, and the sigma-algebra will be updated to be defined on the new sample space with the same atom structure as before. If the event space is empty, then setting the sample space will set the sigma-algebra to be the power set sigma-algebra on the new sample space.

        Parameters
        ----------
        sample_space : SampleSpace
            The new sample space to set.

        Raises
        ------
        TypeError
            If `sample_space` is not a `SampleSpace` instance.
        """
        from .sample_space import SampleSpace

        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")

        if self.sample_space is not None:
            self.sig_alg.sample_space = sample_space
            self._sample_space = sample_space
        else:
            self._sample_space, self._sig_alg = self._generate_components(
                sample_space=sample_space, sig_alg=None
            )

    @property
    def sig_alg(self) -> SigmaAlgebra:
        """Get the sigma-algebra of the event space.

        The `sig_alg` property is settable. If the event space is not empty, the new sigma-algebra must have the same sample space as the current sigma-algebra. If the event space is empty, then setting the sigma-algebra will set the sample space to be the sample space of the new sigma-algebra.

        Returns
        -------
        sig_alg : SigmaAlgebra
            The sigma-algebra of the event space.

        Examples
        --------
        Define a sample space and sigma-algebra.

        >>> from sigalg.core import EventSpace, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...     },
        ... )

        Instantiate an event space and print it.

        >>> event_space = EventSpace(Omega, F)
        >>> print(event_space)  # doctest: +NORMALIZE_WHITESPACE
        Event space (Omega, F)
        ======================
        <BLANKLINE>
        * Sample space 'Omega':
         sample
              0
              1
              2
              3
        <BLANKLINE>
        * Sigma algebra 'F':
                atom_ID
        sample
        0             0
        1             1
        2             2
        3             2

        Define a new sigma-algebra, a sub-sigma-algebra of the first and set the `sig_alg` property of the event space.

        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 1,
        ...     },
        ...     name="G",
        ... )
        >>> event_space.sig_alg = G

        Print the updated event space to check.

        >>> print(event_space)  # doctest: +NORMALIZE_WHITESPACE
        Event space (Omega, G)
        ======================
        <BLANKLINE>
        * Sample space 'Omega':
         sample
              0
              1
              2
              3
        <BLANKLINE>
        * Sigma algebra 'G':
                atom_ID
        sample
        0             0
        1             1
        2             1
        3             1

        Create an empty `EventSpace` instance and set its sigma-algebra property.

        >>> empty_event_space = EventSpace()
        >>> empty_event_space.sig_alg = G

        Print the updated empty event space. Note the sample space was extracted from the sigma-algebra.

        >>> print(empty_event_space)  # doctest: +NORMALIZE_WHITESPACE
        Event space (Omega, G)
        ======================
        <BLANKLINE>
        * Sample space 'Omega':
         sample
              0
              1
              2
              3
        <BLANKLINE>
        * Sigma algebra 'G':
                atom_ID
        sample
        0             0
        1             1
        2             1
        3             1
        """
        return self._sig_alg

    @sig_alg.setter
    def sig_alg(self, sig_alg: SigmaAlgebra) -> None:
        """Set the sigma-algebra of the event space.

        If the event space is not empty, the new sigma-algebra must have the same sample space as the current sigma-algebra. If the event space is empty, then setting the sigma-algebra will set the sample space to be the sample space of the new sigma-algebra.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The new sigma-algebra to set.

        Raises
        ------
        TypeError
            If `sig_alg` is not a `SigmaAlgebra` instance.
        ValueError
            If the new `sig_alg` does not have the same sample space as the current `sig_alg` (when the event space is not empty).
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be a SigmaAlgebra instance.")

        if self.sig_alg is not None:
            if self.sig_alg.sample_space != sig_alg.sample_space:
                raise ValueError(
                    "New sig_alg must have the same sample space as the current sig_alg."
                )
            self._sig_alg = sig_alg
        else:
            self._sample_space, self._sig_alg = self._generate_components(
                sample_space=None, sig_alg=sig_alg
            )

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
        Create an instance of `EventSpace`.

        >>> from sigalg.core import EventSpace, ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> event_space = EventSpace(sample_space=Omega, sig_alg=F)

        Create a probability space with a uniform probability measure

        >>> prob_space = event_space.make_probability_space()
        >>> print(prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, F, U)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
         sample
              0
              1
              2
        <BLANKLINE>
        * Sigma algebra 'F':
                atom_ID
        sample
        0             0
        1             0
        2             1
        <BLANKLINE>
        * Probability measure 'U':
                 probability
        atom_ID
        0                0.5
        1                0.5

        Create a probability space with a custom probability measure

        >>> P = ProbabilityMeasure.on(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.7,
        ...         1: 0.3,
        ...     },
        ... )
        >>> prob_space = event_space.make_probability_space(prob_measure=P)
        >>> print(prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, F, P)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
         sample
              0
              1
              2
        <BLANKLINE>
        * Sigma algebra 'F':
                atom_ID
        sample
        0             0
        1             0
        2             1
        <BLANKLINE>
        * Probability measure 'P':
                 probability
        atom_ID
        0                0.7
        1                0.3
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
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> event_space = EventSpace(sample_space=Omega, sig_alg=F)
        >>> Omega1, F1 = event_space
        >>> print(Omega1)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega':
         sample
              0
              1
              2
        >>> print(F1)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                atom_ID
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
            and sig_alg._sample_space != sample_space
        ):
            raise ValueError(
                "sig_alg's sample_space must match the provided sample_space."
            )
