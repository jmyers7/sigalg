"""A class representing a measurable space."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..sigma_algebras.sigma_algebra import SigmaAlgebraMethods

if TYPE_CHECKING:
    from ...typing.index_like import IndexLike
    from ..measures.measure import Measure
    from ..sigma_algebras import SigmaAlgebra
    from .domain import Domain
    from .measure_space import MeasureSpace


class MeasurableSpace(SigmaAlgebraMethods):
    r"""A class representing a measurable space.

    See the Notes section below for the mathematical details.

    If both `domain` and `sig_alg` are provided during initialization, the `domain` of the provided `sig_alg` must match the provided `domain`. If only one of them is provided, the other will be automatically created to be compatible with it (i.e. if only `domain` is given, a power-set sigma-algebra will be created on that domain; if only `sig_alg` is given, the domain will be taken from the sigma-algebra).

    Parameters
    ----------
    domain : IndexLike | None, default=None
        The domain of the measurable space.
    sig_alg : SigmaAlgebra | None, default=None
        The sigma-algebra of the measurable space.

    Examples
    --------
    Define a domain and sigma-algebra.

    >>> from sigalg.core import Domain, MeasurableSpace, SigmaAlgebra
    >>> X = Domain.from_sequence(size=3)
    >>> F = SigmaAlgebra(
    ...     domain=X,
    ...     mapping={
    ...         0: 0,
    ...         1: 0,
    ...         2: 1,
    ...     },
    ... )

    Create a measurable space and print it.

    >>> measurable_space = MeasurableSpace(domain=X, sig_alg=F)
    >>> print(measurable_space)  # doctest: +NORMALIZE_WHITESPACE
    Measurable space (X, F)
    =======================
    <BLANKLINE>
    * Domain 'X':
     x
     0
     1
     2
    <BLANKLINE>
    * Sigma algebra 'F':
         F
    x
    0    0
    1    0
    2    1

    Notes
    -----
    A *measurable space* is a pair $(X, \mathcal{F})$ consisting of a nonempty set $X$ and a $\sigma$-algebra $\mathcal{F}$ on $X$.
    """

    _repr_name = "MeasurableSpace"
    _str_name = "Measurable space"

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        domain: IndexLike | None = None,
        sig_alg: SigmaAlgebra | None = None,
    ):
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("If given, sig_alg must be a SigmaAlgebra.")

        if sig_alg is not None:
            if domain is not None and sig_alg.domain != domain:
                raise ValueError(
                    "If both the sigma-algebra and domain are given, the domain of the former must equal the latter."
                )
        elif domain is not None:
            sig_alg = SigmaAlgebra.power_set(domain)

        self.sig_alg = sig_alg

    @classmethod
    def _from_validated(cls, *, sig_alg: SigmaAlgebra) -> MeasurableSpace:
        measurable_space = object.__new__(MeasurableSpace)
        measurable_space.sig_alg = sig_alg

        return measurable_space

    # --------------------- properties --------------------- #

    @property
    def domain(self) -> Domain | None:
        """Get the domain of the measurable space.

        Returns
        -------
        domain : Domain | None
            The domain of the measurable space.

        Examples
        --------
        Define a domain and sigma-algebra.

        >>> from sigalg.core import Domain, MeasurableSpace, SigmaAlgebra
        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...     },
        ... )

        Instantiate a measurable space and print its domain.

        >>> measurable_space = MeasurableSpace(domain=X, sig_alg=F)
        >>> print(measurable_space.domain)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'X':
         x
         0
         1
         2
         3
        """
        return self.sig_alg.domain if self.sig_alg is not None else None

    # --------------------- conversion methods --------------------- #

    def make_measure_space(
        self,
        measure: Measure | None = None,
    ) -> MeasureSpace:
        """Convert this measurable space to a measure space by adding a measure.

        Parameters
        ----------
        measure : Measure | None, default=None
            Measure to use. If `None`, the counting measure will be created.

        Returns
        -------
        measure_space : MeasureSpace
            A measure space with this measurable space's domain and sigma-algebra.

        Examples
        --------
        Create an instance of `MeasurableSpace`.

        >>> from sigalg.core import Domain, Measure, MeasurableSpace, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> measurable_space = MeasurableSpace(domain=X, sig_alg=F)

        Create a measure space with a counting measure.

        >>> measure_space = measurable_space.make_measure_space()
        >>> print(measure_space)  # doctest: +NORMALIZE_WHITESPACE
        Measure space (X, F, C)
        =======================
        <BLANKLINE>
        * Domain 'X':
         x
         0
         1
         2
        <BLANKLINE>
        * Sigma algebra 'F':
             F
        x
        0    0
        1    0
        2    1
        <BLANKLINE>
        * Measure 'C':
                C
        u
        0       2
        1       1

        Create a measure space with a custom measure.

        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 7,
        ...         1: 3,
        ...     },
        ... )
        >>> measure_space = measurable_space.make_measure_space(measure=mu)
        >>> print(measure_space)  # doctest: +NORMALIZE_WHITESPACE
        Measure space (X, F, mu)
        ========================
        <BLANKLINE>
        * Domain 'X':
         x
         0
         1
         2
        <BLANKLINE>
        * Sigma algebra 'F':
             F
        x
        0    0
        1    0
        2    1
        <BLANKLINE>
        * Measure 'mu':
                mu
        u
        0        7
        1        3
        """
        from .measure_space import MeasureSpace

        return MeasureSpace(
            domain=self.domain,
            sig_alg=self.sig_alg,
            measure=measure,
        )

    # --------------------- data access methods --------------------- #

    def __iter__(self):
        """Allow unpacking of measurable space components.

        Enables syntax like: `X, F = measurable_space`, where `X` is the domain of the measurable space and `F` is its sigma-algebra.

        Yields
        ------
        domain : Domain
            The domain of the measurable space.
        sig_alg : SigmaAlgebra
            The sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import Domain, MeasurableSpace, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> measurable_space = MeasurableSpace(domain=X, sig_alg=F)
        >>> X1, F1 = measurable_space
        >>> print(X1)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'X':
         x
         0
         1
         2
        >>> print(F1)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
           F
        x
        0  0
        1  0
        2  1
        """
        yield self.domain
        yield self.sig_alg

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a concise string representation of the measurable space.

        Returns
        -------
        repr_str : str
            A string representation showing the measurable space's domain
            and sigma-algebra names.
        """
        if self.domain is not None and self.sig_alg is not None:
            return (
                type(self)._repr_name
                + f"(domain={self.domain.name}, sig_alg={self.sig_alg.name})"
            )
        return type(self)._repr_name + "(empty)"

    def __str__(self) -> str:
        """Return a detailed string representation of the measurable space.

        Returns
        -------
        repr_str : str
            A formatted string showing the measurable space header and detailed
            representations of its components.
        """
        header = type(self)._str_name + f" ({self.domain.name}, {self.sig_alg.name})"
        separator = "=" * len(header)
        return (
            header
            + "\n"
            + separator
            + "\n\n* "
            + str(self.domain)
            + "\n\n* "
            + str(self.sig_alg)
        )

    # --------------------- equality --------------------- #

    def __eq__(self, other: object) -> bool:
        """Check equality with another measurable space.

        Two measurable spaces are equal if they have the same domain and
        sigma-algebra.

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        are_equal : bool
            `True` if the other object is an `MeasurableSpace` with identical `domain` and `sig_alg`, `False` otherwise.
        """
        if not isinstance(other, MeasurableSpace):
            raise TypeError("other must be a MeasurableSpace.")

        return self.domain == other.domain and self.sig_alg == other.sig_alg

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(domain: Domain | None, sig_alg: SigmaAlgebra | None):
        """Validate measurable space construction parameters.

        Parameters
        ----------
        domain : Domain | None
            The domain to validate.
        sig_alg : SigmaAlgebra | None
            The sigma-algebra to validate.

        Raises
        ------
        TypeError
            If `domain` is not a `Domain` instance or `sig_alg`
            is not a `SigmaAlgebra` instance (when provided).
        ValueError
            If `sig_alg`'s domain does not match the provided
            `domain`.
        """
        from ..sigma_algebras import SigmaAlgebra
        from .domain import Domain

        if domain is not None and not isinstance(domain, Domain):
            raise TypeError("domain must be a Domain instance, if given.")
        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be a SigmaAlgebra instance, if given.")

        if domain is not None and sig_alg is not None and sig_alg.domain != domain:
            raise ValueError("sig_alg's domain must match the provided domain.")
