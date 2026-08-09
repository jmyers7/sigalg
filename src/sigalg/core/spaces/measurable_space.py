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

    If both `domain` and `sig_alg` are provided during initialization, the `domain` of the provided `sig_alg` must match the provided `domain`. If only one of them is provided, the other will be automatically created to be compatible with it (i.e. if only `domain` is given, a power-set sigma-algebra will be created on that domain; if only `sig_alg` is given, the domain will be taken from the sigma-algebra). If neither is provided, both will be initialized to `None` and can be set later.

    Parameters
    ----------
    domain : Domain | IndexLike | None, default=None
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
     point
          0
          1
          2
    <BLANKLINE>
    * Sigma algebra 'F':
            atom_ID
    point
    0             0
    1             0
    2             1

    Notes
    -----
    A *measurable space* is a pair $(X, \mathcal{F})$ consisting of a nonempty set $X$ and a $\sigma$-algebra $\mathcal{F}$ on $X$.
    """

    _repr_name = "MeasurableSpace"
    _str_name = "Measurable space"

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        domain: Domain | IndexLike | None = None,
        sig_alg: SigmaAlgebra | None = None,
    ):
        from .domain import Domain

        if domain is not None and not isinstance(domain, Domain):
            domain = Domain(domain)

        self._validate_parameters(domain, sig_alg)
        self._domain, self._sig_alg = self._generate_components(domain, sig_alg)

    def _generate_components(self, domain, sig_alg):
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        parameter_cases = (domain is not None, sig_alg is not None)
        if parameter_cases == (1, 0):
            sig_alg = SigmaAlgebra.power_set(domain)
        if parameter_cases == (0, 1):
            domain = sig_alg._domain

        return domain, sig_alg

    # --------------------- properties --------------------- #

    @property
    def domain(self) -> Domain | None:
        """Get the domain of the measurable space.

        The `domain` parameter is settable. If the measurable space is not empty, the new domain must contain the same number of points as the current domain, and the sigma-algebra will be updated to be defined on the new domain with the same atom structure. If the measurable space is empty, then setting the domain will set the sigma-algebra to be the power-set sigma-algebra on the new domain.

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

        Instantiate a measurable space and print it.

        >>> measurable_space = MeasurableSpace(domain=X, sig_alg=F)
        >>> print(measurable_space)  # doctest: +NORMALIZE_WHITESPACE
        Measurable space (X, F)
        =======================
        <BLANKLINE>
        * Domain 'X':
         point
              0
              1
              2
              3
        <BLANKLINE>
        * Sigma algebra 'F':
                atom_ID
        point
        0             0
        1             1
        2             2
        3             2

        Set the domain of the measurable space to a new domain. Print to check.

        >>> Y = Domain(["a", "b", "c", "d"], name="Y")
        >>> measurable_space.domain = Y
        >>> print(measurable_space)  # doctest: +NORMALIZE_WHITESPACE
        Measurable space (Y, F)
        =======================
        <BLANKLINE>
        * Domain 'Y':
         point
             a
             b
             c
             d
        <BLANKLINE>
        * Sigma algebra 'F':
                atom_ID
        point
        a             0
        b             1
        c             2
        d             2

        Create an empty `MeasurableSpace` instance and set its domain.

        >>> empty_measurable_space = MeasurableSpace()
        >>> empty_measurable_space.domain = Y

        Print the measurable space and note the sigma-algebra is the power-set sigma-algebra by default.

        >>> print(empty_measurable_space)  # doctest: +NORMALIZE_WHITESPACE
        Measurable space (Y, R)
        =======================
        <BLANKLINE>
        * Domain 'Y':
         point
             a
             b
             c
             d
        <BLANKLINE>
        * Sigma algebra 'R':
              point
        point
        a         a
        b         b
        c         c
        d         d
        """
        return self._domain

    @domain.setter
    def domain(self, domain: Domain) -> None:
        """Set the domain of the measurable space.

        If the measurable space is not empty, the new domain must contain the same number of points as the current domain, and the sigma-algebra will be updated to be defined on the new domain with the same atom structure. If the measurable space is empty, then setting the domain will set the sigma-algebra to be the power-set sigma-algebra on the new domain.

        Parameters
        ----------
        domain : Domain
            The new domain to set.

        Raises
        ------
        TypeError
            If `domain` is not a `Domain` instance.
        """
        from .domain import Domain

        if not isinstance(domain, Domain):
            raise TypeError("domain must be a Domain instance.")

        if self.domain is not None:
            self.sig_alg.domain = domain
            self._domain = domain
        else:
            self._domain, self._sig_alg = self._generate_components(
                domain=domain, sig_alg=None
            )

    @property
    def sig_alg(self) -> SigmaAlgebra:
        """Get the sigma-algebra of the measurable space.

        The `sig_alg` property is settable. If the measurable space is not empty, the new sigma-algebra must have the same domain as the current sigma-algebra. If the measurable space is empty, then setting the sigma-algebra will set the domain to be the domain of the new sigma-algebra.

        Returns
        -------
        sig_alg : SigmaAlgebra
            The sigma-algebra of the measurable space.

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

        Instantiate a measurable space and print it.

        >>> measurable_space = MeasurableSpace(domain=X, sig_alg=F)
        >>> print(measurable_space)  # doctest: +NORMALIZE_WHITESPACE
        Measurable space (X, F)
        =======================
        <BLANKLINE>
        * Domain 'X':
         point
              0
              1
              2
              3
        <BLANKLINE>
        * Sigma algebra 'F':
                atom_ID
        point
        0             0
        1             1
        2             2
        3             2

        Define a new sigma-algebra, a sub-sigma-algebra of the first and set the `sig_alg` property of the measurable space.

        >>> G = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 1,
        ...     },
        ...     name="G",
        ... )
        >>> measurable_space.sig_alg = G

        Print the updated measurable space to check.

        >>> print(measurable_space)  # doctest: +NORMALIZE_WHITESPACE
        Measurable space (X, G)
        =======================
        <BLANKLINE>
        * Domain 'X':
         point
              0
              1
              2
              3
        <BLANKLINE>
        * Sigma algebra 'G':
                atom_ID
        point
        0             0
        1             1
        2             1
        3             1

        Create an empty `MeasurableSpace` instance and set its sigma-algebra property.

        >>> empty_measurable_space = MeasurableSpace()
        >>> empty_measurable_space.sig_alg = G

        Print the updated empty measurable space. Note the domain was extracted from the sigma-algebra.

        >>> print(empty_measurable_space)  # doctest: +NORMALIZE_WHITESPACE
        Measurable space (X, G)
        =======================
        <BLANKLINE>
        * Domain 'X':
         point
              0
              1
              2
              3
        <BLANKLINE>
        * Sigma algebra 'G':
                atom_ID
        point
        0             0
        1             1
        2             1
        3             1
        """
        return self._sig_alg

    @sig_alg.setter
    def sig_alg(self, sig_alg: SigmaAlgebra) -> None:
        """Set the sigma-algebra of the measurable space.

        If the measurable space is not empty, the new sigma-algebra must have the same domain as the current sigma-algebra. If the measurable space is empty, then setting the sigma-algebra will set the domain to be the domain of the new sigma-algebra.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The new sigma-algebra to set.

        Raises
        ------
        TypeError
            If `sig_alg` is not a `SigmaAlgebra` instance.
        ValueError
            If the new `sig_alg` does not have the same domain as the current `sig_alg` (when the measurable space is not empty).
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be a SigmaAlgebra instance.")

        if self.sig_alg is not None:
            if self.sig_alg.domain != sig_alg.domain:
                raise ValueError(
                    "New sig_alg must have the same domain as the current sig_alg."
                )
            self._sig_alg = sig_alg
        else:
            self._domain, self._sig_alg = self._generate_components(
                domain=None, sig_alg=sig_alg
            )

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
         point
              0
              1
              2
        <BLANKLINE>
        * Sigma algebra 'F':
                atom_ID
        point
        0             0
        1             0
        2             1
        <BLANKLINE>
        * Measure 'C':
                measure
        atom_ID
        0             2
        1             1

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
         point
             0
             1
             2
        <BLANKLINE>
        * Sigma algebra 'F':
                atom_ID
        point
        0             0
        1             0
        2             1
        <BLANKLINE>
        * Measure 'mu':
                measure
        atom_ID
        0             7
        1             3
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
         point
             0
             1
             2
        >>> print(F1)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                atom_ID
        point
        0             0
        1             0
        2             1
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
