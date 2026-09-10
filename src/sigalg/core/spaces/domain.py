"""A class representing domains of functions and measurable spaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ..indices.index import Index

if TYPE_CHECKING:
    from collections.abc import Hashable

    from ..functions.function import Function
    from ..measures.measure import Measure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from .measurable_space import MeasurableSpace
    from .measure_space import MeasureSpace
    from .sample_space import SampleSpace


class Domain(Index):
    """A class representing domains of functions and measurable spaces.

    Parameters
    ----------
    indices : IndexLike | None, default=None
        The object from which to construct the `Domain`.
    variable_names : list[Hashable] | None, default=None
        A list of variable names for the dimensions of the domain. If `None`, defaults will be generated.
    name : Hashable | None, default=None
        Name identifier for the domain. If `None`, a default name will be generated.

    Examples
    --------
    >>> from sigalg.core import Domain

    Build a `Domain` from a list of indices.

    >>> X = Domain([1, 2, 3])
    >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
    Domain 'X':
     x
     1
     2
     3
    """

    _default_name = "X"
    _repr_name = "Domain"
    _str_name = "Domain"
    _variable_names_prefix = "x"

    # --------------------- conversion methods --------------------- #

    def make_measurable_space(
        self, sig_alg: SigmaAlgebra | None = None
    ) -> MeasurableSpace:
        """Convert this domain to a measurable space by adding a sigma-algebra.

        Parameters
        ----------
        sig_alg : SigmaAlgebra | None, default=None
            Sigma-algebra to use. If `None`, a power set sigma-algebra will be created.

        Returns
        -------
        measurable_space : MeasurableSpace
            A `MeasurableSpace` object with this domain.

        Examples
        --------
        >>> from sigalg.core import Domain, SigmaAlgebra

        Define a domain.

        >>> X = Domain(indices=["s0", "s1", "s2", "s3"], name="X")

        Promote to a `MeasurableSpace` with default power-set sigma-algebra.

        >>> measurable_space = X.make_measurable_space()
        >>> print(measurable_space) # doctest: +NORMALIZE_WHITESPACE
        Measurable space (X, R)
        =======================
        <BLANKLINE>
        * Domain 'X':
          x
         s0
         s1
         s2
         s3
        <BLANKLINE>
        * Sigma algebra 'R':
             R
         x
        s0  s0
        s1  s1
        s2  s2
        s3  s3

        Create a custom sigma-algebra, and promote to a `MeasurableSpace` with this custom object.

        >>> F = SigmaAlgebra(domain=X, mapping={"s0": 0, "s1": 0, "s2": 1, "s3": 1})
        >>> measurable_space = X.make_measurable_space(sig_alg=F)
        >>> print(measurable_space) # doctest: +NORMALIZE_WHITESPACE
        Measurable space (X, F)
        =======================
        <BLANKLINE>
        * Domain 'X':
          x
         s0
         s1
         s2
         s3
        <BLANKLINE>
        * Sigma algebra 'F':
             F
        x
        s0   0
        s1   0
        s2   1
        s3   1
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .measurable_space import MeasurableSpace

        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("If given, sig_alg must be a SigmaAlgebra.")

        return MeasurableSpace(domain=self, sig_alg=sig_alg)

    def make_measure_space(
        self, sig_alg: SigmaAlgebra | None = None, measure: Measure | None = None
    ) -> MeasureSpace:
        """Convert this domain to a measure space by adding a sigma-algebra and measure.

        Parameters
        ----------
        sig_alg : SigmaAlgebra | None, default=None
            Sigma-algebra to use. If `None`, a power set sigma-algebra will be created.
        measure : Measure | None, default=None
            Measure to use. If `None`, the counting measure will be created.

        Returns
        -------
        measure_space : MeasureSpace
            A `MeasureSpace` object with this domain.

        Examples
        --------
        >>> from sigalg.core import Domain, Measure, SigmaAlgebra

        Define a domain.

        >>> X = Domain(indices=["a", "b", "c"])

        Promote to a `MeasureSpace` with default power set sigma-algebra and counting measure.

        >>> measure_space = X.make_measure_space()
        >>> print(measure_space) # doctest: +NORMALIZE_WHITESPACE
        Measure space (X, R, C)
        =======================
        <BLANKLINE>
        * Domain 'X':
         x
         a
         b
         c
        <BLANKLINE>
        * Sigma algebra 'R':
           R
        x
        a  a
        b  b
        c  c
        <BLANKLINE>
        * Measure 'C':
                C
        x
        a       1
        b       1
        c       1

        Create a custom sigma-algebra and measure, and promote to a `MeasureSpace` with these custom objects.

        >>> F = SigmaAlgebra(domain=X, mapping=
        ...     {
        ...         "a": 0,
        ...         "b": 1,
        ...         "c": 1,
        ...     },
        ... )
        >>> mu = Measure(domain=F, mapping=
        ...     {
        ...         0: 2,
        ...         1: 8,
        ...     },
        ... )
        >>> measure_space = X.make_measure_space(sig_alg=F, measure=mu)
        >>> print(measure_space)  # doctest: +NORMALIZE_WHITESPACE
        Measure space (X, F, mu)
        ========================
        <BLANKLINE>
        * Domain 'X':
         x
         a
         b
         c
        <BLANKLINE>
        * Sigma algebra 'F':
           F
        x
        a  0
        b  1
        c  1
        <BLANKLINE>
        * Measure 'mu':
                mu
        F
        0        2
        1        8
        """
        from ..measures.measure import Measure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .measure_space import MeasureSpace

        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("`sig_alg` must be a `SigmaAlgebra` or `None`.")
        if measure is not None and not isinstance(measure, Measure):
            raise TypeError("`measure` must be a `Measure` or `None`.")

        return MeasureSpace(
            domain=self,
            sig_alg=sig_alg,
            measure=measure,
        )

    def to_sample_space(self) -> SampleSpace:
        """Copy this domain and return it as an instance of `SampleSpace`.

        Returns
        -------
        sample_space : SampleSpace
            A new `SampleSpace` object with the same underlying data as this domain.

        Examples
        --------
        >>> from sigalg.core import Domain

        Define a domain.

        >>> X = Domain(indices=["a", "b", "c"], name="X")

        Promote to a `SampleSpace`.

        >>> sample_space = X.to_sample_space()
        >>> print(sample_space) # doctest: +NORMALIZE_WHITESPACE
        Sample space 'X':
         x
         a
         b
         c
        """
        from .sample_space import SampleSpace

        return SampleSpace._from_validated(data=self.data.copy(), name=self.name)

    def to_function(
        self,
        domain_name: Hashable = "Omega",
        domain_variable_name: Hashable = "omega",
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable = "f",
    ) -> Function:
        """Convert the current domain to a `Function` instance.

        Parameters
        ----------
        domain_name : Hashable, default="Omega"
            Name of the domain of the function.
        domain_variable_name : Hashable, default="omega"
            Name of the domain variable.
        index_kind : Literal["Index", "Time"], default="Index"
            Kind of the index.
        index_name : Hashable | None, default=None
            Name of the index.
        name : Hashable, default="f"
            Name of the function.

        Returns
        -------
        function : Function
            A new `Function` object representing the current domain.

        Examples
        --------
        >>> from sigalg.core import Domain

        Define a 2-dimensional domain.

        >>> X = Domain.cartesian_product([["a", "b"], [1, 2]], variable_names=["letter", "number"])

        Convert to a function with default domain parameter values.

        >>> f = X.to_function()
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
        i     letter  number
        omega
        0          a       1
        1          a       2
        2          b       1
        3          b       2

        Convert to a function with custom domain parameters values.

        >>> g = X.to_function(name="g", domain_variable_name="s", domain_name="S")
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
        i letter  number
        s
        0      a       1
        1      a       2
        2      b       1
        3      b       2
        """
        from ..functions.function import Function
        from ..indices.time import Time

        index_class = Index if index_kind == "Index" else Time
        func_data = self.data.to_frame().squeeze(axis=1).reset_index(drop=True)
        func_data.index.name = domain_variable_name
        func_data.columns.name = index_class._variable_names_prefix

        return Function._from_validated(
            data=func_data,
            kind="any",
            domain_kind=type(self).__name__,
            domain_name=domain_name,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
        )

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a concise string representation of the domain.

        Returns
        -------
        repr_str : str
            String representation of the domain.
        """
        if self.data is None:
            return f"{type(self)._repr_name}(empty)"
        else:
            return f"{type(self)._repr_name}(size={len(self.data)}, variable_names={self.variable_names}, name={self.name})"

    # --------------------- equality --------------------- #

    def __eq__(self, other: Domain) -> bool:
        """Check equality with another domain.

        Two domains are equal if they have the same variable names (in the same order) and the same elements (in the same order).

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the domains are considered equal according to the above criteria, `False` otherwise.
        """
        if not isinstance(other, Domain):
            return False
        return super().__eq__(other)
