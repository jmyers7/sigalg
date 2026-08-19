"""A class representing a sigma-algebra."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ...typing.index_like import IndexLike
    from ...typing.mapping_like import MappingLike
    from ..functions.function import Function
    from ..indices.index import Index
    from ..measures.measure import Measure
    from ..spaces.domain import Domain
    from ..spaces.set import Set
    from .lattice import Lattice


class SigmaAlgebra:
    r"""A class representing a sigma-algebra on a domain.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    domain : Domain | IndexLike | None, default=None
        The domain over which the sigma-algebra is defined.
    mapping: MappingLike | None, default=None
        The mapping object assigning points to atom IDs.
    variable_names : list[Hashable] | None, default=None
        The variables names of the atom identifiers of the sigma-algebra.
    name : Hashable, default="F"
        The name of the sigma-algebra.

    Raises
    ------
    ValueError
        If `variables_names` is not a list of hashables, if given.

    Examples
    --------
    >>> from sigalg.core import Domain, Index, SigmaAlgebra

    Construct a `SigmaAlgebra` with two atoms.

    >>> X = Domain.from_sequence(size=3)
    >>> mapping = {
    ...     0: 1,
    ...     1: 0,
    ...     2: 0,
    ... }
    >>> F = SigmaAlgebra(domain=X, mapping=mapping)
    >>> print(F)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'F':
       F
    x
    0  1
    1  0
    2  0

    Print the `atom_space` of the sigma-algebra, which is the domain whose points are the unique atom IDs. Note the default variable name `u`.

    >>> print(F.atom_space)  # doctest: +NORMALIZE_WHITESPACE
    Domain 'F':
     F
     1
     0

    Construct a `SigmaAlgebra` on the same domain with 2-dimensional atom IDs and a custom index.

    >>> mapping = {
    ...     0: (1, 2),
    ...     1: (0, 1),
    ...     2: (0, 1),
    ... }
    >>> I = Index([1, 2])
    >>> G = SigmaAlgebra(domain=X, mapping=mapping, index=I, name="G")
    >>> print(G)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'G':
    i  1  2
    x
    0  1  2
    1  0  1
    2  0  1

    Notes
    -----
    A *$\sigma$-algebra* $\mathcal{F}$ on a nonempty set $X$ is a collection of subsets of $X$ that contains $X$, and is closed under complementation and countable unions. In the case that $X$ is finite (as it always is, in SigAlg), then $\mathcal{F}$ needs only to be closed under finite unions.

    A $\sigma$-algebra $\mathcal{F}$ determines its *atoms*, which are the nonempty sets $A\in \mathcal{F}$ that are *minimal* with respect to subset inclusion, in the following sense: If $B\in \mathcal{F}$ is nonempty and $B\subset A$, then necessarily $A=B$. Conversely, provided that $X$ is finite, the $\sigma$-algebra $\mathcal{F}$ is completely recoverable from its atoms, in the sense that every event $A\in \mathcal{F}$ is a disjoint union of atoms.

    If $\{A_i\}_{i\in I}$ is the set of atoms, indexed by a finite set $I$, then there is a mapping $X \to I$ given by $x \mapsto i$, where $A_i$ is the unique atom that contains $x$. This mapping is what SigAlg uses to represent $\sigma$-algebras. The indices in $I$ are called *atom identifiers*. The atom identifiers may consist of tuples, in which case the $\sigma$-algebra is said to have *multi-dimensional* atom identifiers, and the *dimension* of the $\sigma$-algebra is the common length of the tuples.
    """

    _properties = []

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        domain: IndexLike | None = None,
        mapping: MappingLike | None = None,
        variable_names: list[Hashable] | None = None,
        domain_kind: Literal["Domain", "SampleSpace"] = "Domain",
        domain_name: Hashable | None = None,
        output_name: Hashable | None = None,
        index: IndexLike | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable = "F",
    ) -> None:
        from ...validation.domain_index_validator import DomainIndexValidator
        from ...validation.mapping_validator import MappingValidator

        u = DomainIndexValidator(
            domain=domain,
            domain_kind=domain_kind,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
        )

        domain = u.domain
        index = u.index
        self.domain_kind = u.domain_kind
        self.domain_name = u.domain_name
        self.index_kind = u.index_kind
        self.index_name = u.index_name

        if output_name is None:
            output_name = name

        v = MappingValidator(
            domain=domain,
            mapping=mapping,
            domain_kind=domain_kind,
            output_name=output_name,
            index=index,
            index_kind=index_kind,
            name=name,
        )

        self.data = v.data
        self.name = v.name

        self._validate_variable_names(variable_names, self.dimension)
        self._variable_names = variable_names

    @classmethod
    def _from_validated(
        cls,
        *,
        data: pd.Series | pd.DataFrame,
        variable_names: list[Hashable] | None,
        domain_kind: Literal["Domain", "SampleSpace"],
        domain_name: Hashable,
        index_kind: Literal["Index", "Time"],
        index_name: Hashable | None,
        name: Hashable,
    ) -> SigmaAlgebra:
        sig_alg = object.__new__(cls)
        sig_alg.data = data
        sig_alg._variable_names = variable_names
        sig_alg.name = name
        sig_alg.domain_kind = domain_kind
        sig_alg.domain_name = domain_name
        sig_alg.index_kind = index_kind
        sig_alg.index_name = index_name

        return sig_alg

    @classmethod
    def power_set(
        cls,
        domain: IndexLike,
        domain_kind: Literal["Domain", "SampleSpace"] = "Domain",
        domain_name: Hashable | None = None,
        index_name: Hashable = "I",
        name: Hashable = "R",
    ) -> SigmaAlgebra:
        r"""Create the power-set sigma-algebra over a given domain.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        domain : IndexLike
            The domain over which to create the power-set sigma-algebra.
        name : Hashable, default="R"
            Name identifier for the sigma algebra.

        Returns
        -------
        sig_alg : SigmaAlgebra
            A new `SigmaAlgebra` instance representing the power-set sigma-algebra.

        Examples
        --------
        Create a power-set sigma-algebra.

        >>> from sigalg.core import Domain, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> G = SigmaAlgebra.power_set(X, name="G")
        >>> print(G)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
           G
        x
        0  0
        1  1
        2  2

        The atom space of a power set contructed via this method is the original domain itself.

        >>> print(G.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'X':
         x
         0
         1
         2

        Create another power-set sigma-algebra, this time on a 2-dimensional domain.

        >>> Y = Domain.cartesian_product(
        ...     [[1, 2], ["a", "b"]], name="Y", variable_names=["number", "letter"]
        ... )
        >>> F = SigmaAlgebra.power_set(Y, name="F")
        >>> print(F)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
        i             number  letter
        number letter
        1      a           1       a
               b           1       b
        2      a           2       a
               b           2       b
        >>> print(F.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'Y':
         number letter
             1      a
             1      b
             2      a
             2      b

        Notes
        -----
        The *power-set $\sigma$-algebra* on a nonempty set $X$ consists of all subsets of $X$. Its atoms are all singleton subsets. It is the finest $\sigma$-algebra on $X$.
        """
        from ...validation.domain_index_validator import DomainIndexValidator

        if domain is None:
            raise TypeError("The domain must be given for the power_set method.")

        u = DomainIndexValidator(
            domain=domain,
            domain_kind=domain_kind,
            domain_name=domain_name,
            index_name=index_name,
        )

        domain = u.domain
        domain_kind = u.domain_kind
        domain_name = u.domain_name
        index_name = u.index_name

        if domain.dimension > 1:
            data = domain.data.to_frame()
            data.columns.name = "i"
        else:
            data = domain.data.to_series()
            data.name = name

        return cls._from_validated(
            data=data,
            variable_names=domain.variable_names,
            name=name,
            domain_kind=domain_kind,
            domain_name=domain_name,
            index_kind="Index",
            index_name=index_name,
        )

    @classmethod
    def trivial(
        cls,
        domain: Domain | IndexLike,
        variable_name: Hashable | None = None,
        domain_kind: Literal["Domain", "SampleSpace"] = "Domain",
        domain_name: Hashable | None = None,
        name: Hashable = "T",
    ) -> SigmaAlgebra:
        r"""Create the trivial sigma-algebra over a given domain.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        domain : Domain | IndexLike
            The domain over which to create the trivial sigma-algebra.
        name : Hashable, default="T"
            Name identifier for the sigma-algebra.

        Returns
        -------
        sig_alg : SigmaAlgebra
            A new `SigmaAlgebra` instance representing the trivial sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega1 = SampleSpace.from_sequence(size=3, name="Omega1")
        >>> G = SigmaAlgebra.trivial(Omega1, name="G")
        >>> print(G)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
           G
        s
        0  0
        1  0
        2  0
        >>> print(G.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'G':
         G
         0
        >>> Omega2 = SampleSpace.cartesian_product(
        ...     [[1, 2], ["a", "b"]], name="Omega2", variable_names=["number", "letter"]
        ... )
        >>> F = SigmaAlgebra.trivial(Omega2, name="F")
        >>> print(F)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                       F
        number letter
        1      a       0
               b       0
        2      a       0
               b       0
        >>> print(F.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'F':
         F
         0

        Notes
        -----
        The *trivial $\sigma$-algebra* on a nonempty set $X$ consists of only the sets $X$ and $\emptyset$. Its single atom is $X$ itself. It is the coarsest $\sigma$-algebra on $X$.
        """
        from ...validation.domain_index_validator import DomainIndexValidator

        if domain is None:
            raise TypeError("The domain must be given for the power_set method.")

        u = DomainIndexValidator(
            domain=domain,
            domain_kind=domain_kind,
            domain_name=domain_name,
        )

        domain = u.domain
        domain_kind = u.domain_kind
        domain_name = u.domain_name

        data = pd.Series(0, name=name, index=domain.data)

        return cls._from_validated(
            data=data,
            variable_names=[variable_name] if variable_name else None,
            name=name,
            domain_kind=domain_kind,
            domain_name=domain.name,
            index_kind="Index",
            index_name=None,
        )

    @classmethod
    def from_rand(
        cls,
        domain: IndexLike | None = None,
        super: SigmaAlgebra | None = None,
        num_atoms: int = 1,
        dim: int = 1,
        atom_ID_range: tuple[int, int] | None = None,
        variable_names: list[Hashable] | None = None,
        domain_kind: Literal["Domain", "SampleSpace"] = "Domain",
        domain_name: Hashable | None = None,
        output_name: Hashable | None = None,
        index: IndexLike | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable = "F",
        random_state: int | np.random.Generator | None = None,
    ) -> SigmaAlgebra:
        """Generate a sigma-algebra with random atom identifiers.

        Parameters
        ----------
        domain : Domain | IndexLike | None, default=None
            The domain over which to create the sigma-algebra. If `None`, then `super` must be provided and the domain will be obtained from it.
        super : SigmaAlgebra | None, default=None
            If provided, the randomly generated sigma-algebra will be a sub-sigma-algebra of `super`.
        num_atoms : int, default=1
            The number of atoms in the sigma-algebra. Creates the trivial sigma-algebra by default.
        dim : int, default=1
            The dimension of the atom identifiers.
        atom_ID_range : tuple[int, int] | None, default=None
            A tuple of the form (min, max), or `None`. If not `None`, the atom identifiers will be drawn from the range [min, max). If `None`, the atom identifiers will be drawn from the range [0, num_atoms).
        variable_names : list[Hashable] | None, default=None
            A list of variable names for the atom identifiers.
        name : Hashable, default="F"
            Name identifier for the sigma-algebra.
        random_state : int | np.random.Generator | None, default=None
            An optional seed for the random number generator.

        Raises
        ------
        TypeError
            If `num_atoms` or `dim` is not an integer, if `atom_ID_range` is not a tuple of two integers (if provided), or if `random_state` is not an integer, `np.random.Generator`, or `None`.
        ValueError
            If `num_atoms` or `dim` is not a positive integer, if `atom_ID_range` does not have min < max, or if `num_atoms` is greater than the number of points in the domain.

        Returns
        -------
        random_sig_alg : SigmaAlgebra
            A new `SigmaAlgebra` instance with random atom identifiers.

        Examples
        --------
        Generate a sigma-algebra with three random atoms and 1-dimensional atom identifiers.

        >>> from sigalg.core import Domain, SigmaAlgebra
        >>> X = Domain.from_sequence(size=5)
        >>> F = SigmaAlgebra.from_rand(
        ...     domain=X,
        ...     num_atoms=3,
        ...     random_state=42,
        ... )
        >>> print(F)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
           F
        x
        0  0
        1  0
        2  0
        3  2
        4  1

        Generate a sigma-algebra with three random atoms and 3-dimensional atom identifiers.

        >>> G = SigmaAlgebra.from_rand(
        ...     domain=X,
        ...     num_atoms=3,
        ...     dim=3,
        ...     random_state=42,
        ...     name="G",
        ... )
        >>> print(G)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
        i  0  1  2
        x
        0  2  1  2
        1  0  0  2
        2  2  1  2
        3  2  1  2
        4  1  0  0

        Generate a sigma-algebra with three random atoms and 2-dimensional atom identifiers with values in the range [10, 15).

        >>> H = SigmaAlgebra.from_rand(
        ...     domain=X,
        ...     num_atoms=3,
        ...     atom_ID_range=(10, 15),
        ...     dim=2,
        ...     random_state=42,
        ...     name="H",
        ... )
        >>> print(H)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'H':
        i   0   1
        x
        0  14  14
        1  10  10
        2  14  14
        3  13  13
        4  14  14

        Create a random sub-sigma-algebra of `H` with two atoms:

        >>> K = SigmaAlgebra.from_rand(
        ...     super=H,
        ...     num_atoms=2,
        ...     random_state=42,
        ...     name="K",
        ... )
        >>> print(K)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'K':
           K
        x
        0  0
        1  1
        2  0
        3  1
        4  0
        >>> print(K <= H)
        True
        """
        from ...validation.domain_index_validator import DomainIndexValidator
        from ..indices._helpers import random_tuples
        from ..indices.index import Index
        from ..indices.time import Time

        if not isinstance(num_atoms, int) or num_atoms <= 0:
            raise TypeError("num_atoms must be a positive integer.")
        if not isinstance(dim, int) or dim <= 0:
            raise TypeError("dim must be a positive integer.")
        if atom_ID_range is not None:
            if (
                not isinstance(atom_ID_range, tuple)
                or len(atom_ID_range) != 2
                or not all(isinstance(x, int) for x in atom_ID_range)
            ):
                raise TypeError(
                    "atom_ID_range must be a tuple of two integers (min, max)."
                )
            if atom_ID_range[0] >= atom_ID_range[1]:
                raise ValueError(
                    "atom_ID_range must be a tuple of two integers (min, max) with min < max."
                )
        if random_state is not None and not isinstance(
            random_state, (int, np.random.Generator)
        ):
            raise TypeError(
                "random_state must be an integer, np.random.Generator, or None."
            )
        if (domain is None) == (super is None):
            raise ValueError("Exactly one of domain or super must be provided.")

        if domain is None:
            domain = super.domain
            if num_atoms > super.num_atoms:
                raise ValueError(
                    "num_atoms must be less than or equal to the number of atoms in the super-sigma-algebra."
                )

        u = DomainIndexValidator(
            domain=domain,
            domain_kind=domain_kind,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
        )

        domain = u.domain
        index = u.index
        domain_kind = u.domain_kind
        domain_name = u.domain_name
        index_kind = u.index_kind
        index_name = u.index_name

        if output_name is None:
            output_name = name

        if num_atoms > len(domain):
            raise ValueError(
                "num_atoms must be less than or equal to the number of points."
            )

        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        atom_IDs = random_tuples(
            size=num_atoms,
            sample_range=atom_ID_range,
            dim=dim,
            random_state=rng,
        )

        if super is not None:
            population = super.atom_ids.copy()
            partitioned_atom_IDs = cls._partition(
                population=population, size=num_atoms, random_state=rng
            )
            partitioned = []
            for partition in partitioned_atom_IDs:
                partitioned.append(
                    list(
                        chain.from_iterable(
                            super.atom_id_to_points[atom_id] for atom_id in partition
                        )
                    )
                )

        else:
            population = list(domain)
            partitioned = cls._partition(
                population=population, size=num_atoms, random_state=rng
            )

        mapping = {
            point: atom_ID
            for partition, atom_ID in zip(partitioned, atom_IDs)
            for point in partition
        }
        mapping = {point: mapping[point] for point in domain}

        if dim == 1:
            data = pd.Series(mapping.values(), index=domain.data, name=output_name)
        else:
            data = pd.DataFrame(mapping.values(), index=domain.data)
            if index is None:
                index_class = Index if index_kind == "Index" else Time
                data.columns.name = index_class._variable_names_prefix
                index = index_class._from_validated(data=data.columns, name=index_name)
            data.columns = index.data

        cls._validate_variable_names(variable_names, dim)

        return cls._from_validated(
            data=data,
            variable_names=variable_names,
            name=name,
            domain_kind=domain_kind,
            domain_name=domain_name,
            index_kind=index_kind,
            index_name=index_name,
        )

    @classmethod
    def from_function(
        cls,
        function: Function,
    ) -> SigmaAlgebra:
        r"""Create a sigma-algebra induced by a measurable vector.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        function : Function
            The function from which to generate the sigma-algebra.

        Raises
        ------
        TypeError
            If `function` is not a `Function` instance.

        Returns
        -------
        sig_alg : SigmaAlgebra
            A new `SigmaAlgebra` instance induced by the given function.

        Examples
        --------
        >>> from sigalg.core import Domain, Function, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> f = Function(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (2, 4),
        ...     },
        ... )
        >>> sigma_f = SigmaAlgebra.from_function(f)
        >>> print(sigma_f)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'sigma(f)':
        i  0  1
        x
        0  1  2
        1  1  2
        2  2  4
        >>> print(sigma_f.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'sigma(f)':
         f_0  f_1
           1    2
           2    4
        >>> g = Function(domain=X, mapping=dict(zip(X, [1, 1, 2])), name="g")
        >>> sigma_g = SigmaAlgebra.from_function(g)
        >>> print(sigma_g)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'sigma(g)':
           sigma(g)
        x
        0         1
        1         1
        2         2

        Notes
        -----
        Let $f: X \to \mathbb{R}^d$ be a function defined on a set $X$. The *$\sigma$-algebra induced by $f$*, denoted $\sigma(f)$, is the $\sigma$-algebra generated by the preimages of Borel sets in $\mathbb{R}^d$ under $f$. In SigAlg, in which $X$ is finite and $\sigma$-algebras are determined by their atoms, we may take the atom identifiers to be the unique values of $f$ on $X$.
        """
        from ..functions.function import Function

        if not isinstance(function, Function):
            raise TypeError("function must be a Function instance.")

        if function.name.startswith("(") and function.name.endswith(")"):
            name = f"sigma{function.name}"
        else:
            name = f"sigma({function.name})"

        if isinstance(function.data, pd.DataFrame):
            sig_alg_data = function.data.copy()
        else:
            sig_alg_data = function.data.copy()
            sig_alg_data.name = name

        if function.index is not None:
            variable_names = list(function.component_names.values())
        else:
            variable_names = [function.name]

        return SigmaAlgebra._from_validated(
            data=sig_alg_data,
            variable_names=variable_names,
            name=name,
            domain_kind=type(function.domain).__name__,
            domain_name=function.domain.name,
            index_kind=type(function.index).__name__ if function.index else "Index",
            index_name=function.index.name if function.index else None,
        )

    @classmethod
    def cartesian_product(
        cls,
        factors: list[SigmaAlgebra],
        variable_names: list[Hashable] | None = None,
        domain_name: Hashable | None = None,
        index: IndexLike | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
    ) -> SigmaAlgebra:
        r"""Compute the Cartesian product of a list of sigma-algebras.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        factors : list[SigmaAlgebra]
            The factors of the Cartesian product.
        variable_names : list[Hashable] | None, default=None
            The variable names of the atom identifiers of the resulting sigma-algebra. If `None`, the variable names will be generated automatically.
        name : Hashable | None, default=None
            The name of the resulting sigma-algebra. If `None`, the name will be generated automatically.

        Raises
        ------
        TypeError
            If any element of `factors` is not a `SigmaAlgebra`, or if `variable_names` is not a list of hashables (if provided), or if `name` is not hashable (if provided).
        ValueError
            If the length of `variable_names` does not match the sum of the dimensions of the sigma-algebras in `factors`.

        Returns
        -------
        cartesian_product : SigmaAlgebra
            The Cartesian product of the sigma-algebras in `factors`.

        Examples
        --------
        >>> from sigalg.core import Domain, SigmaAlgebra

        Define two sigma-algebras on two domains.

        >>> X = Domain.from_sequence(size=3, variable_name="x")
        >>> Y = Domain.from_sequence(size=3, variable_name="y", name="Y")
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     },
        ...     variable_names=["u"],
        ... )
        >>> G = SigmaAlgebra(
        ...     domain=Y,
        ...     mapping={
        ...         0: ("a", "b"),
        ...         1: ("a", "b"),
        ...         2: ("c", "d"),
        ...     },
        ...     variable_names=["v", "w"],
        ...     name="G",
        ... )
        >>> print(F)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
           F
        x
        0  0
        1  1
        2  1
        >>> print(G)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
        i  0  1
        y
        0  a  b
        1  a  b
        2  c  d

        Compute the Cartesian product of the two sigma-algebras usings the `cartesian_product` method.

        >>> prod_sig_alg = SigmaAlgebra.cartesian_product([F, G])
        >>> print(prod_sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F x G':
        i    0  1  2
        x y
        0 0  0  a  b
          1  0  a  b
          2  0  c  d
        1 0  1  a  b
          1  1  a  b
          2  1  c  d
        2 0  1  a  b
          1  1  a  b
          2  1  c  d

        Print the atom space.

        >>> print(prod_sig_alg.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'F x G':
         u v w
         0 a b
         0 c d
         1 a b
         1 c d

        Compute the same Cartesian product using the `@` operator.

        >>> prod_sig_alg = SigmaAlgebra.cartesian_product([F, G])
        >>> print(prod_sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F x G':
        i    0  1  2
        x y
        0 0  0  a  b
          1  0  a  b
          2  0  c  d
        1 0  1  a  b
          1  1  a  b
          2  1  c  d
        2 0  1  a  b
          1  1  a  b
          2  1  c  d

        Notes
        -----
        Let $\mathcal{F}$ be a $\sigma$-algebra on a finite nonempty set $X$, and let $\mathcal{G}$ be a $\sigma$-algebra on a finite nonempty set $Y$. The *Cartesian product* of $\mathcal{F}$ and $\mathcal{G}$, denoted $\mathcal{F} \times \mathcal{G}$, is the $\sigma$-algebra on $X\times Y$ whose atoms are all sets of the form $A \times B$, where $A$ is an atom of $\mathcal{F}$ and $B$ is an atom of $\mathcal{G}$. If $I$ is the set of atom identifiers of $\mathcal{F}$ and $J$ is the set of atom identifiers of $\mathcal{G}$, then the atom identifiers of $\mathcal{F} \times \mathcal{G}$ are all tuples $(i, j)$ with $i\in I$ and $j\in J$.
        """
        from ...validation.domain_index_validator import DomainIndexValidator
        from .._utils.utils import subscript_var_names, to_df
        from ..indices.index import Index
        from ..indices.time import Time
        from ..spaces.domain import Domain

        if not all(isinstance(sig_alg, SigmaAlgebra) for sig_alg in factors):
            raise TypeError(
                "All elements of `factors` must be instances of SigmaAlgebra."
            )
        if variable_names is not None and not isinstance(variable_names, list):
            raise TypeError("`variable_names` must be a list or None.")
        if isinstance(variable_names, list) and not all(
            isinstance(name, Hashable) for name in variable_names
        ):
            raise TypeError("All elements of `variable_names` must be hashable.")
        if variable_names is not None and len(variable_names) != sum(
            sig_alg.dimension for sig_alg in factors
        ):
            raise ValueError(
                "The length of `variable_names` must match the sum of the dimensions of the sigma-algebras in `factors`."
            )

        u = DomainIndexValidator(
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
        )

        index = u.index
        index_kind = u.index_kind
        index_name = u.index_name

        if all(sig_alg.is_power_set for sig_alg in factors):
            domain = Domain.cartesian_product([sig_alg.domain for sig_alg in factors])
            return SigmaAlgebra.power_set(domain)

        domain_var_names = subscript_var_names(
            [sig_alg.domain.variable_names for sig_alg in factors],
            grouped=True,
        )
        sig_alg_var_names = subscript_var_names(
            [sig_alg.variable_names for sig_alg in factors],
            grouped=True,
        )

        sig_alg_data = []

        for domain_vars, sig_alg_vars, sig_alg in zip(
            domain_var_names, sig_alg_var_names, factors
        ):
            data = to_df(sig_alg.data)
            data.columns = sig_alg_vars
            data = data.add_suffix("_ID")
            data.index.names = domain_vars
            sig_alg_data.append(data)

        product_data = sig_alg_data[0].reset_index()

        for next_data in sig_alg_data[1:]:
            product_data = pd.merge(
                left=product_data,
                right=next_data.reset_index(),
                how="cross",
            )

        mapping = product_data.set_index(
            [name for lst in domain_var_names for name in lst]
        )

        if index is None:
            index_class = Index if index_kind == "Index" else Time
            index_data = pd.RangeIndex(
                mapping.shape[1], name=index_class._variable_names_prefix
            )
            index = index_class._from_validated(data=index_data, name=index_name)

        mapping.columns = index.data

        if name is None:
            name = " x ".join([sig_alg.name for sig_alg in factors])

        if variable_names is None:
            variable_names = sum(sig_alg_var_names, [])
        if domain_name is None:
            domain_name = " x ".join(sig_alg.domain.name for sig_alg in factors)

        return SigmaAlgebra._from_validated(
            data=mapping,
            variable_names=variable_names,
            name=name,
            domain_kind="Domain",
            domain_name=domain_name,
            index_kind=index_kind,
            index_name=index_name,
        )

    @classmethod
    def cartesian_power(cls, sig_alg: SigmaAlgebra, n: int) -> SigmaAlgebra:
        """Form the Cartesian power of the sigma-algebra.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The base of the Cartesian power.
        n : int
            The power of the Cartesian power.

        Raises
        ------
        TypeError
            If `n` is not an integer, or if `sig_alg` is not a `SigmaAlgebra`.
        ValueError
            If `n` is not a positive integer.

        Returns
        -------
        power : SigmaAlgebra
            The Cartesian power.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: (1, "a"),
        ...         1: (1, "a"),
        ...         2: (2, "b"),
        ...     },
        ... )
        >>> F_3 = SigmaAlgebra.cartesian_power(F, 3)
        >>> print(F_3)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F ^ 3':
        i            0  1  2  3  4  5
        s_0 s_1 s_2
        0   0   0    1  a  1  a  1  a
                1    1  a  1  a  1  a
                2    1  a  1  a  2  b
            1   0    1  a  1  a  1  a
                1    1  a  1  a  1  a
                2    1  a  1  a  2  b
            2   0    1  a  2  b  1  a
                1    1  a  2  b  1  a
                2    1  a  2  b  2  b
        1   0   0    1  a  1  a  1  a
                1    1  a  1  a  1  a
                2    1  a  1  a  2  b
            1   0    1  a  1  a  1  a
                1    1  a  1  a  1  a
                2    1  a  1  a  2  b
            2   0    1  a  2  b  1  a
                1    1  a  2  b  1  a
                2    1  a  2  b  2  b
        2   0   0    2  b  1  a  1  a
                1    2  b  1  a  1  a
                2    2  b  1  a  2  b
            1   0    2  b  1  a  1  a
                1    2  b  1  a  1  a
                2    2  b  1  a  2  b
            2   0    2  b  2  b  1  a
                1    2  b  2  b  1  a
                2    2  b  2  b  2  b
        >>> print(F_3.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'F ^ 3':
         F_0 F_1  F_2 F_3  F_4 F_5
           1   a    1   a    1   a
           1   a    1   a    2   b
           1   a    2   b    1   a
           1   a    2   b    2   b
           2   b    1   a    1   a
           2   b    1   a    2   b
           2   b    2   b    1   a
           2   b    2   b    2   b
        """
        name = f"{sig_alg.name} ^ {n}"
        return SigmaAlgebra.cartesian_product(factors=[sig_alg] * n, name=name)

    # --------------------- dunder operators --------------------- #

    def __matmul__(self, other: SigmaAlgebra) -> SigmaAlgebra:
        """Form the Cartesian product of this instance of `SigmaAlgebra` with another.

        Internally calls the `cartesian_product` method.

        Parameters
        ----------
        other : SigmaAlgebra
            The other sigma-algebra to form the Cartesian product with.

        Returns
        -------
        cartesian_product : SigmaAlgebra
            The tensor product.
        """
        return type(self).cartesian_product(factors=[self, other])

    def __xor__(self, n: int) -> SigmaAlgebra:
        """Form the Cartesian power of this instance of `SigmaAlgebra`.

        Internally calls the `cartesian_power` method.

        Parameters
        ----------
        n : int
            The power of the Cartesian power.

        Returns
        -------
        cartesian_power : SigmaAlgebra
            The Cartesian power.
        """
        return SigmaAlgebra.cartesian_power(sig_alg=self, n=n)

    # --------------------- utils --------------------- #

    @staticmethod
    def _partition(
        population: list,
        size: int,
        shuffle: bool = True,
        random_state: int | np.random.Generator | None = None,
    ) -> list:
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        if shuffle:
            rng.shuffle(population)

        possible_cut_points = list(range(1, len(population)))
        cut_points = sorted(
            rng.choice(possible_cut_points, size=size - 1, replace=False).tolist()
        )
        cut_points.insert(0, 0)
        cut_points.append(len(population))

        partitioned = [
            population[cut_points[i] : cut_points[i + 1]] for i in range(size)
        ]
        return partitioned

    @staticmethod
    def _validate_variable_names(
        variable_names: list[Hashable] | None, dim: int
    ) -> None:
        if variable_names is not None and (
            not isinstance(variable_names, list)
            or not all(isinstance(name, Hashable) for name in variable_names)
            or len(variable_names) != dim
        ):
            raise TypeError(
                "If given, variable_names must be a list of hashables of the same length as the dimension of the atom identifiers of the sigma-algebra."
            )

    # --------------------- properties --------------------- #

    @cached_property
    def domain(self) -> Domain | None:
        """Get the domain over which this sigma-algebra is defined.

        Returns
        -------
        domain : Domain | None
            The domain of this sigma-algebra.

        Examples
        --------
        Define a sigma-algebra

        >>> from sigalg.core import Domain, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 1,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ...     name="F",
        ... )
        >>> print(F.domain)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'X':
         x
         0
         1
         2
        """
        from ..spaces.domain import Domain
        from ..spaces.sample_space import SampleSpace

        if self.data is not None:
            domain_class = Domain if self.domain_kind == "Domain" else SampleSpace

            return domain_class._from_validated(
                data=self.data.index, name=self.domain_name
            )
        else:
            return None

    @property
    def variable_names(self) -> list[Hashable] | None:
        """Get the variable names of the sigma-algebra.

        Returns
        -------
        names : list[Hashable] | None
            The variables names of the sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import Domain, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (0, 1),
        ...     },
        ...     variable_names=["x", "y"],
        ... )
        >>> F.variable_names
        ['x', 'y']
        >>> print(F.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'F':
         x  y
         1  2
         0  1
        """
        if self.data is not None:
            if self._variable_names is None:
                return (
                    [
                        f"{self.name}_{i}".replace("(", "")
                        .replace(")", "")
                        .replace(", ", "_")
                        for i in self.index
                    ]
                    if self.dimension > 1
                    else [self.name]
                )
            else:
                return self._variable_names

    @property
    def index(self) -> Index | None:
        """Get the index of the sigma-algebra."""
        import pandas as pd

        from ..indices.index import Index
        from ..indices.time import Time

        if isinstance(self.data, pd.DataFrame):
            index_class = Index if self.index_kind == "Index" else Time
            index = index_class._from_validated(
                data=self.data.columns, name=self.index_name
            )
            return index
        else:
            return None

    @cached_property
    def up_lattice(self) -> Lattice | None:
        """Pass."""
        from .lattice import Lattice

        if self.data is not None:
            return Lattice(base=self, type="upward")

        else:
            return None

    @cached_property
    def down_lattice(self) -> Lattice | None:
        """Pass."""
        from .lattice import Lattice

        if self.data is not None:
            return Lattice(base=self, type="downward")

        else:
            return None

    @cached_property
    def point_to_atom_id(self) -> Mapping[Hashable, Hashable] | None:
        """Get the mapping from points to atom IDs.

        Returns
        -------
        point_to_atom_id : Mapping[Hashable, Hashable] | None
            A mapping from points to atom IDs.

        Examples
        --------
        >>> from sigalg.core import Domain, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ...     name="F",
        ... )
        >>> print(F.point_to_atom_id)
        {0: 0, 1: 0, 2: 1}
        """
        if self.data is not None:
            if isinstance(self.data, pd.Series):
                point_to_atom_id = self.data.to_dict()
            else:
                point_to_atom_id = self.data.apply(tuple, axis=1).to_dict()
        else:
            point_to_atom_id = None

        return point_to_atom_id

    @cached_property
    def atom_space(self) -> Domain | None:
        """Get the domain consisting of atom identifiers.

        Returns
        -------
        atom_space: Domain | None
            The domain whose points are the atom identifiers of the sigma-algebra.

        Examples
        --------
        Define a sigma-algebra with two atoms.

        >>> from sigalg.core import Domain, Index, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 1,
        ...         1: 0,
        ...         2: 0,
        ...     },
        ... )

        The atom space is an instance of `Domain` consisting of the atom identifiers 0 and 1.

        >>> print(F.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'F':
         F
         1
         0

        Create a second sigma-algebra with 2-dimensional atom IDs.

        >>> J = Index([1, 2], variable_names=["j"], name="J")
        >>> G = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (0, 1),
        ...         2: (0, 1),
        ...     },
        ...     index=J,
        ...     name="G",
        ... )
        >>> print(G.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'G':
         G_1  G_2
           1    2
           0    1

        If the atom identifiers of a sigma-algebra are the points of the underlying domain itself (i.e., if the sigma-algebra is the power set), then the atom space is the domain.

        >>> H = SigmaAlgebra(domain=X, mapping=dict(zip(X, X)), name="H")
        >>> print(H)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'H':
           H
        x
        0  0
        1  1
        2  2
        >>> print(H.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'X':
         x
         0
         1
         2

        We check the atom space of a power-set sigma-algebra on a 2-dimensional domain.

        >>> Y = Domain.cartesian_product(
        ...     [[0, 1], [2, 3]], variable_names=["y_0", "y_1"], name="Y"
        ... )
        >>> K = SigmaAlgebra(domain=Y, mapping=dict(zip(Y, Y)), name="K")
        >>> print(K)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'K':
        i        0  1
        y_0 y_1
        0   2    0  2
            3    0  3
        1   2    1  2
            3    1  3
        >>> print(K.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'Y':
         y_0  y_1
           0    2
           0    3
           1    2
           1    3
        """
        from ..spaces.domain import Domain

        if self.data is not None:
            if (
                self.dimension == self.domain.dimension
                and (
                    self.data.values
                    == self.domain.data.to_frame().squeeze(axis=1).values
                ).all()
            ):
                atom_space = Domain._from_validated(
                    data=self.domain.data, name=self.domain.name
                )
            else:
                if isinstance(self.data, pd.DataFrame):
                    data = pd.MultiIndex.from_tuples(
                        self.atom_ids, names=self.variable_names
                    )
                else:
                    data = pd.Index(self.atom_ids, name=self.variable_names[0])

                atom_space = Domain._from_validated(data=data, name=self.name)
        else:
            atom_space = None

        return atom_space

    @property
    def dimension(self) -> int | None:
        """Get the dimension of the atom identifiers of the sigma-algebra.

        Returns
        -------
        dim : int | None
            The dimension of the atom identifiers of the sigma-algebra.
        """
        if isinstance(self.data, pd.Series):
            dimension = 1
        elif isinstance(self.data, pd.DataFrame):
            dimension = self.data.shape[1]
        else:
            dimension = None

        return dimension

    @cached_property
    def atom_indicator_df(self) -> pd.DataFrame | None:
        """Get a `pd.DataFrame` whose columns are indicators for membership of each point in the atoms of the sigma-algebra.

        Returns
        -------
        atom_indicator_df : pd.DataFrame | None
            A `pd.DataFrame` where each column corresponds to an atom of the sigma-algebra and each row corresponds to a point in the domain. The entries are 1 if the point belongs to the atom and 0 otherwise.

        Examples
        --------
        Define a sigma-algebra with three atoms.

        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: "b",
        ...         1: "b",
        ...         2: "a",
        ...         3: "a",
        ...         4: "c",
        ...         5: "c",
        ...     },
        ... )
        >>> print(F.atom_indicator_df)  # doctest: +NORMALIZE_WHITESPACE
                a  b  c
        s
        0       0  1  0
        1       0  1  0
        2       1  0  0
        3       1  0  0
        4       0  0  1
        5       0  0  1

        Define a sigma-algebra with 2-dimensional atom IDs.

        >>> G = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: ("a", "b"),
        ...         1: ("a", "b"),
        ...         2: ("a", "b"),
        ...         3: ("c", "d"),
        ...         4: ("c", "d"),
        ...         5: ("a", "b"),
        ...     },
        ...     name="G",
        ... )
        >>> print(G.atom_indicator_df)  # doctest: +NORMALIZE_WHITESPACE
                (a, b)  (c, d)
        s
        0            1       0
        1            1       0
        2            1       0
        3            0       1
        4            0       1
        5            1       0
        """
        if self.data is not None:
            if self.dimension == 1:
                atom_indicator_df = pd.get_dummies(self.data).astype(int)
            else:
                atom_indicator_df = pd.get_dummies(
                    self.data.apply(tuple, axis=1)
                ).astype(int)
        else:
            atom_indicator_df = None

        return atom_indicator_df

    @property
    def num_atoms(self) -> int | None:
        """Get the number of atoms in this sigma-algebra.

        Returns
        -------
        num_atoms : int | None
            The number of atoms in this sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import Domain, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> print(F.num_atoms)
        2
        """
        if self.data is not None:
            if isinstance(self.data, pd.DataFrame):
                num_atoms = len(self.data.drop_duplicates())
            else:
                num_atoms = self.data.nunique()

        return num_atoms

    @property
    def atom_ids(self) -> list[Hashable] | None:
        """Get a list of atom IDs in this sigma-algebra.

        Returns
        -------
        atom_ids : list[Hashable] | None
            A list of atom IDs in this sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import Domain, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> print(F.atom_ids)
        [0, 1]
        """
        if self.data is not None:
            if isinstance(self.data, pd.DataFrame):
                atom_ids = list(
                    self.data.drop_duplicates().itertuples(index=False, name=None)
                )
            else:
                atom_ids = list(self.data.drop_duplicates())
        else:
            atom_ids = None

        return atom_ids

    @cached_property
    def atom_id_to_points(self) -> dict[Hashable, list[Hashable]] | None:
        """Get a mapping from atom IDs to lists of points.

        Returns
        -------
        atom_id_to_points : dict[Hashable, list[Hashable]] | None
            A dictionary mapping each atom ID to a list of points contained in that atom.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> print(F.atom_id_to_points)
        {0: [0, 1], 1: [2]}
        """
        if self.data is not None:
            atom_id_to_points = {}
            for point, atom_id in self.point_to_atom_id.items():
                if atom_id not in atom_id_to_points:
                    atom_id_to_points[atom_id] = []
                atom_id_to_points[atom_id].append(point)
            self._atom_id_to_points = atom_id_to_points
        else:
            self.atom_id_to_points = None

        return atom_id_to_points

    @cached_property
    def atom_id_to_atom(self) -> dict[Hashable, Set] | None:
        r"""Get a mapping from atom IDs to `Set` objects in this sigma-algebra.

        Returns
        -------
        atom_id_to_atom : dict[Hashable, Set] | None
            A dictionary mapping each atom ID to its corresponding `Set` object.

        Examples
        --------
        >>> from sigalg.core import Domain, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> for atom_id, atom in F.atom_id_to_atom.items():
        ...     print(f"Atom ID: {atom_id}\n{atom}\n") # doctest: +NORMALIZE_WHITESPACE
        Atom ID: 0
        Set '0':
         x
         0
         1
        <BLANKLINE>
        Atom ID: 1
        Set '1':
         x
         2
        <BLANKLINE>
        """
        if self.data is not None:
            atom_id_to_atom = {
                atom_id: self.get_set(points, name=atom_id)
                for atom_id, points in self.atom_id_to_points.items()
            }
        else:
            atom_id_to_atom = None

        return atom_id_to_atom

    @property
    def atom_id_to_cardinality(self) -> dict[Hashable, int] | None:
        """Get a mapping from atom IDs to their cardinalities in this sigma-algebra.

        Returns
        -------
        atom_id_to_cardinality : dict[Hashable, int] | None
            A dictionary mapping each atom ID to the number of points it contains.

        Examples
        --------
        >>> from sigalg.core import Domain, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> print(F.atom_id_to_cardinality)
        {0: 2, 1: 1}
        """
        if self.data is not None:
            atom_id_to_cardinality = {
                atom_id: len(lst) for atom_id, lst in self.atom_id_to_points.items()
            }
        else:
            atom_id_to_cardinality = None

        return atom_id_to_cardinality

    @property
    def is_power_set(self) -> bool | None:
        """Boolean flag signaling a power-set sigma-algebra.

        Returns
        -------
        is_power_set: bool | None
            A boolean signaling whether the sigma-algebra is the power-set sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import Domain, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> print(F.is_power_set)
        False
        >>> G = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 1,
        ...         1: 6,
        ...         2: 7,
        ...     },
        ...     name="G",
        ... )
        >>> print(G.is_power_set)
        True
        """
        return len(self) == len(self.domain) if self.data is not None else None

    @property
    def is_canonical_power_set(self) -> bool | None:
        """Boolean flag signaling a canonical power-set sigma-algebra.

        In SigAlg, the *canonical* power-set sigma-algebra on a domain is the one created by the `power_set` constructor. Its atom identifiers are the points of the domain itself.

        Returns
        -------
        is_canonical : bool | None
            A flag whether the current sigma-algebra is the canonical power-set sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import Domain, SigmaAlgebra

        Define a 1-dimensional domain and obtain the sigma-algebra created by the `power_set` constructor.

        >>> X = Domain.from_sequence(size=3)
        >>> R = SigmaAlgebra.power_set(X)

        This sigma-algebra is both a power-set sigma-algebra and it is the canonical one.

        >>> R.is_power_set
        True
        >>> R.is_canonical_power_set
        True

        Create a power-set sigma-algebra which is not the canonical one.

        >>> F = SigmaAlgebra(domain=X, mapping=dict(zip(X, ["a", "b", "c"])))
        >>> F.is_power_set
        True
        >>> F.is_canonical_power_set
        False

        Now, run through the same routine with a 2-dimensional domain.

        >>> Y = Domain.cartesian_product(
        ...     [[0, 1], [2, 3]], variable_names=["y_0", "y_1"], name="Y"
        ... )
        >>> S = SigmaAlgebra.power_set(Y, name="S")
        >>> S.is_power_set
        True
        >>> S.is_canonical_power_set
        True
        >>> G = SigmaAlgebra(domain=Y, mapping=dict(zip(Y, [0, 1, 2, 3])), name="G")
        >>> G.is_power_set
        True
        >>> G.is_canonical_power_set
        False
        """
        if self.data is not None:
            R = SigmaAlgebra.power_set(self.domain)
            return self.data.index.equals(R.data.index) and self.data.equals(R.data)

    @property
    def is_trivial(self) -> bool | None:
        """Boolean flag signaling a trivial sigma-algebra.

        Returns
        -------
        is_trivial: bool | None
            A boolean signaling whether the sigma-algebra is the trivial sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import Domain, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> print(F.is_trivial)
        False
        >>> G = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 1,
        ...         1: 1,
        ...         2: 1,
        ...     },
        ...     name="G",
        ... )
        >>> print(G.is_trivial)
        True
        """
        return len(self) == 1 if self.data is not None else None

    @cached_property
    def atoms(self) -> list[Set] | None:
        r"""Get a list of atoms as `Set` objects in this sigma-algebra.

        Returns
        -------
        atoms : list[Set] | None
            A list of `Set` objects representing the atoms in this sigma-algebra.

        Examples
        --------
        Define a sigma-algebra with two atoms.

        >>> from sigalg.core import Domain, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> for atom in F.atoms:
        ...     print(atom, "\n") # doctest: +NORMALIZE_WHITESPACE
        Set '0':
         x
         0
         1
        <BLANKLINE>
        Set '1':
         x
         2
        <BLANKLINE>
        """
        if self.data is not None:
            atoms = list(self.atom_id_to_atom.values())
        else:
            atoms = None

        return atoms

    # --------------------- atom and set methods --------------------- #

    def get_set(self, indices: list[Hashable], name: Hashable = "A") -> Set:
        """Extract a measurable set from a list of points.

        Parameters
        ----------
        indices : list[Hashable]
            List of points to include in the measurable set.
        name : Hashable, default="A"
            Name identifier for the set.

        Returns
        -------
        measurable_set : Set
            A `Set` object containing the specified points.

        Examples
        --------
        Create a sigma-algebra with two atoms.

        >>> from sigalg.core import Domain, SigmaAlgebra
        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: (0, 1),
        ...         1: (0, 1),
        ...         2: (1, 2),
        ...         3: (1, 2),
        ...     },
        ... )

        Extract a measurable set.

        >>> A = F.get_set([0, 1], name="A")
        >>> print(A)  # doctest: +NORMALIZE_WHITESPACE
        Set 'A':
         x
         0
         1

        Try to extract a non-measurable set.

        >>> B = F.get_set([0, 2], name="B")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        NonMeasurableError: The candidate set is not measurable.
        """
        from ..spaces.set import Set
        from .lattice import NonMeasurableError

        measurable_set = Set(indices, domain=self.domain, name=name)

        if self in measurable_set.lattice:
            return measurable_set
        else:
            raise NonMeasurableError("The candidate set is not measurable.")

    def get_random_set(
        self,
        num_atoms: int,
        name: Hashable = "A",
        random_state: int | np.random.Generator | None = None,
    ) -> Set:
        """Get a random measurable set consisting of a specified number of atoms.

        Parameters
        ----------
        num_atoms : int
            The number of atoms to include in the random measurable set.
        name : Hashable, default="A"
            Name identifier for the set.
        random_state : int | np.random.Generator | None, default=None
            The random state for reproducibility.

        Raises
        ------
        TypeError
            If `num_atoms` is not an integer or if `random_state` is not an integer, `np.random.Generator`, or `None`.
        ValueError
            If `num_atoms` is not a non-negative integer or if it exceeds the number of atoms in the sigma-algebra, or if the sigma-algebra has no data.

        Returns
        -------
        random_set : Set
            A `Set` object representing the random measurable set.
        """
        if self.data is None:
            raise ValueError("The sigma-algebra must contain data.")

        if not isinstance(num_atoms, int):
            raise TypeError("num_atoms must be an integer.")
        if not (0 <= num_atoms <= len(self)):
            raise ValueError(
                "num_atoms must be a non-negative integer not larger than the number of atoms in the sigma-algebra."
            )
        if random_state is not None and not isinstance(
            random_state, (int, np.random.Generator)
        ):
            raise TypeError(
                "random_state must be an integer, np.random.Generator, or None."
            )

        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        atom_IDs = rng.choice(
            list(self.atom_id_to_points.keys()), size=num_atoms, replace=False
        )

        points = [
            point
            for id in atom_IDs
            for point in self.atom_id_to_points[self._normalize_key(id)]
        ]

        return self.get_set(points, name=name)

    @staticmethod
    def _normalize_key(key: Hashable | np.ndarray):
        if isinstance(key, np.ndarray):
            return tuple(key)
        else:
            return key

    def get_atom_containing(self, point: Hashable) -> Set:
        """Get the atom containing a given point.

        Parameters
        ----------
        point : Hashable
            The point for which to retrieve the containing atom.

        Raises
        ------
        ValueError
            If `point` is not in the domain of this sigma-algebra.

        Returns
        -------
        atom : Set
            The `Set` object representing the atom that contains the given point.

        Examples
        --------
        >>> from sigalg.core import Domain, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> print(F.get_atom_containing(0))  # doctest: +NORMALIZE_WHITESPACE
        Set '0':
         x
         0
         1
        """
        if not isinstance(point, Hashable):
            raise TypeError("The point parameter must be hashable.")

        if point not in self.point_to_atom_id:
            raise ValueError("The point is not in the domain of the sigma-algebra.")

        atom_id = self.point_to_atom_id[point]
        return self.atom_id_to_atom[atom_id]

    def non_null_atoms(self, measure: Measure) -> list[Set]:
        """Get the non-null atoms of this sigma-algebra with respect to a given measure.

        Parameters
        ----------
        measure : Measure
            A measure defined on a super-sigma-algebra of this sigma-algebra.

        Raises
        ------
        TypeError
            If `measure` is not an instance of `Measure`.
        ValueError
            If the sigma-algebra of the measure is not a super-sigma-algebra of this sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import Domain, Measure, SigmaAlgebra
        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 0,
        ...     },
        ... )
        >>> for A in F.non_null_atoms(measure=mu):
        ...     print(f"Atom id of non-null atom: {A.name}")
        Atom id of non-null atom: 0
        Atom id of non-null atom: 1
        """
        from ..measures.measure import Measure

        if not isinstance(measure, Measure):
            raise TypeError("measure must be an instance of Measure.")
        if not self <= measure.sig_alg:
            raise ValueError(
                "The sigma-algebra of the measure must be a super-sigma-algebra of this sigma-algebra."
            )

        prob_data = (measure | self).data
        atom_ids = list(prob_data[prob_data > 1e-8].index)

        return [atom for atom in self.atoms if atom.name in atom_ids]

    def restrict_to(
        self,
        subset: Set | list[Hashable],
        subset_name: Hashable = "A",
        name: Hashable | None = None,
    ) -> SigmaAlgebra:
        """Pass."""
        from ..spaces.set import Set

        if not isinstance(subset, Set):
            subset = Set(subset, domain=self.domain, name=subset_name)
        if subset not in self:
            raise ValueError("Subset must be in the sigma-algebra.")

        if name is None:
            name = f"{self.name}|{subset.name}"

        data = self.data.loc[subset.data]

        return SigmaAlgebra._from_validated(
            data=data.rename(name),
            variable_names=self.variable_names,
            domain_kind=type(self.domain).__name__,
            domain_name=subset.name,
            index_kind=type(self.index).__name__ if self.index else "Index",
            index_name=self.index.name if self.index else None,
            name=name,
        )

    # --------------------- conversion methods --------------------- #

    def with_name(self, name: Hashable) -> SigmaAlgebra:
        """Set the name of the sigma-algebra and return self for chaining.

        Parameters
        ----------
        name : Hashable
            The new name for the sigma-algebra.

        Returns
        -------
        self : SigmaAlgebra
            The current instance with the updated name.

        Examples
        --------
        >>> from sigalg.core import Domain, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> print(F.with_name("G"))  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
           F
        x
        0  0
        1  0
        2  1
        """
        self.name = name
        return self

    # --------------------- measurability methods --------------------- #

    def is_measurable(
        self,
        candidate: list[Hashable] | Set,
    ) -> bool:
        """Check if a candidate set is measurable with respect to this sigma-algebra.

        Parameters
        ----------
        candidate : Set | list[Hashable]
            The set to check for measurability.

        Raises
        ------
        TypeError
            If `candidate` is not a `Set` instance or a list of hashables.
        ValueError
            If `candidate` is a `Set` instance and its domain does not match the domain of this sigma-algebra.

        Returns
        -------
        is_measurable : bool
            `True` if the set is measurable with respect to this sigma-algebra, `False` otherwise.

        Examples
        --------
        Define a sigma-algebra with four atoms.

        >>> import numpy as np
        >>> from sigalg.core import Domain, SigmaAlgebra
        >>> rng = np.random.default_rng(101)
        >>> X = Domain.from_sequence(size=10)
        >>> F = SigmaAlgebra.from_rand(
        ...     num_atoms=4,
        ...     domain=X,
        ...     random_state=rng,
        ... )
        >>> print(F)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
           F
        x
        0  0
        1  3
        2  2
        3  2
        4  1
        5  1
        6  1
        7  3
        8  3
        9  2

        Define a sub-sigma-algebra of the first with two atoms.

        >>> G = SigmaAlgebra.from_rand(
        ...     num_atoms=2,
        ...     super=F,
        ...     name="G",
        ...     random_state=rng,
        ... )
        >>> print(G)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
           G
        x
        0  1
        1  1
        2  0
        3  0
        4  1
        5  1
        6  1
        7  1
        8  1
        9  0

        Extract a measurable set from the sub-sigma-algebra.

        >>> A = G.get_set([0, 1, 4, 5, 6, 7, 8], name="A")

        Check that the set is measurable with respect to the super-sigma-algebra.

        >>> F.is_measurable(A)
        True

        Check measurability of a non-measurable set using the `in` operators.

        >>> [0, 1, 4, 5] in F
        False
        """
        from ..spaces import Set

        measurable_set = Set(candidate, domain=self.domain)

        return self in measurable_set.lattice

    def __contains__(self, candidate: Set | list[Hashable] | Function) -> bool:
        """Check if a candidate set is measurable with respect to this sigma-algebra.

        Parameters
        ----------
        candidate : Set | list[Hashable]
            The set to check for measurability.

        Returns
        -------
        contains : bool
            `True` if the set is measurable with respect to this sigma-algebra, `False` otherwise.
        """
        from ..functions.function import Function
        from ..spaces.set import Set

        if isinstance(candidate, Function):
            return self in candidate.lattice

        elif isinstance(candidate, Set | list):
            return self.is_measurable(candidate)

        else:
            raise TypeError(
                "candidate must be a Set instance, a list of points in the domain, or a Function instance."
            )

    # --------------------- sequence methods --------------------- #

    def __iter__(self) -> iter:
        """Iterate over the atoms (as `Set`s) in this sigma-algebra.

        Returns
        -------
        iterator : iter
            An iterator over the atoms (as `Set`s) in the sigma-algebra.
        """
        return iter(self.atoms)

    def __len__(self) -> int:
        """Get the number of atoms in the sigma-algebra.

        Returns
        -------
        length : int
            The number of atoms in the sigma-algebra.
        """
        return self.num_atoms

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a concise string representation of the sigma-algebra.

        Returns
        -------
        repr_str : str
            A string representation of the sigma-algebra.
        """
        if self.data is None:
            return "SigmaAlgebra(empty)"
        else:
            return (
                f"SigmaAlgebra(domain={self.domain.name}, "
                f"num_atoms={self.num_atoms}, "
                f"variable_names={self.variable_names}, "
                f"name={self.name})"
            )

    def __str__(self) -> str:
        """Return a detailed string representation of the sigma-algebra.

        Returns
        -------
        repr_str : str
            A string representation of the sigma-algebra.
        """
        if self.data is None:
            return f"Sigma algebra '{self.name}': empty"
        elif self.dimension == 1:
            return f"Sigma algebra '{self.name}':\n{self.data.to_frame()}"
        else:
            return f"Sigma algebra '{self.name}':\n{self.data}"

    # --------------------- equality --------------------- #

    def __eq__(self, other: SigmaAlgebra) -> bool:
        """Check equality with another sigma-algebra.

        Two sigma-algebras are equal if they have the same domain and contain the same atoms.

        Parameters
        ----------
        other : SigmaAlgebra
            The other sigma-algebra to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the other object is a `SigmaAlgebra` with the same domain and atoms, `False` otherwise.
        """
        from .._utils import align_index, pandas_all_equal

        if not isinstance(other, SigmaAlgebra):
            return False
        if self.domain != other.domain:
            return False
        if pandas_all_equal(self.data, other.data):
            return True
        if self.is_power_set and other.is_power_set:
            return True
        if self.num_atoms != other.num_atoms:
            return False

        try:
            self_data = align_index(self.data, by=other.data.index)
        except ValueError:
            return False

        return (
            len(pd.concat([self_data, other.data], axis=1).drop_duplicates())
            == self.num_atoms
        )

    # --------------------- lattice methods --------------------- #

    def __or__(self, other: SigmaAlgebra | Set | list[Hashable]) -> SigmaAlgebra:
        r"""Get the join (least upper bound) of this sigma-algebra with another.

        Internally calls `Lattice.join`. See the documentation there for more details.

        Parameters
        ----------
        other : SigmaAlgebra
            The other sigma-algebra to join with.

        Returns
        -------
        join : SigmaAlgebra
            A new `SigmaAlgebra` instance representing the join of the two sigma-algebras.

        Examples
        --------
        >>> from sigalg.core import Domain, SigmaAlgebra
        >>> X = Domain.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 0,
        ...         3: 1,
        ...         4: 1,
        ...         5: 1,
        ...     },
        ... )
        >>> G = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 1,
        ...         4: 0,
        ...         5: 0,
        ...     },
        ...     name="G",
        ... )
        >>> join = F | G
        >>> print(join)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F v G':
        i  0  1
        x
        0  0  0
        1  0  1
        2  0  1
        3  1  1
        4  1  0
        5  1  0
        """
        from .lattice import Lattice

        if isinstance(other, SigmaAlgebra):
            return Lattice.join([self, other])

        else:
            return self.restrict_to(other)

    def __le__(self, other: SigmaAlgebra) -> bool:
        """Check if this sigma-algebra is a sub-algebra of another.

        Parameters
        ----------
        other : SigmaAlgebra
            The other sigma-algebra to compare with.

        Returns
        -------
        is_subalgebra : bool
            `True` if this sigma-algebra is a sub-algebra of the other, `False` otherwise.
        """
        if not isinstance(other, SigmaAlgebra):
            raise TypeError("other must be an instance of SigmaAlgebra")

        return self in other.down_lattice

    def __lt__(self, other: SigmaAlgebra) -> bool:
        """Check if this sigma-algebra is a proper sub-algebra of another.

        Parameters
        ----------
        other : SigmaAlgebra
            The other sigma-algebra to compare with.

        Returns
        -------
        is_proper_subalgebra : bool
            `True` if this sigma-algebra is a proper sub-algebra of the other, `False` otherwise.
        """
        if not isinstance(other, SigmaAlgebra):
            raise TypeError("other must be an instance of SigmaAlgebra")

        return self <= other and self != other

    def __ge__(self, other: SigmaAlgebra) -> bool:
        """Check if this sigma-algebra is a super-algebra of another.

        Parameters
        ----------
        other : SigmaAlgebra
            The other sigma-algebra to compare with.

        Returns
        -------
        is_superalgebra : bool
            `True` if this sigma-algebra is a super-algebra of the other, `False` otherwise.
        """
        if not isinstance(other, SigmaAlgebra):
            raise TypeError("other must be an instance of SigmaAlgebra")

        return self in other.up_lattice

    def __gt__(self, other: SigmaAlgebra) -> bool:
        """Check if this sigma-algebra is a proper super-algebra of another.

        Parameters
        ----------
        other : SigmaAlgebra
            The other sigma-algebra to compare with.

        Returns
        -------
        is_proper_superalgebra : bool
            `True` if this sigma-algebra is a proper super-algebra of the other, `False` otherwise.
        """
        if not isinstance(other, SigmaAlgebra):
            raise TypeError("other must be an instance of SigmaAlgebra")

        return self >= other and self != other


class SigmaAlgebraMethods:
    """Mixin class providing sigma-algebra methods to other classes."""

    def get_set(self, indices: list[Hashable], name: Hashable = "A") -> Set:
        """Extract a measurable set from the sigma-algebra.

        Calls `SigmaAlgebra.get_set`.

        Parameters
        ----------
        indices : list[Hashable]
            List of points to include in the measurable set.
        name : Hashable, default="A"
            Name identifier for the set.

        Returns
        -------
        set : Set
            An `Set` object containing the specified points.
        """
        return self.sig_alg.get_set(indices, name)

    def is_measurable(self, candidate: Set) -> bool:
        """Check if a candidate set is measurable with respect to the sigma-algebra.

        Calls `SigmaAlgebra.is_measurable`.

        Parameters
        ----------
        candidate : Set | list[Hashable]
            The set to check for measurability.

        Returns
        -------
        is_measurable : bool
            `True` if the set is measurable with respect to this sigma-algebra, `False` otherwise.
        """
        return self.sig_alg.is_measurable(candidate)
