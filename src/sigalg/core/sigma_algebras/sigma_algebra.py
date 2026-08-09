"""A class representing a sigma-algebra."""

from __future__ import annotations

import copy
from collections.abc import Hashable, Mapping
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ...typing.index_like import IndexLike
    from ...typing.mapping_like import MappingLike
    from ..functions.measurable_vector import MeasurableVector
    from ..measures.measure import Measure
    from ..spaces.domain import Domain
    from ..spaces.measurable_set import MeasurableSet


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
    Construct a `SigmaAlgebra` with two atoms.

    >>> from sigalg.core import Index, Domain, SigmaAlgebra
    >>> X = Domain.from_sequence(size=3)
    >>> mapping = {
    ...     0: 1,
    ...     1: 0,
    ...     2: 0,
    ... }
    >>> F = SigmaAlgebra(domain=X, mapping=mapping, variable_names=["u"])
    >>> print(F)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'F':
           u
    point
    0      1
    1      0
    2      0

    Construct a `SigmaAlgebra` on the same sample space with 2-dimensional atom IDs.

    >>> mapping = {
    ...     0: (1, 2),
    ...     1: (0, 1),
    ...     2: (0, 1),
    ... }
    >>> G = SigmaAlgebra(domain=X, mapping=mapping, variable_names=["a", "b"], name="G")
    >>> print(G)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'G':
           a  b
    point
    0      1  2
    1      0  1
    2      0  1

    Notes
    -----
    A *$\sigma$-algebra* $\mathcal{F}$ on a nonempty set $X$ is a collection of subsets of $X$ that contains $X$, and is closed under complementation and countable unions. In the case that $X$ is finite (as it always is, in SigAlg), then $\mathcal{F}$ needs only to be closed under finite unions.

    A $\sigma$-algebra $\mathcal{F}$ determines its *atoms*, which are the nonempty sets $A\in \mathcal{F}$ that are *minimal* with respect to subset inclusion, in the following sense: If $B\in \mathcal{F}$ is nonempty and $B\subset A$, then necessarily $A=B$. Conversely, provided that $X$ is finite, the $\sigma$-algebra $\mathcal{F}$ is completely recoverable from its atoms, in the sense that every event $A\in \mathcal{F}$ is a disjoint union of atoms.

    If $\{A_i\}_{i\in I}$ is the set of atoms, indexed by a finite set $I$, then there is a mapping $X \to I$ given by $x \mapsto i$, where $A_i$ is the unique atom that contains $x$. This mapping is what SigAlg uses to represent $\sigma$-algebras. The indices in $I$ are called *atom identifiers*. The atom identifiers may consist of tuples, in which case the $\sigma$-algebra is said to have *multi-dimensional* atom identifiers, and the *dimension* of the $\sigma$-algebra is the common length of the tuples.
    """

    _properties = [
        "_point_to_atom_id",
        "_dimension",
        "_atom_space",
        "_atom_indicator_df",
        "_num_atoms",
        "_atom_ids",
        "_atom_id_to_points",
        "_atom_id_to_atom",
        "_atom_id_to_cardinality",
        "_is_power_set",
        "_is_trivial",
        "_atoms",
        "_variable_names",
    ]

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        domain: Domain | IndexLike | None = None,
        mapping: MappingLike | None = None,
        variable_names: list[Hashable] | None = None,
        name: Hashable = "F",
    ) -> None:
        from ...validation.mapping_validator import MappingValidator
        from ..spaces.domain import Domain

        if domain is not None and not isinstance(domain, Domain):
            domain = Domain(domain)

        v = MappingValidator(
            mapping=mapping,
            domain=domain,
            name=name,
            output_name="atom_ID",
        )
        self._data = v.mapping

        if variable_names is not None and (
            not isinstance(variable_names, list)
            or any(not isinstance(name, Hashable) for name in variable_names)
        ):
            raise ValueError(
                "If given, variable_names must be a list of hashable items."
            )

        self._initialize_property_caches()

        self._domain = v.domain
        self._name = v.name
        self._index = v.index

        if variable_names is None:
            if self.dimension is not None:
                self._variable_names = (
                    [f"atom_ID_{i}" for i in range(self.dimension)]
                    if self.dimension > 1
                    else ["atom_ID"]
                )
            else:
                self._variable_names = None
        else:
            self._variable_names = variable_names

        if self._data is not None:
            if isinstance(self._data, pd.DataFrame):
                self._data.columns = self._variable_names
            else:
                self._data.name = self._variable_names[0]

    def _initialize_property_caches(self) -> None:
        for property in self._properties:
            setattr(self, property, None)

    @classmethod
    def power_set(
        cls,
        domain: IndexLike,
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
        >>> X1 = Domain.from_sequence(size=3, variable_name="x1", name="X1")
        >>> G = SigmaAlgebra.power_set(X1, name="G")
        >>> print(G)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
              x1
        x1
        0      0
        1      1
        2      2
        >>> print(G.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'X1':
         x1
          0
          1
          2

        Create another power-set sigma-algebra.

        >>> X2 = Domain.cartesian_product(
        ...     [[1, 2], ["a", "b"]], name="X2", variable_names=["number", "letter"]
        ... )
        >>> F = SigmaAlgebra.power_set(X2, name="F")
        >>> print(F)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                      number  letter
        number letter
        1      a           1       a
               b           1       b
        2      a           2       a
               b           2       b
        >>> print(F.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'X2':
         number letter
             1      a
             1      b
             2      a
             2      b

        Notes
        -----
        The *power-set $\sigma$-algebra* on a nonempty set $X$ consists of all subsets of $X$. Its atoms are all singleton subsets. It is the finest $\sigma$-algebra on $X$.
        """
        from ..spaces.domain import Domain

        domain = Domain(domain)
        mapping = dict(zip(domain, domain))
        return cls(
            domain=domain,
            mapping=mapping,
            name=name,
            variable_names=domain.variable_names,
        )

    @classmethod
    def trivial(
        cls,
        domain: Domain | IndexLike,
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
               atom_ID
        sample
        0            0
        1            0
        2            0
        >>> print(G.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'G':
         atom_ID
               0
        >>> Omega2 = SampleSpace.cartesian_product(
        ...     [[1, 2], ["a", "b"]], name="Omega2", variable_names=["number", "letter"]
        ... )
        >>> F = SigmaAlgebra.trivial(Omega2, name="F")
        >>> print(F)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                      atom_ID
        number letter
        1      a            0
               b            0
        2      a            0
               b            0
        >>> print(F.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'F':
         atom_ID
               0

        Notes
        -----
        The *trivial $\sigma$-algebra* on a nonempty set $X$ consists of only the sets $X$ and $\emptyset$. Its single atom is $X$ itself. It is the coarsest $\sigma$-algebra on $X$.
        """
        from ..spaces.domain import Domain

        if not isinstance(domain, Domain):
            domain = Domain(domain)

        mapping = dict.fromkeys(domain.data, 0)
        return cls(domain=domain, mapping=mapping, name=name)

    @classmethod
    def from_rand(
        cls,
        domain: Domain | IndexLike | None = None,
        super: SigmaAlgebra | None = None,
        num_atoms: int = 1,
        dim: int = 1,
        atom_ID_range: tuple[int, int] | None = None,
        variable_names: list[Hashable] | None = None,
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
                atom_ID
        point
        0             0
        1             0
        2             0
        3             2
        4             1

        Generate a sigma-algebra with three random atoms and 3-dimensional atom identifiers.

        >>> G = SigmaAlgebra.from_rand(
        ...     domain=X,
        ...     num_atoms=3,
        ...     dim=3,
        ...     random_state=42,
        ...     name="G",
        ...     variable_names=["a", "b", "c"],
        ... )
        >>> print(G)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
                  a  b  c
        point
        0         2  1  2
        1         0  0  2
        2         2  1  2
        3         2  1  2
        4         1  0  0

        Generate a sigma-algebra with three random atoms and 2-dimensional atom identifiers with values in the range [10, 15).

        >>> H = SigmaAlgebra.from_rand(
        ...     domain=X,
        ...     num_atoms=3,
        ...     atom_ID_range=(10, 15),
        ...     dim=2,
        ...     random_state=42,
        ...     name="H",
        ...     variable_names=["x", "y"],
        ... )
        >>> print(H)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'H':
                 x   y
        point
        0        14  14
        1        10  10
        2        14  14
        3        13  13
        4        14  14

        Create a random sub-sigma-algebra of `H` with two atoms:

        >>> K = SigmaAlgebra.from_rand(
        ...     super=H,
        ...     num_atoms=2,
        ...     random_state=42,
        ...     name="K",
        ... )
        >>> print(K)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'K':
                atom_ID
        point
        0             0
        1             1
        2             0
        3             1
        4             0
        >>> print(K <= H)
        True
        """
        from ..indices.index import Index
        from ..spaces.domain import Domain

        if domain is not None and not isinstance(domain, Domain):
            domain = Domain(domain)
        if not isinstance(num_atoms, int):
            raise TypeError("num_atoms must be an integer.")
        if num_atoms <= 0:
            raise ValueError("num_atoms must be a positive integer.")
        if not isinstance(dim, int):
            raise TypeError("dim must be an integer.")
        if dim <= 0:
            raise ValueError("dim must be a positive integer.")
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

        if num_atoms > len(domain):
            raise ValueError(
                "num_atoms must be less than or equal to the number of points."
            )

        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        atom_IDs = Index._random_tuples(
            size=num_atoms,
            domain=atom_ID_range,
            dim=dim,
            random_state=rng,
        )

        if super is not None:
            population = copy.deepcopy(super.atom_ids)
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
            population = list(domain.copy())
            partitioned = cls._partition(
                population=population, size=num_atoms, random_state=rng
            )

        mapping = {
            point: atom_ID
            for partition, atom_ID in zip(partitioned, atom_IDs)
            for point in partition
        }
        mapping = {point: mapping[point] for point in domain}

        return cls(
            domain=domain,
            mapping=mapping,
            name=name,
            variable_names=variable_names,
        )

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

    @classmethod
    def from_set(cls, measurable_set: MeasurableSet) -> SigmaAlgebra:
        r"""Create the sigma-algebra generated by a measurable set.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        measurable_set : MeasurableSet
            The measurable to generate the sigma-algebra from.

        Raises
        ------
        TypeError
            If `measurable_set` is not a `MeasurableSet` instance.
        ValueError
            If `measurable_set` is empty.

        Returns
        -------
        sig_alg : SigmaAlgebra
            A new `SigmaAlgebra` instance generated by the given measurable set.

        Examples
        --------
        >>> from sigalg.core import Domain, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(X)
        >>> A = F.get_set([0, 2])
        >>> sigma_A = SigmaAlgebra.from_set(A)
        >>> print(sigma_A) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'sigma(A)':
               atom_ID
        point
        0            1
        1            0
        2            1

        Notes
        -----
        Let $A$ be a subset of a finite set $X$. The *$\sigma$-algebra generated by $A$*, denoted $\sigma(A)$, has two atoms given by $A$ and its complement $A^c$.
        """
        from ..spaces import MeasurableSet

        if not isinstance(measurable_set, MeasurableSet):
            raise TypeError("The measurable set must be an MeasurableSet instance.")

        mapping = measurable_set.indicator.data

        name = f"sigma({measurable_set.name})"

        return cls(
            domain=measurable_set.domain,
            mapping=mapping.rename("atom_ID"),
            name=name,
        )

    @classmethod
    def from_measurable_vector(
        cls,
        vector: MeasurableVector,
    ) -> SigmaAlgebra:
        r"""Create a sigma-algebra induced by a measurable vector.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        vector : MeasurableVector
            The measurable vector from which to generate the sigma-algebra.

        Raises
        ------
        TypeError
            If `vector` is not a `MeasurableVector` instance.

        Returns
        -------
        sig_alg : SigmaAlgebra
            A new `SigmaAlgebra` instance induced by the given measurable vector.

        Examples
        --------
        >>> from sigalg.core import Domain, MeasurableVector, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> f = MeasurableVector(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (2, 4),
        ...     },
        ... )
        >>> sigma_f = SigmaAlgebra.from_measurable_vector(vector=f)
        >>> print(sigma_f)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'sigma(f)':
               f_0  f_1
        point
        0        1    2
        1        1    2
        2        2    4

        Notes
        -----
        Let $f: X \to \mathbb{R}^d$ be a function defined on a set $X$. The *$\sigma$-algebra induced by $f$*, denoted $\sigma(f)$, is the $\sigma$-algebra generated by the preimages of Borel sets in $\mathbb{R}^d$ under $f$. In SigAlg, in which $X$ is finite and $\sigma$-algebras are determined by their atoms, we may take the atom identifiers to be the unique values of $f$ on $X$.
        """
        from ..functions.measurable_vector import MeasurableVector

        if not isinstance(vector, MeasurableVector):
            raise TypeError("vector must be a MeasurableVector instance.")

        if vector.name.startswith("(") and vector.name.endswith(")"):
            name = f"sigma{vector.name}"
        else:
            name = f"sigma({vector.name})"

        if isinstance(vector.data, pd.DataFrame):
            mapping = vector.data.apply(tuple, axis=1).to_dict()
        else:
            mapping = vector.data.to_dict()

        return cls(
            domain=vector.domain,
            mapping=mapping,
            name=name,
            variable_names=vector.component_names,
        )

    @classmethod
    def cartesian_product(
        cls,
        factors: list[SigmaAlgebra],
        variable_names: list[Hashable] | None = None,
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
        Define two sigma-algebras on two domains.

        >>> from sigalg.core import Domain, SigmaAlgebra
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
            u
        x
        0   0
        1   1
        2   1
        >>> print(G)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
           v  w
        y
        0  a  b
        1  a  b
        2  c  d

        Compute the Cartesian product of the two sigma-algebras usings the `cartesian_product` method.

        >>> prod_sig_alg = SigmaAlgebra.cartesian_product([F, G])
        >>> print(prod_sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F x G':
             u  v  w
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


        Compute the same Cartesian product using the `@` operator.

        >>> prod_sig_alg = SigmaAlgebra.cartesian_product([F, G])
        >>> print(prod_sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F x G':
             u  v  w
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
        from ..spaces.domain import Domain
        from .._utils.utils import _subscript_var_names, _to_df

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
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("`name` must be hashable or None.")

        if all(sig_alg.is_power_set for sig_alg in factors):
            domain = Domain.cartesian_product([sig_alg.domain for sig_alg in factors])
            return SigmaAlgebra.power_set(domain)

        domain_var_names = _subscript_var_names(
            [sig_alg.domain.variable_names for sig_alg in factors],
            grouped=True,
        )
        sig_alg_var_names = _subscript_var_names(
            [sig_alg.variable_names for sig_alg in factors],
            grouped=True,
        )

        sig_alg_data = []

        for domain_vars, sig_alg_vars, sig_alg in zip(
            domain_var_names, sig_alg_var_names, factors
        ):
            data = _to_df(sig_alg.data)
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

        if name is None:
            name = " x ".join([sig_alg.name for sig_alg in factors])

        if variable_names is None:
            variable_names = [name.strip("_ID") for name in mapping.columns]

        domain = Domain(
            indices=mapping.index,
            name=" x ".join([sig_alg.domain.name for sig_alg in factors]),
            variable_names=mapping.index.names,
            bypass_validation=True,
        )

        return cls(
            domain=domain,
            mapping=mapping,
            variable_names=variable_names,
            name=name,
        )

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
        >>> Omega = SampleSpace.from_sequence(size=3, variable_name="omega")
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: (1, "a"),
        ...         1: (1, "a"),
        ...         2: (2, "b"),
        ...     },
        ...     variable_names=["x", "y"],
        ... )
        >>> F_3 = SigmaAlgebra.cartesian_power(F, 3)
        >>> print(F_3)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F ^ 3':
                                 x_0 y_0  x_1 y_1  x_2 y_2
        omega_0 omega_1 omega_2
        0       0       0          1   a    1   a    1   a
                        1          1   a    1   a    1   a
                        2          1   a    1   a    2   b
                1       0          1   a    1   a    1   a
                        1          1   a    1   a    1   a
                        2          1   a    1   a    2   b
                2       0          1   a    2   b    1   a
                        1          1   a    2   b    1   a
                        2          1   a    2   b    2   b
        1       0       0          1   a    1   a    1   a
                        1          1   a    1   a    1   a
                        2          1   a    1   a    2   b
                1       0          1   a    1   a    1   a
                        1          1   a    1   a    1   a
                        2          1   a    1   a    2   b
                2       0          1   a    2   b    1   a
                        1          1   a    2   b    1   a
                        2          1   a    2   b    2   b
        2       0       0          2   b    1   a    1   a
                        1          2   b    1   a    1   a
                        2          2   b    1   a    2   b
                1       0          2   b    1   a    1   a
                        1          2   b    1   a    1   a
                        2          2   b    1   a    2   b
                2       0          2   b    2   b    1   a
                        1          2   b    2   b    1   a
                        2          2   b    2   b    2   b
        """
        name = f"{sig_alg.name} ^ {n}"
        return SigmaAlgebra.cartesian_product(factors=[sig_alg] * n, name=name)

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

    # --------------------- properties --------------------- #

    @property
    def domain(self) -> Domain | None:
        """Get the domain over which this sigma-algebra is defined.

        The `domain` property is settable. If the `SigmaAlgebra` instance already has a domain, the new domain must contain the same number of points.

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
         point
              0
              1
              2

        Set a new domain on the sigma-algebra. Notice that the new domain is in bijective correspondence with the old one.

        >>> Y = Domain(["a", "b", "c"], name="Y")
        >>> F.domain = Y
        >>> print(F.domain)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'Y':
         point
              a
              b
              c
        >>> print(F)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
               atom_ID
        point
        a            1
        b            0
        c            1
        """
        return self._domain

    @domain.setter
    def domain(self, domain: Domain | IndexLike) -> None:
        """Set the domain of this sigma-algebra.

        If the `SigmaAlgebra` instance already has a domain, the new domain must contain the same number of points.

        Parameters
        ----------
        domain : Domain | IndexLike
            The new domain for this sigma-algebra.

        Raises
        ------
        ValueError
            If the new sample space does not have the same number of points as the existing sample space.
        """
        from ..spaces.sample_space import Domain

        if not isinstance(domain, Domain):
            domain = Domain(domain)

        if self.domain is not None:
            if len(domain) != len(self.domain):
                raise ValueError(
                    "New domain must have the same number of points as the existing domain."
                )

            if self.data is not None:
                self.data.index = domain.data

            self._point_to_atom_id = None
            self._atom_id_to_points = None
            self._atom_id_to_atom = None
            self._atoms = None
            self._atom_space = None

        self._domain = domain

    @property
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
        if self._point_to_atom_id is None and self.data is not None:
            if isinstance(self.data, pd.Series):
                self._point_to_atom_id = self.data.to_dict()
            else:
                self._point_to_atom_id = self.data.apply(tuple, axis=1).to_dict()
        return self._point_to_atom_id

    @property
    def data(self) -> pd.Series | None:
        """Get the underlying `pd.Series`.

        Returns
        -------
        data: pd.Series | None
            A `pd.Series` mapping points to atom IDs.

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
        >>> print(F.data) # doctest: +NORMALIZE_WHITESPACE
        point
        0    0
        1    0
        2    1
        Name: atom_ID, dtype: int64
        """
        return self._data

    @property
    def atom_space(self) -> Domain | None:
        """Get the domain consisting of atom identifiers.

        Returns
        -------
        atom_space: Domain | None
            The domain whose points are the atom identifiers of the sigma-algebra.

        Examples
        --------
        Define a sigma-algebra with two atoms.

        >>> from sigalg.core import Index, Domain, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 1,
        ...         1: 0,
        ...         2: 0,
        ...     },
        ... )

        The atom space is an instance of `Domain` consisting of the atom IDs 0 and 1.

        >>> print(F.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'F':
         atom_ID
               1
               0

        Create a second sigma-algebra with 2-dimensional atom IDs.

        >>> G = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (0, 1),
        ...         2: (0, 1),
        ...     },
        ...     name="G",
        ... )
        >>> print(G.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'G':
         atom_ID_0  atom_ID_1
                 1       2
                 0       1

        Define a third sigma-algebra with custom variable names for its atom space.

        >>> H = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (0, 1),
        ...         2: (0, 1),
        ...     },
        ...     name="H",
        ...     variable_names=["x", "y"],
        ... )
        >>> print(H.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'H':
         x  y
         1  2
         0  1
        """
        from ..spaces.domain import Domain

        if self._atom_space is None and self.data is not None:
            if self.is_power_set:
                self._atom_space = self.domain
            else:
                self._atom_space = Domain(
                    self.atom_ids,
                    name=self.name,
                    variable_names=self.variable_names,
                )

        return self._atom_space

    @property
    def variable_names(self) -> list[Hashable] | None:
        """Get the variable names of the sigma-algebra.

        The `variable_names` property is settable.

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
        >>> print(F.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'F':
         x  y
         1  2
         0  1
        >>> F.variable_names = ["u", "v"]
        >>> print(F.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'F':
         u  v
         1  2
         0  1
        """
        return self._variable_names

    @variable_names.setter
    def variable_names(self, variable_names: list[Hashable]) -> None:
        """Set the variable names of the sigma-algebra.

        Parameters
        ----------
        variable_names : list[Hashable]
            The new variable names of the sigma-algebra.

        Raises
        ------
        ValueError
            If `variable_names` is not a list of hashables, or if its length does not match the dimension of the sigma-algebra.
        """
        if not isinstance(variable_names, list) or any(
            not isinstance(name, Hashable) for name in variable_names
        ):
            raise ValueError("If given, names must be a list of hashable items.")
        if len(variable_names) != self.dimension:
            raise ValueError(
                "The number of variables names must match the dimension of the sigma-algebra."
            )

        self._variable_names = variable_names
        self._atom_space = None

    @property
    def dimension(self) -> int | None:
        """Get the dimension of the atom identifiers of the sigma-algebra.

        Returns
        -------
        dim : int | None
            The dimension of the atom identifiers of the sigma-algebra.
        """
        if self._dimension is None and self.data is not None:
            if isinstance(self.data, pd.DataFrame):
                self._dimension = self.data.shape[1]
            else:
                self._dimension = 1
        return self._dimension

    @property
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
        sample
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
        sample
        0            1       0
        1            1       0
        2            1       0
        3            0       1
        4            0       1
        5            1       0
        """
        if self._atom_indicator_df is None and self.data is not None:
            if self.dimension == 1:
                self._atom_indicator_df = pd.get_dummies(self.data).astype(int)
            else:
                self._atom_indicator_df = pd.get_dummies(
                    self.data.apply(tuple, axis=1)
                ).astype(int)

        return self._atom_indicator_df

    @property
    def name(self) -> Hashable | None:
        """Get the name identifier for this sigma-algebra.

        Returns
        -------
        name : Hashable | None
            The name of this sigma-algebra.
        """
        return self._name

    @name.setter
    def name(self, name: Hashable) -> None:
        """Set the name identifier for this sigma-algebra.

        Parameters
        ----------
        name : Hashable
            New name for this sigma-algebra.

        Raises
        ------
        TypeError
            If `name` is not a hashable.
        """
        if not isinstance(name, Hashable):
            raise TypeError("name must be a hashable type.")
        self._name = name
        self._atom_space = None

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
               atom_ID
        point
        0            0
        1            0
        2            1
        """
        self.name = name
        return self

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
        if self._num_atoms is None and self.data is not None:
            if isinstance(self.data, pd.DataFrame):
                self._num_atoms = len(self.data.drop_duplicates())
            else:
                self._num_atoms = self.data.nunique()
        return self._num_atoms

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
        if self._atom_ids is None and self.data is not None:
            if isinstance(self.data, pd.DataFrame):
                self._atom_ids = list(
                    self.data.drop_duplicates().itertuples(index=False, name=None)
                )
            else:
                self._atom_ids = list(self.data.drop_duplicates())
        return self._atom_ids

    @property
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
        if self._atom_id_to_points is None and self.point_to_atom_id is not None:
            atom_id_to_sample_ids = {}
            for sample_id, atom_id in self.point_to_atom_id.items():
                if atom_id not in atom_id_to_sample_ids:
                    atom_id_to_sample_ids[atom_id] = []
                atom_id_to_sample_ids[atom_id].append(sample_id)
            self._atom_id_to_points = atom_id_to_sample_ids
        return self._atom_id_to_points

    @property
    def atom_id_to_atom(self) -> dict[Hashable, MeasurableSet] | None:
        r"""Get a mapping from atom IDs to `MeasurableSet` objects in this sigma-algebra.

        Returns
        -------
        atom_id_to_atom : dict[Hashable, MeasurableSet] | None
            A dictionary mapping each atom ID to its corresponding `MeasurableSet` object.

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
        Measurable set '0':
         point
              0
              1
        <BLANKLINE>
        Atom ID: 1
        Measurable set '1':
         point
              2
        <BLANKLINE>
        """
        if self._atom_id_to_atom is None and self.atom_id_to_points is not None:
            atom_id_to_atom = {
                atom_id: self.get_set(points, name=atom_id)
                for atom_id, points in self.atom_id_to_points.items()
            }
            self._atom_id_to_atom = atom_id_to_atom
        return self._atom_id_to_atom

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
        if self._atom_id_to_cardinality is None and self.atom_id_to_points is not None:
            self._atom_id_to_cardinality = {
                atom_id: len(lst) for atom_id, lst in self.atom_id_to_points.items()
            }
        return self._atom_id_to_cardinality

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
        if self._is_power_set is None and self.data is not None:
            self._is_power_set = len(self) == len(self._domain)
        return self._is_power_set

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
        if self._is_trivial is None and self.data is not None:
            self._is_trivial = len(self) == 1
        return self._is_trivial

    @property
    def atoms(self) -> list[MeasurableSet] | None:
        r"""Get a list of atoms as `MeasurableSet` objects in this sigma-algebra.

        Returns
        -------
        atoms : list[MeasurableSet] | None
            A list of `MeasurableSet` objects representing the atoms in this sigma-algebra.

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
        Measurable set '0':
         point
              0
              1
        <BLANKLINE>
        Measurable set '1':
         point
              2
        <BLANKLINE>
        """
        if self._atoms is None and self.atom_id_to_atom is not None:
            self._atoms = list(self.atom_id_to_atom.values())
        return self._atoms

    # --------------------- atom and event methods --------------------- #

    def get_set(self, indices: list[Hashable], name: Hashable = "A") -> MeasurableSet:
        """Extract a measurable set from a list of points.

        Parameters
        ----------
        indices : list[Hashable]
            List of points to include in the measurable set.
        name : Hashable, default="A"
            Name identifier for the set.

        Returns
        -------
        measurable_set : MeasurableSet
            An `MeasurableSet` object containing the specified points.

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
        Measurable set 'A':
         point
              0
              1

        Try to extract a non-measurable set.

        >>> try:
        ...     B = F.get_set([0, 2], name="B")
        ... except ValueError as e:
        ...     print(e)
        The candidate set is not measurable.
        """
        from ..spaces.measurable_set import MeasurableSet

        return MeasurableSet.from_list(indices=indices, sig_alg=self, name=name)

    def get_random_set(
        self,
        num_atoms: int,
        name: Hashable = "A",
        random_state: int | np.random.Generator | None = None,
    ) -> MeasurableSet:
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
        random_set : MeasurableSet
            A `MeasurableSet` object representing the random measurable set.
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

    def get_atom_containing(self, point: Hashable) -> MeasurableSet:
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
        atom : MeasurableSet
            The `MeasurableSet` object representing the atom that contains the given point.

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
        Measurable set '0':
         point
              0
              1
        """
        if point not in self.point_to_atom_id:
            raise ValueError("The point is not in the domain of the sigma-algebra.")

        atom_id = self.point_to_atom_id[point]
        return self.atom_id_to_atom[atom_id]

    def non_null_atoms(self, measure: Measure) -> list[MeasurableSet]:
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

    # --------------------- measurability methods --------------------- #

    def is_measurable(
        self,
        candidate: MeasurableSet | list[Hashable],
    ) -> bool:
        """Check if a candidate set is measurable with respect to this sigma-algebra.

        Parameters
        ----------
        candidate : MeasurableSet | list[Hashable]
            The set to check for measurability.

        Raises
        ------
        TypeError
            If `candidate` is not an `MeasurableSet` instance or a list of hashables.
        ValueError
            If `candidate` is an `MeasurableSet` instance and its domain does not match the domain of this sigma-algebra.

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
               atom_ID
        point
        0            0
        1            3
        2            2
        3            2
        4            1
        5            1
        6            1
        7            3
        8            3
        9            2

        Define a sub-sigma-algebra of the first with two atoms.

        >>> G = SigmaAlgebra.from_rand(
        ...     num_atoms=2,
        ...     super=F,
        ...     name="G",
        ...     random_state=rng,
        ... )
        >>> print(G)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
               atom_ID
        point
        0            1
        1            1
        2            0
        3            0
        4            1
        5            1
        6            1
        7            1
        8            1
        9            0

        Extract a measurable set from the sub-sigma-algebra.

        >>> A = G.get_set([0, 1, 4, 5, 6, 7, 8], name="A")

        Check that the set is measurable with respect to the super-sigma-algebra.

        >>> print(F.is_measurable(A))
        True

        Check measurability of a non-measurable set using the `in` operators.

        >>> print([0, 1, 4, 5] in F)
        False
        """
        from ..spaces import MeasurableSet

        if not isinstance(candidate, MeasurableSet) and not isinstance(candidate, list):
            raise TypeError(
                "candidate must be a MeasurableSet instance or a list of hashables."
            )
        if isinstance(candidate, MeasurableSet) and candidate.domain != self._domain:
            raise ValueError(
                "candidate must have the same domain as the sigma-algebra."
            )

        if isinstance(candidate, MeasurableSet) and candidate.sig_alg <= self:
            return True

        return MeasurableSet.is_measurable(
            candidate=candidate, sig_alg=self, verbose=False
        )

    def __contains__(self, candidate: MeasurableSet | list[Hashable]) -> bool:
        """Check if a candidate set is measurable with respect to this sigma-algebra.

        Parameters
        ----------
        candidate : MeasurableSet | list[Hashable]
            The set to check for measurability.

        Returns
        -------
        contains : bool
            `True` if the set is measurable with respect to this sigma-algebra, `False` otherwise.
        """
        return self.is_measurable(candidate)

    # --------------------- sequence methods --------------------- #

    def __iter__(self) -> iter:
        """Iterate over the atoms (as `MeasurableSets`) in this sigma-algebra.

        Returns
        -------
        iterator : iter
            An iterator over the atoms (as `MeasurableSets`) in the sigma-algebra.
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
        if not isinstance(other, SigmaAlgebra):
            return False
        if self.domain != other.domain:
            return False
        if self.num_atoms != other.num_atoms:
            return False
        if self.is_power_set and other.is_power_set:
            return True

        if isinstance(other.data.index, pd.MultiIndex):
            other_data = other.data.reorder_levels(self.domain.variable_names)
        else:
            other_data = other.data

        self_sorted = self.data.sort_index()
        other_sorted = other_data.sort_index()

        return (
            len(pd.concat([self_sorted, other_sorted], axis=1).drop_duplicates())
            == self.num_atoms
        )

    # --------------------- lattice methods --------------------- #

    def __or__(self, other: SigmaAlgebra) -> SigmaAlgebra:
        r"""Get the join (least upper bound) of this sigma-algebra with another.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        other : SigmaAlgebra
            The other sigma-algebra to join with.

        Returns
        -------
        join_sigma_algebra : SigmaAlgebra
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
        Sigma algebra 'join':
               atom_ID_0  atom_ID_1
        point
        0              0          0
        1              0          1
        2              0          1
        3              1          1
        4              1          0
        5              1          0

        Notes
        -----
        Let $\{\mathcal{F}_i\}_{k\in K}$ be a finite collection of $\sigma$-algebras on a finite set $X$. The *join* (or *least upper bound*) of the collection, denoted $\bigvee_{k\in K} \mathcal{F}_k$, is the coarsest $\sigma$-algebra that contains all of the $\mathcal{F}_k$. Its atoms are given by the nonempty intersections of atoms from each $\mathcal{F}_k$. In particular, the atom identifiers for the join can be represented as tuples of the atom identifiers from each $\mathcal{F}_k$.
        """
        from .lattice import Lattice

        return Lattice.join([self, other])

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
        from .lattice import Lattice

        if not isinstance(other, SigmaAlgebra):
            return NotImplemented

        return Lattice.is_subalgebra(sub_algebra=self, super_algebra=other)

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
            return NotImplemented

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
        from .lattice import Lattice

        if not isinstance(other, SigmaAlgebra):
            return NotImplemented

        return Lattice.is_subalgebra(sub_algebra=other, super_algebra=self)

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
            return NotImplemented

        return self >= other and self != other


class SigmaAlgebraMethods:
    """Mixin class providing sigma-algebra methods to other classes."""

    def get_set(self, indices: list[Hashable], name: Hashable = "A") -> MeasurableSet:
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
        measurable_set : MeasurableSet
            An `MeasurableSet` object containing the specified points.
        """
        return self.sig_alg.get_set(indices, name)

    def is_measurable(self, candidate: MeasurableSet) -> bool:
        """Check if a candidate set is measurable with respect to the sigma-algebra.

        Calls `SigmaAlgebra.is_measurable`.

        Parameters
        ----------
        candidate : MeasurableSet | list[Hashable]
            The set to check for measurability.

        Returns
        -------
        is_measurable : bool
            `True` if the set is measurable with respect to this sigma-algebra, `False` otherwise.
        """
        return self.sig_alg.is_measurable(candidate)
