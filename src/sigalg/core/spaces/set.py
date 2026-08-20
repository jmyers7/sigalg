"""A class representing a set."""

from __future__ import annotations

from functools import cached_property
from itertools import product
from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd

from ..indices.index import Index

if TYPE_CHECKING:
    from collections.abc import Hashable

    from ...typing.index_like import IndexLike
    from ..functions.function import Function
    from ..sigma_algebras.lattice import Lattice
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from .domain import Domain
    from .sample_space import SampleSpace


class Set(Index):
    r"""A class representing a set.

    See the Notes section below for the mathematical details.

    Examples
    --------
    >>> from sigalg.core import Domain, Set, SigmaAlgebra

    Extract a set from a domain with four points.

    >>> X = Domain.from_sequence(size=4)
    >>> A = Set([1, 2], domain=X)

    Print it.

    >>> print(A)  # doctest: +NORMALIZE_WHITESPACE
    Set 'A':
     x
     1
     2
    """

    _default_name = "A"
    _repr_name = "Set"
    _str_name = "Set"

    # --------------------- constructors --------------------- #

    def __init__(
        self, indices: IndexLike, domain: Domain, name: Hashable | None = None
    ) -> None:
        super().__init__(
            indices=indices,
            name=name,
            variable_names=domain.variable_names,
        )
        if not set(self) <= set(domain):
            raise ValueError("The set of points is not a subset of the domain.")
        self.domain = domain

    @classmethod
    def _from_validated(cls, *, data: pd.Index, domain: Domain, name: Hashable) -> Set:
        measurable_set = object.__new__(Set)
        measurable_set.data = data
        measurable_set.domain = domain
        measurable_set.name = name
        return measurable_set

    @classmethod
    def cartesian_product(
        cls,
        factors: list[Set],
        name: Hashable | None = None,
    ) -> Index:
        """Pass."""
        from .._utils.utils import flatten

        domain = type(factors[0].domain).cartesian_product(
            [factor.domain for factor in factors]
        )

        if name is None:
            name = " x ".join([str(factor.name) for factor in factors])

        product_indices = list(product(*factors))
        flattened_indices = [flatten(t) for t in product_indices]

        return cls(indices=flattened_indices, domain=domain, name=name)

    # --------------------- properties --------------------- #

    @cached_property
    def lattice(self) -> Lattice:
        r"""Get the (upward) lattice of sigma-algebras containing this set.

        See the Notes section below for the mathematical details.

        Returns
        -------
        lattice : Lattice
            The (upward) lattice of sigma-algebras containing this set.

        Examples
        --------
        >>> from sigalg.core import Domain, Set, SigmaAlgebra

        Define two sigma-algebras on a domain.

        >>> X = Domain.from_sequence(size=4, variable_name="x")
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ... )
        >>> G = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 0,
        ...     },
        ...     name="G",
        ... )
        >>> H = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 0,
        ...         3: 0,
        ...     },
        ...     name="H",
        ... )

        Extract a subset of the domain.

        >>> A = Set([1, 2], domain=X)

        We may test whether the set is measurable with respect to a sigma-algebra by using the `in` operator with the `lattice` attribute.

        >>> F in A.lattice
        True

        We may get the indicator values of the set on the atoms of the sigma-algebra by calling the `get_atom_data` method.

        >>> print(A.lattice.get_atom_data(F))  # doctest: +NORMALIZE_WHITESPACE
        F
        0    0
        1    1
        2    0
        Name: sigma(A), dtype: int64

        Whenever a measurability check is executed, and the result is `True`, the sigma-algebra is added to the internal `lattice`.

        >>> A.lattice
        Lattice(base=sigma(A), type=upward, num_sig_algs=2)

        Perform another measurability check, inspect the `lattice` attribute to see the updated list of contents, and print the atom data.

        >>> G in A.lattice
        True
        >>> A.lattice
        Lattice(base=sigma(A), type=upward, num_sig_algs=3)
        >>> print(A.lattice.get_atom_data(G))  # doctest: +NORMALIZE_WHITESPACE
        G
        0    0
        1    1
        Name: sigma(A), dtype: int64

        Notice that the set is not measurable with respect to the third sigma-algebra. The measurability check accordingly returns `False`, and the contents of `lattice` is not changed.

        >>> H in A.lattice
        False
        >>> A.lattice
        Lattice(base=sigma(A), type=upward, num_sig_algs=3)

        Notes
        -----
        Let $A$ be a subset of a nonempty finite set $X$. The *lattice* of $\sigma$-algebras associated with $A$ is the collection of all $\sigma$-algebras on $X$ that contain $A$. A $\sigma$-algebra $\mathcal{F}$ is in this lattice if and only if the set is $\mathcal{F}$-measurable.
        """
        from ..sigma_algebras.lattice import Lattice

        return Lattice(base=self.generated_sig_alg, type="upward")

    @cached_property
    def indicator_data(self) -> pd.Series | None:
        """Get the underlying data of the indicator function of the set as a `pd.Series`.

        Returns
        -------
        indicator_data : pd.Series | None
            The data of the indicator function of the set.

        Examples
        --------
        >>> from sigalg.core import Domain, Set
        >>> X = Domain.from_sequence(size=4)
        >>> A = Set([0, 1], domain=X)
        >>> print(A.indicator_data)  # doctest: +NORMALIZE_WHITESPACE
             x
        0    1
        1    1
        2    0
        3    0
        Name: I_A, dtype: int64
        """
        if self.data is not None:
            name = f"I_{self.name}"
            ones = pd.Series(1, index=self.data, name=name)
            return ones.reindex(self.domain.data, fill_value=0)

        else:
            return None

    @cached_property
    def indicator(self) -> Function | None:
        """Get the indicator function of the set.

        Returns
        -------
        indicator : Function | None
            The indicator function of the set.

        Examples
        --------
        >>> from sigalg.core import Domain, Set
        >>> X = Domain.from_sequence(size=4)
        >>> A = Set([0, 1], domain=X)
        >>> print(A.indicator)  # doctest: +NORMALIZE_WHITESPACE
        Function 'I_A':
                I_A
        x
        0        1
        1        1
        2        0
        3        0
        """
        from ..functions.function import Function

        if self.data is not None:
            name = f"I_{self.name}"

            return Function._from_validated(
                data=self.indicator_data,
                kind="any",
                name=name,
                domain_kind=type(self.domain).__name__,
                domain_name=self.domain.name,
                index_kind="Index",
                index_name=None,
            )

        else:
            return None

    @cached_property
    def generated_sig_alg(self) -> SigmaAlgebra | None:
        r"""Get the sigma-algebra generated by the set.

        Returns
        -------
        gen_sig_alg : SigmaAlgebra | None
            The sigma-algebra generated by the set.

        Notes
        -----
        Let $A$ be a nonempty, proper subset of a nonempty finite set $X$. The $\sigma$-algebra generated by $A$, denoted $\sigma(A)$, is the $\sigma$-algebra whose atoms are $A$ and its complement $A^c$. If $A$ is either empty or coincides with the whole set $X$, then $\sigma(A)$ is the trivial $\sigma$-algebra whose single atom is $X$ itself.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if self.data is not None:
            return SigmaAlgebra._from_validated(
                data=self.indicator_data.rename(self.name),
                variable_names=[self.name],
                name=f"sigma({self.name})",
                domain_kind=type(self.domain).__name__,
                domain_name=self.domain.name,
                index_kind="Index",
                index_name=None,
            )

    # --------------------- measurable methods --------------------- #

    def is_atom(self, sig_alg: SigmaAlgebra) -> bool | None:
        """Return whether this set is an atom in a given sigma-algebra.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The sigma-algebra with respect to which to check if this set is an atom.

        Returns
        -------
        is_atom : bool | None
            Whether the current set is an atom or not relative to the given sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, Set, SigmaAlgebra

        Define a sigma-algebra on a sample space.

        >>> Omega = SampleSpace.from_sequence(size=5)
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 1,
        ...         1: 1,
        ...         2: 0,
        ...         3: 0,
        ...         4: 2,
        ...     },
        ... )

        Extract a set and check whether it is an atom.

        >>> A = Set([0, 1], domain=Omega)
        >>> A.is_atom(F)
        True

        Extract a second set and perform the same check.

        >>> B = Set([0, 1, 2, 3], domain=Omega, name="B")
        >>> B.is_atom(F)
        False

        Check whether a non-measurable set is an atom.

        >>> C = Set([0, 1, 2], domain=Omega, name="C")
        >>> C.is_atom(F)
        False
        """
        from ..sigma_algebras.lattice import NonMeasurableError

        if self.data is not None:
            try:
                _ = self.lattice.add(sig_alg)
                atom_indicator_data = self.lattice[sig_alg]
            except NonMeasurableError:
                return False

            return bool(atom_indicator_data.sum() == 1)

        else:
            return None

    def atom_id(self, sig_alg: SigmaAlgebra) -> Hashable | None:
        """Return the atom identifiers of this set relative to a given sigma-algebra provided that it is an atom, or `None` otherwise.

        Returns
        -------
        atom_id : Hashable | None
            The atom ID of the current set if it is an atom of the given sigma-algebra, and `None` otherwise.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, Set, SigmaAlgebra

        Define a sigma-algebra on a sample space.

        >>> Omega = SampleSpace.from_sequence(size=5)
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 1,
        ...         1: 1,
        ...         2: 0,
        ...         3: 0,
        ...         4: 2,
        ...     },
        ... )

        Extract a set and get its atom identifier.

        >>> A = Set([0, 1], domain=Omega)
        >>> A.atom_id(F)
        {'F': 1}

        Attempt to access the atom identifier of a non-atom measurable set.

        >>> B = Set([0, 1, 2, 3], domain=Omega, name="B")
        >>> print(B.atom_id(F))
        None

        Attempt to access the atom identifier of a non-measurable set.

        >>> C = Set([0, 1, 2], domain=Omega, name="C")
        >>> print(C.atom_id(F))
        None
        """
        from ..sigma_algebras.lattice import NonMeasurableError

        if self.data is not None and self.is_atom(sig_alg):
            try:
                _ = self.lattice.add(sig_alg)
                atom_indicator_data = self.lattice[sig_alg]
            except NonMeasurableError:
                return None

            atom_id = atom_indicator_data[atom_indicator_data == 1].index

            if atom_id.nlevels > 1:
                return {
                    name: value.astype(Real) if hasattr(value, "astype") else value
                    for name, value in zip(atom_id.names, atom_id[0])
                }
            else:
                return {
                    atom_id.name: atom_id[0].astype(Real)
                    if hasattr(atom_id[0], "astype")
                    else atom_id[0]
                }

            # if isinstance(atom_id, tuple):
            #     return tuple(
            #         x.astype(Real) if hasattr(x, "astype") else x for x in atom_id
            #     )
            # else:
            #     return atom_id.astype(Real) if hasattr(atom_id, "astype") else atom_id

        else:
            return None

    def indicator_atom_data(self, sig_alg: SigmaAlgebra) -> pd.Series | None:
        """Pass."""
        import pandas as pd

        from .._utils.utils import to_df

        if self.indicator_data is not None:
            sig_alg_data = to_df(sig_alg.data)
            sig_alg_cols = list(sig_alg_data.columns)

            data = (
                pd.concat([self.indicator_data, sig_alg_data], axis=1)
                .drop_duplicates(sig_alg_cols)
                .set_index(sig_alg_cols)
            )
            data.index.names = sig_alg.variable_names

            return data

        else:
            return None

    # --------------------- set-theoretic operations --------------------- #

    def complement(self) -> Set:
        """Return the complement of this set.

        Returns
        -------
        set : Set
            A set containing all points not in this set.

        Examples
        --------
        >>> from sigalg.core import Set, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_set([0])
        >>> print(A.complement()) # doctest: +NORMALIZE_WHITESPACE
        Set 'A complement':
         s
         1
         2
        """
        return ~self

    def intersection(self, other: Set) -> Set:
        """Return the intersection of this set with another set.

        Parameters
        ----------
        other : Set
            Another set from the same domain.

        Returns
        -------
        set : Set
            A set containing all points in both sets.

        Examples
        --------
        >>> from sigalg.core import Set, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_set([0, 1])
        >>> B = F.get_set([1, 2], name="B")
        >>> print(A.intersection(B)) # doctest: +NORMALIZE_WHITESPACE
        Set 'A intersect B':
         s
         1
        """
        return self & other

    def union(self, other: Set) -> Set:
        """Return the union of this set with another set.

        Parameters
        ----------
        other : Set
            Another set from the same domain.

        Returns
        -------
        set : Set
            A set containing all points in either set.

        Examples
        --------
        >>> from sigalg.core import Set, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_set([0])
        >>> B = F.get_set([1], name="B")
        >>> print(A.union(B)) # doctest: +NORMALIZE_WHITESPACE
        Set 'A union B':
         s
         0
         1
        """
        return self | other

    def difference(self, other: Set) -> Set:
        """Return the set difference of this set and another set.

        Parameters
        ----------
        other : Set
            Another set from the same domain.

        Returns
        -------
        set : Set
            A set containing all points in this set but not in `other`.

        Examples
        --------
        >>> from sigalg.core import Set, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_set([0, 1])
        >>> B = F.get_set([1, 2], name="B")
        >>> print(A.difference(B)) # doctest: +NORMALIZE_WHITESPACE
        Set 'A difference B':
         s
         0
        """
        return self - other

    def __invert__(self) -> Set:
        """Return the complement of this set (`~` operator).

        Returns
        -------
        set : Set
            A set containing all points not in this set.
        """
        space = self.domain.data
        pts = set(self.data)
        comp = [idx for idx in space if idx not in pts]
        if self.domain.dimension > 1:
            data = pd.MultiIndex.from_tuples(comp, names=self.domain.variable_names)
        else:
            data = pd.Index(comp, name=self.domain.variable_names[0])

        return type(self)._from_validated(
            data=data, domain=self.domain, name=f"{self.name} complement"
        )

    def __and__(self, other: Set) -> Set:
        """Return the intersection of this set with another set (`&` operator).

        Parameters
        ----------
        other : Set
            Another set from the same domain.

        Raises
        ------
        ValueError
            If sets are from different domains.

        Returns
        -------
        set : Set
            A set containing all points in both sets.
        """
        if self.domain != other.domain:
            raise ValueError("Sets must belong to the same domain.")

        pts = set(self.data) & set(other.data)

        if isinstance(self.data, pd.MultiIndex):
            data = pd.MultiIndex.from_tuples(pts, names=self.domain.variable_names)
        else:
            data = pd.Index(pts, name=self.domain.variable_names[0])

        return type(self)._from_validated(
            data=data, domain=self.domain, name=f"{self.name} intersect {other.name}"
        )

    def __or__(self, other: Set) -> Set:
        """Return the union of this set with another set (`|` operator).

        Parameters
        ----------
        other : Set
            Another set from the same domain.

        Raises
        ------
        ValueError
            If sets are from different domains.

        Returns
        -------
        set : Set
            A set containing all points in either set.
        """
        if self.domain != other.domain:
            raise ValueError("Sets must belong to the same domain.")

        pts = set(self.data) | set(other.data)

        if isinstance(self.data, pd.MultiIndex):
            data = pd.MultiIndex.from_tuples(pts, names=self.domain.variable_names)
        else:
            data = pd.Index(pts, name=self.domain.variable_names[0])

        return type(self)._from_validated(
            data=data, domain=self.domain, name=f"{self.name} union {other.name}"
        )

    def __sub__(self, other: Set) -> Set:
        """Return the set difference of this set and another set from the same domain (`-` operator).

        Parameters
        ----------
        other : Set
            Another set from the same domain.

        Raises
        ------
        ValueError
            If sets are from different domains.

        Returns
        -------
        set : Set
            A set containing all points in this set but not in `other`.
        """
        if self.domain != other.domain:
            raise ValueError("Sets must belong to the same domain.")

        pts = set(self.data) - set(other.data)

        if isinstance(self.data, pd.MultiIndex):
            data = pd.MultiIndex.from_tuples(pts, names=self.domain.variable_names)
        else:
            data = pd.Index(pts, name=self.domain.variable_names[0])

        return type(self)._from_validated(
            data=data, domain=self.domain, name=f"{self.name} difference {other.name}"
        )

    # --------------------- sub/superset methods --------------------- #

    def __le__(self, other: Set) -> bool:
        """Check if this set is a subset of another set (`<=` operator).

        Parameters
        ----------
        other : Set
            Another set from the same domain.

        Raises
        ------
        ValueError
            If sets are from different domains.

        Returns
        -------
        is_le : bool
            True if this set is a subset of the other set.
        """
        if self.domain != other.domain:
            raise ValueError("Measurable sets must belong to the same domain.")

        return set(self.data).issubset(set(other.data))

    def __lt__(self, other: Set) -> bool:
        """Check if this set is a proper subset of another set (`<` operator).

        Parameters
        ----------
        other : Set
            Another set from the same domain.

        Raises
        ------
        ValueError
            If sets are from different domains.

        Returns
        -------
        is_lt : bool
            True if this set is a proper subset of the other set.
        """
        if self.domain != other.domain:
            raise ValueError("Measurable sets must belong to the same domain.")

        return set(self.data) < set(other.data)

    def __ge__(self, other: Set) -> bool:
        """Check if this set is a superset of another set (`>=` operator).

        Parameters
        ----------
        other : Set
            Another set from the same domain.

        Raises
        ------
        ValueError
            If sets are from different domains.

        Returns
        -------
        is_ge : bool
            True if this set is a superset of the other set.
        """
        if self.domain != other.domain:
            raise ValueError("Measurable sets must belong to the same domain.")

        return set(self.data).issuperset(set(other.data))

    def __gt__(self, other: Set) -> bool:
        """Check if this set is a proper superset of another set (`>` operator).

        Parameters
        ----------
        other : Set
            Another set from the same domain.

        Raises
        ------
        ValueError
            If sets are from different domains.

        Returns
        -------
        is_gt : bool
            True if this set is a proper superset of the other set.
        """
        if self.domain != other.domain:
            raise ValueError("Measurable sets must belong to the same domain.")

        return set(self.data) > set(other.data)

    # --------------------- equality --------------------- #

    def __eq__(self, other) -> bool:
        """Check equality with another set.

        Two sets are equal if they belong to the same domain and are equal as instances of `Index`.

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the other object is a `Set` with identical domain
            and values, `False` otherwise.
        """
        return (
            isinstance(other, Set)
            and self.domain == other.domain
            and super().__eq__(other)
        )

    # --------------------- conversion methods --------------------- #

    def to_domain(self) -> Domain:
        """Convert this set to a domain.

        Creates a new `Domain` containing only the points in this set.

        Returns
        -------
        domain : Domain
            A domain containing this set's points.

        Examples
        --------
        >>> from sigalg.core import Set, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_set([0, 1])
        >>> print(A.to_domain()) # doctest: +NORMALIZE_WHITESPACE
        Domain 'A':
         x
         0
         1
        """
        from .domain import Domain

        return Domain(indices=self.data.to_list(), name=self.name)

    def to_sample_space(self) -> SampleSpace:
        """Convert this set to a sample space.

        Creates a new `SampleSpace` containing only the points in this set.

        Returns
        -------
        sample_space : SampleSpace
            A sample space containing this set's points.

        Examples
        --------
        >>> from sigalg.core import Set, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_set([0, 1])
        >>> print(A.to_sample_space()) # doctest: +NORMALIZE_WHITESPACE
        Sample space 'A':
         s
         0
         1
        """
        from .sample_space import SampleSpace

        return SampleSpace(indices=self.data.to_list(), name=self.name)

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a concise string representation of the set.

        Returns
        -------
        repr_str : str
            String representation of the set.
        """
        if self.data is None:
            return f"{type(self)._repr_name}(empty)"
        else:
            return f"{type(self)._repr_name}(domain={self.domain.name}, num_points={len(self.data)}, name={self.name})"
