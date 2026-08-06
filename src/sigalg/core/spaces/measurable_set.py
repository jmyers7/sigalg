"""A class representing a measurable subset."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..indices.index import Index

if TYPE_CHECKING:
    from ..functions.measurable_function import MeasurableFunction
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from .domain import Domain
    from .measurable_space import MeasurableSpace
    from .sample_space import SampleSpace


class MeasurableSet(Index):
    r"""A class representing a measurable subset.

    The constructor should not be used directly. Instead, the user should use the `from_list` class method or the `get_set` method of a `SigmaAlgebra` instance to create a measurable set.

    See the Notes section below for the mathematical details.

    Examples
    --------
    Extract a measurable set by calling the `from_list` class method.

    >>> from sigalg.core import MeasurableSet, SampleSpace, SigmaAlgebra
    >>> Omega = SampleSpace.from_sequence(size=4)
    >>> F = SigmaAlgebra.power_set(Omega)
    >>> A = MeasurableSet.from_list([0, 2], sig_alg=F)
    >>> print(A) # doctest: +NORMALIZE_WHITESPACE
    Measurable set 'A':
     sample
          0
          2

    Extract a measurable set directly from the sigma-algebra

    >>> B = F.get_set([1, 3], name="B")
    >>> print(B) # doctest: +NORMALIZE_WHITESPACE
    Measurable set 'B':
     sample
          1
          3

    Notes
    -----
    Let $\mathcal{F}$ be a $\sigma$-algebra on a nonempty set $X$. An *$\mathcal{F}$-measurable set* is a subset $A$ of $X$ in $\mathcal{F}$. In the case that $X$ is finite (as it always is, in SigAlg), a set $A$ is $\mathcal{F}$-measurable if and only if it is a union of atoms of $\mathcal{F}$.
    """

    _properties = Index._properties + [
        "_sig_alg",
        "_is_atom",
        "_indicator_mapping",
        "_indicator",
        "_atom_id",
        "_measurable_space",
    ]
    _default_name = "A"
    _repr_name = "MeasurableSet"
    _str_name = "Measurable set"
    _variable_names_prefix = "point"

    # --------------------- constructors --------------------- #

    @classmethod
    def from_list(
        cls,
        indices: list[Hashable],
        sig_alg: SigmaAlgebra,
        name: Hashable = "A",
    ) -> MeasurableSet:
        """Create a measurable set from a list of points.

        Parameters
        ----------
        indices : list[Hashable]
            List of points to include in the measurable set.
        sig_alg : SigmaAlgebra
            The sigma-algebra to which the measurable set belongs.
        name : Hashable, default="A"
            Name of the measurable set.

        Raises
        ------
        TypeError
            If `indices` is not a list of hashable elements, or if `sig_alg` is not an instance of `SigmaAlgebra`.
        ValueError
            If the measurable set defined by `indices` is not measurable with respect to the sigma-algebra.

        Returns
        -------
        measurable_set : MeasurableSet
            The measurable set instance with the specified sample points.

        Examples
        --------
        Define a sigma-algebra with three atoms.

        >>> from sigalg.core import MeasurableSet, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=5)
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (0, 1),
        ...         3: (0, 1),
        ...         4: (2, 3),
        ...     },
        ... )

        Get a measurable set from the sigma-algebra.

        >>> A = MeasurableSet.from_list(indices=[0, 1], sig_alg=F)
        >>> print(A)  # doctest: +NORMALIZE_WHITESPACE
        Measurable set 'A':
         sample
              0
              1

        Try to build a non-measurable set.

        >>> B = MeasurableSet.from_list(indices=[0, 2], sig_alg=F, name="B")
        Traceback (most recent call last):
            ...
        ValueError: The candidate set is not measurable.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .measurable_space import MeasurableSpace

        if not isinstance(indices, list) or not all(
            isinstance(x, Hashable) for x in indices
        ):
            raise TypeError("indices must be a list of hashable elements.")
        if not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be a SigmaAlgebra")

        is_measurable, is_atom, atom_id, indicator_mapping = (
            MeasurableSet.is_measurable(
                candidate=indices, sig_alg=sig_alg, verbose=True
            )
        )

        if not is_measurable:
            raise ValueError("The candidate set is not measurable.")

        measurable_space = MeasurableSpace(sig_alg=sig_alg)

        measurable_set = cls(
            indices=indicator_mapping[indicator_mapping == 1].index,
            name=name,
            variable_names=measurable_space.domain.variable_names,
        )

        measurable_set._measurable_space = measurable_space
        measurable_set._is_atom = is_atom
        measurable_set._atom_id = atom_id
        measurable_set._indicator_mapping = indicator_mapping.rename(None)

        return measurable_set

    @staticmethod
    def is_measurable(
        candidate: list[Hashable] | MeasurableSet,
        sig_alg: SigmaAlgebra,
        verbose: bool = False,
    ) -> bool | tuple[bool, bool, Hashable | None, pd.Series]:
        """Check if a candidate set is measurable with respect to a given sigma-algebra, and return atom information and indicator mapping.

        Parameters
        ----------
        candidate : list[Hashable] | MeasurableSet
            The candidate set (list or MeasurableSet) to check for measurability.
        sig_alg : SigmaAlgebra
            The sigma-algebra to check measurability against.
        verbose : bool, default=False
            If `True`, return additional information about whether the candidate set is an atom and its atom ID, along with the indicator mapping as a `pd.Series`. If `False`, only return whether the candidate set is measurable.

        Raises
        ------
        TypeError
            If `sig_alg` is not an instance of `SigmaAlgebra`, or if `candidate` is not a list of hashable elements.
        ValueError
            If the candidate set is not a subset of the domain of the sigma-algebra.

        Returns
        -------
        measurability_info : bool | tuple[bool, bool, Hashable | None, pd.Series]
            If `verbose` is `False`, returns a boolean indicating whether the candidate set is measurable. If `verbose` is `True`, returns a tuple containing:
            - `is_measurable`: Whether the candidate set is measurable with respect to the sigma-algebra.
            - `is_atom`: Whether the candidate set is an atom in the sigma-algebra.
            - `atom_id`: The identifier of the atom if the candidate set is an atom, otherwise `None`.
            - `indicator_mapping`: A `pd.Series` indicating membership of each element in the candidate set.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import Domain, MeasurableSet, SigmaAlgebra
        >>> rng = np.random.default_rng(42)
        >>> X = Domain.from_rand(
        ...     size=10,
        ...     dim=2,
        ...     variable_names=["x_0", "x_1"],
        ...     random_state=rng,
        ... )
        >>> F = SigmaAlgebra.from_rand(
        ...     num_atoms=4,
        ...     domain=X,
        ...     dim=2,
        ...     variable_names=["u", "v"],
        ...     random_state=rng,
        ... )
        >>> print(F)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                 u  v
        x_0 x_1
        2   8    0  3
        9   4    3  3
        1   5    2  3
        6   3    0  3
        3   1    3  3
        8   9    0  3
        5   7    0  3
        7   6    1  0
        4   4    2  3
        0   8    1  0
        >>> candidate1 = [
        ...     (3, 1),
        ...     (9, 4),
        ...     (1, 5),
        ...     (4, 4),
        ... ]
        >>> is_measurable, is_atom, atom_id, indicator_mapping = MeasurableSet.is_measurable(
        ...     candidate=candidate1,
        ...     sig_alg=F,
        ...     verbose=True,
        ... )
        >>> print(is_measurable, is_atom, atom_id)
        True False None
        >>> print(indicator_mapping)  # doctest: +NORMALIZE_WHITESPACE
        x_0  x_1
        2    8      0
        9    4      1
        1    5      1
        6    3      0
        3    1      1
        8    9      0
        5    7      0
        7    6      0
        4    4      1
        0    8      0
        Name: indicator, dtype: int64
        >>> candidate2 = [
        ...     (7, 6),
        ...     (0, 8),
        ... ]
        >>> is_measurable, is_atom, atom_id, indicator_mapping = MeasurableSet.is_measurable(
        ...     candidate=candidate2,
        ...     sig_alg=F,
        ...     verbose=True,
        ... )
        >>> print(is_measurable, is_atom, atom_id)
        True True (1, 0)
        >>> print(indicator_mapping)  # doctest: +NORMALIZE_WHITESPACE
        x_0  x_1
        2    8      0
        9    4      0
        1    5      0
        6    3      0
        3    1      0
        8    9      0
        5    7      0
        7    6      1
        4    4      0
        0    8      1
        Name: indicator, dtype: int64
        >>> candidate3 = [
        ...     (2, 8),
        ...     (9, 4),
        ... ]
        >>> is_measurable, is_atom, atom_id, indicator_mapping = MeasurableSet.is_measurable(
        ...     candidate=candidate3,
        ...     sig_alg=F,
        ...     verbose=True,
        ... )
        >>> print(is_measurable, is_atom, atom_id)
        False False None
        >>> print(indicator_mapping)  # doctest: +NORMALIZE_WHITESPACE
        x_0  x_1
        2    8      1
        9    4      1
        1    5      0
        6    3      0
        3    1      0
        8    9      0
        5    7      0
        7    6      0
        4    4      0
        0    8      0
        Name: indicator, dtype: int64
        >>> candidate4 = []
        >>> is_measurable, is_atom, atom_id, indicator_mapping = MeasurableSet.is_measurable(
        ...     candidate=candidate4,
        ...     sig_alg=F,
        ...     verbose=True,
        ... )
        >>> print(is_measurable, is_atom, atom_id)
        True False None
        >>> print(indicator_mapping)  # doctest: +NORMALIZE_WHITESPACE
        x_0  x_1
        2    8      0
        9    4      0
        1    5      0
        6    3      0
        3    1      0
        8    9      0
        5    7      0
        7    6      0
        4    4      0
        0    8      0
        Name: indicator, dtype: int64
        """
        from ..indices.index import Index
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be an instance of SigmaAlgebra.")
        if not isinstance(candidate, MeasurableSet) and (
            not isinstance(candidate, list)
            or not all(isinstance(x, Hashable) for x in candidate)
        ):
            raise TypeError(
                "candidate must be a list of hashable elements or a MeasurableSet."
            )

        if len(candidate) == 0:
            zeros = pd.Series(
                np.zeros(len(sig_alg.domain), dtype=int),
                index=sig_alg.domain.data,
                name="indicator",
            )
            if verbose:
                return True, False, None, zeros
            else:
                return True

        if not set(candidate) <= set(sig_alg.domain):
            raise ValueError(
                "The candidate set is not a subset of the domain of the sigma-algebra."
            )

        if not verbose and sig_alg.is_power_set:
            return True

        if isinstance(candidate, list):
            candidate = Index(candidate, variable_names=sig_alg.domain.variable_names)

        ones = pd.Series(
            np.ones(len(candidate), dtype=int),
            index=candidate.data,
            name="indicator",
        )

        mapping = pd.merge(
            left=sig_alg.data,
            right=ones,
            left_index=True,
            right_index=True,
            how="left",
        ).fillna(0)

        measurable_test_data = mapping.drop_duplicates()
        is_measurable = len(measurable_test_data) == sig_alg.num_atoms

        if verbose:
            is_atom = bool(measurable_test_data["indicator"].sum() == 1)

            if is_atom:
                atom_id = measurable_test_data.loc[
                    measurable_test_data["indicator"] == 1, sig_alg.variable_names
                ]
                atom_id = tuple(atom_id.iloc[0])
                atom_id = atom_id[0] if len(atom_id) == 1 else atom_id
            else:
                atom_id = None

            return (
                is_measurable,
                is_atom,
                atom_id,
                mapping.drop(columns=sig_alg.variable_names)
                .squeeze(axis=1)
                .astype(int),
            )
        else:
            return is_measurable

    # --------------------- properties --------------------- #

    @property
    def measurable_space(self) -> MeasurableSpace | None:
        """Get the measurable space associated with this set.

        Returns
        -------
        measurable_space : MeasurableSpace | None
            The measurable space associated with this set.
        """
        return self._measurable_space

    @property
    def domain(self) -> Domain | None:
        """Get the domain associated with this set.

        Returns
        -------
        domain : Domain | None
            The ambient domain of the set.
        """
        return (
            self.measurable_space.domain if self.measurable_space is not None else None
        )

    @property
    def sig_alg(self) -> SigmaAlgebra | None:
        """Get the sigma-algebra containing this set.

        The `sig_alg` property is settable. The new sigma-algebra must be a sub-sigma-algebra of the existing one, and the set must be measurable with respect to the new sigma-algebra.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra containing this set.

        Examples
        --------
        Define a measurable space.

        >>> from sigalg.core import MeasurableSpace, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=5)
        >>> F = SigmaAlgebra(
        ...    domain=Omega,
        ...    mapping={
        ...        0: 1,
        ...        1: 1,
        ...        2: 0,
        ...        3: 0,
        ...        4: 2,
        ...    },
        ... )
        >>> measurable_space = MeasurableSpace(Omega, F)

        Extract a measurable set from the measurable space and print its `sig_alg` property.

        >>> A = measurable_space.get_set([0, 1])
        >>> print(A.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                atom_ID
        sample
        0             1
        1             1
        2             0
        3             0
        4             2

        Define a new sigma-algebra, a sub-sigma-algebra of the first.

        >>> G = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 1,
        ...     },
        ...     name="G",
        ... )

        Set the sigma-algebra and print the updated measurable space.

        >>> A.sig_alg = G
        >>> print(A.measurable_space)  # doctest: +NORMALIZE_WHITESPACE
        Measurable space (Omega, G)
        ===========================
        <BLANKLINE>
        * Sample space 'Omega':
         sample
              0
              1
              2
              3
              4
        <BLANKLINE>
        * Sigma algebra 'G':
                atom_ID
        sample
        0             0
        1             0
        2             1
        3             1
        4             1
        """
        return (
            self.measurable_space.sig_alg if self.measurable_space is not None else None
        )

    @sig_alg.setter
    def sig_alg(self, sig_alg: SigmaAlgebra) -> None:
        """Set the sigma-algebra associated with this measurable set.

        The new sigma-algebra must be a sub-sigma-algebra of the existing one, and the set must be measurable with respect to the new sigma-algebra.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The new sigma-algebra.

        Raises
        ------
        TypeError
            If `sig_alg` is not an instance of `SigmaAlgebra`.
        ValueError
            If the current instance of `MeasurableSet` has a `measurable_space` attribute equal to `None`, or if the current instance is not in the new sigma-algebra.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be an instance of SigmaAlgebra.")
        if self.measurable_space is None:
            raise ValueError(
                "Cannot set a new sigma-algebra for a measurable set whose measurable_space attribute is `None`."
            )
        if self not in sig_alg:
            raise ValueError("The measurable set must be in the new sigma-algebra.")

        self.measurable_space.sig_alg = sig_alg
        self._indicator = None

    @property
    def indicator(self) -> MeasurableFunction | None:
        """Get the indicator function of this measurable set.

        Returns
        -------
        indicator : MeasurableFunction | None
            The indicator function of this measurable set.

        Examples
        --------
        >>> from sigalg.core import Domain, MeasurableSet, SigmaAlgebra
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
        >>> A = MeasurableSet.from_list(
        ...     indices=[0, 1],
        ...     sig_alg=F,
        ... )
        >>> print(A.indicator)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'I_A':
               I_A
        point
        0        1
        1        1
        2        0
        3        0
        """
        from ..functions.measurable_function import MeasurableFunction

        if self._indicator is None and self._indicator_mapping is not None:
            self._indicator = MeasurableFunction(
                domain=self.domain,
                sig_alg=self.sig_alg,
                mapping=self._indicator_mapping,
                name=f"I_{self.name}",
            )

        return self._indicator

    @property
    def is_atom(self) -> bool | None:
        """Return whether this measurable set is an atom in the sigma-algebra.

        Returns
        -------
        is_atom : bool | None
            Whether the current measurable set is an atom or not.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
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
        >>> A = F.get_set([0, 1])
        >>> print(A.is_atom)
        True
        >>> B = F.get_set([0, 1, 2, 3], name="B")
        >>> print(B.is_atom)
        False
        """
        return self._is_atom

    @property
    def atom_id(self) -> Hashable | None:
        """Return the atom ID if this measurable set is an atom, or `None` otherwise.

        Returns
        -------
        atom_id : Hashable | None
            The atom ID of the current measurable set if it is an atom, and `None` otherwise.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
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
        >>> A = F.get_set([0, 1])
        >>> print(A.atom_id)
        1
        >>> B = F.get_set([0, 1, 2, 3], name="B")
        >>> print(B.atom_id)
        None
        """
        return self._atom_id

    # --------------------- data access methods --------------------- #

    def __getitem__(self, key: any) -> MeasurableSet | Hashable:
        """Get a measurable set from the current measurable set by indexing.

        If `key` is an integer, a measurable set is created from a single point retrieved by position given by `key`; a slice creates a measurable set with a slice of sample points, a tuple `(index, name)` creates a measurable set with a custom name, and a `list` creates a measurable set with multiple sample points.

        Parameters
        ----------
        key : any
            Indexing key for accessing sample points by position.

        Returns
        -------
        sub_index : MeasurableSet | Hashable
            A `MeasurableSet` object containing the indexed sample points, or a single hashable if `key` is an `int`.

        Examples
        --------
        Define the power-set sigma-algebra on a sample space and extract a measurable set.

        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=5)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_set(indices=[0, 2, 4], name="A")

        Access the element in position 1 of the measurable.

        >>> E = A[1, "E"]
        >>> print(E)
        2

        Access elements of the measurable set by passing a slice of positions.

        >>> D = A[1:3, "D"]
        >>> print(D)  # doctest: +NORMALIZE_WHITESPACE
        Measurable set 'D':
         sample
              2
              4

        Access elements of the measurable set by passing a list of positions.

        >>> C = A[[0, 2], "C"]
        >>> print(C)  # doctest: +NORMALIZE_WHITESPACE
        Measurable set 'C':
         sample
              0
              4
        """
        if isinstance(key, tuple):
            if len(key) != 2:
                raise TypeError(
                    "Use `MeasurableSet[idx]` or `MeasurableSet[idx, name]`."
                )
            item_idx, name = key
            if not isinstance(name, Hashable):
                raise TypeError("Measurable set name must be hashable.")
        else:
            item_idx, name = key, "A"

        if not isinstance(item_idx, (int, slice, list)):
            raise TypeError("Index must be an int, slice, or list[int].")

        item = self.data[item_idx]

        if isinstance(item_idx, int):
            return item
        else:
            return self.sig_alg.get_set(item.to_list(), name)

    # --------------------- set-theoretic operations --------------------- #

    def complement(self) -> MeasurableSet:
        """Return the complement of this measurable set.

        Returns
        -------
        measurable_set : MeasurableSet
            A measurable set containing all points not in this measurable set.

        Examples
        --------
        >>> from sigalg.core import MeasurableSet, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_set([0])
        >>> print(A.complement()) # doctest: +NORMALIZE_WHITESPACE
        Measurable set 'A complement':
         sample
              1
              2
        """
        return ~self

    def intersection(self, other: MeasurableSet) -> MeasurableSet:
        """Return the intersection of this measurable set with another measurable set.

        Parameters
        ----------
        other : MeasurableSet
            Another measurable set from the same domain.

        Returns
        -------
        measurable_set : MeasurableSet
            A measurable set containing all points in both measurable sets.

        Examples
        --------
        >>> from sigalg.core import MeasurableSet, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_set([0, 1])
        >>> B = F.get_set([1, 2], name="B")
        >>> print(A.intersection(B)) # doctest: +NORMALIZE_WHITESPACE
        Measurable set 'A intersect B':
         sample
              1
        """
        return self & other

    def union(self, other: MeasurableSet) -> MeasurableSet:
        """Return the union of this measurable set with another measurable set.

        Parameters
        ----------
        other : MeasurableSet
            Another measurable set from the same domain.

        Returns
        -------
        measurable_set : MeasurableSet
            A measurable set containing all points in either measurable set.

        Examples
        --------
        >>> from sigalg.core import MeasurableSet, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_set([0])
        >>> B = F.get_set([1], name="B")
        >>> print(A.union(B)) # doctest: +NORMALIZE_WHITESPACE
        Measurable set 'A union B':
         sample
              0
              1
        """
        return self | other

    def difference(self, other: MeasurableSet) -> MeasurableSet:
        """Return the set difference of this measurable set and another measurable set.

        Parameters
        ----------
        other : MeasurableSet
            Another measurable set from the same domain.

        Returns
        -------
        measurable_set : MeasurableSet
            A measurable set containing all points in this measurable set but not in `other`.

        Examples
        --------
        >>> from sigalg.core import MeasurableSet, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_set([0, 1])
        >>> B = F.get_set([1, 2], name="B")
        >>> print(A.difference(B)) # doctest: +NORMALIZE_WHITESPACE
        Measurable set 'A difference B':
         sample
              0
        """
        return self - other

    def __invert__(self) -> MeasurableSet:
        """Return the complement of this measurable set (`~` operator).

        Returns
        -------
        measurable_set : MeasurableSet
            A measurable set containing all points not in this measurable set.
        """
        space = self.domain.data
        pts = set(self.data)
        comp = [idx for idx in space if idx not in pts]
        return self.sig_alg.get_set(comp, name=f"{self.name} complement")

    def __and__(self, other: MeasurableSet) -> MeasurableSet:
        """Return the intersection of this measurable set with another measurable set (`&` operator).

        Parameters
        ----------
        other : MeasurableSet
            Another measurable set from the same sigma-algebra.

        Raises
        ------
        ValueError
            If measurable sets are from different sigma-algebras.

        Returns
        -------
        measurable_set : MeasurableSet
            A measurable set containing all points in both measurable sets.
        """
        if self.sig_alg <= other.sig_alg:
            super_sig_alg = other.sig_alg
        elif other.sig_alg <= self.sig_alg:
            super_sig_alg = self.sig_alg
        else:
            raise ValueError("Measurable sets must belong to the same sigma-algebra.")

        pts = set(self.data) & set(other.data)
        return super_sig_alg.get_set(
            list(pts), name=f"{self.name} intersect {other.name}"
        )

    def __or__(self, other: MeasurableSet) -> MeasurableSet:
        """Return the union of this measurable set with another measurable set (`|` operator).

        Parameters
        ----------
        other : MeasurableSet
            Another measurable set from the same sigma-algebra.

        Raises
        ------
        ValueError
            If measurable sets are from different sigma-algebras.

        Returns
        -------
        measurable_set : MeasurableSet
            A measurable set containing all points in either measurable set.
        """
        if self.sig_alg <= other.sig_alg:
            super_sig_alg = other.sig_alg
        elif other.sig_alg <= self.sig_alg:
            super_sig_alg = self.sig_alg
        else:
            raise ValueError("Measurable sets must belong to the same sigma-algebra.")

        pts = set(self.data) | set(other.data)
        return super_sig_alg.get_set(list(pts), name=f"{self.name} union {other.name}")

    def __sub__(self, other: MeasurableSet) -> MeasurableSet:
        """Return the set difference of this measurable set and another measurable set (`-` operator).

        Parameters
        ----------
        other : MeasurableSet
            Another measurable set from the same sigma-algebra.

        Raises
        ------
        ValueError
            If measurable sets are from different sigma-algebras.

        Returns
        -------
        measurable_set : MeasurableSet
            A measurable set containing all points in this measurable set but not in `other`.
        """
        if self.sig_alg <= other.sig_alg:
            super_sig_alg = other.sig_alg
        elif other.sig_alg <= self.sig_alg:
            super_sig_alg = self.sig_alg
        else:
            raise ValueError("Measurable sets must belong to the same sigma-algebra.")

        pts = set(self.data) - set(other.data)
        return super_sig_alg.get_set(
            list(pts), name=f"{self.name} difference {other.name}"
        )

    # --------------------- sub/superset methods --------------------- #

    def __le__(self, other: MeasurableSet) -> bool:
        """Check if this measurable set is a subset of another measurable set (`<=` operator).

        Parameters
        ----------
        other : MeasurableSet
            Another measurable set from the same sigma-algebra.

        Raises
        ------
        ValueError
            If measurable sets are from different sigma-algebras.

        Returns
        -------
        is_le : bool
            True if this measurable set is a subset of the other measurable set.
        """
        if self.sig_alg != other.sig_alg:
            raise ValueError("Measurable sets must belong to the same sigma-algebra.")
        return set(self.data).issubset(set(other.data))

    def __lt__(self, other: MeasurableSet) -> bool:
        """Check if this measurable set is a proper subset of another measurable set (`<` operator).

        Parameters
        ----------
        other : MeasurableSet
            Another measurable set from the same sigma-algebra.

        Raises
        ------
        ValueError
            If measurable sets are from different sigma-algebras.

        Returns
        -------
        is_lt : bool
            True if this measurable set is a proper subset of the other measurable set.
        """
        if self.sig_alg != other.sig_alg:
            raise ValueError("Measurable sets must belong to the same sigma-algebra.")
        return set(self.data) < set(other.data)

    def __ge__(self, other: MeasurableSet) -> bool:
        """Check if this measurable set is a superset of another measurable set (`>=` operator).

        Parameters
        ----------
        other : MeasurableSet
            Another measurable set from the same sigma-algebra.

        Raises
        ------
        ValueError
            If measurable sets are from different sigma-algebras.

        Returns
        -------
        is_ge : bool
            True if this measurable set is a superset of the other measurable set.
        """
        if self.sig_alg != other.sig_alg:
            raise ValueError("Measurable sets must belong to the same sigma-algebra.")
        return set(self.data).issuperset(set(other.data))

    def __gt__(self, other: MeasurableSet) -> bool:
        """Check if this measurable set is a proper superset of another measurable set (`>` operator).

        Parameters
        ----------
        other : MeasurableSet
            Another measurable set from the same sigma-algebra.

        Raises
        ------
        ValueError
            If measurable sets are from different sigma-algebras.

        Returns
        -------
        is_gt : bool
            True if this measurable set is a proper superset of the other measurable set.
        """
        if self.sig_alg != other.sig_alg:
            raise ValueError("Measurable sets must belong to the same sigma-algebra.")
        return set(self.data) > set(other.data)

    # --------------------- equality --------------------- #

    def __eq__(self, other) -> bool:
        """Check equality with another measurable set.

        Two measurable sets are equal if they belong to the same sigma-algebra and
        contain the same sample points in the same order.

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the other object is a `MeasurableSet` with identical sigma-algebra
            and values, `False` otherwise.
        """
        return (
            isinstance(other, MeasurableSet)
            and self.sig_alg == other.sig_alg
            and self.data.equals(other.data)
        )

    # --------------------- conversion methods --------------------- #

    def to_domain(self) -> Domain:
        """Convert this measurable set to a domain.

        Creates a new `Domain` containing only the points in this measurable set.

        Returns
        -------
        domain : Domain
            A domain containing this measurable set's points.

        Examples
        --------
        >>> from sigalg.core import MeasurableSet, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_set([0, 1])
        >>> print(A.to_domain()) # doctest: +NORMALIZE_WHITESPACE
        Domain 'A':
         point
             0
             1
        """
        from .domain import Domain

        return Domain(indices=self.data.to_list(), name=self.name)

    def to_sample_space(self) -> SampleSpace:
        """Convert this measurable set to a sample space.

        Creates a new `SampleSpace` containing only the points in this measurable set.

        Returns
        -------
        sample_space : SampleSpace
            A sample space containing this measurable set's points.

        Examples
        --------
        >>> from sigalg.core import MeasurableSet, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_set([0, 1])
        >>> print(A.to_sample_space()) # doctest: +NORMALIZE_WHITESPACE
        Sample space 'A':
         sample
              0
              1
        """
        from .sample_space import SampleSpace

        return SampleSpace(indices=self.data.to_list(), name=self.name)

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a concise string representation of the measurable set.

        Returns
        -------
        repr_str : str
            String representation of the measurable set.
        """
        if self.data is None:
            return f"{type(self)._repr_name}(empty)"
        else:
            return f"{type(self)._repr_name}(domain={self.domain.name}, sig_alg={self.sig_alg.name}, num_points={len(self.data)}, name={self.name})"
