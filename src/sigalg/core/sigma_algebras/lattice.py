"""A class representing a lattice of sigma-algebras."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pandas as pd

    from ...typing.index_like import IndexLike
    from ..functions.function import Function
    from ..spaces.set import Set
    from .sigma_algebra import SigmaAlgebra

    MeasurableObject = Set | Function


class NonMeasurableError(Exception):  # noqa: D101
    pass


# TODO: finish the docstring mathematical section
class Lattice:
    r"""A class representing a lattice of sigma-algebras.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    base : SigmaAlgebra
        The base sigma-algebra of the lattice.
    type : Literal["upward", "downward"]
        The type of the lattice, indicating whether it is an upward or downward lattice.

    Notes
    -----
    Mathematically, a *lattice* is a partially-ordered set in which every pair of elements has a unique supremum (least upper bound, or *join*) and an infimum (greatest lower bound, or *meet*).

    An example is the set of all $\sigma$-algebras on a fixed nonempty set $X$, which we assume to be finite for simplicity.

    1. The *partial order* is given by set inclusion, i.e., for two $\sigma$-algebras $\mathcal{F}$ and $\mathcal{G}$ on $X$, we say $\mathcal{F} \leq \mathcal{G}$ if and only if $\mathcal{F}$ is a subset of $\mathcal{G}$.

    2. The *join* of two $\sigma$-algebras $\mathcal{F}$ and $\mathcal{G}$, denoted by $\mathcal{F} \vee \mathcal{G}$, is the $\sigma$-algebra whose atoms are the nonempty intersections of the atoms of $\mathcal{F}$ and $\mathcal{G}$.

    3. The *meet* of two $\sigma$-algebras $\mathcal{F}$ and $\mathcal{G}$, denoted by $\mathcal{F} \wedge \mathcal{G}$, is the $\sigma$-algebra whose atoms are ...
    """

    # --------------------- constructors --------------------- #

    def __init__(self, base: SigmaAlgebra, type: Literal["upward", "downward"]) -> None:
        import pandas as pd

        from .sigma_algebra import SigmaAlgebra

        if not isinstance(base, SigmaAlgebra):
            raise TypeError("The base object must be a SigmaAlgebra.")
        if type not in ["upward", "downward"]:
            raise ValueError("type must either be 'upward' or 'downward'")

        self._items = []
        self._ruled_out = []
        self.base = base
        self.type = type

        self_base_data = self.base.data.copy()
        if self.base.dimension > 1:
            new_index = pd.MultiIndex.from_frame(
                self.base.data, names=self.base.variable_names
            )
        else:
            new_index = pd.Index(self.base.data, name=self.base.variable_names[0])
        self_base_data.index = new_index

        self[self.base] = self_base_data

    # --------------------- cache and data access methods --------------------- #

    def add(self, sig_alg: SigmaAlgebra) -> None:
        """Add a sigma-algebra to the lattice, if possible.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The sigma-algebra to add.

        Raises
        ------
        ValueError
            If `sig_alg` is not a `SigmaAlgebra` with the same domain as the base sigma-algebra.
        NonMeasurableError
            If the sigma-algebra is not contained in the lattice.

        Examples
        --------
        >>> from sigalg.core import Domain, SigmaAlgebra

        Define three sigma-algebras on a domain.

        >>> X = Domain.from_sequence(size=5)
        >>> F = SigmaAlgebra(domain=X, mapping=dict(zip(X, [0, 1, 1, 2, 2])))
        >>> H = SigmaAlgebra(
        ...     domain=X, mapping=dict(zip(X, [0, 1, 1, 2, 3])), name="H")
        >>> G = SigmaAlgebra(
        ...     domain=X, mapping=dict(zip(X, [0, 1, 1, 1, 1])), name="G")

        The sigma-algebra `G` is a sub-sigma-algebra of `F`, so it may be added to the downward lattice of the latter.

        >>> F.down_lattice.add(G)
        >>> G in F.down_lattice
        True

        However, the sigma-algebra `H` is not a sub-sigma-algebra of `F`, so attempting to add it raises an exception.

        >>> F.down_lattice.add(H)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        NonMeasurableError: The candidate set is not measurable.
        >>> H in F.down_lattice
        False

        But `H` is a super-sigma-algebra of `F`, so it may be added to the upward lattice of the latter.

        >>> F.up_lattice.add(H)
        >>> H in F.up_lattice
        True
        """
        from .sigma_algebra import SigmaAlgebra

        if not isinstance(sig_alg, SigmaAlgebra) or sig_alg.domain != self.base.domain:
            raise ValueError(
                "You may only add a SigmaAlgebra with the same domain as the base sigma-algebra."
            )

        if any(key is sig_alg for key, _ in self.items()):
            return

        if sig_alg.is_canonical_power_set and self.type == "upward":
            self[sig_alg] = self.base.data
            sig_alg.down_lattice[self.base] = self.base.data
            return

        if any(key is sig_alg for key in self._ruled_out):
            sub_or_super = "sub" if self.type == "downward" else "super"
            raise NonMeasurableError(
                f"The given sigma-algebra is not a {sub_or_super}-algebra of the base sigma-algebra."
            )

        self._add_with_no_checks(sig_alg)

    def get_atom_data(self, sig_alg: SigmaAlgebra) -> pd.Series | pd.DataFrame:
        """Get the atom data of the base sigma-algebra relative to given a sigma-algebra.

        If the lattice is`type='upward'`, then `sig_alg` is a super-sigma-algebra of the base sigma-algebra. In this case, every atom of `sig_alg` is contained in a unique atom of the base sigma-algebra. The pandas structure returned by this method encodes this mapping. If the lattice is `type='downward'`, then everything is reversed. See the Examples below.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The sigma-algebra relative to which the atom data is requested.

        Returns
        -------
        data : pd.DataFrame
            The atom data of the base sigma-algebra relative to the given sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import Domain, SigmaAlgebra

        Define three sigma-algebras on a domain.

        >>> X = Domain.from_sequence(size=5)
        >>> F = SigmaAlgebra(domain=X, mapping=dict(zip(X, [0, 1, 1, 2, 2])))
        >>> H = SigmaAlgebra(
        ...     domain=X, mapping=dict(zip(X, [0, 1, 1, 2, 3])), name="H")
        >>> G = SigmaAlgebra(
        ...     domain=X, mapping=dict(zip(X, [0, 1, 1, 1, 1])), name="G")

        The sigma-algebra `G` is a sub-sigma-algebra of `F`, so it may be added to the downward lattice of the latter.

        >>> F.down_lattice.add(G)

        Every atom of `F` is contained in a unique atom of `G` according to the following mapping on identifiers: `0->0`, `1->1`, and `2->1`. This mapping is encoded in the atom data returned by the `get_atom_data` method.

        >>> F_to_G_atom_data = F.down_lattice.get_atom_data(G)
        >>> print(F_to_G_atom_data)  # doctest: +NORMALIZE_WHITESPACE
        F
        0    0
        1    1
        2    1
        Name: G, dtype: int64

        Symmetrically, the same atom data may be obtained through the upward lattice of `G`. Note that we do not need to add `F` to this lattice — the call to `add` above already did this.

        >>> same_atom_data = G.up_lattice.get_atom_data(F)
        >>> print(same_atom_data)  # doctest: +NORMALIZE_WHITESPACE
        F
        0    0
        1    1
        2    1
        Name: G, dtype: int64

        The sigma-algebra `H` is a super-sigma-algebra of `F`, so it may be added to the upward lattice of the latter.

        >>> F.up_lattice.add(H)

        There is now a mapping from the atom IDs of `H` to those of `F`.

        >>> H_to_F_atom_data = F.up_lattice.get_atom_data(H)
        >>> print(H_to_F_atom_data)  # doctest: +NORMALIZE_WHITESPACE
        H
        0    0
        1    1
        2    2
        3    2
        Name: F, dtype: int64
        """
        return self[sig_alg]

    def items(self) -> list[SigmaAlgebra, pd.Series | pd.DataFrame]:
        """Return a list of tuples `(sig_alg, data)` of a sigma-algebra and the atom data relative to the base sigma-algebra.

        Returns
        -------
        items : list[SigmaAlgebra, pd.Series | pd.DataFrame]
            The above mentioned list.
        """
        return list(self._items)

    # --------------------- internal methods --------------------- #

    def _add_with_no_checks(
        self, sig_alg: SigmaAlgebra, return_data: bool = False
    ) -> None:
        import pandas as pd

        from .._utils import to_df

        if self.type == "upward":
            sub_alg = self.base
            super_alg = sig_alg
        else:
            sub_alg = sig_alg
            super_alg = self.base

        sub_data = to_df(sub_alg.data, "_sub")
        super_data = to_df(super_alg.data, "_super")
        test_data = pd.concat([sub_data, super_data], axis=1).drop_duplicates()

        if len(test_data) != super_alg.num_atoms:
            self._rule_out(sig_alg)

            if self.type == "upward":
                sig_alg.down_lattice._rule_out(self.base)
            else:
                sig_alg.up_lattice._rule_out(self.base)

            sub_or_super = "sub" if self.type == "downward" else "super"
            raise NonMeasurableError(
                f"The given sigma-algebra is not a {sub_or_super}-algebra of the base sigma-algebra."
            )

        data = test_data.set_index(list(super_data.columns)).squeeze(axis=1)

        if isinstance(data, pd.DataFrame):
            data.columns = sub_alg.data.columns
        else:
            data.name = sub_alg.name

        if isinstance(data.index, pd.MultiIndex):
            data.index.names = super_alg.variable_names
        else:
            data.index.name = super_alg.variable_names[0]

        self[sig_alg] = data

        if self.type == "upward":
            sig_alg.down_lattice[sub_alg] = data
        else:
            sig_alg.up_lattice[super_alg] = data

        return data if return_data else None

    def _rule_out(self, sig_alg: SigmaAlgebra) -> None:
        """Mark a sigma-algebra as ruled out, if not already."""
        for existing in self._ruled_out:
            if existing is sig_alg:
                return
        self._ruled_out.append(sig_alg)

    def __getitem__(self, sig_alg: SigmaAlgebra) -> pd.DataFrame:
        """Get the atom data of the base sigma-algebra relative to a given sigma-algebra."""
        import pandas as pd

        from .sigma_algebra import SigmaAlgebra

        if not isinstance(sig_alg, SigmaAlgebra) or sig_alg.domain != self.base.domain:
            raise ValueError(
                "sig_alg must be an instance of SigmaAlgebra with the same domain as the base of the lattice."
            )

        for key, data in self._items:
            if key is sig_alg:
                return data

        if sig_alg.is_power_set and self.type == "upward":
            ordered_sig_alg_data = sig_alg.data.reindex(self.base.data.index)

            if isinstance(ordered_sig_alg_data, pd.DataFrame):
                new_index = pd.MultiIndex.from_frame(
                    ordered_sig_alg_data, names=sig_alg.variable_names
                )
            else:
                new_index = pd.Index(
                    ordered_sig_alg_data, name=sig_alg.variable_names[0]
                )

            data = self.base.data.copy()
            data.index = new_index
            self[sig_alg] = data
            sig_alg.down_lattice[self.base] = data
            return data

        if any(key is sig_alg for key in self._ruled_out):
            sub_or_super = "sub" if self.type == "downward" else "super"
            raise NonMeasurableError(
                f"The given sigma-algebra is not a {sub_or_super}-algebra of the base sigma-algebra."
            )

        return self._add_with_no_checks(sig_alg, return_data=True)

    def __setitem__(self, sig_alg: SigmaAlgebra, data: pd.DataFrame) -> None:
        """Set the atom data of the base sigma-algebra relative to a given sigma-algebra."""
        for i, (key, _) in enumerate(self._items):
            if key is sig_alg:
                self._items[i] = (sig_alg, data)
                return
        self._items.append((sig_alg, data))

    # --------------------- lattice methods --------------------- #

    def __contains__(self, sig_alg: SigmaAlgebra) -> bool:
        """Test whether a given sigma-algebra is contained in the lattice.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The sigma-algebra to test membership in the lattice..

        Returns
        -------
        is_measurable : bool
            `True` if the sigma-algebra is in the lattice, `False` otherwise.
        """
        from .sigma_algebra import SigmaAlgebra

        if not isinstance(sig_alg, SigmaAlgebra) or sig_alg.domain != self.base.domain:
            raise ValueError(
                "sig_alg must be an instance of SigmaAlgebra with the same domain as the base of the lattice."
            )

        if sig_alg is self.base or any(key is sig_alg for key, _ in self.items()):
            return True

        if sig_alg.is_canonical_power_set and self.type == "upward":
            self[sig_alg] = self.base.data
            sig_alg.down_lattice[self.base] = self.base.data
            return True

        if any(key is sig_alg for key in self._ruled_out):
            return False

        try:
            self._add_with_no_checks(sig_alg)
        except NonMeasurableError:
            return False

        return True

    @staticmethod
    def is_subalgebra(sub_algebra: SigmaAlgebra, super_algebra: SigmaAlgebra) -> bool:
        """Test whether a sigma-algebra is a sub-sigma-algebra of another.

        Parameters
        ----------
        sub_algebra : SigmaAlgebra
            The sigma-algebra who will be tested in the sub-sigma-algebra position.
        super_algebra : SigmaAlgebra
            The sigma-algebra who will be test in the sub-sigma-algebra position.

        Raises
        ------
        ValueError
            If the sigma-algebras are not instances of `SigmaAlgebra` with the same domain.

        Returns
        -------
        is_sub_alg : bool
            `True` if `sub_algebra` is a sub-sigma-algebra of `super_algebra`; `False` otherwise.

        Examples
        --------
        >>> from sigalg.core import Domain, Lattice, SigmaAlgebra

        Define three sigma-algebras on a domain.

        >>> X = Domain.from_sequence(size=5)
        >>> F = SigmaAlgebra(domain=X, mapping=dict(zip(X, [0, 1, 1, 2, 2])))
        >>> H = SigmaAlgebra(
        ...     domain=X, mapping=dict(zip(X, [0, 1, 1, 2, 3])), name="H")
        >>> G = SigmaAlgebra(
        ...     domain=X, mapping=dict(zip(X, [0, 1, 1, 1, 1])), name="G")

        The sigma-algebras sit in a chain `G <= F <= H`. Test this.

        >>> Lattice.is_subalgebra(G, G)
        True
        >>> Lattice.is_subalgebra(G, F)
        True
        >>> Lattice.is_subalgebra(G, H)
        True
        >>> Lattice.is_subalgebra(F, G)
        False
        >>> Lattice.is_subalgebra(F, F)
        True
        >>> Lattice.is_subalgebra(F, H)
        True
        >>> Lattice.is_subalgebra(H, G)
        False
        >>> Lattice.is_subalgebra(H, F)
        False
        >>> Lattice.is_subalgebra(H, H)
        True
        """
        from .._utils import pandas_all_equal
        from .sigma_algebra import SigmaAlgebra

        if (
            not isinstance(sub_algebra, SigmaAlgebra)
            or not isinstance(super_algebra, SigmaAlgebra)
            or (sub_algebra.domain != super_algebra.domain)
        ):
            raise ValueError("Both sigma-algebras must have the same domain")

        if (
            pandas_all_equal(sub_algebra.data, super_algebra.data)
            or super_algebra.is_power_set
        ):
            return True

        return sub_algebra in super_algebra.down_lattice

    @staticmethod
    def join(
        sigma_algebras: list[SigmaAlgebra],
        variable_names: list[Hashable] | None = None,
        index: IndexLike | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
    ) -> SigmaAlgebra:
        """Compute the join (least upper bound) of a list of sigma-algebras on the same domain.

        Parameters
        ----------
        sigma_algebras : list[SigmaAlgebra]
            A list of sigma-algebras instances to join.
        variable_names : list[Hashable] | None, default=None
            The list of variable names for the join. If `None`, defaults will be generated.
        name : Hashable | None, default=None
            Name identifier for the resulting sigma algebra. If `None`, a default will be generated.

        Raises
        ------
        ValueError
            If `sigma_algebras` is not a nonempty list of sigma-algebras on the same domain.

        Examples
        --------
        >>> from sigalg.core import Domain, Lattice, SigmaAlgebra
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: (0, 1),
        ...         1: (3, 4),
        ...         2: (6, 7),
        ...     },
        ... )
        >>> G = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 2,
        ...         1: 5,
        ...         2: 8,
        ...     },
        ...     name="G",
        ... )
        >>> join = Lattice.join([F, G])
        >>> print(join)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F v G':
        i  0  1  2
        x
        0  0  1  2
        1  3  4  5
        2  6  7  8
        >>> F <= join
        True
        >>> G <= join
        True
        >>> print(join.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'F v G':
         F_0  F_1  G
           0    1  2
           3    4  5
           6    7  8
        """
        import pandas as pd

        from ...validation.domain_index_validator import DomainIndexValidator
        from .._utils import subscript_var_names, to_df
        from ..indices.index import Index
        from ..indices.time import Time
        from .sigma_algebra import SigmaAlgebra

        if name is not None and not isinstance(name, Hashable):
            raise TypeError("If given, name must be hashable.")
        if variable_names is not None:
            if not isinstance(variable_names, list) or not all(
                isinstance(name, Hashable) for name in variable_names
            ):
                raise ValueError(
                    "If given, variable_names must be a list of hashables."
                )
            if len(variable_names) != sum(
                len(sig_alg.variable_names) for sig_alg in sigma_algebras
            ):
                raise ValueError(
                    "If given, the number of variable_names must equal the total number of variable names of the sigma-algebras."
                )
        if (
            not isinstance(sigma_algebras, list)
            or not all(isinstance(alg, SigmaAlgebra) for alg in sigma_algebras)
            or len(sigma_algebras) == 0
        ):
            raise TypeError(
                "sigma_algebras must be a list of instances of SigmaAlgebra."
            )

        if len(sigma_algebras) == 1:
            return sigma_algebras[0]

        domain = sigma_algebras[0].domain
        if not all(sig_alg.domain == domain for sig_alg in sigma_algebras):
            raise ValueError("All sigma-algebras must have the same domain.")

        u = DomainIndexValidator(
            index=index,
            index_kind=index_kind,
            index_name=index_name,
        )

        index = u.index
        index_kind = u.index_kind
        index_name = index_name

        data = pd.concat(
            [to_df(sig_alg.data, f"_{k}") for k, sig_alg in enumerate(sigma_algebras)],
            axis=1,
        )

        if variable_names is None:
            variable_names = subscript_var_names(
                [sig_alg.variable_names for sig_alg in sigma_algebras]
            )
        if index is None:
            index_class = Index if index_kind == "Index" else Time
            index_data = pd.RangeIndex(
                data.shape[1], name=index_class._variable_names_prefix
            )
            index = index_class._from_validated(data=index_data, name=index_name)

        data.columns = index.data

        if name is None:
            name = " v ".join([sig_alg.name for sig_alg in sigma_algebras])

        return SigmaAlgebra._from_validated(
            data=data,
            variable_names=variable_names,
            name=name,
            domain_kind=type(domain).__name__,
            domain_name=domain.name,
            index_kind=index_kind,
            index_name=index_name,
        )

    @staticmethod
    def meet(
        sigma_algebras: list[SigmaAlgebra],
        variable_name: Hashable | None = None,
        name: Hashable | None = None,
    ) -> SigmaAlgebra:
        """Compute the meet (greatest lower bound) of a list of sigma-algebras on the same domain.

        Parameters
        ----------
        sigma_algebras : list[SigmaAlgebra]
            A list of sigma-algebras instances to meet.
        name : Hashable | None, default=None
            Name identifier for the resulting sigma algebra. If `None`, a default will be generated.

        Raises
        ------
        ValueError
            If `sigma_algebras` is not a nonempty list of sigma-algebras on the same domain.

        Examples
        --------
        >>> from sigalg.core import Domain, Lattice, SigmaAlgebra
        >>> X = Domain.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...         4: 3,
        ...         5: 4,
        ...     },
        ... )
        >>> G = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 3,
        ...         5: 4,
        ...     },
        ...     name="G",
        ... )
        >>> H = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...         4: 3,
        ...         5: 4,
        ...     },
        ...     name="H",
        ... )
        >>> meet = Lattice.meet([F, G, H])
        >>> print(meet)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F ^ G ^ H':
           F ^ G ^ H
        x
        0          0
        1          0
        2          0
        3          0
        4          1
        5          2
        >>> meet <= F
        True
        >>> meet <= G
        True
        >>> meet <= H
        True
        """
        import numpy as np
        import pandas as pd
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        from .._utils import add_subscript, to_df
        from .sigma_algebra import SigmaAlgebra

        if name is not None and not isinstance(name, Hashable):
            raise TypeError("If given, name must be hashable.")

        if (
            not isinstance(sigma_algebras, list)
            or not all(isinstance(alg, SigmaAlgebra) for alg in sigma_algebras)
            or len(sigma_algebras) == 0
        ):
            raise TypeError(
                "sigma_algebras must be a list of instances of SigmaAlgebra."
            )

        if len(sigma_algebras) == 1:
            return sigma_algebras[0]

        domain = sigma_algebras[0].domain
        if not all(sig_alg.domain == domain for sig_alg in sigma_algebras):
            raise ValueError("All sigma-algebras must have the same domain.")

        left_domain_names = add_subscript(domain.variable_names, "0")
        right_domain_names = add_subscript(domain.variable_names, "1")

        equiv_relations = []

        for sig_alg in sigma_algebras:
            equiv_relation = pd.merge(
                left=to_df(sig_alg.data).reset_index(names=left_domain_names),
                right=to_df(sig_alg.data).reset_index(names=right_domain_names),
            )[left_domain_names + right_domain_names]

            equiv_relations.append(equiv_relation)

        union_relation = pd.concat(equiv_relations).drop_duplicates()

        idx = {x: i for i, x in enumerate(domain)}

        if domain.dimension > 1:
            rows = pd.MultiIndex.from_frame(union_relation[left_domain_names]).map(idx)
            cols = pd.MultiIndex.from_frame(union_relation[right_domain_names]).map(idx)

        else:
            rows = union_relation[left_domain_names].squeeze(axis=1).map(idx)
            cols = union_relation[right_domain_names].squeeze(axis=1).map(idx)

        n = len(domain)
        adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        _, atom_ids = connected_components(adj, directed=False)

        if name is None:
            name = " ^ ".join([sig_alg.name for sig_alg in sigma_algebras])
        if variable_name is None:
            variable_name = name

        data = pd.Series(atom_ids, index=domain.data, name=variable_name)

        return SigmaAlgebra._from_validated(
            data=data,
            variable_names=[variable_name],
            name=name,
            domain_kind=type(domain).__name__,
            domain_name=domain.name,
            index_kind="Index",
            index_name=None,
        )

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a concise string representation of the lattice.

        Returns
        -------
        repr_str : str
            A concise string representation of the lattice.
        """
        return (
            "Lattice("
            + f"base={self.base.name}, "
            + f"type={self.type}, "
            + f"num_sig_algs={len(self.items())})"
        )
