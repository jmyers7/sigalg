"""Class for lattice operations on sigma-algebras."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .sigma_algebra import SigmaAlgebra


class Lattice:
    """Class containing lattice operations on sigma-algebras.

    The class does not have an `__init__` method, and all methods are class methods.
    """

    @classmethod
    def is_subalgebra(
        cls, sub_algebra: SigmaAlgebra, super_algebra: SigmaAlgebra
    ) -> bool:
        """Check if one sigma-algebra is a subalgebra of another.

        Parameters
        ----------
        sub_algebra : SigmaAlgebra
            The candidate subalgebra.
        super_algebra : SigmaAlgebra
            The candidate superalgebra.

        Raises
        ------
        TypeError
            If either `sub_algebra` or `super_algebra` is not a `SigmaAlgebra` instance.
        ValueError
            If `sub_algebra` and `super_algebra` do not have the same domain.

        Returns
        -------
        is_subalgebra : bool
            True if `sub_algebra` is a subalgebra of `super_algebra`, False otherwise.

        Examples
        --------
        >>> from sigalg.core import Lattice, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...     },
        ... )
        >>> G = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ...     name="G",
        ... )
        >>> print(Lattice.is_subalgebra(F, G))
        True
        """
        from .sigma_algebra import SigmaAlgebra

        if not isinstance(sub_algebra, SigmaAlgebra) or not isinstance(
            super_algebra, SigmaAlgebra
        ):
            raise TypeError("Both arguments must be SigmaAlgebra instances")
        if sub_algebra.domain != super_algebra.domain:
            raise ValueError("Both sigma-algebras must have the same domain")

        if super_algebra.is_power_set:
            return True

        sub_df = sub_algebra.data.to_frame(name="sub")
        super_df = super_algebra.data.to_frame(name="super")
        df = pd.concat([sub_df, super_df], axis=1)

        return bool(df.groupby("super")["sub"].nunique().max() == 1)

    @classmethod
    def join(
        cls, sigma_algebras: list[SigmaAlgebra], name: Hashable | None = "join"
    ) -> SigmaAlgebra:
        r"""Compute the join (least upper bound) of a list of sigma-algebras.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        sigma_algebras : list[SigmaAlgebra]
            A list of sigma-algebras instances to join.
        name : Hashable | None, default="join"
            Name identifier for the resulting sigma algebra.

        Raises
        ------
        TypeError
            If the input is not a list of sigma-algebras.
        ValueError
            If the list is empty or if the sigma-algebras do not share the same domain.

        Examples
        --------
        >>> from sigalg.core import Lattice, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
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
        ...     domain=Omega,
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
        >>> join = Lattice.join([F, G])
        >>> print(join)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'join':
               atom_ID
        sample
        0       (0, 0)
        1       (0, 1)
        2       (0, 1)
        3       (1, 1)
        4       (1, 0)
        5       (1, 0)

        Notes
        -----
        Let $\{\mathcal{F}_i\}_{k\in K}$ be a finite collection of $\sigma$-algebras on a finite set $\Omega$. The *join* (or *least upper bound*) of the collection, denoted $\bigvee_{k\in K} \mathcal{F}_k$, is the coarsest $\sigma$-algebra that contains all of the $\mathcal{F}_k$. Its atoms are given by the nonempty intersections of atoms from each $\mathcal{F}_k$. In particular, the atom identifiers for the join can be represented as tuples of the atom identifiers from each $\mathcal{F}_k.
        """
        from .sigma_algebra import SigmaAlgebra

        if name is not None and not isinstance(name, Hashable):
            raise TypeError("name must be a Hashable or None")
        if not isinstance(sigma_algebras, list):
            raise TypeError("Expected a list of sigma-algebras")
        if not all(isinstance(alg, SigmaAlgebra) for alg in sigma_algebras):
            raise TypeError("All elements of the list must be a SigmaAlgebra")
        if len(sigma_algebras) == 0:
            raise ValueError(
                "The join of an empty list of sigma-algebras is the trivial algebra on the domain"
            )
        if len(sigma_algebras) == 1:
            return sigma_algebras[0]
        domain = sigma_algebras[0].domain
        if not all(alg._domain == domain for alg in sigma_algebras):
            raise ValueError("All sigma-algebras must have the same domain")

        for alg in sigma_algebras:
            alg.data.rename(alg.name, inplace=True)
        df = pd.concat([alg.data for alg in sigma_algebras], axis=1)

        sample_id_to_atom_id = df.apply(lambda row: tuple(row), axis=1).to_dict()

        return SigmaAlgebra(domain=domain, mapping=sample_id_to_atom_id, name=name)
