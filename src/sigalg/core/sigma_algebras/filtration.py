"""A class representing a filtration of a sigma-algebra."""

from __future__ import annotations

from collections.abc import Hashable, Iterator
from typing import TYPE_CHECKING

import pandas as pd

from ...validation.filtration_validator import FiltrationLike, FiltrationValidator

if TYPE_CHECKING:
    from ..base.index import Index
    from .sigma_algebra import SigmaAlgebra


class Filtration:
    r"""A class representing a filtration of sigma-algebras.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    time : Index | None
        An index for the sigma-algebras in the filtration.
    name : Hashable | None, default="F"
        An name for the filtration.

    Raises
    ------
    TypeError
        If `name` is not a hashable or None.

    Examples
    --------
    >>> from sigalg.core import Filtration, Index, SampleSpace, SigmaAlgebra
    >>> Omega = SampleSpace.from_sequence(size=5)
    >>> A = SigmaAlgebra(
    ...     sample_space=Omega,
    ...     mapping={
    ...         0: 0,
    ...         1: 0,
    ...         2: 0,
    ...         3: 1,
    ...         4: 1,
    ...     },
    ...     name="A",
    ... )
    >>> B = SigmaAlgebra(
    ...     sample_space=Omega,
    ...     mapping={
    ...         0: 0,
    ...         1: 1,
    ...         2: 1,
    ...         3: 2,
    ...         4: 2,
    ...     },
    ...     name="B",
    ... )
    >>> C = SigmaAlgebra(
    ...     sample_space=Omega,
    ...     mapping={
    ...         0: 0,
    ...         1: 1,
    ...         2: 1,
    ...         3: 2,
    ...         4: 3,
    ...     },
    ...     name="C",
    ... )
    >>> I = Index(["coarset", "middle", "finest"])
    >>> F = Filtration(sig_algs=[A, B, C], index=I)
    >>> print(F)  # doctest: +NORMALIZE_WHITESPACE
    Filtration 'F'
    ==============
    <BLANKLINE>
    * Index 'I':
    index
    coarset
    middle
    finest
    <BLANKLINE>
    * At index coarset:
    Sigma algebra 'F_coarset':
            atom_ID
    sample
    0             0
    1             0
    2             0
    3             1
    4             1
    <BLANKLINE>
    * At index middle:
    Sigma algebra 'F_middle':
            atom_ID
    sample
    0             0
    1             1
    2             1
    3             2
    4             2
    <BLANKLINE>
    * At index finest:
    Sigma algebra 'F_finest':
            atom_ID
    sample
    0             0
    1             1
    2             1
    3             2
    4             3

    Notes
    -----
    A $\sigma$-algebra $\mathcal{F}$ on a set $\Omega$ is called a *filtered $\sigma$-algebra* if it equipped with a collection $\{\mathcal{F}_t\}_{t\in T}$ of $\sigma$-algebras on $\Omega$, indexed by some linearly ordered set $T$, such that $\mathcal{F}_t \subset \mathcal{F}$ for every $t\in T$, and $\mathcal{F}_s \subset \mathcal{F}_t$ for all $s,t\in T$ with $s\leq t$. In this case, the collection $\{\mathcal{F}_t\}_{t\in T}$ is called a *filtration*.
    """

    # --------------------- constructors --------------------- #

    _properties = ["_sample_space", "_coarsest", "_finest"]

    def __init__(
        self,
        sig_algs: FiltrationLike | None = None,
        index: Index | None = None,
        variable_names: dict | None = None,
        name: Hashable = "F",
    ) -> None:
        v = FiltrationValidator(sig_algs=sig_algs, index=index, name=name)
        self._data = v.sig_algs
        self._index = v.index
        self._name = v.name
        self._variable_names = variable_names

        self._initialize_property_caches()

    def _initialize_property_caches(self) -> None:
        for property in self._properties:
            setattr(self, property, None)

    # --------------------- properties --------------------- #

    @property
    def data(self) -> pd.DataFrame | None:
        """Get the underlying `pd.DataFrame` of the filtration.

        Returns
        -------
        data : pd.DataFrame | None
            The underlying data of the filtration, if set.

        Examples
        --------
        >>> from sigalg.core import Filtration, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=5)
        >>> A = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 0,
        ...         3: 1,
        ...         4: 1,
        ...     },
        ...     name="A",
        ... )
        >>> B = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 2,
        ...     },
        ...     name="B",
        ... )
        >>> C = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 3,
        ...     },
        ...     name="C",
        ... )
        >>> F = Filtration(sig_algs=[A, B, C])
        >>> print(F.data)  # doctest: +NORMALIZE_WHITESPACE
        index   0  1  2
        sample
        0       0  0  0
        1       0  1  1
        2       0  1  1
        3       1  2  2
        4       1  2  3
        """
        return self._data

    @property
    def name(self) -> Hashable:
        """Get the name of the filtration.

        Returns
        -------
        name : Hashable | None
            The name of the filtration.
        """
        return self._name

    @name.setter
    def name(self, name: Hashable) -> None:
        if not isinstance(name, Hashable):
            raise TypeError("name must be a hashable.")
        self._name = name

    @property
    def variable_names(self) -> dict | None:
        """Pass."""
        return self._variable_names

    @property
    def index(self) -> Index | None:
        """Get the index of the filtration.

        Returns
        -------
        index : Index | None
            The index of the filtration, if set.

        Examples
        --------
        >>> from sigalg.core import Filtration, Index, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=5)
        >>> A = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 0,
        ...         3: 1,
        ...         4: 1,
        ...     },
        ...     name="A",
        ... )
        >>> B = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 2,
        ...     },
        ...     name="B",
        ... )
        >>> C = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 3,
        ...     },
        ...     name="C",
        ... )
        >>> I = Index(["a", "b", "c"])
        >>> F = Filtration(sig_algs=[A, B, C], index=I)
        >>> print(F.index)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I':
        index
            a
            b
            c
        """
        return self._index

    @property
    def coarsest(self) -> SigmaAlgebra:
        """Get the coarsest sigma algebra in the filtration.

        Returns
        -------
        coarsest : SigmaAlgebra
            The coarsest sigma algebra in the filtration.

        Examples
        --------
        >>> from sigalg.core import Filtration, Index, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=5)
        >>> A = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 0,
        ...         3: 1,
        ...         4: 1,
        ...     },
        ...     name="A",
        ... )
        >>> B = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 2,
        ...     },
        ...     name="B",
        ... )
        >>> C = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 3,
        ...     },
        ...     name="C",
        ... )
        >>> F = Filtration(sig_algs=[A, B, C])
        >>> print(F.coarsest)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F_0':
                atom_ID
        sample
        0             0
        1             0
        2             0
        3             1
        4             1
        """
        from .sigma_algebra import SigmaAlgebra

        if self._coarsest is None and self.data is not None:
            mapping = self.data.iloc[:, 0].rename("atom_ID")
            first_mapping = self.index[0]
            self._coarsest = SigmaAlgebra(
                sample_space=self.sample_space,
                mapping=mapping,
                name=f"{self.name}_{first_mapping}",
            )

        return self._coarsest

    @property
    def finest(self) -> SigmaAlgebra:
        """Get the finest sigma-algebra in the filtration.

        Returns
        -------
        finest : SigmaAlgebra
            The finest sigma-algebra in the filtration.

        Examples
        --------
        >>> from sigalg.core import Filtration, Index, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=5)
        >>> A = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 0,
        ...         3: 1,
        ...         4: 1,
        ...     },
        ...     name="A",
        ... )
        >>> B = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 2,
        ...     },
        ...     name="B",
        ... )
        >>> C = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 3,
        ...     },
        ...     name="C",
        ... )
        >>> F = Filtration(sig_algs=[A, B, C])
        >>> print(F.finest)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F_2':
                atom_ID
        sample
        0             0
        1             1
        2             1
        3             2
        4             3
        """
        from .sigma_algebra import SigmaAlgebra

        if self._finest is None and self.data is not None:
            mapping = self.data.iloc[:, -1].rename("atom_ID")
            last_id = self.index[-1]
            self._finest = SigmaAlgebra(
                sample_space=self.sample_space,
                mapping=mapping,
                name=f"{self.name}_{last_id}",
            )

        return self._finest

    @property
    def sample_space(self):
        """Get the sample space underlying the sigma-algebras in the filtration.

        Returns
        -------
        sample_space : SampleSpace
            The sample space common to all sigma-algebras in the filtration.

        Examples
        --------
        >>> from sigalg.core import Filtration, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=5)
        >>> A = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 0,
        ...         3: 1,
        ...         4: 1,
        ...     },
        ...     name="A",
        ... )
        >>> B = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 2,
        ...     },
        ...     name="B",
        ... )
        >>> C = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 3,
        ...     },
        ...     name="C",
        ... )
        >>> F = Filtration(sig_algs=[A, B, C])
        >>> print(F.sample_space)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega':
         sample
              0
              1
              2
              3
              4
        """
        from ..base.sample_space import SampleSpace

        if self._sample_space is None and self.data is not None:
            self._sample_space = SampleSpace(indices=self.data.index)

        return self._sample_space

    # --------------------- data access methods --------------------- #

    def __getitem__(self, index: Hashable) -> SigmaAlgebra | None:
        """Get the sigma-algebra at a specific position in the filtration.

        Parameters
        ----------
        index : Hashable
            The index at which to retrive the sigma-algebra in the filtration.

        Raises
        ------
        ValueError
            If the provided index is not in the index of the filtration.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra at the specified position in the filtration, or `None` if the filtration is empty.

        Examples
        --------
        >>> from sigalg.core import Filtration, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=5)
        >>> A = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 0,
        ...         3: 1,
        ...         4: 1,
        ...     },
        ...     name="A",
        ... )
        >>> B = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 2,
        ...     },
        ...     name="B",
        ... )
        >>> C = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 3,
        ...     },
        ...     name="C",
        ... )
        >>> F = Filtration(sig_algs=[A, B, C])
        >>> F_2 = F[2]
        >>> print(F_2)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F_2':
                atom_ID
        sample
        0             0
        1             1
        2             1
        3             2
        4             3
        """
        from .sigma_algebra import SigmaAlgebra

        if index not in self.index:
            raise ValueError(
                "The provided index is not in the index of the filtration."
            )

        if self.data is not None:
            mapping = self.data[index].rename("atom_ID")
            return SigmaAlgebra(
                sample_space=self.sample_space,
                mapping=mapping,
                name=f"{self.name}_{index}",
                variable_names=self.variable_names[index]
                if self.variable_names is not None
                else None,
            )
        else:
            return None

    @property
    def at(self) -> Filtration._FiltrationIndexer:
        """Get an indexer for accessing sigma-algebras at specific times.

        Returns
        -------
        indexer : Filtration._FiltrationIndexer
            An indexer for accessing sigma-algebras at specific times.

        Examples
        --------
        >>> from sigalg.core import Filtration, SampleSpace, SigmaAlgebra, Time
        >>> Omega = SampleSpace.from_sequence(size=5)
        >>> A = SigmaAlgebra(
        ...    sample_space=Omega,
        ...    mapping={
        ...        0: 0,
        ...        1: 0,
        ...        2: 0,
        ...        3: 1,
        ...        4: 1,
        ...    },
        ...    name="A",
        ... )
        >>> B = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 2,
        ...     },
        ...     name="B",
        ... )
        >>> C = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 3,
        ...     },
        ...     name="C",
        ... )
        >>> T = Time.continuous(start=0.0, stop=1.5, num_points=3)
        >>> F = Filtration(sig_algs=[A, B, C], index=T)
        >>> print(F) # doctest: +NORMALIZE_WHITESPACE
        Filtration 'F'
        ==============
        <BLANKLINE>
        * Time 'T':
        time
        0.00
        0.75
        1.50
        <BLANKLINE>
        * At index 0.0:
        Sigma algebra 'F_0.0':
                atom_ID
        sample
        0             0
        1             0
        2             0
        3             1
        4             1
        <BLANKLINE>
        * At index 0.75:
        Sigma algebra 'F_0.75':
                atom_ID
        sample
        0             0
        1             1
        2             1
        3             2
        4             2
        <BLANKLINE>
        * At index 1.5:
        Sigma algebra 'F_1.5':
                atom_ID
        sample
        0             0
        1             1
        2             1
        3             2
        4             3
        >>> # Access sigma algebra at time 0.0
        >>> print(F.at[0.0]) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F_0.0':
                atom_ID
        sample
        0             0
        1             0
        2             0
        3             1
        4             1
        >>> # Access sigma algebra at time 0.5 (returns the same as at time 0.75)
        >>> print(F.at[0.5]) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F_0.75':
                atom_ID
        sample
        0             0
        1             1
        2             1
        3             2
        4             2
        >>> # Access sigma algebra at time 0.75
        >>> print(F.at[0.75]) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F_0.75':
                atom_ID
        sample
        0             0
        1             1
        2             1
        3             2
        4             2
        >>> # Access sigma algebra at time 1.2 (returns the same as at time 1.5)
        >>> print(F.at[1.2]) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F_1.5':
                atom_ID
        sample
        0             0
        1             1
        2             1
        3             2
        4             3
        >>> # Access sigma algebra at time 1.5
        >>> print(F.at[1.5]) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F_1.5':
                atom_ID
        sample
        0             0
        1             1
        2             1
        3             2
        4             3
        """
        from ..base.time import Time

        if not isinstance(self.index, Time):
            raise TypeError(
                "The index of the filtration must be a Time object to use the 'at' property."
            )

        return Filtration._FiltrationIndexer(self)

    class _FiltrationIndexer:
        def __init__(self, filtration):
            self.filtration = filtration

        def __getitem__(self, time) -> SigmaAlgebra:
            return self.filtration[self.filtration.index.find_nearest_time(time)]

    # --------------------- sequence methods --------------------- #

    def __len__(self) -> int:
        """Get the number of sigma-algebras in the filtration.

        Returns
        -------
        length : int
            The length of the filtration.

        Examples
        --------
        >>> from sigalg.core import Filtration, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=5)
        >>> A = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 0,
        ...         3: 1,
        ...         4: 1,
        ...     },
        ...     name="A",
        ... )
        >>> B = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 2,
        ...     },
        ...     name="B",
        ... )
        >>> C = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 3,
        ...     },
        ...     name="C",
        ... )
        >>> F = Filtration(sig_algs=[A, B, C])
        >>> print(len(F))
        3
        """
        return len(self.data.columns)

    def __iter__(self) -> Iterator[(Hashable, SigmaAlgebra)]:
        r"""Iterate over the indices and sigma-algebras in the filtration.

        Returns
        -------
        iterator : Iterator[(Hashble, SigmaAlgebra)]
            An iterator over the indices and sigma-algebras in the filtration.

        Examples
        --------
        >>> from sigalg.core import Filtration, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=5)
        >>> A = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 0,
        ...         3: 1,
        ...         4: 1,
        ...     },
        ...     name="A",
        ... )
        >>> B = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 2,
        ...     },
        ...     name="B",
        ... )
        >>> C = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...         4: 3,
        ...     },
        ...     name="C",
        ... )
        >>> F = Filtration(sig_algs=[A, B, C])
        >>> for idx, sig_alg in F:
        ...     print(f"Index: {idx}\n{sig_alg}")  # doctest: +NORMALIZE_WHITESPACE
        Index: 0
        Sigma algebra 'F_0':
                atom_ID
        sample
        0             0
        1             0
        2             0
        3             1
        4             1
        Index: 1
        Sigma algebra 'F_1':
                atom_ID
        sample
        0             0
        1             1
        2             1
        3             2
        4             2
        Index: 2
        Sigma algebra 'F_2':
                atom_ID
        sample
        0             0
        1             1
        2             1
        3             2
        4             3
        """
        from .sigma_algebra import SigmaAlgebra

        if self.data is not None:
            for idx, col in self.data.items():
                yield (
                    idx,
                    SigmaAlgebra(
                        sample_space=self.sample_space,
                        mapping=col.rename("atom_ID"),
                        name=f"{self.name}_{idx}",
                    ),
                )
        else:
            yield None

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Get the string representation of the filtration.

        Returns
        -------
        representation : str
            The string representation of the filtration.
        """
        return f"Filtration(name='{self._name}', length={len(self)})"

    def __str__(self) -> str:
        """Get a detailed string representation of the filtration.

        Returns
        -------
        detailed_representation : str
            A detailed string representation of the filtration.
        """
        header = f"Filtration '{self.name}'"
        separator = "=" * len(header)

        result = header + "\n" + separator + "\n\n* " + repr(self.index)

        for time, sig_alg in self:
            result += f"\n\n* At index {time}:\n{sig_alg}"

        return result
