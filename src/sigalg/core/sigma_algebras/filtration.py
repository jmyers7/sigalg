"""A class representing a filtration of a sigma-algebra."""

from __future__ import annotations

from collections.abc import Hashable, Iterator
from typing import TYPE_CHECKING

import pandas as pd

from ...validation.filtration_validator import FiltrationLike, FiltrationValidator

if TYPE_CHECKING:
    from ...typing.index_like import IndexLike
    from ..indices.time import Time
    from ..spaces.domain import Domain
    from .sigma_algebra import SigmaAlgebra


class Filtration:
    r"""A class representing a filtration of sigma-algebras.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    sig_algs : FiltrationLike | None, default=None
        A list of sigma-algebras or a `pd.DataFrame` representing the filtration.
    time : Time | IndexLike | None
        An index for the sigma-algebras in the filtration.
    variable_names : dict[Hashable, list[Hashable]] | None, default=None
        A dictionary mapping each index to a list of variable names for the corresponding sigma-algebra. This parameter is primarily used when `sig_algs` is a `pd.DataFrame` and the variable names need to be specified for each sigma-algebra in the filtration. See the documentation for the `variable_names` property for usage examples.
    name : Hashable, default="F"
        A name for the filtration.

    Examples
    --------
    >>> import numpy as np
    >>> from sigalg.core import Domain, Filtration, SigmaAlgebra
    >>> rng = np.random.default_rng(42)
    >>> X = Domain.from_sequence(size=5)
    >>> C = SigmaAlgebra.from_rand(domain=X, num_atoms=4, name="C", random_state=rng)
    >>> B = SigmaAlgebra.from_rand(super=C, num_atoms=3, name="B", random_state=rng)
    >>> A = SigmaAlgebra.from_rand(super=B, num_atoms=2, name="A", random_state=rng)
    >>> F = Filtration(sig_algs=[A, B, C])
    >>> print(F)  # doctest: +NORMALIZE_WHITESPACE
    Filtration 'F'
    ==============
    <BLANKLINE>
    * Time 'T':
     time
        0
        1
        2
    <BLANKLINE>
    * At index 0:
    Sigma algebra 'F_0':
        atom_ID
    point
    0            0
    1            0
    2            1
    3            0
    4            1
    <BLANKLINE>
    * At index 1:
    Sigma algebra 'F_1':
        atom_ID
    point
    0            0
    1            0
    2            2
    3            1
    4            2
    <BLANKLINE>
    * At index 2:
    Sigma algebra 'F_2':
        atom_ID
    point
    0            2
    1            3
    2            0
    3            1
    4            0

    Notes
    -----
    A $\sigma$-algebra $\mathcal{F}$ on a nonempty set $X$ is called a *filtered $\sigma$-algebra* if it is equipped with a collection $\{\mathcal{F}_t\}_{t\in T}$ of $\sigma$-algebras on $X$, indexed by some linearly ordered set $T$, such that $\mathcal{F}_t \subset \mathcal{F}$ for every $t\in T$, and $\mathcal{F}_s \subset \mathcal{F}_t$ for all $s,t\in T$ with $s\leq t$. In this case, the collection $\{\mathcal{F}_t\}_{t\in T}$ is called a *filtration*.
    """

    # --------------------- constructors --------------------- #

    _properties = ["_domain", "_coarsest", "_finest"]

    def __init__(
        self,
        sig_algs: FiltrationLike | None = None,
        time: Time | IndexLike | None = None,
        variable_names: dict[Hashable, list[Hashable]] | None = None,
        name: Hashable = "F",
    ) -> None:
        from ..indices.time import Time
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if time is not None and not isinstance(time, Time):
            time = Time(time)

        v = FiltrationValidator(sig_algs=sig_algs, index=time, name=name)
        self._data = v.sig_algs
        self._index = v.index
        self._name = v.name

        if variable_names is None and self.data is not None:
            if isinstance(sig_algs, list) and all(
                isinstance(sig_alg, SigmaAlgebra) for sig_alg in sig_algs
            ):
                variable_names = {
                    time: sig_alg.variable_names
                    for time, sig_alg in zip(self.index, sig_algs)
                }
            elif isinstance(sig_algs, pd.DataFrame):
                variable_names = dict.fromkeys(self.index)
            else:
                variable_names = None

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
        Generate a filtration with three sigma-algebras.

        >>> import numpy as np
        >>> from sigalg.core import Domain, Filtration, SigmaAlgebra
        >>> rng = np.random.default_rng(42)
        >>> X = Domain.from_sequence(size=5)
        >>> C = SigmaAlgebra.from_rand(domain=X, num_atoms=4, name="C", random_state=rng)
        >>> B = SigmaAlgebra.from_rand(super=C, num_atoms=3, name="B", random_state=rng)
        >>> A = SigmaAlgebra.from_rand(super=B, num_atoms=2, name="A", random_state=rng)
        >>> F = Filtration(sig_algs=[A, B, C])

        Print the underlying data. The columns specify the atom identifiers for each sigma-algebra in the filtration.

        >>> print(F.data)  # doctest: +NORMALIZE_WHITESPACE
        time   0  1  2
        point
        0      0  0  2
        1      0  0  3
        2      1  2  0
        3      0  1  1
        4      1  2  0
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
        """Set the name of the filtration.

        Parameters
        ----------
        name : Hashable
            The new name for the filtration.

        Raises
        ------
        TypeError
            If the provided name is not hashable.
        """
        if not isinstance(name, Hashable):
            raise TypeError("name must be a hashable.")
        self._name = name

    @property
    def variable_names(self) -> dict[Hashable, list[Hashable]] | None:
        """Get the variable names for the sigma-algebras in the filtration.

        Returns
        -------
        variable_names : dict[Hashable, list[Hashable]] | None
            A dictionary mapping each index to a list of variable names for the corresponding sigma-algebra, or `None` if the variable names are not set.

        Examples
        --------
        Explicitly define a filtration from two instances of `SigmaAlgebra` with custom variable names.

        >>> import numpy as np
        >>> import pandas as pd
        >>> from sigalg.core import Domain, Filtration, SigmaAlgebra
        >>> rng = np.random.default_rng(42)
        >>> X = Domain.from_sequence(size=5)
        >>> B = SigmaAlgebra.from_rand(
        ...     domain=X,
        ...     num_atoms=4,
        ...     dim=2,
        ...     variable_names=["x", "y"],
        ...     name="B",
        ...     random_state=rng,
        ... )
        >>> A = SigmaAlgebra.from_rand(
        ...     super=B,
        ...     num_atoms=2,
        ...     variable_names=["u"],
        ...     name="A",
        ...     random_state=rng,
        ... )
        >>> F = Filtration([A, B], time=[1, 2])

        Leaving the `variable_names` parameter as `None` will automatically extract the variable names from the sigma-algebras in the filtration.

        >>> print(F.variable_names)
        {1: ['u'], 2: ['x', 'y']}

        Passing in a custom dictionary for `variable_names` to the `Filtration` constructor will override the variable names from the sigma-algebras. Note that when we index into the filtration to retrieve a specific sigma-algebra, the variable names for that sigma-algebra will reflect the custom names provided in the `variable_names` dictionary and not the original names from the sigma-algebras.

        >>> G = Filtration([A, B], time=[1, 2], name="G", variable_names={1: ["v"], 2: ["a", "b"]})
        >>> print(repr(G[2]))
        SigmaAlgebra(domain=X, num_atoms=4, variable_names=['a', 'b'], name=G_2)

        Instead of generating a filtration from a list of `SigmaAlgebra` instances, we can also create a filtration directly from a `pd.DataFrame`.

        >>> sig_algs = pd.DataFrame(
        ...     [
        ...         (0, (0, 0)),
        ...         (0, (3, 0)),
        ...         (0, (0, 0)),
        ...         (1, (2, 2)),
        ...         (1, (1, 0)),
        ...     ],
        ...     columns=[1, 2],
        ... )
        >>> H = Filtration(sig_algs, name="H")

        If the `variable_names` parameter is not provided, the variable names for each sigma-algebra in the filtration will default to `None`.

        >>> print(H.variable_names)
        {1: None, 2: None}

        However, this does not mean that the sigma-algebras themselves have no variable names. When we index into the filtration to retrieve a specific sigma-algebra, the variable names for that sigma-algebra will reflect defaults from the `SigmaAlgebra` constructor.

        >>> print(repr(H[2]))
        SigmaAlgebra(domain=X, num_atoms=4, variable_names=['atom_ID_0', 'atom_ID_1'], name=H_2)

        Finally, if we construct a filtration from a `pd.DataFrame` and provide a custom dictionary for `variable_names`, the variable names for each sigma-algebra in the filtration will reflect the custom names provided in the `variable_names` dictionary.

        >>> K = Filtration(sig_algs, name="K", variable_names={1: ["u"], 2: ["x", "y"]})
        >>> print(K.variable_names)
        {1: ['u'], 2: ['x', 'y']}
        >>> print(repr(K[2]))
        SigmaAlgebra(domain=X, num_atoms=4, variable_names=['x', 'y'], name=K_2)
        """
        return self._variable_names

    @property
    def index(self) -> Time | None:
        """Get the index of the filtration.

        Returns
        -------
        time : Time | None
            The index of the filtration, if set.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import Domain, Filtration, SigmaAlgebra
        >>> rng = np.random.default_rng(42)
        >>> X = Domain.from_sequence(size=5)
        >>> C = SigmaAlgebra.from_rand(domain=X, num_atoms=4, name="C", random_state=rng)
        >>> B = SigmaAlgebra.from_rand(super=C, num_atoms=3, name="B", random_state=rng)
        >>> A = SigmaAlgebra.from_rand(super=B, num_atoms=2, name="A", random_state=rng)
        >>> F = Filtration(sig_algs=[A, B, C], time=[1, 2, 3])
        >>> print(F.index)  # doctest: +NORMALIZE_WHITESPACE
        Time 'T':
         time
            1
            2
            3
        """
        return self._index

    @property
    def coarsest(self) -> SigmaAlgebra | None:
        """Get the coarsest sigma algebra in the filtration.

        Returns
        -------
        coarsest : SigmaAlgebra | None
            The coarsest sigma algebra in the filtration.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import Domain, Filtration, SigmaAlgebra
        >>> rng = np.random.default_rng(42)
        >>> X = Domain.from_sequence(size=5)
        >>> C = SigmaAlgebra.from_rand(domain=X, num_atoms=4, name="C", random_state=rng)
        >>> B = SigmaAlgebra.from_rand(super=C, num_atoms=3, name="B", random_state=rng)
        >>> A = SigmaAlgebra.from_rand(super=B, num_atoms=2, name="A", random_state=rng)
        >>> F = Filtration(sig_algs=[A, B, C])
        >>> print(F.coarsest)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F_0':
               atom_ID
        point
        0            0
        1            0
        2            1
        3            0
        4            1
        """
        from .sigma_algebra import SigmaAlgebra

        if self._coarsest is None and self.data is not None:
            mapping = self.data.iloc[:, 0].rename("atom_ID")
            first_mapping = self.index[0]
            self._coarsest = SigmaAlgebra(
                domain=self.domain,
                mapping=mapping,
                name=f"{self.name}_{first_mapping}",
                variable_names=self.variable_names[first_mapping],
            )

        return self._coarsest

    @property
    def finest(self) -> SigmaAlgebra | None:
        """Get the finest sigma-algebra in the filtration.

        Returns
        -------
        finest : SigmaAlgebra | None
            The finest sigma-algebra in the filtration.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import Domain, Filtration, SigmaAlgebra
        >>> rng = np.random.default_rng(42)
        >>> X = Domain.from_sequence(size=5)
        >>> C = SigmaAlgebra.from_rand(domain=X, num_atoms=4, name="C", random_state=rng)
        >>> B = SigmaAlgebra.from_rand(super=C, num_atoms=3, name="B", random_state=rng)
        >>> A = SigmaAlgebra.from_rand(super=B, num_atoms=2, name="A", random_state=rng)
        >>> F = Filtration(sig_algs=[A, B, C])
        >>> print(F.finest)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F_2':
               atom_ID
        point
        0            2
        1            3
        2            0
        3            1
        4            0
        """
        from .sigma_algebra import SigmaAlgebra

        if self._finest is None and self.data is not None:
            mapping = self.data.iloc[:, -1].rename("atom_ID")
            last_id = self.index[-1]
            self._finest = SigmaAlgebra(
                domain=self.domain,
                mapping=mapping,
                name=f"{self.name}_{last_id}",
                variable_names=self.variable_names[last_id],
            )

        return self._finest

    @property
    def domain(self) -> Domain | None:
        """Get the domain underlying the sigma-algebras in the filtration.

        Returns
        -------
        domain : Domain
            The domain underlying the sigma-algebras in the filtration.
        """
        return self._domain

    # --------------------- data access methods --------------------- #

    def __getitem__(self, index: Hashable) -> SigmaAlgebra | None:
        """Get the sigma-algebra at a specific index in the filtration.

        Parameters
        ----------
        index : Hashable
            The index at which to retrieve the sigma-algebra in the filtration.

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
        >>> import numpy as np
        >>> from sigalg.core import Domain, Filtration, SigmaAlgebra
        >>> rng = np.random.default_rng(42)
        >>> X = Domain.from_sequence(size=5)
        >>> C = SigmaAlgebra.from_rand(domain=X, num_atoms=4, name="C", random_state=rng)
        >>> B = SigmaAlgebra.from_rand(super=C, num_atoms=3, name="B", random_state=rng)
        >>> A = SigmaAlgebra.from_rand(super=B, num_atoms=2, name="A", random_state=rng)
        >>> F = Filtration(sig_algs=[A, B, C], time=[1, 2, 3])
        >>> print(F[2])  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F_2':
               atom_ID
        point
        0            0
        1            0
        2            2
        3            1
        4            2
        """
        from .sigma_algebra import SigmaAlgebra

        if index not in self.index:
            raise ValueError(
                "The provided index is not in the index of the filtration."
            )

        if self.data is not None:
            mapping = self.data[index].rename("atom_ID")
            return SigmaAlgebra(
                domain=self.domain,
                mapping=mapping,
                name=f"{self.name}_{index}",
                variable_names=self.variable_names[index],
            )
        else:
            return None

    @property
    def at(self) -> Filtration._FiltrationIndexer:
        """Get an indexer for accessing sigma-algebras at specific times.

        The difference between `F.at[time]` and `F[time]` is that `F.at[time]` will return the sigma-algebra at the nearest time in the filtration, while `F[time]` will raise a `ValueError` if the specified time is not in the index of the filtration.

        Returns
        -------
        indexer : Filtration._FiltrationIndexer
            An indexer for accessing sigma-algebras at specific times.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import Domain, Filtration, SigmaAlgebra, Time
        >>> rng = np.random.default_rng(42)
        >>> X = Domain.from_sequence(size=5)
        >>> C = SigmaAlgebra.from_rand(domain=X, num_atoms=4, name="C", random_state=rng)
        >>> B = SigmaAlgebra.from_rand(super=C, num_atoms=3, name="B", random_state=rng)
        >>> A = SigmaAlgebra.from_rand(super=B, num_atoms=2, name="A", random_state=rng)
        >>> T = Time.continuous(start=0.0, stop=1.5, num_points=3)
        >>> F = Filtration(sig_algs=[A, B, C], time=T)
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
        point
        0            0
        1            0
        2            1
        3            0
        4            1
        <BLANKLINE>
        * At index 0.75:
        Sigma algebra 'F_0.75':
            atom_ID
        point
        0            0
        1            0
        2            2
        3            1
        4            2
        <BLANKLINE>
        * At index 1.5:
        Sigma algebra 'F_1.5':
            atom_ID
        point
        0            2
        1            3
        2            0
        3            1
        4            0

        Access sigma algebra at time 0.0.

        >>> print(F.at[0.0]) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F_0.0':
                atom_ID
        point
        0             0
        1             0
        2             1
        3             0
        4             1

        Access sigma algebra at time 0.5 (returns the same as at time 0.75).

        >>> print(F.at[0.5]) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F_0.75':
                atom_ID
        point
        0             0
        1             0
        2             2
        3             1
        4             2

        Access sigma algebra at time 0.75.

        >>> print(F.at[0.75]) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F_0.75':
                atom_ID
        point
        0             0
        1             0
        2             2
        3             1
        4             2

        Access sigma algebra at time 1.2 (returns the same as at time 1.5).

        >>> print(F.at[1.2]) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F_1.5':
                atom_ID
        point
        0             2
        1             3
        2             0
        3             1
        4             0

        Access sigma algebra at time 1.5.

        >>> print(F.at[1.5]) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F_1.5':
                atom_ID
        point
        0             2
        1             3
        2             0
        3             1
        4             0
        """
        return Filtration._FiltrationIndexer(self)

    class _FiltrationIndexer:
        def __init__(self, filtration):
            self.filtration = filtration

        def __getitem__(self, time) -> SigmaAlgebra:
            return self.filtration[self.filtration.index.find_nearest_time(time)]

    # --------------------- sequence methods --------------------- #

    def __len__(self) -> int | None:
        """Get the number of sigma-algebras in the filtration.

        Returns
        -------
        length : int | None
            The length of the filtration.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import Domain, Filtration, SigmaAlgebra, Time
        >>> rng = np.random.default_rng(42)
        >>> X = Domain.from_sequence(size=5)
        >>> C = SigmaAlgebra.from_rand(domain=X, num_atoms=4, name="C", random_state=rng)
        >>> B = SigmaAlgebra.from_rand(super=C, num_atoms=3, name="B", random_state=rng)
        >>> A = SigmaAlgebra.from_rand(super=B, num_atoms=2, name="A", random_state=rng)
        >>> F = Filtration(sig_algs=[A, B, C])
        >>> print(len(F))
        3
        """
        return len(self.data.columns) if self.data is not None else None

    def __iter__(self) -> Iterator[SigmaAlgebra]:
        r"""Iterate over the sigma-algebras in the filtration.

        Returns
        -------
        iterator : Iterator[SigmaAlgebra]
            An iterator over the sigma-algebras in the filtration.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import Domain, Filtration, SigmaAlgebra, Time
        >>> rng = np.random.default_rng(42)
        >>> X = Domain.from_sequence(size=5)
        >>> C = SigmaAlgebra.from_rand(domain=X, num_atoms=4, name="C", random_state=rng)
        >>> B = SigmaAlgebra.from_rand(super=C, num_atoms=3, name="B", random_state=rng)
        >>> A = SigmaAlgebra.from_rand(super=B, num_atoms=2, name="A", random_state=rng)
        >>> F = Filtration(sig_algs=[A, B, C])
        >>> F_0, F_1, F_2 = F
        >>> print(repr(F_0))
        SigmaAlgebra(domain=X, num_atoms=2, variable_names=['atom_ID'], name=F_0)
        >>> print(repr(F_1))
        SigmaAlgebra(domain=X, num_atoms=3, variable_names=['atom_ID'], name=F_1)
        >>> print(repr(F_2))
        SigmaAlgebra(domain=X, num_atoms=4, variable_names=['atom_ID'], name=F_2)
        """
        for time in self.index:
            yield self[time]

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Get a concise string representation of the filtration.

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

        result = header + "\n" + separator + "\n\n* " + str(self.index)

        for time, sig_alg in zip(self.index, self):
            result += f"\n\n* At index {time}:\n{sig_alg}"

        return result
