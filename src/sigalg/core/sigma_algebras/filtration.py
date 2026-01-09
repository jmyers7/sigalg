"""Filtration module.

Provides the `Filtration` class representing a filtration of sigma algebras.

Classes
-------
Filtration
    Class representing a filtration of sigma algebras.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..base.index import Index
    from .sigma_algebra import SigmaAlgebra


class Filtration:
    """A class representing a nested sequence of sigma algebras.

    A filtration is an increasing sequence of sigma algebras defined on the same sample space.

    Parameters
    ----------
    time : Index | None
        An index for the time points corresponding to each sigma algebra in the filtration. Does not have to be an instance of `Time`; may be an instance of the parent class `Index`.
    name : Hashable | None, default="Ft"
        An optional name for the filtration.

    Raises
    ------
    TypeError
        If `name` is not a hashable or None.

    Examples
    --------
    >>> from sigalg.core import Filtration, SampleSpace, SigmaAlgebra, Time
    >>> # Define sample space and sigma algebras
    >>> sample_space = SampleSpace.generate_sequence(size=3)
    >>> F = SigmaAlgebra.trivial(sample_space=sample_space, name="F")
    >>> G = SigmaAlgebra(sample_space=sample_space, name="G").from_dict(
    ...     sample_id_to_atom_id={"omega_0": 0, "omega_1": 0, "omega_2": 1},
    ... )
    >>> H = SigmaAlgebra.power_set(sample_space=sample_space, name="H")
    >>> # Define continous time index
    >>> time = Time.continuous(start=0.0, stop=1.5, num_points=3)
    >>> # Create and print filtration
    >>> Ft = Filtration(time=time, name="Ft").from_list([F, G, H])
    >>> print(Ft) # doctest: +NORMALIZE_WHITESPACE
    Filtration (Ft)
    ===============
    <BLANKLINE>
    * Time 'T':
    [0.0, 0.75, 1.5]
    <BLANKLINE>
    * At time 0.0:
    Sigma algebra 'F':
            atom ID
    sample
    omega_0        0
    omega_1        0
    omega_2        0
    <BLANKLINE>
    * At time 0.75:
    Sigma algebra 'G':
            atom ID
    sample
    omega_0        0
    omega_1        0
    omega_2        1
    <BLANKLINE>
    * At time 1.5:
    Sigma algebra 'H':
            atom ID
    sample
    omega_0        0
    omega_1        1
    omega_2        2
    """

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        time: Index | None = None,
        name: Hashable | None = "Ft",
    ) -> None:
        self._validate_parameters(time=time)
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("name must be a hashable or None.")

        self._time = time
        self._name = name

        # caches for properties
        self._sigma_algebras: list[SigmaAlgebra] | None = None
        self._data: pd.DataFrame | None = None
        self._time_to_pos: dict | None = None

    def from_list(self, sigma_algebras: list[SigmaAlgebra]) -> Filtration:
        """Initialize the filtration from a list of sigma algebras.

        If the `time` parameter was not provided at initialization, it will be set to a discrete time index of the same length as the provided list of sigma algebras.

        Parameters
        ----------
        sigma_algebras : list[SigmaAlgebra]
            A list of sigma algebras that form a filtration. The order of the list determines the order of the filtration (i.e., the first element is the coarsest sigma algebra and the last element is the finest sigma algebra).

        Returns
        -------
        filtration : Filtration
            The filtration initialized from the provided list of sigma algebras.
        """
        from ..base.time import Time

        self._validate_parameters(sigma_algebras=sigma_algebras, time=self.time)

        if self._time is None:
            self._time = Time().discrete(length=len(sigma_algebras))

        self._sigma_algebras = sigma_algebras
        return self

    def from_pandas(self, data: pd.DataFrame) -> Filtration:
        """Initialize the filtration from a `pd.DataFrame`.

        The columns of the DataFrame represent the atom IDs of the sigma algebras in the filtration.

        If the `time` parameter was not provided at initialization, it will be set to an index matching the columns of the provided DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame where each column represents the atom IDs of a sigma algebra in the filtration. The order of the columns determines the order of the filtration (i.e., the first column is the coarsest sigma algebra and the last column is the finest sigma algebra).

        Raises
        ------
        TypeError
            If `data` is not a pandas DataFrame or if the time index of the filtration (if given) does not match the columns of the provided DataFrame.
        ValueError
            If the provided data does not represent a valid filtration (i.e., if the atom IDs in the columns do not form a nested sequence of sigma algebras).

        Returns
        -------
        filtration : Filtration
            The filtration initialized from the provided DataFrame.
        """
        from ..base.index import Index

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        columns = data.columns
        if self._time is not None and not columns.equals(self._time.data):
            raise ValueError(
                "If given, the time index of the filtration must match the columns of the provided data."
            )
        if self._time is None:
            self._time = Index().from_pandas(pd.Index(columns))

        for curr_alg, next_alg in zip(columns[:-1], columns[1:]):
            if data.groupby(next_alg)[curr_alg].nunique().max() > 1:
                raise ValueError(
                    "The provided data does not represent a valid filtration. "
                    f"Column '{curr_alg}' is not a subalgebra of column '{next_alg}'."
                )

        self._data = data.copy()
        return self

    # --------------------- properties --------------------- #

    @property
    def sigma_algebras(self) -> list[SigmaAlgebra]:
        """Get the list of sigma algebras in the filtration.

        Returns
        -------
        sigma_algebras : list[SigmaAlgebra]
            The list of sigma algebras in the filtration.
        """
        from .sigma_algebra import SigmaAlgebra

        if self._sigma_algebras is None:
            sigma_algebras = []
            for col in self._data.columns:
                alg = SigmaAlgebra(name=col).from_pandas(self._data[col])
                sigma_algebras.append(alg)
            self._sigma_algebras = sigma_algebras
        return self._sigma_algebras

    @property
    def data(self) -> pd.DataFrame:
        """Get the underlying data of the filtration.

        Returns
        -------
        data : pd.DataFrame
            The underlying data of the filtration.
        """
        if self._data is None:
            data_dict = {alg.name: alg.data for alg in self._sigma_algebras}
            self._data = pd.DataFrame(data_dict)
        return self._data

    @property
    def name(self) -> Hashable | None:
        """Get the name of the filtration.

        Returns
        -------
        name : Hashable | None
            The name of the filtration.
        """
        return self._name

    @name.setter
    def name(self, name: Hashable | None) -> None:
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("name must be a hashable or None.")
        self._name = name

    @property
    def time(self) -> Index:
        """Get the time index of the filtration.

        Returns
        -------
        time : Index
            The time index of the filtration.
        """
        return self._time

    @property
    def time_to_pos(self) -> dict:
        """Get the mapping from time points to positions in the sigma algebras list.

        Returns
        -------
        time_to_pos : dict
            A mapping from time points to positions in the sigma algebras list.
        """
        if self._time_to_pos is None:
            self._time_to_pos = {time: idx for idx, time in enumerate(self.time)}
        return self._time_to_pos

    @property
    def coarsest(self) -> SigmaAlgebra:
        """Get the coarsest sigma algebra in the filtration.

        Returns
        -------
        coarsest : SigmaAlgebra
            The coarsest sigma algebra in the filtration.
        """
        return self.sigma_algebras[0]

    @property
    def finest(self) -> SigmaAlgebra:
        """Get the finest sigma algebra in the filtration.

        Returns
        -------
        finest : SigmaAlgebra
            The finest sigma algebra in the filtration.
        """
        return self.sigma_algebras[-1]

    @property
    def sample_space(self):
        """Get the sample space of the filtration.

        Returns
        -------
        sample_space : SampleSpace
            The sample space common to all sigma algebras in the filtration.
        """
        return self.sigma_algebras[0].sample_space

    # --------------------- data access methods --------------------- #

    def __getitem__(self, time) -> SigmaAlgebra:
        """Get the sigma algebra at a specific position in the filtration."""
        return self.at[time]

    @property
    def at(self) -> Filtration._FiltrationIndexer:
        """Get an indexer for accessing sigma algebras at specific times.

        Returns
        -------
        indexer : Filtration._FiltrationIndexer
            An indexer for accessing sigma algebras at specific times.

        Examples
        --------
        >>> from sigalg.core import Filtration, SampleSpace, SigmaAlgebra, Time
        >>> # Define sample space and sigma algebras
        >>> sample_space = SampleSpace.generate_sequence(size=3)
        >>> F = SigmaAlgebra.trivial(sample_space=sample_space, name="F")
        >>> G = SigmaAlgebra(sample_space=sample_space, name="G").from_dict(
        ...     sample_id_to_atom_id={"omega_0": 0, "omega_1": 0, "omega_2": 1},
        ... )
        >>> H = SigmaAlgebra.power_set(sample_space=sample_space, name="H")
        >>> # Define continous time index
        >>> time = Time.continuous(start=0.0, stop=1.5, num_points=3)
        >>> # Create and print filtration
        >>> Ft = Filtration(time=time, name="Ft").from_list([F, G, H])
        >>> print(Ft) # doctest: +NORMALIZE_WHITESPACE
        Filtration (Ft)
        ===============
        <BLANKLINE>
        * Time 'T':
        [0.0, 0.75, 1.5]
        <BLANKLINE>
        * At time 0.0:
        Sigma algebra 'F':
                atom ID
        sample
        omega_0        0
        omega_1        0
        omega_2        0
        <BLANKLINE>
        * At time 0.75:
        Sigma algebra 'G':
                atom ID
        sample
        omega_0        0
        omega_1        0
        omega_2        1
        <BLANKLINE>
        * At time 1.5:
        Sigma algebra 'H':
                atom ID
        sample
        omega_0        0
        omega_1        1
        omega_2        2
        >>> # Access sigma algebra at time 0.0
        >>> print(Ft.at[0.0]) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                atom ID
        sample
        omega_0        0
        omega_1        0
        omega_2        0
        >>> # Access sigma algebra at time 0.5 (returns the same as at time 0.0)
        >>> print(Ft.at[0.5]) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                atom ID
        sample
        omega_0        0
        omega_1        0
        omega_2        0
        >>> # Access sigma algebra at time 0.75
        >>> print(Ft.at[0.75]) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
                atom ID
        sample
        omega_0        0
        omega_1        0
        omega_2        1
        >>> # Access sigma algebra at time 1.2 (returns the same as at time 0.75)
        >>> print(Ft.at[1.2]) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
                atom ID
        sample
        omega_0        0
        omega_1        0
        omega_2        1
        >>> # Access sigma algebra at time 1.5
        >>> print(Ft.at[1.5]) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'H':
                atom ID
        sample
        omega_0        0
        omega_1        1
        omega_2        2
        """
        from ..base.time import Time

        if not isinstance(self.time, Time):
            raise TypeError(
                "Time index must be a Time object to use the 'at' property."
            )

        return Filtration._FiltrationIndexer(self)

    class _FiltrationIndexer:
        def __init__(self, filtration):
            self.filtration = filtration

        def __getitem__(self, time) -> SigmaAlgebra:
            time_index = self.filtration.time

            if time in time_index:
                pos_idx = self.filtration.time_to_pos[time]
                return self.filtration.sigma_algebras[pos_idx]

            time_series = pd.Series(time_index.data)

            if time < time_series.min():
                raise ValueError(
                    f"Time {time} is before the start of the filtration "
                    f"(min time: {time_series.min()})"
                )
            if time > time_series.max():
                raise ValueError(
                    f"Time {time} is after the end of the filtration "
                    f"(max time: {time_series.max()})"
                )

            pos_idx = time_series.searchsorted(time, side="right") - 1
            return self.filtration.sigma_algebras[pos_idx]

    # --------------------- sequence methods --------------------- #

    def __len__(self) -> int:
        """Get the length of the filtration.

        The length is defined as the number of sigma algebras minus one.

        Returns
        -------
        length : int
            The length of the filtration.
        """
        return len(self.sigma_algebras) - 1

    def __iter__(self):
        """Iterate over the sigma algebras in the filtration.

        Returns
        -------
        iterator : Iterator[SigmaAlgebra]
            An iterator over the sigma algebras in the filtration.
        """
        yield from self.sigma_algebras

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
        header = f"Filtration ({self.name})"
        separator = "=" * len(header)

        result = header + "\n" + separator + "\n\n* " + repr(self.time)

        for time, sigma_algebra in zip(self.time, self.sigma_algebras):
            result += f"\n\n* At time {time}:\n{sigma_algebra}"

        return result

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        sigma_algebras: list[SigmaAlgebra] | None = None,
        time: Index | None = None,
    ) -> None:
        from ..base.index import Index
        from .comparison import is_subalgebra
        from .sigma_algebra import SigmaAlgebra

        if sigma_algebras is not None:
            if not isinstance(sigma_algebras, list) or len(sigma_algebras) == 0:
                raise ValueError("sigma_algebras must be a non-empty list.")
            for alg in sigma_algebras:
                if not isinstance(alg, SigmaAlgebra):
                    raise ValueError(
                        "All sigma algebras need to be instances of SigmaAlgebra."
                    )

        if time is not None and not isinstance(time, Index):
            raise TypeError("time must be an Index object.")

        if sigma_algebras is not None and time is not None:
            if len(sigma_algebras) != len(time):
                raise ValueError(
                    "The number of sigma algebras must match the length of the time index."
                )
            if len(sigma_algebras) >= 2:
                sample_space = sigma_algebras[0].sample_space
                for alg in sigma_algebras[1:]:
                    if alg.sample_space != sample_space:
                        raise ValueError(
                            "All sigma algebras must have the same sample space"
                        )
                for sub_algebra, super_algebra in zip(
                    sigma_algebras[:-1], sigma_algebras[1:]
                ):
                    if not is_subalgebra(sub_algebra, super_algebra):
                        raise ValueError(
                            "The provided sigma algebras do not form a valid filtration."
                        )
