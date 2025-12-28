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
    from ..base.time import Time
    from .sigma_algebra import SigmaAlgebra


class Filtration:
    """Class representing a filtration of sigma algebras.

    A filtration is a sequence of sigma algebras indexed by time, where each
    sigma algebra is a subset of the next one in the sequence.

    Parameters
    ----------
    sigma_algebras : list[SigmaAlgebra]
        A list of sigma algebras forming the filtration.
    time : Time
        A time index corresponding to the sigma algebras.
    name : Hashable | None, default="Ft"
        An optional name for the filtration.

    Raises
    ------
    ValueError
        If the sigma algebras do not form a valid filtration or if the lengths
        of sigma algebras and time index do not match.
    TypeError
        If the provided parameters are of incorrect types.

    Examples
    --------
    >>> from sigalg.core import Filtration, SampleSpace, SigmaAlgebra, Time
    >>> # Define sample space and sigma algebras
    >>> sample_space = SampleSpace.generate_default(size=3)
    >>> F = SigmaAlgebra.trivial(sample_space=sample_space, name="F")
    >>> G = SigmaAlgebra(
    ...     sample_id_to_atom_id={"omega0": 0, "omega1": 0, "omega2": 1},
    ...     sample_space=sample_space,
    ...     name="G",
    ... )
    >>> H = SigmaAlgebra.power_set(sample_space=sample_space, name="H")
    >>> # Define continous time index
    >>> time = Time.continuous(start=0.0, stop=1.5, num_points=3)
    >>> # Create and print filtration
    >>> Ft = Filtration(sigma_algebras=[F, G, H], time=time, name="Ft")
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
    omega0        0
    omega1        0
    omega2        0
    <BLANKLINE>
    * At time 0.75:
    Sigma algebra 'G':
            atom ID
    sample
    omega0        0
    omega1        0
    omega2        1
    <BLANKLINE>
    * At time 1.5:
    Sigma algebra 'H':
            atom ID
    sample
    omega0        0
    omega1        1
    omega2        2
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        sigma_algebras: list[SigmaAlgebra],
        time: Time,
        name: Hashable | None = "Ft",
    ) -> None:

        self._validate_parameters(sigma_algebras=sigma_algebras, time=time, name=name)
        self.sigma_algebras = sigma_algebras
        self.time = time
        self.sample_space = sigma_algebras[0].sample_space
        self._name = name
        self._time_to_pos = {t: idx for idx, t in enumerate(self.time)}

    # --------------------- coarsest --------------------- #

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

    # --------------------- data access methods --------------------- #

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
        >>> sample_space = SampleSpace.generate_default(size=3)
        >>> F = SigmaAlgebra.trivial(sample_space=sample_space, name="F")
        >>> G = SigmaAlgebra(
        ...     sample_id_to_atom_id={"omega0": 0, "omega1": 0, "omega2": 1},
        ...     sample_space=sample_space,
        ...     name="G",
        ... )
        >>> H = SigmaAlgebra.power_set(sample_space=sample_space, name="H")
        >>> # Define continous time index
        >>> time = Time.continuous(start=0.0, stop=1.5, num_points=3)
        >>> # Create and print filtration
        >>> Ft = Filtration(sigma_algebras=[F, G, H], time=time, name="Ft")
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
        omega0        0
        omega1        0
        omega2        0
        <BLANKLINE>
        * At time 0.75:
        Sigma algebra 'G':
                atom ID
        sample
        omega0        0
        omega1        0
        omega2        1
        <BLANKLINE>
        * At time 1.5:
        Sigma algebra 'H':
                atom ID
        sample
        omega0        0
        omega1        1
        omega2        2
        >>> # Access sigma algebra at time 0.0
        >>> print(Ft.at[0.0]) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                atom ID
        sample
        omega0        0
        omega1        0
        omega2        0
        >>> # Access sigma algebra at time 0.5 (returns the same as at time 0.0)
        >>> print(Ft.at[0.5]) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                atom ID
        sample
        omega0        0
        omega1        0
        omega2        0
        >>> # Access sigma algebra at time 0.75
        >>> print(Ft.at[0.75]) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
                atom ID
        sample
        omega0        0
        omega1        0
        omega2        1
        >>> # Access sigma algebra at time 1.2 (returns the same as at time 0.75)
        >>> print(Ft.at[1.2]) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
                atom ID
        sample
        omega0        0
        omega1        0
        omega2        1
        >>> # Access sigma algebra at time 1.5
        >>> print(Ft.at[1.5]) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'H':
                atom ID
        sample
        omega0        0
        omega1        1
        omega2        2
        """
        return Filtration._FiltrationIndexer(self)

    class _FiltrationIndexer:
        def __init__(self, filtration):
            self.filtration = filtration

        def __getitem__(self, time) -> SigmaAlgebra:
            time_index = self.filtration.time

            if time in time_index:
                pos_idx = self.filtration._time_to_pos[time]
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
        sigma_algebras: list[SigmaAlgebra],
        time: Time,
        name: Hashable | None,
    ) -> None:
        from ..base.time import Time
        from .comparison import is_subalgebra
        from .sigma_algebra import SigmaAlgebra

        if not isinstance(sigma_algebras, list) or len(sigma_algebras) == 0:
            raise ValueError("sigma_algebras must be a non-empty list.")
        for alg in sigma_algebras:
            if not isinstance(alg, SigmaAlgebra):
                raise ValueError(
                    "All sigma algebras need to be instances of SigmaAlgebra."
                )
        if not isinstance(time, Time):
            raise TypeError("time must be a Time object.")
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("name must be hashable, if provided.")
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
