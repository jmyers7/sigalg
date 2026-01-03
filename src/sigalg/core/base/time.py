"""Time module.

This module provides the `Time` class, which represents time indices for
stochastic processes and other objects. Time indices can be discrete (integer-valued) or continuous (real-valued).

Classes
-------
Time
    Represents a time index for temporal processes.

Examples
--------
>>> from sigalg.core import Time
>>> # Discrete time
>>> time_discrete = Time.discrete(start=0, length=5)
>>> time_discrete # doctest: +NORMALIZE_WHITESPACE
Time 'T':
[0, 1, 2, 3, 4]
>>> # Continuous time
>>> time_continuous = Time.continuous(start=0.0, stop=1.0, num_points=9)
>>> time_continuous # doctest: +NORMALIZE_WHITESPACE
Time 'T':
[0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
"""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Real

import numpy as np
import pandas as pd

from ...validation.time_in import TimeIn
from .index import Index


class Time(Index):
    """A class representing a time index.

    Parameters
    ----------
    name : Hashable | None, default="T"
        Name identifier for the index.
    data_name : Hashable | None, default="time"
        Name for the internal `pd.Index`.

    Examples
    --------
    >>> from sigalg.core import Time
    >>> # Discrete time
    >>> time_discrete = Time.discrete(start=0, length=5)
    >>> time_discrete # doctest: +NORMALIZE_WHITESPACE
    Time 'T':
    [0, 1, 2, 3, 4]
    >>> # Continuous time
    >>> time_continuous = Time.continuous(start=0.0, stop=1.0, num_points=9)
    >>> time_continuous # doctest: +NORMALIZE_WHITESPACE
    Time 'T':
    [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    """

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        name: Hashable | None = "T",
        data_name: Hashable | None = "time",
    ) -> None:
        super().__init__(name=name, data_name=data_name)

    def from_list(
        self,
        indices: list[Real],
        is_discrete: bool = True,
    ) -> Index:
        """Create a `Time` from a list of time points.

        The time points can represent either discrete time steps (integers) or
        continuous time points (real numbers). They must be monotonically
        increasing and are used as the temporal dimension for stochastic processes and other objects.

        Parameters
        ----------
        indices : list[Real]
            List of real-valued time points to use for the index.
        is_discrete : bool, default=True
            Whether the time index represents discrete (`True`) or continuous (`False`) time.

        Returns
        -------
        self : Time
            The current `Time` instance with updated indices.
        """
        v = TimeIn(indices=indices, is_discrete=is_discrete)
        self.is_discrete = v.is_discrete
        self._indices = v.indices
        return self

    # --------------------- factory methods --------------------- #

    @classmethod
    def discrete(
        cls,
        length: int,
        start: int = 0,
        name: Hashable | None = "T",
        data_name: Hashable | None = "time",
    ) -> Time:
        """Create a discrete time index with integer time steps.

        Generates a time index with consecutive integer time points starting
        from the specified start value.

        Parameters
        ----------
        length : int
            Number of time points to generate. Must be positive.
        start : int, default=0
            Starting time point.
        name : Hashable | None, default="T"
            Name identifier for the index.
        data_name : Hashable | None, default="time"
            Name for the internal `pd.Index`.

        Returns
        -------
        time : Time
            A discrete time index with integer time points.

        Raises
        ------
        ValueError
            If `length` is not a positive integer.
        TypeError
            If `start` is not an integer.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> time = Time.discrete(start=0, length=5)
        >>> list(time)
        [0, 1, 2, 3, 4]
        >>> time.is_discrete
        True
        """
        if not isinstance(length, int) or length <= 0:
            raise ValueError("length must be a positive integer.")
        if not isinstance(start, int):
            raise TypeError("start must be an integer.")
        indices = list(range(start, start + length))
        return cls(name=name, data_name=data_name).from_list(indices, is_discrete=True)

    @classmethod
    def continuous(
        cls,
        start: Real,
        stop: Real,
        dt: Real | None = None,
        num_points: int | None = None,
        name: Hashable | None = "T",
        data_name: Hashable | None = "time",
    ) -> Time:
        """Create a continuous time index with real-valued time points.

        Generates a time index with real-valued time points either by specifying
        the time step (`dt`) or the number of points (`num_points`). Exactly one of
        these parameters must be provided.

        Parameters
        ----------
        start : Real
            Starting time point.
        stop : Real
            Ending time point.
        dt : Real, optional
            Time step between consecutive points. Mutually exclusive with `num_points`.
        num_points : int, optional
            Number of evenly-spaced points to generate. Mutually exclusive with `dt`.
        name : Hashable | None, default="T"
            Name identifier for the index.
        data_name : Hashable | None, default="time"
            Name for the internal `pd.Index`.

        Returns
        -------
        time : Time
            A continuous time index with real-valued time points.

        Raises
        ------
        ValueError
            If both `dt` and `num_points` are specified, or if neither is specified. Also raised if `start` is not less than `stop`, or if `dt` is not positive, or if `num_points` is less than 2.
        TypeError
            If `start`, `stop`, or `dt` (if given) are not real numbers, or if `num_points` (if given) is not an integer.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> # Using num_points
        >>> time1 = Time.continuous(start=0.0, stop=1.0, num_points=3)
        >>> list(time1)
        [0.0, 0.5, 1.0]
        >>> # Using dt
        >>> time2 = Time.continuous(start=0.0, stop=1.0, dt=0.25)
        >>> len(time2)
        4
        """
        if (dt is None) == (num_points is None):
            raise ValueError("Specify exactly one of dt or num_points.")
        if not isinstance(start, Real) or not isinstance(stop, Real):
            raise TypeError("start and stop must be real numbers.")
        if start >= stop:
            raise ValueError("start must be less than stop.")
        if dt is not None and (not isinstance(dt, Real) or dt <= 0):
            raise ValueError("If given, dt must be a positive real number.")
        if num_points is not None and (
            not isinstance(num_points, int) or num_points < 2
        ):
            raise ValueError("If given, num_points must be an integer >= 2.")
        if num_points is not None:
            indices = list(np.linspace(start, stop, num_points))
        else:
            indices = list(np.arange(start, stop, dt))
        return cls(name=name, data_name=data_name).from_list(indices, is_discrete=False)

    # --------------------- data access methods --------------------- #

    def _getitem_hook(self, pos: int | list[int] | slice) -> Time:
        """Internal hook for indexing operations.

        This method is called by `__getitem__` from the parent `Index` class. In `Time`, the purpose of this method is to ensure that `__getitem__` returns an instance of `Time`. Times are retrieved by position.

        Parameters
        ----------
        pos : int | list[int] | slice
            Index, slice, or other key for accessing elements positionally.

        Returns
        -------
        time : Time
            A `Time` object containing the indexed time points.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> time = Time.discrete(start=0, length=5)
        >>> print(time) # doctest: +NORMALIZE_WHITESPACE
        Time 'T':
        [0, 1, 2, 3, 4]
        >>> # Access via integer index
        >>> print(time[0])
        0
        >>> # Access via slice
        >>> print(time[1:3]) # doctest: +NORMALIZE_WHITESPACE
        Time:
        [1, 2]
        >>> # Access via list of positions
        >>> print(time[[0, 2]]) # doctest: +NORMALIZE_WHITESPACE
        Time:
        [0, 2]
        """  # noqa: D401
        if not isinstance(pos, (int, list, slice)):
            raise TypeError("pos must be int | list[int] | slice.")
        if isinstance(pos, list) and not all(isinstance(i, int) for i in pos):
            raise TypeError("pos list must contain only int.")

        data = self.data[pos]
        if isinstance(data, pd.Index):
            return Time(data_name=self.data.name, name=None).from_list(
                indices=data.to_list(), is_discrete=self.is_discrete
            )
        else:
            return data

    def find_nearest_time(self, time_point: Real) -> Real:
        """Find the nearest time point to the given value.

        Parameters
        ----------
        time_point : Real
            The time point to find the nearest index for.

        Returns
        -------
        time : Real
            The nearest time point in the Time index.

        Raises
        ------
        ValueError
            If the Time index is empty.
        """
        if len(self) == 0:
            raise ValueError("Time index is empty.")
        array = np.array(self.data)
        if time_point < array[0]:
            raise ValueError(
                f"time_point {time_point} is before the start of the Time index."
            )
        if time_point > array[-1]:
            raise ValueError(
                f"time_point {time_point} is after the end of the Time index."
            )
        nearest_idx = (np.abs(array - time_point)).argmin()
        return self.data[nearest_idx]

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a string representation of the index.

        Returns
        -------
        repr_str : str
            String representation of the index.
        """
        if self.name is None:
            return f"Time:\n{self.data.to_list()}"
        else:
            return f"Time '{self.name}':\n{self.data.to_list()}"

    # --------------------- equality --------------------- #

    def __eq__(self, other: Time) -> bool:
        """Check equality with another time index.

        Two time indices are equal if they have the same time points in the
        same order and the same discrete/continuous flag.

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the other object is a `Time` with identical values and
            `is_discrete` flag, `False` otherwise.
        """
        return (
            isinstance(other, Time)
            and super().__eq__(other)
            and self.is_discrete == other.is_discrete
        )
