"""A class representing a time index."""

from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .index import Index

if TYPE_CHECKING:
    from collections.abc import Hashable

    from ...typing.index_like import IndexLike


class Time(Index):
    """A class representing a time index.

    Parameters
    ----------
    indices : IndexLike | None, default=None
        An `IndexLike` object of real numbers for the time index.
    variable_names : list[Hashable] | None, default=None
        A list consisting of a single hashable item for the variable name of the index. If `None`, a default variable name will be generated.
    name : Hashable | None, default=None
        Name identifier for the index. If `None`, a default name will be generated.

    Examples
    --------
    Create a discrete `Time` instance.

    >>> from sigalg.core import Time
    >>> T_discrete = Time.discrete(start=0, length=5, name="T_discrete")
    >>> print(T_discrete)  # doctest: +NORMALIZE_WHITESPACE
    Time 'T_discrete':
     t
     0
     1
     2
     3
     4
     5

    Create a continuous `Time` instance.

    >>> T_continuous = Time.continuous(start=0.0, stop=1.0, num_points=9, name="T_continuous")
    >>> print(T_continuous)  # doctest: +NORMALIZE_WHITESPACE
    Time 'T_continuous':
         t
     0.000
     0.125
     0.250
     0.375
     0.500
     0.625
     0.750
     0.875
     1.000
    """

    _default_name = "T"
    _repr_name = "Time"
    _str_name = "Time"
    _variable_names_prefix = "t"

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        indices: IndexLike | None = None,
        variable_names: list[Hashable] | None = None,
        name: Hashable | None = None,
    ) -> None:
        from ...validation.index_validator import IndexValidator

        if name is None:
            name = type(self)._default_name

        v = IndexValidator(
            indices=indices,
            variable_names=variable_names,
            variable_names_prefix=type(self)._variable_names_prefix,
            kind="Time",
            name=name,
        )

        time = type(self)._from_validated(data=v.data, name=v.name)
        self.__dict__.update(time.__dict__)

    @classmethod
    def discrete(
        cls,
        length: int | None = None,
        start: int = 0,
        stop: int | None = None,
        variable_name: Hashable = "t",
        name: Hashable = "T",
    ) -> Time:
        """Create a discrete time index with integer time steps.

        Generates a time index with consecutive integer time points starting
        from the specified `start`. The user may pass either the `length` of the time interval, or the `stop` value, but not both. The relation between the three parameters is `length = stop - start`.

        Parameters
        ----------
        length : int | None, default=None
            Number of time points to generate. Must be positive.
        start : int, default=0
            Starting time point.
        stop : int | None, default=None
            Ending time point. Mutually exclusive with `length`.
        variable_name : Hashable, default="time"
            Variable name for the time index.
        name : Hashable, default="T"
            Name of the time index.

        Returns
        -------
        time : Time
            A discrete time index with integer time points.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> T = Time.discrete(start=0, length=5)
        >>> print(T) # doctest: +NORMALIZE_WHITESPACE
        Time 'T':
         t
         0
         1
         2
         3
         4
         5
        >>> print(T.is_discrete)
        True
        """
        if not isinstance(start, int):
            raise TypeError("start must be an integer.")
        if length is not None and (not isinstance(length, int) or length <= 0):
            raise ValueError("length must be a positive integer.")
        if stop is not None and (not isinstance(stop, int) or stop <= start):
            raise ValueError("stop must be an integer greater than start.")
        if (length is None) == (stop is None):
            raise ValueError("Specify exactly one of length or stop.")

        if stop is not None:
            length = stop - start

        indices = list(range(start, start + length + 1))
        data = pd.Index(indices, name=variable_name)

        return cls._from_validated(data=data, name=name)

    @classmethod
    def continuous(
        cls,
        start: Real,
        stop: Real,
        dt: Real | None = None,
        num_points: int | None = None,
        variable_name: Hashable = "t",
        name: Hashable = "T",
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
        dt : Real | None, default=None
            Time step between consecutive points. Mutually exclusive with `num_points`.
        num_points : int | None, default=None
            Number of evenly-spaced points to generate. Mutually exclusive with `dt`.
        variable_name : Hashable, default="t"
            Variable name for the time index.
        name : Hashable, default="T"
            Name of the time index.

        Returns
        -------
        time : Time
            A continuous time index with real-valued time points.

        Examples
        --------
        Create a continuous `Time` instance by specifying `num_points`.

        >>> from sigalg.core import Time
        >>> T1 = Time.continuous(start=0.0, stop=1.0, num_points=3, name="T1")
        >>> print(T1)  # doctest: +NORMALIZE_WHITESPACE
        Time 'T1':
           t
         0.0
         0.5
         1.0
        >>> print(T1.is_discrete)
        False

        Create a continuous `Time` instance by specifying `dt`.

        >>> T2 = Time.continuous(start=0.0, stop=1.0, dt=0.25, name="T2")
        >>> print(T2) # doctest: +NORMALIZE_WHITESPACE
        Time 'T2':
            t
         0.00
         0.25
         0.50
         0.75
         1.00
        >>> print(T2.is_discrete)
        False
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
            num_steps = int(np.round((stop - start) / dt)) + 1
            indices = list(np.linspace(start, stop, num_steps))

        data = pd.Index(indices, name=variable_name)

        return cls._from_validated(data=data, name=name)

    # --------------------- properties --------------------- #

    @property
    def is_discrete(self) -> bool | None:
        """Get whether the time index represents discrete or continuous time.

        Returns
        -------
        is_discrete : bool | None
            `True` if the time index represents discrete time, `False` if it represents continuous time, or `None` if not set.
        """
        return (
            all(isinstance(x, int) for x in self.data)
            if self.data is not None
            else None
        )

    # --------------------- data access methods --------------------- #

    def find_nearest_time(self, time_point: Real) -> Real:
        """Find the nearest time point to the given value.

        Parameters
        ----------
        time_point : Real
            The time point to find the nearest time for.

        Returns
        -------
        time : Real
            The nearest time point in the Time index.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> T = Time.discrete(start=0, length=5, name="T")
        >>> print(T) # doctest: +NORMALIZE_WHITESPACE
        Time 'T':
         t
         0
         1
         2
         3
         4
         5
        >>> print(T.find_nearest_time(2.3))
        2
        >>> print(T.find_nearest_time(4.7))
        5
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

    def insert_time(self, time: Real) -> Time:
        """Insert a new time point into the time index.

        Parameters
        ----------
        time : Real
            The time point to insert.

        Returns
        -------
        new_time : Time
            A new Time object with the inserted time point.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> T = Time.discrete(start=0, length=5, name="T")
        >>> print(T) # doctest: +NORMALIZE_WHITESPACE
        Time 'T':
         t
         0
         1
         2
         3
         4
         5
        >>> new_time = T.insert_time(6)
        >>> print(new_time) # doctest: +NORMALIZE_WHITESPACE
        Time 'insert(T)':
         t
         0
         1
         2
         3
         4
         5
         6
        """
        if not isinstance(time, Real):
            raise TypeError("time must be a real number.")
        if self.data is None:
            raise ValueError("Time index is empty.")
        if time in self.data:
            raise ValueError(f"time {time} already exists in the Time index.")

        data = self.data.copy()
        pos = data.searchsorted(time)
        new_data = data.insert(pos, time)
        new_name = f"insert({self.name})"
        return Time._from_validated(data=new_data, name=new_name)

    def remove_time(self, time: Real | None = None, pos: int | None = None) -> Time:
        """Remove a time point from the time index.

        Parameters
        ----------
        time : Real | None, default=None
            The time point to remove. Must be specified if `pos` is not provided.
        pos : int | None, default=None
            The position of the time point to remove. Must be specified if `time` is not provided.

        Returns
        -------
        new_time : Time
            A new Time object with the specified time point removed.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> T = Time.discrete(start=0, length=5, name="T")
        >>> print(T) # doctest: +NORMALIZE_WHITESPACE
        Time 'T':
         t
         0
         1
         2
         3
         4
         5
        >>> new_time = T.remove_time(time=2)
        >>> print(new_time) # doctest: +NORMALIZE_WHITESPACE
        Time 'remove(T)':
         t
         0
         1
         3
         4
         5
        """
        if self.data is None:
            raise ValueError("Time index is empty.")
        if time is not None and not isinstance(time, Real):
            raise TypeError("If provided, time must be a real number.")
        if pos is not None and not isinstance(pos, int):
            raise TypeError("If provided, pos must be an integer.")
        if time is not None and time not in self.data:
            raise ValueError(f"time {time} does not exist in the Time index.")
        if (time is None) == (pos is None):
            raise ValueError("Specify exactly one of time or pos.")
        if pos is not None and (pos < 0 or pos >= len(self.data)):
            raise ValueError(f"pos {pos} is out of bounds.")

        data = self.data.copy()
        if pos is None:
            pos = data.get_loc(time)
        new_data = data.delete(pos)
        new_name = f"remove({self.name})"
        return Time._from_validated(data=new_data, name=new_name)

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a concise string representation of the time index.

        Returns
        -------
        repr_str : str
            String representation of the time index.
        """
        if self.data is None:
            return f"{type(self)._repr_name}(empty)"
        else:
            return f"{type(self)._repr_name}(start={self.data[0]}, stop={self.data[-1]}, is_discrete={self.is_discrete}, name={self.name})"
