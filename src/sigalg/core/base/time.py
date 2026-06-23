"""A class representing a time index."""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Real

import numpy as np
import pandas as pd

from ...validation.index_validator import IndexLike, IndexValidator
from .index import Index


class Time(Index):
    """A class representing a time index.

    Parameters
    ----------
    name : Hashable, default="T"
        Name identifier for the index.

    Raises
    ------
    ValueError
        If the underlying data of the time index is a `pd.MultiIndex`, if any element in the time index is not a real number, or if the time index is not in ascending order.

    Examples
    --------
    Create a discrete `Time` instance.

    >>> from sigalg.core import Time
    >>> T_discrete = Time.discrete(start=0, length=5, name="T_discrete")
    >>> print(T_discrete)  # doctest: +NORMALIZE_WHITESPACE
    Time 'T_discrete':
    [0, 1, 2, 3, 4, 5]

    Create a continuous `Time` instance.

    >>> T_continuous = Time.continuous(start=0.0, stop=1.0, num_points=9, name="T_continuous")
    >>> print(T_continuous)  # doctest: +NORMALIZE_WHITESPACE
    Time 'T_continuous':
    [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    """

    _properties = Index._properties + ["_is_discrete"]

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        indices: IndexLike | None = None,
        name: Hashable = "T",
        variable_names: list[Hashable] | None = None,
        **kwargs,
    ) -> None:
        v = IndexValidator(indices=indices, name=name, variable_names=variable_names)

        if v.indices is not None:
            if isinstance(v.indices, pd.MultiIndex):
                raise ValueError(
                    "The underlying data of a time index cannot be a pd.MultiIndex"
                )
            if not all(isinstance(x, Real) for x in v.indices):
                raise ValueError("All elements in the time index must be real numbers")
            if not v.indices.is_monotonic_increasing:
                raise ValueError("Time index must be in ascending order")

        super().__init__(
            indices=v.indices,
            name=name,
            variable_names=v.variable_names,
            bypass_validation=True,
            **kwargs,
        )

    @classmethod
    def discrete(
        cls,
        length: int | None = None,
        start: int = 0,
        stop: int | None = None,
        name: Hashable = "T",
        variable_name: Hashable | None = None,
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
        name : Hashable, default="T"
            Name identifier for the index.
        variable_name : Hashable | None, default=None
            Name for the internal `pd.Index`. If `None`, a default `"time"` will be used.

        Returns
        -------
        time : Time
            A discrete time index with integer time points.

        Raises
        ------
        ValueError
            If `length` is not a positive integer or if `stop` is not an integer greater than `start`, or if both `length` and `stop` are specified, or if neither is specified.
        TypeError
            If `start` is not an integer.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> T = Time.discrete(start=0, length=5)
        >>> print(T) # doctest: +NORMALIZE_WHITESPACE
        Time 'T':
        [0, 1, 2, 3, 4, 5]
        >>> print(T.is_discrete)
        True
        """
        if not isinstance(start, int):
            raise TypeError("start must be an integer.")
        if length is not None and (not isinstance(length, int) or length <= 0):
            raise ValueError("length must be a positive integer.")
        if stop is not None and (not isinstance(stop, int) or stop <= start):
            raise ValueError("stop must be an integer greater than start.")
        if length is not None and stop is not None:
            raise ValueError("Specify exactly one of length or stop.")
        if length is None and stop is None:
            raise ValueError("Specify exactly one of length or stop.")

        if stop is not None:
            length = stop - start

        if variable_name is None:
            variable_name = "time"

        indices = list(range(start, start + length + 1))
        return cls(indices, variable_names=[variable_name], name=name)

    @classmethod
    def continuous(
        cls,
        start: Real,
        stop: Real,
        dt: Real | None = None,
        num_points: int | None = None,
        name: Hashable = "T",
        variable_name: list | None = None,
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
        name : Hashable, default="T"
            Name identifier for the index.
        variable_name : list | None, default=None
            Name for the internal `pd.Index`. If `None`, a default `["time"]` will be used.

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
        Create a continuous `Time` instance by specifying `num_points`.

        >>> from sigalg.core import Time
        >>> T1 = Time.continuous(start=0.0, stop=1.0, num_points=3, name="T1")
        >>> print(T1)  # doctest: +NORMALIZE_WHITESPACE
        Time 'T1':
        [0.0, 0.5, 1.0]
        >>> print(T1.is_discrete)
        False

        Create a continuous `Time` instance by specifying `dt`.

        >>> T2 = Time.continuous(start=0.0, stop=1.0, dt=0.25, name="T2")
        >>> print(T2) # doctest: +NORMALIZE_WHITESPACE
        Time 'T2':
        [0.0, 0.25, 0.5, 0.75, 1.0]
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
        return cls(indices, variable_names=[variable_name], name=name)

    # --------------------- properties --------------------- #

    @property
    def is_discrete(self) -> bool | None:
        """Get whether the time index represents discrete or continuous time.

        Returns
        -------
        is_discrete : bool | None
            `True` if the time index represents discrete time, `False` if it represents continuous time, or `None` if not set.
        """
        if self._is_discrete is None and self.data is not None:
            self._is_discrete = all(isinstance(x, int) for x in self.data)
        return self._is_discrete

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
        >>> T = Time.discrete(start=0, length=5, name="T")
        >>> print(T) # doctest: +NORMALIZE_WHITESPACE
        Time 'T':
        [0, 1, 2, 3, 4, 5]
        >>> # Access via integer index
        >>> print(T[0])
        0
        >>> # Access via slice
        >>> print(T[1:3]) # doctest: +NORMALIZE_WHITESPACE
        Time 'T_slice':
        [1, 2]
        >>> # Access via list of positions
        >>> print(T[[0, 2]]) # doctest: +NORMALIZE_WHITESPACE
        Time 'T_slice':
        [0, 2]
        """  # noqa: D401
        if not isinstance(pos, (int, list, slice)):
            raise TypeError("pos must be int | list[int] | slice.")
        if isinstance(pos, list) and not all(isinstance(i, int) for i in pos):
            raise TypeError("pos list must contain only int.")

        data = self.data[pos]
        if isinstance(data, pd.Index):
            return Time(
                indices=data.to_list(),
                variable_names=self.variable_names,
                name=f"{self.name}_slice",
            )
        else:
            return data

    def find_nearest_time(self, time_point: Real) -> Real:
        """Find the nearest time point to the given value.

        Parameters
        ----------
        time_point : Real
            The time point to find the nearest index for.

        Raises
        ------
        ValueError
            If the Time index is empty, or if `time_point` is outside the range of the Time index.

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
        [0, 1, 2, 3, 4, 5]
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
            The time point to insert. Must be an integer for discrete Time indices.

        Raises
        ------
        TypeError
            If `time` is not a real number.
        ValueError
            If the Time index is empty or if `time` already exists in the Time index.

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
        [0, 1, 2, 3, 4, 5]
        >>> new_time = T.insert_time(6)
        >>> print(new_time) # doctest: +NORMALIZE_WHITESPACE
        Time 'insert(T)':
        [0, 1, 2, 3, 4, 5, 6]
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
        new_name = f"insert({self.name})" if self.name is not None else None
        new_time = Time(indices=new_data, name=new_name)
        return new_time

    def remove_time(self, time: Real | None = None, pos: int | None = None) -> Time:
        """Remove a time point from the time index.

        Parameters
        ----------
        time : Real | None, default=None
            The time point to remove. Must be specified if `pos` is not provided.
        pos : int | None, default=None
            The position of the time point to remove. Must be specified if `time` is not provided.

        Raises
        ------
        TypeError
            If `time` is not a real number or `pos` is not an integer.
        ValueError
            If the Time index is empty, if `time` does not exist in the Time index, if `pos` is out of bounds, if both `time` and `pos` are provided, or if neither is provided.

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
        [0, 1, 2, 3, 4, 5]
        >>> new_time = T.remove_time(time=2)
        >>> print(new_time) # doctest: +NORMALIZE_WHITESPACE
        Time 'remove(T)':
        [0, 1, 3, 4, 5]
        """
        if self.data is None:
            raise ValueError("Time index is empty.")
        if time is not None and not isinstance(time, Real):
            raise TypeError("If provided, time must be a real number.")
        if pos is not None and not isinstance(pos, int):
            raise TypeError("If provided, pos must be an integer.")
        if time is not None and time not in self.data:
            raise ValueError(f"time {time} does not exist in the Time index.")
        if time is None and pos is None:
            raise ValueError("Either time or pos must be specified.")
        if time is not None and pos is not None:
            raise ValueError("Only one of time or pos must be specified.")
        if pos is not None and (pos < 0 or pos >= len(self.data)):
            raise ValueError(f"pos {pos} is out of bounds.")

        data = self.data.copy()
        if pos is None:
            pos = data.get_loc(time)
        new_data = data.delete(pos)
        new_name = f"remove({self.name})" if self.name is not None else None
        new_time = Time(indices=new_data, name=new_name)
        return new_time

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a string representation of the time index.

        Returns
        -------
        repr_str : str
            String representation of the time index.
        """
        if self.data is None:
            if self.name is None:
                return "Time: empty"
            else:
                return f"Time '{self.name}': empty"
        else:
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

    # --------------------- set-theoretic operations --------------------- #

    def __and__(self, other: Time) -> Time:
        """Return the intersection of this time index with another (`&` operator).

        Parameters
        ----------
        other : Time
            Another index from the same sample space.

        Returns
        -------
        time : Time
            A time index containing elements present in both time indices.

        Raises
        ------
        ValueError
            If the time indices have different `is_discrete` values.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> T = Time.discrete(start=0, length=5, name="T")
        >>> S = Time.discrete(start=3, length=5, name="S")
        >>> print(T) # doctest: +NORMALIZE_WHITESPACE
        Time 'T':
        [0, 1, 2, 3, 4, 5]
        >>> print(S) # doctest: +NORMALIZE_WHITESPACE
        Time 'S':
        [3, 4, 5, 6, 7, 8]
        >>> intersection = T & S
        >>> print(intersection) # doctest: +NORMALIZE_WHITESPACE
        Time 'T intersect S':
        [3, 4, 5]
        """
        if self.is_discrete != other.is_discrete:
            raise ValueError(
                "Cannot intersect Time indices with different is_discrete values."
            )
        pts = set(self.data) & set(other.data)
        return Time(indices=list(pts), name=f"{self.name} intersect {other.name}")
