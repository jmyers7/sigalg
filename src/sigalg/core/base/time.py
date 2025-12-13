"""Time indices for temporal processes.

This module provides the Time class, which represents time indices for
temporal stochastic processes and other objects. Time indices can be discrete (integer-valued) or continuous (real-valued).

Classes
-------
Time
    Represents a time index for temporal processes.

Examples
--------
>>> import sigalg as sa
>>> # Discrete time
>>> time_discrete = sa.Time.discrete(start=0, length=5)
>>> # Continuous time
>>> time_continuous = sa.Time.continuous(start=0.0, stop=1.0, num_points=10)
"""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd

from .index import Index


class Time(Index):
    """A time index for representing temporal sequences.

    Time indices can represent either discrete time steps (integers) or
    continuous time points (real numbers). They must be monotonically
    increasing and are used as the temporal dimension for stochastic processes and other objects.

    Parameters
    ----------
    indices : list of Real, optional
        List of time points. Must be sorted in ascending order.
        Mutually exclusive with values.
    values : pd.Index, optional
        pandas Index object containing time points.
        Mutually exclusive with indices.
    name : str, default="T"
        Name identifier for the time index.
    values_name : str, default="time"
        Name for the index of values.
    is_discrete : bool, default=True
        Whether the time index represents discrete (True) or continuous (False) time.

    Raises
    ------
    ValueError
        If indices is empty, not sorted, or values is empty/not sorted.
    TypeError
        If indices or values contain non-numeric values.

    Examples
    --------
    >>> import sigalg as sa
    >>> # Discrete time from 0 to 4
    >>> time_discrete = sa.Time.discrete(start=0, length=5)
    >>> list(time_discrete)
    [0, 1, 2, 3, 4]
    >>> # Continuous time from 0.0 to 1.0
    >>> time_continuous = sa.Time.continuous(start=0.0, stop=1.0, num_points=11)
    >>> time_continuous.is_discrete
    False
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        indices: list[Real] | None = None,
        values: pd.Index | None = None,
        name: str = "T",
        values_name: str = "time",
        is_discrete: bool = True,
    ) -> None:
        super().__init__(
            indices=indices, values=values, name=name, values_name=values_name
        )
        self._validate_time_parameters(
            indices=indices, values=values, is_discrete=is_discrete
        )
        self.is_discrete = is_discrete

    # --------------------- factory methods --------------------- #

    @classmethod
    def discrete(cls, start: int = 0, length: int = 10) -> Time:
        """Create a discrete time index with integer time steps.

        Generates a time index with consecutive integer time points starting
        from the specified start value.

        Parameters
        ----------
        start : int, default=0
            Starting time point.
        length : int, default=10
            Number of time points to generate. Must be positive.

        Returns
        -------
        Time
            A discrete time index with integer time points.

        Raises
        ------
        ValueError
            If length is not a positive integer.
        TypeError
            If start is not an integer.

        Examples
        --------
        >>> import sigalg as sa
        >>> time = sa.Time.discrete(start=0, length=5)
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
        return cls(indices=indices, is_discrete=True, values_name="time")

    @classmethod
    def continuous(
        cls,
        start: Real = 0.0,
        stop: Real = 1.0,
        *,
        dt: Real | None = None,
        num_points: int | None = None,
    ) -> Time:
        """Create a continuous time index with real-valued time points.

        Generates a time index with real-valued time points either by specifying
        the time step (dt) or the number of points (num_points). Exactly one of
        these parameters must be provided.

        Parameters
        ----------
        start : Real, default=0.0
            Starting time point.
        stop : Real, default=1.0
            Ending time point.
        dt : Real, optional
            Time step between consecutive points. Mutually exclusive with num_points.
        num_points : int, optional
            Number of evenly-spaced points to generate. Mutually exclusive with dt.

        Returns
        -------
        Time
            A continuous time index with real-valued time points.

        Raises
        ------
        ValueError
            If both dt and num_points are specified, or if neither is specified.

        Examples
        --------
        >>> import sigalg as sa
        >>> # Using num_points
        >>> time1 = sa.Time.continuous(start=0.0, stop=1.0, num_points=3)
        >>> list(time1)
        [0.0, 0.5, 1.0]
        >>> # Using dt
        >>> time2 = sa.Time.continuous(start=0.0, stop=1.0, dt=0.25)
        >>> len(time2)
        4
        """
        if (dt is None) == (num_points is None):
            raise ValueError("Specify exactly one of dt or num_points.")
        if num_points is not None:
            indices = list(np.linspace(start, stop, num_points))
        else:
            indices = list(np.arange(start, stop, dt))
        return cls(indices=indices, is_discrete=False, values_name="time")

    # --------------------- data access methods --------------------- #

    def _getitem_hook(self, key):
        """Internal hook for indexing operations to create events.

        This method is called by __getitem__ from the parent Index class. In Time, the purpose of this method is to ensure that __getitem__ returns an instance of Time. Times are retrieved by position.

        Parameters
        ----------
        key : int, slice, or list
            Indexing key. Can be:
            - An integer: Creates single-element feature index
            - A slice: Creates feature index with slice of features
            - A list: Creates feature index with multiple features

        Returns
        -------
        FeatureIndex
            A FeatureIndex object containing the indexed features.

        Examples
        --------
        >>> import sigalg as sa
        >>> time = sa.Time.discrete(start=0, length=5)
        >>> # Access via integer index
        >>> time1 = time[0]
        >>> # Access via slice
        >>> time2 = time[1:3]
        >>> # Access via list of positions
        >>> time3 = time[[0, 2]]
        """
        if isinstance(key, int):
            result = [self.values[key]]
        else:
            result = self.values[key].to_list()
        return Time(
            indices=result, is_discrete=self.is_discrete, values_name=self.values_name
        )

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a string representation of the time index.

        Returns
        -------
        str
            A formatted string showing the time points and whether the time
            index is discrete or continuous.
        """
        return f"Time(times={self.values.to_list()}, is_discrete={self.is_discrete})"

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
        bool
            True if the other object is a Time with identical values and
            is_discrete flag, False otherwise.
        """
        return (
            isinstance(other, Time)
            and super().__eq__(other)
            and self.is_discrete == other.is_discrete
        )

    # --------------------- validation methods --------------------- #
    @staticmethod
    def _validate_time_parameters(
        indices: list[Real] | None, values: pd.Index | None, is_discrete: bool
    ) -> None:
        """Validate time index construction parameters.

        Parameters
        ----------
        indices : list of Real, optional
            List of time points to validate.
        values : pd.Index, optional
            pandas Index of time points to validate.
        is_discrete : bool
            Whether the time index is discrete.

        Raises
        ------
        ValueError
            If indices is empty or not sorted in ascending order, or if values
            is empty or not monotonically increasing.
        TypeError
            If indices or values contain non-numeric values.
        """
        if indices is not None:
            if len(indices) == 0:
                raise ValueError("indices list cannot be empty.")
            if not all(isinstance(idx, Real) for idx in indices):
                raise TypeError("all indices must be real numbers.")
            if indices != sorted(indices):
                raise ValueError("indices must be sorted in ascending order.")
        if values is not None:
            if len(values) == 0:
                raise ValueError("values cannot be empty.")
            if not all(isinstance(val, Real) for val in values):
                raise TypeError("all values must be real numbers.")
            if not values.is_monotonic_increasing:
                raise ValueError("values must be sorted in ascending order.")
