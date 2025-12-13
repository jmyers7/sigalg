from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd

from .index import Index


class Time(Index):

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
        if (dt is None) == (num_points is None):
            raise ValueError("Specify exactly one of dt or num_points.")
        if num_points is not None:
            indices = list(np.linspace(start, stop, num_points))
        else:
            indices = list(np.arange(start, stop, dt))
        return cls(indices=indices, is_discrete=False, values_name="time")

    # --------------------- data access methods --------------------- #

    def _getitem_hook(self, key):
        result = self.values[key].to_list()
        return Time(
            indices=result, is_discrete=self.is_discrete, values_name=self.values_name
        )

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"Time(times={self.values.to_list()}, is_discrete={self.is_discrete})"

    # --------------------- equality --------------------- #

    def __eq__(self, other: Time) -> bool:
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
