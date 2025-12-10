from __future__ import annotations

from numbers import Real

import numpy as np

from .index import Index


class Time(Index):

    def __init__(
        self, indices: list[Real], is_discrete: bool = True, values_name: str = "time"
    ) -> None:
        self._validate_time_parameters(indices)
        super().__init__(indices=sorted(indices), values_name=values_name)
        self._is_discrete = is_discrete

    # --------------------- properties --------------------- #

    @property
    def is_discrete(self) -> bool:
        return self._is_discrete

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
    def continuous(cls, start: Real = 0.0, stop: Real = 1.0, step: Real = 0.1) -> Time:
        if not isinstance(start, Real):
            raise TypeError("start must be a real number.")
        if not isinstance(stop, Real):
            raise TypeError("stop must be a real number.")
        if not isinstance(step, Real) or step <= 0:
            raise ValueError("step must be a positive real number.")
        if stop <= start:
            raise ValueError("stop must be greater than start.")
        num_points = int((stop - start) / step) + 1
        indices = list(np.linspace(start, stop, num_points))
        return cls(indices=indices, is_discrete=False, values_name="time")

    # --------------------- data access methods --------------------- #

    def _getitem_hook(self, key):
        result = self.values[key].to_list()
        return Time(
            indices=result, is_discrete=self.is_discrete, values_name=self.values_name
        )

    # --------------------- equality --------------------- #

    def __eq__(self, other: Time) -> bool:
        return (
            isinstance(other, Time)
            and super().__eq__(other)
            and self.is_discrete == other.is_discrete
        )

    # --------------------- validation methods --------------------- #
    @staticmethod
    def _validate_time_parameters(indices: list[Real]) -> None:
        if not isinstance(indices, list):
            raise TypeError("indices must be a list of real numbers.")
        if len(indices) == 0:
            raise ValueError("indices list cannot be empty.")
        for idx in indices:
            if not isinstance(idx, Real):
                raise TypeError("all indices must be real numbers.")
