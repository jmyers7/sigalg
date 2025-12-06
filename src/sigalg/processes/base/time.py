from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd


class Time:

    # --------------------- constructor --------------------- #

    def __init__(self, idx: pd.Index, discrete: bool) -> None:
        self._values = idx
        self._discrete = discrete

    # --------------------- properties --------------------- #

    @property
    def values(self) -> pd.Index:
        return self._values

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    # --------------------- factory methods --------------------- #

    @classmethod
    def discrete(cls, start: int, length: int) -> Time:
        if not isinstance(length, int) or length <= 0:
            raise ValueError("length must be a positive integer.")
        if not isinstance(start, int):
            raise TypeError("start must be an integer.")
        idx = pd.Index(range(start, start + length), name="time")
        return cls(idx=idx, discrete=True)

    @classmethod
    def continuous(cls, start: Real, stop: Real, step: Real) -> Time:
        if not isinstance(start, Real):
            raise TypeError("start must be a real number.")
        if not isinstance(stop, Real):
            raise TypeError("stop must be a real number.")
        if not isinstance(step, Real) or step <= 0:
            raise ValueError("step must be a positive real number.")
        if stop <= start:
            raise ValueError("stop must be greater than start.")
        num_points = int((stop - start) / step) + 1
        idx = pd.Index(
            np.linspace(start, stop, num_points),
            name="time",
        )
        return cls(idx=idx, discrete=False)
