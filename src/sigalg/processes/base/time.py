from __future__ import annotations

# from collections.abc import Hashable
from numbers import Real

import numpy as np

from ...core import FeatureIndex


class Time(FeatureIndex):

    # --------------------- factory methods --------------------- #

    @classmethod
    def discrete(cls, start: int = 0, length: int = 10) -> Time:
        if not isinstance(length, int) or length <= 0:
            raise ValueError("length must be a positive integer.")
        if not isinstance(start, int):
            raise TypeError("start must be an integer.")
        indices = list(range(start, start + length))
        return cls(indices=indices, values_name="time")

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
        return cls(indices=indices, values_name="time")
