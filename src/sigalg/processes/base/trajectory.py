from collections.abc import Hashable

import pandas as pd

from ...core import SamplePointFeatures


class Trajectory(SamplePointFeatures):

    # --------------------- constructor --------------------- #

    def __init__(self, values: pd.Series, name: Hashable) -> None:
        super().__init__(values=values, name=name)
        self._values.index.name = "time"

    # --------------------- data access methods --------------------- #

    @property
    def value_at(self):
        return self._iLocIndexer(self)

    class _iLocIndexer:
        def __init__(self, parent) -> None:
            self.parent = parent

        def __getitem__(self, key):
            if key not in self.parent.values.index:
                raise ValueError(f"Time {key} not in trajectory time index")
            return self.parent.values[key]

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"Trajectory(name={self.name}, length={len(self)})"
