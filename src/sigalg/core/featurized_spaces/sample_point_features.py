from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import pandas as pd


class SamplePointFeatures:

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        features: pd.Series | list | dict[Hashable, Any] = None,
        sample_index: Hashable = "omega",
        feature_index: list[Hashable] = None,
        overwrite_default_sample_index: bool = True,
        overwrite_default_feature_index: bool = True,
        initial_feature_index: int = 0,
        feature_prefix: str = "X",
        dtype=None,
    ) -> None:
        self._values = pd.Series(data=features, dtype=dtype).copy()
        n_features = len(self._values)

        is_default_feature_index = self._values.index.equals(
            pd.RangeIndex(start=0, stop=n_features)
        )
        if feature_index is None:
            if n_features == 1:
                feature_index = [f"{feature_prefix}"]
            else:
                feature_index = [
                    f"{feature_prefix}{i + initial_feature_index}"
                    for i in range(n_features)
                ]
        if is_default_feature_index and overwrite_default_feature_index:
            self._values.index = feature_index

        is_default_sample_index = self._values.name is None
        if is_default_sample_index and overwrite_default_sample_index:
            self._values.name = sample_index

    # --------------------- properties --------------------- #

    @property
    def sample_index(self) -> Hashable:
        return self._values.name

    @property
    def feature_index(self) -> pd.Index:
        return self._values.index

    @property
    def features(self) -> pd.Series:
        return self._values.copy()

    @property
    def values(self) -> pd.Series:
        return self._values.copy()

    # --------------------- access & iter methods --------------------- #

    @property
    def feature_at(self):
        return self._iLocIndexer(self)

    class _iLocIndexer:
        def __init__(self, parent) -> None:
            self.parent = parent

        def __getitem__(self, key: int | slice | list[int]):
            return self.parent._values.iloc[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def sum(self) -> Any:
        return self.values.sum()

    # --------------------- conversion methods --------------------- #

    def to_dict(self) -> dict[Hashable, any]:
        return self._values.to_dict()

    def to_list(self) -> list:
        return self._values.tolist()

    def to_pandas(self) -> pd.Series:
        return self._values.copy()

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        series_repr = repr(self._values)
        lines = series_repr.split("\n")
        data_lines = [
            line
            for line in lines
            if not line.startswith(("Name:", "Length:", "dtype:"))
        ]
        data_str = "\n".join(data_lines)
        return (
            f"Sample point features '{self.sample_index}'\n"
            f"Number of features: {len(self)}\n\n"
            f"{data_str}"
        )
