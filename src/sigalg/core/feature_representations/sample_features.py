from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import pandas as pd

from .array_like import ArrayLike


class SampleFeatures(ArrayLike):

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
        self._data = pd.Series(data=features, dtype=dtype).copy()
        n_features = len(self._data)

        is_default_feature_index = self._data.index.equals(
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
            self._data.index = feature_index

        is_default_sample_index = self._data.name is None
        if is_default_sample_index and overwrite_default_sample_index:
            self._data.name = sample_index

    # --------------------- properties --------------------- #

    @property
    def sample_index(self) -> Hashable:
        return self._data.name

    @property
    def feature_index(self) -> pd.Index:
        return self._data.index

    # --------------------- access methods --------------------- #

    class _iLocIndexer:
        def __init__(self, parent: SampleFeatures) -> None:
            self.parent = parent

        def __getitem__(self, key: int | slice | list[int]):
            return self.parent._data.iloc[key]

    @property
    def feature_at(self) -> _iLocIndexer:
        return self._iLocIndexer(self)

    # --------------------- conversion methods --------------------- #

    def to_dict(self) -> dict[Hashable, any]:
        return self._data.to_dict()

    def to_list(self) -> list:
        return self._data.tolist()

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"SampleFeatures(sample='{self.sample_index}',\n{self._data})"
