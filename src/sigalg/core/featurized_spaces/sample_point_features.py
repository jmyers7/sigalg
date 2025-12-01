from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .featurized_sample_space import FeaturizedSampleSpace


class SamplePointFeatures:

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        name: Hashable,
        features: pd.Series,
    ) -> None:
        self._validate_parameters(name, features)
        self._values = features.copy()
        self._name = name
        self._fss = None

    # --------------------- properties --------------------- #

    @property
    def values(self) -> pd.Series:
        return self._values.copy()

    @property
    def name(self) -> Hashable:
        return self._name

    @property
    def fss(self) -> FeaturizedSampleSpace | None:
        return self._fss

    # --------------------- access & iter methods --------------------- #

    @property
    def feature_at(self) -> _iLocIndexer:
        return self._iLocIndexer(self)

    class _iLocIndexer:
        def __init__(self, parent) -> None:
            self.parent = parent

        def __getitem__(self, key: int | slice | list[int]):
            return self.parent._values.iloc[key]

    def __iter__(self) -> iter:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def sum(self) -> Any:
        return self.values.sum()

    # --------------------- class methods --------------------- #

    @classmethod
    def from_fss(
        cls,
        sample_index: Hashable,
        fss: FeaturizedSampleSpace,
    ) -> SamplePointFeatures:
        features = fss.feature_embedding.values.loc[sample_index]
        spf = cls(name=sample_index, features=features)
        spf._fss = fss
        return spf

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"Sample features of {self.name}:\n{self.values.to_frame()}"

    # --------------------- equality --------------------- #

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SamplePointFeatures):
            return False
        return self.name == other.name and self.values.equals(other.values)

    # --------------------- validation methods --------------------- #

    def _validate_parameters(
        self,
        name: Hashable,
        features: pd.Series,
    ) -> None:
        if not isinstance(name, Hashable):
            raise TypeError("name must be hashable.")
        if not isinstance(features, pd.Series):
            raise TypeError("features must be a pandas Series.")
        if features.name != name:
            raise ValueError("features.name must match the given name.")
