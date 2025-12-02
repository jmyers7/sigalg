from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .feature_embedding import FeatureEmbedding


class SamplePointFeatures:

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        values: pd.Series,
        name: Hashable,
    ) -> None:
        self._validate_parameters(values=values, name=name)
        self._values = values.copy()
        self._name = name
        self._feature_embedding = None

    # --------------------- properties --------------------- #

    @property
    def values(self) -> pd.Series:
        return self._values.copy()

    @property
    def name(self) -> Hashable:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        self._name = name

    @property
    def feature_embedding(self) -> FeatureEmbedding | None:
        return self._feature_embedding

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
    def from_feature_embedding(
        cls,
        sample_index: Hashable,
        feature_embedding: FeatureEmbedding,
    ) -> SamplePointFeatures:
        values = feature_embedding.values.loc[sample_index]
        spf = cls(values=values, name=sample_index)
        spf._feature_embedding = feature_embedding
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
        values: pd.Series,
        name: Hashable,
    ) -> None:
        if not isinstance(name, Hashable):
            raise TypeError("name must be a Hashable.")
        if not isinstance(values, pd.Series):
            raise TypeError("values must be a pandas Series.")
        if values.name != name:
            raise ValueError("values.name must match the given name.")
