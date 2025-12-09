from __future__ import annotations

from collections.abc import Callable, Hashable
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..base.sample_space import SampleSpaceMethods

if TYPE_CHECKING:
    from ..base.feature_index import FeatureIndex
    from ..base.sample_space import SampleSpace
    from ..random_objects.random_variable import RandomVariable
    from .featurized_probability_space import FeaturizedProbabilitySpace
    from .sample_point_features import SamplePointFeatures


class FeatureEmbedding(SampleSpaceMethods):

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        sample_space: SampleSpace,
        feature_index: FeatureIndex,
        values: pd.DataFrame,
        name: str = "X",
    ) -> None:
        self._validate_parameters(
            sample_space=sample_space, feature_index=feature_index, values=values
        )
        self._sample_space = sample_space
        self._feature_index = feature_index
        self._values = values.copy()
        self._name = name

    # --------------------- properties --------------------- #

    @property
    def values(self) -> pd.DataFrame:
        return self._values.copy()

    @property
    def sample_space(self) -> SampleSpace:
        return self._sample_space

    @property
    def feature_index(self) -> FeatureIndex:
        return self._feature_index

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        self._name = name

    # --------------------- array methods --------------------- #

    @property
    def shape(self) -> tuple[int, int]:
        return self._values.shape

    def __len__(self) -> int:
        return len(self._values)

    # --------------------- class methods --------------------- #

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        name: str = "X",
        sample_values_name: str = "sample",
        feature_index_name: str = "feature",
    ) -> FeatureEmbedding:
        from ..base.feature_index import FeatureIndex
        from ..base.sample_space import SampleSpace

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        sample_space = SampleSpace(list(df.index), name=f"{name}_sample_space")
        feature_index = FeatureIndex(list(df.columns))
        df.index.name = sample_values_name
        df.columns.name = feature_index_name

        return cls(
            sample_space=sample_space, feature_index=feature_index, values=df, name=name
        )

    @classmethod
    def from_numpy(
        cls,
        array: np.ndarray,
        name: str = "X",
        sample_values_name: str = "sample",
        feature_index_name: str = "feature",
    ) -> FeatureEmbedding:
        from ..base.feature_index import FeatureIndex
        from ..base.sample_space import SampleSpace

        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a numpy ndarray.")
        n_rows, n_cols = array.shape
        sample_space = SampleSpace(list(range(n_rows)), name=f"{name}_sample_space")
        feature_index = FeatureIndex(list(range(n_cols)))

        df = pd.DataFrame(array)
        df.index.name = sample_values_name
        df.columns.name = feature_index_name

        return cls(
            sample_space=sample_space,
            feature_index=feature_index,
            values=df,
            name=name,
        )

    # --------------------- data access methods --------------------- #

    def get_sample_features(self, sample_index: Hashable) -> SamplePointFeatures:
        from .sample_point_features import SamplePointFeatures

        if sample_index not in self.sample_space:
            raise ValueError(f"Sample index {sample_index} not found in sample_space.")
        return SamplePointFeatures.from_feature_embedding(
            sample_index=sample_index,
            feature_embedding=self,
        )

    def get_event_features(
        self, event_indices: list[Hashable], name: str = "A"
    ) -> FeatureEmbedding:
        from ..base.event import Event

        for idx in event_indices:
            if idx not in self.sample_space:
                raise ValueError(f"Sample index {idx} not found in sample_space.")

        event = Event(
            sample_space=self.sample_space,
            event_indices=event_indices,
            name=name,
        )
        event_sample_space = event.to_sample_space()
        event_sample_space.name = name
        event_values = self.values.loc[event_indices].copy()
        event_values.index.name = name

        return FeatureEmbedding(
            sample_space=event_sample_space,
            feature_index=self.feature_index,
            values=event_values,
            name=self.name,
        )

    def get_feature_rv(self, feature_index: Hashable) -> RandomVariable:
        from ..random_objects.random_variable import RandomVariable

        values = self.values[feature_index]
        name = values.name
        return RandomVariable.from_values(
            domain=self.sample_space, values=values, name=name
        )

    def get_sub_features(self, feature_indices: list[Hashable]) -> FeatureEmbedding:
        from ..base.feature_index import FeatureIndex

        values = self.values[feature_indices]
        sub_feature_index = FeatureIndex(
            list(feature_indices), values_name=self.feature_index.values.name
        )
        return FeatureEmbedding(
            sample_space=self.sample_space,
            feature_index=sub_feature_index,
            values=values,
            name=self.name + "_sub",
        )

    def iter_sample_features(self):
        for sample_index in self.values.index:
            yield sample_index, self.get_sample_features(sample_index)

    @property
    def get_sample_features_at(self):
        return self._SampleFeaturesIndexer(self)

    class _SampleFeaturesIndexer:
        def __init__(self, feature_embedding) -> None:
            self.feature_embedding = feature_embedding

        def __getitem__(self, key: int) -> SamplePointFeatures:
            from .sample_point_features import SamplePointFeatures

            features = self.feature_embedding.values.iloc[key]
            return SamplePointFeatures.from_feature_embedding(
                sample_index=features.name, feature_embedding=self.feature_embedding
            )

    @property
    def get_event_features_at(self):
        return self._EventIndexer(self)

    class _EventIndexer:
        def __init__(self, feature_embedding) -> None:
            self.feature_embedding = feature_embedding

        def __getitem__(self, key) -> FeatureEmbedding:
            if isinstance(key, tuple) and len(key) == 2:
                index_key, name = key
            else:
                index_key = key
                name = "A"

            event = self.feature_embedding.sample_space.get_event_at[index_key, name]
            event_indices = event.values.to_list()
            return self.feature_embedding.get_event_features(
                event_indices=event_indices,
                name=name,
            )

    # --------------------- apply methods --------------------- #

    def apply_to_features(
        self, function: Callable[[SamplePointFeatures], any]
    ) -> pd.Series:
        from .sample_point_features import SamplePointFeatures

        def wrapper(row):
            sp = SamplePointFeatures(name=row.name, values=row)
            return function(sp)

        return self.values.apply(wrapper, axis=1)

    # --------------------- equality --------------------- #

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureEmbedding):
            return False
        return (
            self.sample_space == other.sample_space
            and self.values.equals(other.values)
            and self.name == other.name
        )

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"Feature embedding {self.name}:\n{self.values}"

    # --------------------- probability methods --------------------- #

    def add_probability_measure_from_features(
        self, pmf: Callable[[SamplePointFeatures], Real]
    ) -> FeaturizedProbabilitySpace:
        from ..base import ProbabilitySpace
        from ..probability_measures import ProbabilityMeasure
        from .featurized_probability_space import FeaturizedProbabilitySpace

        probabilities = {
            sample_index: pmf(sample_features)
            for sample_index, sample_features in self.iter_sample_features()
        }
        probability_measure = ProbabilityMeasure(
            sample_space=self.sample_space, probabilities=probabilities
        )
        probability_space = ProbabilitySpace(
            sample_space=self.sample_space,
            probability_measure=probability_measure,
        )
        return FeaturizedProbabilitySpace(
            sample_space=self.sample_space,
            sigma_algebra=probability_space.sigma_algebra,
            probability_measure=probability_measure,
            feature_embedding=self,
        )

    # --------------------- validation --------------------- #

    @staticmethod
    def _validate_parameters(
        sample_space: SampleSpace,
        feature_index: FeatureIndex,
        values: pd.DataFrame,
    ) -> None:
        from ..base.feature_index import FeatureIndex
        from ..base.sample_space import SampleSpace

        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")
        if not isinstance(feature_index, FeatureIndex):
            raise TypeError("feature_index must be a FeatureIndex instance.")
        if not isinstance(values, pd.DataFrame):
            raise TypeError("values must be a pandas DataFrame.")
        if not values.index.equals(sample_space.values):
            raise ValueError("The indices of `values` must match sample_space.")
        if not values.columns.equals(feature_index.values):
            raise ValueError("The columns of `values` must match feature_index.")


class FeatureEmbeddingMethods:

    def get_sample_features(self, sample_index: Hashable) -> SamplePointFeatures:
        return self.feature_embedding.get_sample_features(sample_index)

    def get_event_features(self, event_indices: list[Hashable]) -> FeatureEmbedding:
        return self.feature_embedding.get_event_features(event_indices)

    @property
    def get_sample_features_at(self):
        return self.feature_embedding._SampleFeaturesIndexer(self.feature_embedding)

    @property
    def get_event_features_at(self):
        return self.feature_embedding._EventIndexer(self.feature_embedding)

    def get_feature_rv(self, feature_index: Hashable) -> RandomVariable:
        return self.feature_embedding.get_feature_rv(feature_index)

    def get_sub_features(self, feature_indices: list[Hashable]):
        return self.feature_embedding.get_sub_features(feature_indices)

    def apply_to_features(
        self, function: Callable[[SamplePointFeatures], any]
    ) -> pd.Series:
        return self.feature_embedding.apply_to_features(function)

    def iter_sample_features(self):
        return self.feature_embedding.iter_sample_features()
