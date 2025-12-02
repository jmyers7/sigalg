from __future__ import annotations

from collections.abc import Callable, Hashable
from itertools import product
from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd

from ..spaces.sample_space import SampleSpaceMethods

if TYPE_CHECKING:
    from ..random_objects.random_variable import RandomVariable
    from ..spaces.sample_space import SampleSpace
    from .featurized_probability_space import FeaturizedProbabilitySpace
    from .sample_point_features import SamplePointFeatures


class FeatureEmbedding(SampleSpaceMethods):

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        sample_space: SampleSpace,
        values: pd.DataFrame,
        name: str = "E",
    ) -> None:
        self._validate_parameters(sample_space=sample_space, values=values)
        self._sample_space = sample_space
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
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        self._name = name

    # --------------------- class methods --------------------- #

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        name: str = "E",
        overwrite_default_sample_index: bool = True,
        overwrite_default_feature_index: bool = True,
        initial_sample_index: int = 0,
        initial_feature_index: int = 0,
        sample_prefix: str = "omega",
        sample_space_name: str = "Omega",
    ) -> FeatureEmbedding:
        n_rows = len(df)
        n_cols = len(df.columns)

        df.columns = cls._generate_feature_index(
            current_index=df.columns,
            n_cols=n_cols,
            overwrite_default=overwrite_default_feature_index,
            initial_index=initial_feature_index,
            name=name,
        )

        sample_space = cls._generate_sample_space(
            current_index=df.index,
            n_rows=n_rows,
            overwrite_default=overwrite_default_sample_index,
            initial_index=initial_sample_index,
            sample_prefix=sample_prefix,
            name=sample_space_name,
        )
        df.index = sample_space.values

        return cls(sample_space=sample_space, values=df, name=name)

    @classmethod
    def _generate_feature_index(
        cls,
        current_index: pd.Index,
        n_cols: int,
        overwrite_default: bool,
        initial_index: int,
        name: str,
    ) -> pd.Index:
        if overwrite_default and current_index.equals(
            pd.RangeIndex(start=0, stop=n_cols)
        ):
            if n_cols == 1:
                return pd.Index([name])
            return pd.Index([f"{name}{i + initial_index}" for i in range(n_cols)])
        else:
            return current_index

    @classmethod
    def _generate_sample_space(
        cls,
        current_index: pd.Index,
        n_rows: int,
        overwrite_default: bool,
        initial_index: int,
        sample_prefix: str,
        name: str,
    ) -> SampleSpace:
        from ..spaces.sample_space import SampleSpace

        if overwrite_default and current_index.equals(
            pd.RangeIndex(start=0, stop=n_rows)
        ):
            if n_rows == 1:
                indices = [sample_prefix]
            else:
                indices = [f"{sample_prefix}{i + initial_index}" for i in range(n_rows)]
            return SampleSpace(indices=indices, name=name)
        else:
            return SampleSpace(list(current_index), name=name)

    @classmethod
    def from_sequences(
        cls,
        state_space: list,
        sequence_length: int,
        name: str = "E",
        initial_sample_index: int = 0,
        initial_feature_index: int = 0,
        sample_prefix: str = "omega",
        sample_space_name: str = "Omega",
        threshold: int = 1000,
    ) -> FeatureEmbedding:
        if not isinstance(state_space, list):
            raise TypeError("state_space must be a list.")
        state_space_list = list(state_space)
        if len(state_space_list) == 0:
            raise ValueError("state_space must be non-empty")
        if not isinstance(sequence_length, int) or sequence_length < 1:
            raise ValueError("sequence_length must be a positive integer")
        if not isinstance(threshold, int) or threshold < 1:
            raise ValueError("threshold must be a positive integer")

        sample_space_cardinality = len(state_space_list) ** sequence_length
        if sample_space_cardinality > threshold:
            raise ValueError(
                f"Sample space size {sample_space_cardinality} exceeds threshold of {threshold}."
            )

        sequences = list(product(state_space_list, repeat=sequence_length))
        df = pd.DataFrame(sequences)

        return cls.from_df(
            df=df,
            name=name,
            sample_space_name=sample_space_name,
            overwrite_default_sample_index=True,
            overwrite_default_feature_index=True,
            initial_sample_index=initial_sample_index,
            initial_feature_index=initial_feature_index,
            sample_prefix=sample_prefix,
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
        from ..spaces.event import Event

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
        values = self.values[feature_indices]
        return FeatureEmbedding(
            sample_space=self.sample_space, values=values, name=self.name + "_sub"
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
        return (
            f"FeaturizedSampleSpace("
            f"sample_space={self.sample_space.name}, "
            f"feature_embedding={self.name})"
        )

    def __str__(self) -> str:
        header = (
            "Featurized sample space (" f"{self.sample_space.name}, " f"{self.name})"
        )
        separator = "=" * len(header)
        return (
            header
            + "\n"
            + separator
            + "\n\n* "
            + repr(self.sample_space)
            + "\n\n* "
            + repr(self.values)
        )

    # --------------------- probability methods --------------------- #

    def add_probability_measure_from_features(
        self, pmf: Callable[[SamplePointFeatures], Real]
    ) -> FeaturizedProbabilitySpace:
        from ..probability_measures import ProbabilityMeasure
        from ..spaces import ProbabilitySpace
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
        values: pd.DataFrame,
    ) -> None:
        from ..spaces.sample_space import SampleSpace

        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")
        if not values.index.equals(sample_space.values):
            raise ValueError("The indices of `values` must match sample_space.")


class FeatureEmbeddingMethods:
    @property
    def feature_embedding(self) -> FeatureEmbedding:
        return self.feature_embedding.feature_embedding

    @property
    def sample_space(self) -> SampleSpace:
        return self.feature_embedding.sample_space

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
