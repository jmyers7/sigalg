from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from itertools import product
from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd

from ..probability_measures import ProbabilityMeasure
from ..random_objects import RandomVariable
from ..spaces import SampleSpace, SampleSpaceMethods

if TYPE_CHECKING:
    from .featurized_probability_space import FeaturizedProbabilitySpace
    from .sample_point_features import SamplePointFeatures


class FeaturizedSampleSpace(SampleSpaceMethods):

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        features,
        sample_space: SampleSpace = None,
        feature_index: list[Hashable] = None,
        overwrite_default_sample_space: bool = True,
        overwrite_default_feature_index: bool = True,
        initial_sample_index: int = 0,
        initial_feature_index: int = 0,
        sample_prefix: str = "omega",
        feature_prefix: str = "X",
        dtype=None,
    ) -> None:
        self._values = pd.DataFrame(data=features, dtype=dtype).copy()
        self._validate_parameters(self._values, sample_space, feature_index)
        n_rows = len(self._values)
        n_cols = len(self._values.columns)

        is_default_feature_index = self._values.columns.equals(
            pd.RangeIndex(start=0, stop=n_cols)
        )
        if feature_index is None:
            if n_cols == 1:
                feature_index = [f"{feature_prefix}"]
            else:
                feature_index = [
                    f"{feature_prefix}{i + initial_feature_index}"
                    for i in range(n_cols)
                ]
        if is_default_feature_index and overwrite_default_feature_index:
            self._values.columns = feature_index

        is_default_sample_space = self._values.index.equals(
            pd.RangeIndex(start=0, stop=n_rows)
        )
        if sample_space is not None:
            self._values.index = sample_space.values
            self._sample_space = sample_space
        else:
            if is_default_sample_space and overwrite_default_sample_space:
                if n_rows == 1:
                    sample_space = SampleSpace([f"{sample_prefix}"])
                else:
                    indices = [
                        f"{sample_prefix}{i + initial_sample_index}"
                        for i in range(n_rows)
                    ]
                    sample_space = SampleSpace(indices)
                self._values.index = sample_space.values
                self._sample_space = sample_space
            else:
                self._sample_space = SampleSpace(list(self._values.index))

    # --------------------- properties --------------------- #

    @property
    def sample_space(self) -> SampleSpace:
        return self._sample_space

    @property
    def features(self) -> pd.DataFrame:
        return self._values.copy()

    @property
    def values(self) -> pd.DataFrame:
        return self._values.copy()

    @property
    def n_samples(self) -> int:
        return len(self._values)

    @property
    def n_features(self) -> int:
        return len(self._values.columns)

    @property
    def feature_index(self) -> pd.Index:
        return self._values.columns

    @property
    def shape(self) -> tuple[int, int]:
        return self._values.shape

    # --------------------- data access & iter methods --------------------- #

    def get_sample_features(self, sample_index: Hashable):
        from .sample_point_features import SamplePointFeatures

        if sample_index not in self._values.index:
            raise ValueError(
                f"Sample index {sample_index} not found in featurized_sample_space."
            )
        return SamplePointFeatures(
            features=self._values.loc[sample_index], sample_index=sample_index
        )

    def get_event_features(self, event_indices: list[Hashable]):
        from .featurized_event import FeaturizedEvent

        for idx in event_indices:
            if idx not in self._values.index:
                raise ValueError(
                    f"Sample index {idx} not found in featurized_sample_space."
                )
        return FeaturizedEvent(
            featurized_sample_space=self,
            event_indices=event_indices,
        )

    def get_feature_rv(self, feature_index: Hashable) -> RandomVariable:
        values = self._values[feature_index]
        name = values.name
        return RandomVariable.from_values(
            domain=self.sample_space, values=values, name=name
        )

    def get_sub_features(self, feature_indices: list[Hashable]):
        features = self._values[feature_indices]
        return FeaturizedSampleSpace(features=features)

    def iter_sample_features(self):
        for sample_index in self.values.index:
            yield sample_index, self.get_sample_features(sample_index)

    @property
    def get_sample_features_at(self):
        return self._iLocIndexer(self)

    @property
    def get_event_features_at(self):
        return self._iLocIndexer(self)

    class _iLocIndexer:
        def __init__(self, parent) -> None:
            self.parent = parent

        def __getitem__(self, key: int | slice | list[int]):
            from .featurized_event import FeaturizedEvent
            from .sample_point_features import SamplePointFeatures

            features = self.parent._values.iloc[key]
            if isinstance(key, list) or isinstance(key, slice):
                if isinstance(self.parent, FeaturizedEvent):
                    sample_space_features = self.parent.sample_space_features
                else:
                    sample_space_features = self.parent

                return FeaturizedEvent(
                    featurized_sample_space=sample_space_features,
                    event_indices=features.index.tolist(),
                )
            else:
                return SamplePointFeatures(features=features)

    # --------------------- apply methods --------------------- #

    def apply_to_features(
        self, function: Callable[[SamplePointFeatures], any]
    ) -> pd.Series:
        from .sample_point_features import SamplePointFeatures

        def wrapper(row):
            sp = SamplePointFeatures(features=row)
            return function(sp)

        return self._values.apply(wrapper, axis=1)

    # --------------------- class methods --------------------- #

    @classmethod
    def from_sequences(
        cls,
        state_space: Iterable[Hashable],
        sequence_length: int,
        feature_index=None,
        initial_sample_index: int = 0,
        initial_feature_index: int = 0,
        sample_prefix: str = "omega",
        feature_prefix: str = "X",
        threshold: int = 1000,
    ) -> FeaturizedSampleSpace:
        if not isinstance(state_space, Iterable):
            raise TypeError("state_space must be an iterable")
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
        return cls(
            features=sequences,
            sample_prefix=sample_prefix,
            feature_index=feature_index,
            feature_prefix=feature_prefix,
            initial_sample_index=initial_sample_index,
            initial_feature_index=initial_feature_index,
        )

    # --------------------- probability methods --------------------- #

    def add_probability_measure_from_features(
        self, pmf: Callable[[SamplePointFeatures], Real]
    ) -> FeaturizedProbabilitySpace:
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
            probability_space=probability_space,
            featurized_sample_space=self,
        )

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        data: pd.DataFrame,
        sample_space: SampleSpace | None,
        feature_names: list[Hashable] | None,
    ):
        if data.empty:
            raise ValueError("features cannot be empty")

        if sample_space is not None:
            if not isinstance(sample_space, SampleSpace):
                raise TypeError("sample_space must be a SampleSpace instance")
            if len(data) != len(sample_space):
                raise ValueError(
                    f"Number of feature rows ({len(data)}) must match "
                    f"the size of the sample_space ({len(sample_space)})"
                )

        if feature_names is not None:
            if not isinstance(feature_names, list):
                raise TypeError("feature_names must be a list")
            if len(data.columns) != len(feature_names):
                raise ValueError(
                    f"Number of feature columns ({len(data.columns)}) must match "
                    f"the length of feature_names ({len(feature_names)})"
                )


class FeaturizedSampleSpaceMethods(SampleSpaceMethods):
    @property
    def features(self) -> pd.DataFrame:
        return self.featurized_sample_space.values

    @property
    def values(self) -> pd.DataFrame:
        return self.featurized_sample_space.values

    def get_sample_features(self, sample_index: Hashable):
        return self.featurized_sample_space.get_sample_features(sample_index)

    def get_event_features(self, event_indices: list[Hashable]):
        return self.featurized_sample_space.get_event_features(event_indices)

    @property
    def get_sample_features_at(self):
        return self.featurized_sample_space._iLocIndexer(self.featurized_sample_space)

    @property
    def get_event_features_at(self):
        return self.featurized_sample_space._iLocIndexer(self.featurized_sample_space)

    def get_feature_rv(self, feature_index: Hashable) -> RandomVariable:
        return self.featurized_sample_space.get_feature_rv(feature_index)

    def get_sub_features(self, feature_indices: list[Hashable]):
        return self.featurized_sample_space.get_sub_features(feature_indices)

    def apply_to_features(
        self, function: Callable[[SamplePointFeatures], any]
    ) -> pd.Series:
        return self.featurized_sample_space.apply_to_features(function)
