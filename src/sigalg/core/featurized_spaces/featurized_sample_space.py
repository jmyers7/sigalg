from collections.abc import Callable, Hashable, Iterable
from itertools import product
from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..random_objects.random_variable import RandomVariable
    from ..spaces.sample_space import SampleSpace
    from .feature_embedding import FeatureEmbedding
    from .featurized_event import FeaturizedEvent
    from .featurized_probability_space import FeaturizedProbabilitySpace
    from .sample_point_features import SamplePointFeatures


class FeaturizedSampleSpace:

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        sample_space: SampleSpace,
        feature_embedding: FeatureEmbedding,
    ) -> None:
        self._validate_parameters(sample_space, feature_embedding)
        self._sample_space = sample_space
        self._feature_embedding = feature_embedding

    # --------------------- properties --------------------- #

    @property
    def feature_embedding(self) -> FeatureEmbedding:
        return self._feature_embedding

    @property
    def sample_space(self) -> SampleSpace:
        return self._sample_space

    # --------------------- setter methods --------------------- #

    def set_feature_embedding(self, feature_embedding: FeatureEmbedding) -> None:
        self._validate_parameters(self.sample_space, feature_embedding)
        self._feature_embedding = feature_embedding

    def set_sample_space(self, sample_space: SampleSpace) -> None:
        self._validate_parameters(sample_space, self.feature_embedding)
        self._sample_space = sample_space

    # --------------------- class methods --------------------- #

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        embedding_name: str = "X",
        overwrite_default_sample_index: bool = True,
        overwrite_default_feature_index: bool = True,
        initial_sample_index: int = 0,
        initial_feature_index: int = 0,
        sample_prefix: str = "omega",
        sample_space_name: str = "Omega",
    ):
        from .feature_embedding import FeatureEmbedding

        n_rows = len(df)
        n_cols = len(df.columns)

        df.columns = cls._generate_feature_index(
            current_index=df.columns,
            n_cols=n_cols,
            overwrite_default=overwrite_default_feature_index,
            initial_index=initial_feature_index,
            name=embedding_name,
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
        feature_embedding = FeatureEmbedding(features=df, name=embedding_name)

        return cls(sample_space=sample_space, feature_embedding=feature_embedding)

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
        state_space: Iterable[Hashable],
        sequence_length: int,
        embedding_name: str = "X",
        sample_space_name: str = "Omega",
        feature_index: list[Hashable] | None = None,
        initial_sample_index: int = 0,
        initial_feature_index: int = 0,
        sample_prefix: str = "omega",
        threshold: int = 1000,
    ) -> "FeaturizedSampleSpace":
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
        df = pd.DataFrame(sequences)

        return cls.from_df(
            df=df,
            embedding_name=embedding_name,
            sample_space_name=sample_space_name,
            overwrite_default_sample_index=True,
            overwrite_default_feature_index=(feature_index is None),
            initial_sample_index=initial_sample_index,
            initial_feature_index=initial_feature_index,
            sample_prefix=sample_prefix,
        )

    # --------------------- data access methods --------------------- #

    def get_sample_features(self, sample_index: Hashable) -> SamplePointFeatures:
        from .sample_point_features import SamplePointFeatures

        if sample_index not in self.sample_space:
            raise ValueError(f"Sample index {sample_index} not found in sample_space.")
        return SamplePointFeatures.from_fss(
            sample_index=sample_index,
            fss=self,
        )

    def get_event_features(
        self, event_indices: list[Hashable], name: str = "A"
    ) -> FeaturizedEvent:
        from .featurized_event import FeaturizedEvent

        for idx in event_indices:
            if idx not in self.sample_space:
                raise ValueError(f"Sample index {idx} not found in sample_space.")
        return FeaturizedEvent.from_indices(
            fss=self,
            event_indices=event_indices,
            event_name=name,
        )

    def get_feature_rv(self, feature_index: Hashable) -> RandomVariable:
        from ..random_objects.random_variable import RandomVariable

        values = self.feature_embedding.values[feature_index]
        name = values.name
        return RandomVariable.from_values(
            domain=self.sample_space, values=values, name=name
        )

    def get_sub_features(
        self, feature_indices: list[Hashable]
    ) -> "FeaturizedSampleSpace":
        from .feature_embedding import FeatureEmbedding

        df = self.feature_embedding.values[feature_indices]
        feature_embedding = FeatureEmbedding(
            features=df, name=self.feature_embedding.name + "_sub"
        )
        return FeaturizedSampleSpace(
            feature_embedding=feature_embedding, sample_space=self.sample_space
        )

    def iter_sample_features(self):
        for sample_index in self.values.index:
            yield sample_index, self.get_sample_features(sample_index)

    @property
    def get_sample_features_at(self) -> "_SampleFeaturesIndexer":
        return self._SampleFeaturesIndexer(self)

    class _SampleFeaturesIndexer:
        def __init__(self, fss) -> None:
            self.fss = fss

        def __getitem__(self, key: int) -> SamplePointFeatures:
            from .sample_point_features import SamplePointFeatures

            features = self.fss.feature_embedding.values.iloc[key]
            return SamplePointFeatures.from_fss(
                sample_index=features.name, fss=self.fss
            )

    @property
    def get_event_features_at(self) -> "_EventIndexer":
        return self._EventIndexer(self)

    class _EventIndexer:
        def __init__(self, fss) -> None:
            self.fss = fss

        def __getitem__(self, key) -> FeaturizedEvent:
            from .featurized_event import FeaturizedEvent

            if isinstance(key, tuple) and len(key) == 2:
                index_key, name = key
            else:
                index_key = key
                name = "A"

            event = self.fss.sample_space.get_event_at[index_key, name]
            event_indices = event.values.to_list()
            return FeaturizedEvent.from_indices(
                fss=self.fss, event_indices=event_indices, event_name=name
            )

    # --------------------- apply methods --------------------- #

    def apply_to_features(
        self, function: Callable[[SamplePointFeatures], any]
    ) -> pd.Series:
        from .sample_point_features import SamplePointFeatures

        def wrapper(row):
            sp = SamplePointFeatures(name=row.name, features=row)
            return function(sp)

        return self.feature_embedding.values.apply(wrapper, axis=1)

    # --------------------- equality --------------------- #

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeaturizedSampleSpace):
            return False
        return (
            self.sample_space == other.sample_space
            and self.feature_embedding == other.feature_embedding
        )

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return (
            f"FeaturizedSampleSpace("
            f"sample_space={self.sample_space.name}, "
            f"feature_embedding={self.feature_embedding.name})"
        )

    def __str__(self) -> str:
        header = (
            "Featurized sample space ("
            f"{self.sample_space.name}, "
            f"{self.feature_embedding.name})"
        )
        separator = "=" * len(header)
        return (
            header
            + "\n"
            + separator
            + "\n\n* "
            + repr(self.sample_space)
            + "\n\n* "
            + repr(self.feature_embedding)
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
            for sample_index, sample_features in self.feature_embedding.iter_sample_features()
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
            feature_embedding=self.feature_embedding,
        )

    # --------------------- validation --------------------- #

    @staticmethod
    def _validate_parameters(
        sample_space: SampleSpace,
        feature_embedding: FeatureEmbedding,
    ) -> None:
        from ..spaces.sample_space import SampleSpace
        from .feature_embedding import FeatureEmbedding

        if not isinstance(feature_embedding, FeatureEmbedding):
            raise TypeError("embedding must be a FeatureEmbedding instance.")
        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")
        if not feature_embedding.values.index.equals(sample_space.values):
            raise ValueError(
                "The indices of embedding must match the values of sample_space."
            )


class FeaturizedSampleSpaceMethods:
    @property
    def feature_embedding(self) -> FeatureEmbedding:
        return self.featurized_sample_space.feature_embedding

    @property
    def sample_space(self) -> SampleSpace:
        return self.featurized_sample_space.sample_space

    def get_sample_features(self, sample_index: Hashable) -> SamplePointFeatures:
        return self.featurized_sample_space.get_sample_features(sample_index)

    def get_event_features(self, event_indices: list[Hashable]) -> FeaturizedEvent:
        return self.featurized_sample_space.get_event_features(event_indices)

    @property
    def get_sample_features_at(self):
        return self.featurized_sample_space._SampleFeaturesIndexer(
            self.featurized_sample_space
        )

    @property
    def get_event_features_at(self):
        return self.featurized_sample_space._EventIndexer(self.featurized_sample_space)

    def get_feature_rv(self, feature_index: Hashable) -> RandomVariable:
        return self.featurized_sample_space.get_feature_rv(feature_index)

    def get_sub_features(self, feature_indices: list[Hashable]):
        return self.featurized_sample_space.get_sub_features(feature_indices)

    def apply_to_features(
        self, function: Callable[[SamplePointFeatures], any]
    ) -> pd.Series:
        return self.featurized_sample_space.apply_to_features(function)
