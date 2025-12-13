from __future__ import annotations

from collections.abc import Callable, Hashable
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..base.sample_space import SampleSpaceMethods

if TYPE_CHECKING:
    from ..base.index import Index
    from ..random_objects.random_variable import RandomVariable
    from .featurized_probability_space import FeaturizedProbabilitySpace
    from .sample_point_features import SamplePointFeatures


class FeatureEmbedding(SampleSpaceMethods):

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        random_variables: list[RandomVariable] | None = None,
        feature_index: Index | None = None,
        values: pd.DataFrame | None = None,
        domain_name: str | None = None,
        name: str = "X",
    ) -> None:
        from ..base.feature_index import FeatureIndex
        from ..base.sample_space import SampleSpace

        self._validate_parameters(
            random_variables=random_variables,
            feature_index=feature_index,
            values=values,
            domain_name=domain_name,
            name=name,
        )

        if values is not None:
            self._values = values
            self._random_variables = None
            if domain_name is None:
                domain_name = "Omega"
            self.domain_name = domain_name
            self.domain = SampleSpace(
                indices=self.values.index.to_list(),
                name=domain_name,
                values_name=self.values.index.name,
            )
            self.feature_index = FeatureIndex(
                indices=values.columns.to_list(), values_name=values.columns.name
            )
        elif random_variables is not None:
            self._values = None
            self._random_variables = random_variables
            self.domain = random_variables[0].domain
            if domain_name is None:
                self.domain_name = self.domain.name
            else:
                self.domain_name = domain_name
                self.domain.name = domain_name
            if feature_index is None:
                self.feature_index = FeatureIndex(
                    indices=[rv.name for rv in random_variables]
                )
            else:
                self.feature_index = feature_index
                for pos, rv in enumerate(self.random_variables):
                    rv.name = str(self.feature_index.values[pos])
        self._name = name

    # --------------------- properties --------------------- #

    @property
    def values(self) -> pd.DataFrame:
        if self._values is None:
            self._values = pd.concat(
                [rv.values for rv in self.random_variables], axis=1
            )
            self._values.columns = self.feature_index
            self._values.columns.name = self.feature_index.values.name
        return self._values

    @property
    def random_variables(self) -> list[RandomVariable]:
        from ..random_objects.random_variable import RandomVariable

        if self._random_variables is None:
            self._random_variables = [
                RandomVariable(
                    outputs=self.values[col].to_dict(),
                    domain=self.domain,
                    name=str(col),
                )
                for col in self.values.columns
            ]
        return self._random_variables

    # @property
    # def domain(self) -> SampleSpace:
    #     from ..base.sample_space import SampleSpace

    #     if self._domain is None:
    #         self._domain = SampleSpace(
    #             indices=self.values.index.to_list(), values_name=self.values.index.name
    #         )
    #     return self._domain

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
        return self.values.shape

    def __len__(self) -> int:
        return len(self.values)

    # --------------------- factory methods --------------------- #

    @classmethod
    def from_numpy(cls, array: np.ndarray, name: str = "X") -> FeatureEmbedding:
        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a numpy ndarray.")
        values = pd.DataFrame(array)
        values.index.name = "sample"
        values.columns.name = "feature"
        return cls(values=values, name=name)

    # --------------------- data access methods --------------------- #

    def get_sample_features(self, sample_index: Hashable) -> SamplePointFeatures:
        from .sample_point_features import SamplePointFeatures

        if sample_index not in self.domain:
            raise ValueError(f"Sample index {sample_index} not found in domain.")
        return SamplePointFeatures.from_feature_embedding(
            sample_index=sample_index,
            feature_embedding=self,
        )

    def get_event_features(
        self, event_indices: list[Hashable], name: str = "A"
    ) -> FeatureEmbedding:

        for idx in event_indices:
            if idx not in self.domain:
                raise ValueError(f"Sample index {idx} not found in sample_space.")

        event_features = FeatureEmbedding(
            values=self.values.loc[event_indices], name=self.name
        )
        event_features.domain.name = name
        return event_features

    def get_feature_rv(self, key: Hashable) -> RandomVariable:
        idx_pos = self.feature_index.values.get_loc(key)
        return self.random_variables[idx_pos]

    def get_sub_features(self, feature_indices: list[Hashable]) -> FeatureEmbedding:
        values = self.values[feature_indices]
        return FeatureEmbedding(values=values, name=self.name + "_sub")

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
            self.domain == other.domain
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
            sample_space=self.domain, probabilities=probabilities
        )
        probability_space = ProbabilitySpace(
            sample_space=self.domain,
            probability_measure=probability_measure,
        )
        return FeaturizedProbabilitySpace(
            sample_space=self.domain,
            sigma_algebra=probability_space.sigma_algebra,
            probability_measure=probability_measure,
            feature_embedding=self,
        )

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        random_variables: list[RandomVariable],
        feature_index: Index | None,
        values: pd.DataFrame | None = None,
        domain_name: str | None = None,
        name: str | None = None,
    ) -> None:
        from ..base.index import Index
        from ..random_objects.random_variable import RandomVariable

        if (
            random_variables is not None or feature_index is not None
        ) and values is not None:
            raise ValueError(
                "Cannot specify both random_variables/feature_index and values."
            )
        if random_variables is None and values is None:
            raise ValueError("Must specify either random_variables or values.")
        if feature_index is not None and not isinstance(feature_index, Index):
            raise TypeError("feature_index must be an Index instance.")
        if random_variables is not None:
            if not isinstance(random_variables, list):
                raise TypeError(
                    "random_variables must be a list of RandomVariable instances."
                )
            if not all(isinstance(rv, RandomVariable) for rv in random_variables):
                raise TypeError(
                    "All elements in random_variables must be instances of RandomVariable."
                )
            if feature_index is not None and len(feature_index) != len(
                random_variables
            ):
                raise ValueError(
                    "feature_index and random_variables must have the same length."
                )
        if values is not None and not isinstance(values, pd.DataFrame):
            raise TypeError("values must be a pandas DataFrame.")
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if domain_name is not None and not isinstance(domain_name, str):
            raise TypeError("domain_name must be a string.")


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
