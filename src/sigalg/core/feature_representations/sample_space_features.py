from itertools import product

import pandas as pd

from ..spaces import ProbabilitySpace, SampleSpace
from .array_like import ArrayLike


class SampleSpaceFeatures(ArrayLike):
    def __init__(
        self,
        features,
        sample_space: SampleSpace = None,
        feature_index=None,
        overwrite_default_sample_space: bool = True,
        overwrite_default_rv_index: bool = True,
        initial_sample_index: int = 0,
        initial_feature_index: int = 0,
        sample_prefix: str = "omega",
        feature_prefix: str = "X",
        dtype=None,
    ):
        self._validate_parameters(features, sample_space, feature_index)
        self._data = pd.DataFrame(data=features, dtype=dtype)
        n_rows = len(self._data)
        n_cols = len(self._data.columns)

        is_default_feature_index = self._data.columns.equals(
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
        if is_default_feature_index and overwrite_default_rv_index:
            self._data.columns = feature_index

        is_default_sample_space = self._data.index.equals(
            pd.RangeIndex(start=0, stop=n_rows)
        )
        if sample_space is not None:
            self._data.index = sample_space.index
            self._sample_space = sample_space.sample_space
            if isinstance(sample_space, ProbabilitySpace):
                self._probability_space = sample_space
            else:
                self._probability_space = None
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
                self._data.index = sample_space.index
                self._sample_space = sample_space
            else:
                self._sample_space = SampleSpace(self._data.index.tolist())

    class _iLocIndexer:
        def __init__(self, parent):
            self.parent = parent

        def __getitem__(self, key):
            from .sample_features import SampleFeatures

            result = self.parent._data.iloc[key]
            return SampleFeatures(features=result)

    @property
    def sample_space(self):
        return self._sample_space

    @property
    def probability_space(self):
        return self._probability_space

    @property
    def get_sample_features_at(self):
        return self._iLocIndexer(self)

    @staticmethod
    def _validate_parameters(features, sample_space, feature_index):
        if sample_space is not None and not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance")
        if sample_space is not None and len(features) != len(sample_space):
            raise ValueError(
                "Number of feature rows must match the size of the sample_space"
            )
        if feature_index is not None and len(features[0]) != len(feature_index):
            raise ValueError(
                "Number of feature columns must match the length of feature_index"
            )

    @classmethod
    def from_sequences(
        cls,
        state_space: list,
        sequence_length: int,
        initial_sample_index: int = 0,
        initial_feature_index: int = 0,
        sample_prefix: str = "omega",
        feature_prefix: str = "X",
        threshold: int = 1000,
    ):
        if not isinstance(state_space, list) or len(state_space) == 0:
            raise ValueError("state_space must be a non-empty list")
        sample_space_cardinality = len(state_space) ** sequence_length

        if sample_space_cardinality > threshold:
            raise ValueError(
                f"Sample space size {sample_space_cardinality} exceeds threshold of {threshold}. "
            )

        sequences = list(product(state_space, repeat=sequence_length))
        return cls(
            features=sequences,
            sample_prefix=sample_prefix,
            feature_prefix=feature_prefix,
            initial_sample_index=initial_sample_index,
            initial_feature_index=initial_feature_index,
        )
