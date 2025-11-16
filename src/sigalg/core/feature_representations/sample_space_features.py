from __future__ import annotations

from collections.abc import Hashable, Iterable
from itertools import product
from typing import TYPE_CHECKING

import pandas as pd

from ..spaces import SampleSpace
from .array_like import ArrayLike

if TYPE_CHECKING:
    from .event_features import EventFeatures
    from .sample_features import SampleFeatures


class SampleSpaceFeatures(ArrayLike):

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
        self._data = pd.DataFrame(data=features, dtype=dtype).copy()
        self._validate_parameters(self._data, sample_space, feature_index)
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
        if is_default_feature_index and overwrite_default_feature_index:
            self._data.columns = feature_index

        is_default_sample_space = self._data.index.equals(
            pd.RangeIndex(start=0, stop=n_rows)
        )
        if sample_space is not None:
            self._data.index = sample_space.index
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
                self._data.index = sample_space.index
                self._sample_space = sample_space
            else:
                self._sample_space = SampleSpace(list(self._data.index))

    # --------------------- properties --------------------- #

    @property
    def sample_space(self) -> SampleSpace:
        return self._sample_space

    # --------------------- access methods --------------------- #

    class _iLocIndexer:
        def __init__(self, parent: SampleSpaceFeatures) -> None:
            self.parent = parent

        def __getitem__(
            self, key: int | slice | list[int]
        ) -> SampleFeatures | EventFeatures:
            from .event_features import EventFeatures
            from .sample_features import SampleFeatures

            result = self.parent._data.iloc[key]
            if isinstance(key, int):
                return SampleFeatures(features=result)
            elif isinstance(key, slice) or (
                isinstance(key, list) and all(isinstance(k, int) for k in key)
            ):
                indices = list(result.index)
                return EventFeatures(
                    sample_space=self.parent.sample_space,
                    feature_data=result,
                    event_indices=indices,
                )
            else:
                raise TypeError("Invalid key type for iloc indexer.")

    @property
    def get_sample_features_at(self) -> _iLocIndexer:
        return self._iLocIndexer(self)

    # --------------------- class methods --------------------- #

    @classmethod
    def from_sequences(
        cls,
        state_space: Iterable[Hashable],
        sequence_length: int,
        initial_sample_index: int = 0,
        initial_feature_index: int = 0,
        sample_prefix: str = "omega",
        feature_prefix: str = "X",
        threshold: int = 1000,
    ) -> SampleSpaceFeatures:
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
            feature_prefix=feature_prefix,
            initial_sample_index=initial_sample_index,
            initial_feature_index=initial_feature_index,
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
