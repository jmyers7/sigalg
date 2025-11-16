from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import pandas as pd

from ..spaces import SampleSpace
from .array_like import ArrayLike

if TYPE_CHECKING:
    from .sample_features import SampleFeatures


class EventFeatures(ArrayLike):

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        sample_space: SampleSpace,
        feature_data: pd.DataFrame | dict | list,
        event_indices: list[Hashable],
        feature_names: list[Hashable] = None,
        dtype=None,
    ) -> None:
        # Convert to DataFrame first
        if isinstance(feature_data, pd.DataFrame):
            self._data = feature_data.copy()
            if dtype is not None:
                self._data = self._data.astype(dtype)
        else:
            self._data = pd.DataFrame(data=feature_data, dtype=dtype)

        # Validate parameters
        self._validate_parameters(
            self._data, sample_space, event_indices, feature_names
        )

        self._sample_space = sample_space
        self._event_indices = list(event_indices)

        # Set the index to event_indices
        self._data.index = event_indices

        # Set feature names if provided
        if feature_names is not None:
            self._data.columns = feature_names

    # --------------------- properties --------------------- #

    @property
    def sample_space(self) -> SampleSpace:
        """The sample space this event belongs to."""
        return self._sample_space

    @property
    def event_indices(self) -> list[Hashable]:
        """The indices of samples in this event."""
        return self._event_indices.copy()

    @property
    def feature_names(self) -> pd.Index:
        """The names of the features."""
        return self._data.columns

    @property
    def n_samples(self) -> int:
        """The number of samples in this event."""
        return len(self._data)

    @property
    def n_features(self) -> int:
        """The number of features."""
        return len(self._data.columns)

    # --------------------- access methods --------------------- #

    class _iLocIndexer:
        """Indexer for integer-location based indexing of samples."""

        def __init__(self, parent: EventFeatures) -> None:
            self.parent = parent

        def __getitem__(self, key: int | slice | list[int]) -> SampleFeatures:
            from .sample_features import SampleFeatures

            result = self.parent._data.iloc[key]

            if isinstance(key, int):
                # Return a single SampleFeatures
                return SampleFeatures(
                    features=result,
                    sample_index=result.name,
                    feature_names=list(result.index),
                )
            elif isinstance(key, slice) or (
                isinstance(key, list) and all(isinstance(k, int) for k in key)
            ):
                # Return a new EventFeatures with subset of samples
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
        """Access sample features by integer position."""
        return self._iLocIndexer(self)

    # --------------------- conversion methods --------------------- #

    def to_pandas(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return self._data.copy()

    def to_dict(self, orient: str = "dict") -> dict:
        """Convert to dictionary."""
        return self._data.to_dict(orient=orient)

    def to_list(self) -> list[list]:
        """Convert to list of lists."""
        return self._data.values.tolist()

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        event_str = f"[{', '.join(str(idx) for idx in self._event_indices)}]"
        return f"EventFeatures(event={event_str}, n_samples={self.n_samples}, n_features={self.n_features},\n{self._data})"

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        data: pd.DataFrame,
        sample_space: SampleSpace,
        event_indices: list[Hashable],
        feature_names: list[Hashable] | None,
    ) -> None:
        if data.empty:
            raise ValueError("feature_data cannot be empty")

        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance")

        if not isinstance(event_indices, list):
            raise TypeError("event_indices must be a list")

        if len(event_indices) == 0:
            raise ValueError("event_indices cannot be empty")

        if len(data) != len(event_indices):
            raise ValueError(
                f"Number of rows in feature_data ({len(data)}) must match "
                f"the length of event_indices ({len(event_indices)})"
            )

        # Check that all event indices are in the sample space
        for idx in event_indices:
            if idx not in sample_space.index:
                raise ValueError(f"Event index '{idx}' not found in sample space")

        if feature_names is not None:
            if not isinstance(feature_names, list):
                raise TypeError("feature_names must be a list")
            if len(data.columns) != len(feature_names):
                raise ValueError(
                    f"Number of feature columns ({len(data.columns)}) must match "
                    f"the length of feature_names ({len(feature_names)})"
                )
