from __future__ import annotations

from collections.abc import Hashable

from ..spaces import Event, SampleSpace
from .array_like import ArrayLike
from .sample_space_features import SampleSpaceFeatures


class EventFeatures(ArrayLike):

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        sample_space_features: SampleSpaceFeatures,
        event_indices: list[Hashable],
    ) -> None:
        self._validate_parameters(sample_space_features, event_indices)
        self._sample_space_features = sample_space_features
        self._sample_space = sample_space_features.sample_space
        self._event = Event(
            sample_space=self._sample_space, event_indices=event_indices
        )
        # Use the event's index to ensure consistency with Event object
        self._data = sample_space_features._data.loc[self._event.index].copy()

    # --------------------- properties --------------------- #

    @property
    def sample_space(self) -> SampleSpace:
        return self._sample_space

    @property
    def event(self) -> Event:
        return self._event

    @property
    def sample_space_features(self) -> SampleSpaceFeatures:
        return self._sample_space_features

    # --------------------- access methods --------------------- #

    @property
    def get_sample_features_at(self):
        return self._iLocIndexer(self)

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        sample_space_features: SampleSpaceFeatures,
        event_indices: list[Hashable],
    ):
        if not isinstance(sample_space_features, SampleSpaceFeatures):
            raise TypeError(
                "sample_space_features must be an instance of SampleSpaceFeatures."
            )
        if not isinstance(event_indices, list):
            raise TypeError("event_indices must be a list of sample indices.")
        for idx in event_indices:
            if idx not in sample_space_features.sample_index:
                raise ValueError(
                    f"Sample index {idx} not found in sample_space_features."
                )
