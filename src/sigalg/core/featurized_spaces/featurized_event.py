from collections.abc import Hashable

import pandas as pd

from ..spaces import Event, SampleSpace
from .featurized_sample_space import FeaturizedSampleSpace


class FeaturizedEvent:

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        featurized_sample_space: FeaturizedSampleSpace,
        event_indices: list[Hashable],
    ) -> None:
        self._validate_parameters(featurized_sample_space, event_indices)
        self._featurized_sample_space = featurized_sample_space
        self._sample_space = featurized_sample_space.sample_space
        self._event = Event(
            sample_space=self._sample_space, event_indices=event_indices
        )
        self._values = featurized_sample_space._values.loc[self._event.index].copy()

    # --------------------- properties --------------------- #

    @property
    def featurized_sample_space(self) -> FeaturizedSampleSpace:
        return self._featurized_sample_space

    @property
    def sample_space(self) -> SampleSpace:
        return self._sample_space

    @property
    def event(self) -> Event:
        return self._event

    @property
    def features(self) -> pd.DataFrame:
        return self._values.copy()

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        featurized_sample_space: FeaturizedSampleSpace,
        event_indices: list[Hashable],
    ):
        if not isinstance(featurized_sample_space, FeaturizedSampleSpace):
            raise TypeError(
                "featurized_sample_space must be an instance of FeaturizedSampleSpace."
            )
        if not isinstance(event_indices, list):
            raise TypeError("event_indices must be a list of sample indices.")
        for idx in event_indices:
            if idx not in featurized_sample_space.sample_space.values:
                raise ValueError(
                    f"Sample index {idx} not found in featurized_sample_space."
                )
