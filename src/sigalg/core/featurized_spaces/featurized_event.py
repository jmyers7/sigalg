from collections.abc import Hashable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..spaces import Event, SampleSpace
    from .feature_embedding import FeatureEmbedding
    from .featurized_sample_space import FeaturizedSampleSpace


class FeaturizedEvent:

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        fss: FeaturizedSampleSpace,
        event: Event,
    ) -> None:
        from .feature_embedding import FeatureEmbedding

        self._validate_parameters(fss, event)
        self._featurized_sample_space = fss
        self._sample_space = fss.sample_space
        self._event = event
        embedding_name = fss.feature_embedding.name + "|" + event.name
        self._feature_embedding = FeatureEmbedding(
            features=fss.feature_embedding.values.loc[self._event.values],
            name=embedding_name,
        )

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
    def feature_embedding(self) -> FeatureEmbedding:
        return self._feature_embedding

    # --------------------- class methods --------------------- #

    @classmethod
    def from_indices(
        cls,
        fss: FeaturizedSampleSpace,
        event_indices: list[Hashable],
        event_name: str = "A",
    ) -> "FeaturizedEvent":
        from ..spaces.event import Event

        event = Event(
            sample_space=fss.sample_space,
            event_indices=event_indices,
            name=event_name,
        )
        return cls(fss=fss, event=event)

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        header = f"Featurized event ({self.event.name}, {self.feature_embedding.name}) in featurized sample space ({self.sample_space.name}, {self.featurized_sample_space.feature_embedding.name})"
        separator = "=" * len(header)
        return (
            header
            + "\n"
            + separator
            + "\n\n* "
            + repr(self.sample_space)
            + "\n\n* "
            + repr(self.event)
            + "\n\n* "
            + repr(self.feature_embedding)
        )

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        fss: FeaturizedSampleSpace,
        event_indices: list[Hashable],
    ):
        # if not isinstance(fss, FeaturizedSampleSpace):
        #     raise TypeError(
        #         "featurized_sample_space must be an instance of FeaturizedSampleSpace."
        #     )
        # if not isinstance(event_indices, list):
        #     raise TypeError("event_indices must be a list of sample indices.")
        # for idx in event_indices:
        #     if idx not in fss.sample_space.values:
        #         raise ValueError(
        #             f"Sample index {idx} not found in featurized_sample_space."
        #         )
        pass
