from .sample_space_features import SampleSpaceFeatures


class EventFeatures(SampleSpaceFeatures):

    def __init__(
        self, sample_space_features: SampleSpaceFeatures, event_indices
    ) -> None:
        self._validate_event_parameters(sample_space_features, event_indices)
        super().__init__(features=sample_space_features._data.loc[event_indices])
        self._sample_space_features = sample_space_features

    @property
    def sample_space_features(self):
        return self._sample_space_features

    @staticmethod
    def _validate_event_parameters(sample_space_features, event_indices):
        if not isinstance(sample_space_features, SampleSpaceFeatures):
            raise TypeError(
                "sample_space_features must be a SampleSpaceFeatures instance."
            )
        if not isinstance(event_indices, list):
            raise TypeError("event_indices must be a list.")
        for idx in event_indices:
            if idx not in sample_space_features.sample_space:
                raise IndexError(
                    f"Sample index '{idx}' not found in the sample space features."
                )
