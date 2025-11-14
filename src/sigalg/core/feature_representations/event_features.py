from .sample_space_features import SampleSpaceFeatures


class EventFeatures(SampleSpaceFeatures):

    def __init__(
        self, sample_space_features: SampleSpaceFeatures, sample_index
    ) -> None:
        self._validate_parameters(sample_space_features, sample_index)
        super().__init__(data=sample_space_features._data.loc[sample_index])
        self._sample_space_features = sample_space_features

    @property
    def sample_space_features(self):
        return self._sample_space_features

    @staticmethod
    def _validate_parameters(sample_space_features, sample_index):
        if not isinstance(sample_space_features, SampleSpaceFeatures):
            raise TypeError(
                "sample_space_features must be a SampleSpaceFeatures instance."
            )
        if not isinstance(sample_index, list):
            raise TypeError("sample_index must be a list.")
        for idx in sample_index:
            if idx not in sample_space_features.sample_index:
                raise IndexError(
                    f"Sample index '{idx}' not found in the sample space features."
                )
