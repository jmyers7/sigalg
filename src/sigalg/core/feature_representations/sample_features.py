import pandas as pd

from .array_like import ArrayLike
from .sample_space_features import SampleSpaceFeatures


class SampleFeatures(ArrayLike):

    def __init__(
        self,
        sample_space_features=None,
        features=None,
        sample_index=None,
        feature_index=None,
        dtype=None,
    ) -> None:
        if sample_space_features is None:
            if isinstance(features, pd.Series):
                preexisting_sample_index = features.name
                preexisting_feature_index = features.index
                is_sample_index_default = preexisting_sample_index is None
                is_feature_index_default = preexisting_feature_index.equals(
                    pd.RangeIndex(start=0, stop=len(features))
                )
            else:
                is_sample_index_default = sample_index is None
                is_feature_index_default = feature_index is None

            self._data = pd.Series(data=features, dtype=dtype)

            if sample_index is not None:
                self._data.name = sample_index
            if feature_index is not None:
                self._data.index = feature_index
            if sample_index is None and is_sample_index_default:
                self._data.name = "omega"
            if feature_index is None and is_feature_index_default:
                if len(self._data) == 1:
                    self._data.index = pd.Index(["X"])
                else:
                    self._data.index = [f"X{i}" for i in range(len(self._data))]
        else:
            self._validate_parameters(sample_space_features, sample_index)
            if isinstance(sample_index, int):
                self._data = sample_space_features._data.iloc[sample_index]
                self._sample_index = sample_space_features.sample_index[sample_index]
            else:
                self._data = sample_space_features._data.loc[sample_index]
                self._sample_index = sample_index
            self._feature_index = sample_space_features.feature_index
            self._n_features = sample_space_features.n_features

    def get_sample_features(self, key):
        raise IndexError(
            "SamplePoint object is 1-dimensional and has no sample points."
        )

    class _iLocIndexer:
        def __init__(self, parent):
            self.parent = parent

        def __getitem__(self, key):
            return self.parent._data.iloc[key]

    @property
    def feature_at(self):
        return self._iLocIndexer(self)

    @staticmethod
    def _validate_parameters(sample_space_features, sample_index):
        if not isinstance(sample_space_features, SampleSpaceFeatures):
            raise TypeError(
                "sample_space_features must be a SampleSpaceFeatures instance."
            )
        if not isinstance(sample_index, (str, int)):
            raise TypeError("sample_index must be a string or an integer.")
        if isinstance(sample_index, str):
            if sample_index not in sample_space_features.sample_index:
                raise IndexError(
                    f"Sample index '{sample_index}' not found in the sample space."
                )
        elif isinstance(sample_index, int):
            if sample_index < 0 or sample_index >= len(
                sample_space_features.sample_index
            ):
                raise IndexError(f"Sample index '{sample_index}' is out of bounds.")
