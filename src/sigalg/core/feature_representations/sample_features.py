from .array_like import ArrayLike
from .sample_space_features import SampleSpaceFeatures
import pandas as pd
from typing import Any


class SampleFeatures(ArrayLike):

    def __init__(
        self,
        sample_space: SampleSpaceFeatures = None,
        data: Any = None,
        sample_index: Any = None,
        feature_index: Any = None,
        dtype=None,
    ) -> None:
        if sample_space is None:
            if isinstance(data, pd.Series):
                preexisting_sample_index = data.name
                preexisting_feature_index = data.index
                is_sample_index_default = preexisting_sample_index is None
                is_feature_index_default = preexisting_feature_index.equals(
                    pd.RangeIndex(start=0, stop=len(data))
                )
            else:
                is_sample_index_default = sample_index is None
                is_feature_index_default = feature_index is None

            self._data = pd.Series(data=data, dtype=dtype)

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
            self._validate_parameters(sample_space, sample_index)
            if isinstance(sample_index, int):
                self._data = sample_space._data.iloc[sample_index]
                self._sample_index = sample_space.sample_index[sample_index]
            else:
                self._data = sample_space._data.loc[sample_index]
                self._sample_index = sample_index
            self._feature_index = sample_space.feature_index
            self._n_features = sample_space.n_features

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
    def _validate_parameters(sample_space, sample_index):
        if not isinstance(sample_space, SampleSpaceFeatures):
            raise TypeError("sample_space must be a SampleSpaceFeatures instance.")
        if not isinstance(sample_index, (str, int)):
            raise TypeError("sample_index must be a string or an integer.")
        if isinstance(sample_index, str):
            if sample_index not in sample_space.sample_index:
                raise IndexError(
                    f"Sample index '{sample_index}' not found in the sample space."
                )
        elif isinstance(sample_index, int):
            if sample_index < 0 or sample_index >= len(sample_space.sample_index):
                raise IndexError(f"Sample index '{sample_index}' is out of bounds.")
