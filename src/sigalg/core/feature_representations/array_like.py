import numpy as np
import pandas as pd


class ArrayLike:

    # --------------------- properties --------------------- #

    @property
    def is_1d(self) -> bool:
        return isinstance(self._values, pd.Series)

    @property
    def sample_index(self) -> pd.Index:
        if self.is_1d:
            return self._values.name
        else:
            return self._values.index

    @property
    def feature_index(self) -> pd.Index:
        if self.is_1d:
            return self._values.index
        else:
            return self._values.columns

    @property
    def n_samples(self) -> int:
        if self.is_1d:
            return 1
        else:
            return len(self._values)

    @property
    def n_features(self) -> int:
        if self.is_1d:
            return len(self._values)
        else:
            return len(self._values.columns)

    @property
    def shape(self):
        if self.is_1d:
            return (len(self._values),)
        else:
            return self._values.shape

    # --------------------- access methods --------------------- #

    def __getitem__(self, key):
        from .event_features import EventFeatures
        from .sample_features import SampleFeatures

        if self.is_1d:
            return self._values.loc[key]
        else:
            if isinstance(key, list):
                return EventFeatures(features=self._values.loc[key], event_indices=key)
            else:
                return SampleFeatures(features=self._values.loc[key], sample_index=key)

    def get_sample_features(self, key):
        return self[key]

    def get_event(self, key):
        if not isinstance(key, list):
            raise TypeError("key must be a list of sample indices.")
        return self.get_sample_features(key)

    def get_feature_rv(self, key):
        from ..random_objects.random_variable import RandomVariable  # lazy import

        if self.is_1d:
            return self._values[key]
        else:
            column_values = self._values[key]
            values = column_values.to_dict()
            return RandomVariable(
                domain_features=self, outputs=values, name=column_values.name
            )

    class _iLocIndexer:
        def __init__(self, parent) -> None:
            self.parent = parent

        def __getitem__(self, key: int | slice | list[int]):
            features = self.parent._values.iloc[key]
            if self.parent.is_1d:
                return features
            else:
                from .event_features import EventFeatures
                from .sample_features import SampleFeatures

                if isinstance(key, list) or isinstance(key, slice):
                    if isinstance(self.parent, EventFeatures):
                        sample_space_features = self.parent.sample_space_features
                    else:
                        sample_space_features = self.parent

                    return EventFeatures(
                        sample_space_features=sample_space_features,
                        event_indices=features.index.tolist(),
                    )
                else:
                    return SampleFeatures(features=features)

    # --------------------- iteration methods --------------------- #

    def __iter__(self):
        for idx in self._values.index:
            yield self[idx]

    def iter_samples(self):
        for idx in self._values.index:
            yield self.get_sample_features(idx)

    # --------------------- conversion methods --------------------- #

    def to_pandas(self):
        return self._values.copy()

    def to_numpy(self) -> np.ndarray:
        return self._values.to_numpy().copy()

    def __array__(self, dtype=None) -> np.ndarray:
        if dtype is None:
            return self.to_numpy()
        else:
            return self.to_numpy().astype(dtype)

    # --------------------- transformation methods --------------------- #

    def sum(self):
        if self.is_1d:
            return self._values.sum()
        else:
            return self._values.sum(axis=1)

    def apply_to_row(self, function):
        if self.is_1d:
            return function(self)
        else:
            from .sample_features import SampleFeatures

            def wrapper(row):
                sp = SampleFeatures(features=row)
                return function(sp)

            return self._values.apply(wrapper, axis=1)

    def apply_to_index(self, idx_function):
        return self._values.index.to_series().apply(idx_function)

    # --------------------- special methods --------------------- #

    def __len__(self) -> int:
        return len(self._values)

    def __eq__(self, other) -> bool:
        if not isinstance(other, ArrayLike):
            if self.is_1d and len(self._values) == 1:
                return self._values.iloc[0] == other
            return False
        return self._values.equals(other._values)
