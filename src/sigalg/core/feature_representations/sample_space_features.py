from .array_like import ArrayLike
import pandas as pd
from itertools import product


class SampleSpaceFeatures(ArrayLike):
    def __init__(
        self,
        data,
        sample_index=None,
        feature_index=None,
        dtype=None,
        overwrite_default_sample_index: bool = True,
        overwrite_default_rv_index: bool = True,
        initial_sample_index: int = 0,
        initial_feature_index: int = 0,
        sample_prefix: str = "omega",
        feature_rv_prefix: str = "X",
    ):
        from ..sigma_algebras import SigmaAlgebra

        self._data = pd.DataFrame(data=data, dtype=dtype)
        n_rows = len(self._data)
        n_cols = len(self._data.columns)

        if sample_index is None:
            if n_rows == 1:
                sample_index = [f"{sample_prefix}"]
            else:
                sample_index = [
                    f"{sample_prefix}{i + initial_sample_index}" for i in range(n_rows)
                ]
        if feature_index is None:
            if n_cols == 1:
                feature_index = [f"{feature_rv_prefix}"]
            else:
                feature_index = [
                    f"{feature_rv_prefix}{i + initial_feature_index}"
                    for i in range(n_cols)
                ]

        is_default_sample_index = self._data.index.equals(
            pd.RangeIndex(start=0, stop=n_rows)
        )
        is_default_feature_index = self._data.columns.equals(
            pd.RangeIndex(start=0, stop=n_cols)
        )

        if is_default_sample_index and overwrite_default_sample_index:
            self._data.index = sample_index
        if is_default_feature_index and overwrite_default_rv_index:
            self._data.columns = feature_index

        atom_ids = dict(zip(self._data.index, range(len(self._data))))
        self._sigma_algebra = SigmaAlgebra(sample_space_features=self, atom_ids=atom_ids)

    class _iLocIndexer:
        def __init__(self, parent):
            self.parent = parent

        def __getitem__(self, key):
            from .sample_features import SampleFeatures

            result = self.parent._data.iloc[key]
            return SampleFeatures(data=result)

    @property
    def get_sample_features_at(self):
        return self._iLocIndexer(self)

    @property
    def sigma_algebra(self):
        return self._sigma_algebra

    def set_sigma_algebra(self, sigma_algebra):
        from ..sigma_algebras import SigmaAlgebra

        if not isinstance(sigma_algebra, SigmaAlgebra):
            raise TypeError("sigma_algebra must be a SigmaAlgebra instance.")
        if sigma_algebra._sample_space_features is not self:
            raise ValueError(
                "sigma_algebra must have the same space_features as this SampleSpaceFeatures instance."
            )
        self._sigma_algebra = sigma_algebra

    @classmethod
    def from_sequences(
        cls,
        state_space: list,
        sequence_length: int,
        initial_sample_index: int = 0,
        initial_feature_index: int = 0,
        sample_prefix: str = "omega",
        feature_rv_prefix: str = "X",
        threshold: int = 1000,
    ):
        if not isinstance(state_space, list) or len(state_space) == 0:
            raise ValueError("state_space must be a non-empty list")
        sample_space_cardinality = len(state_space) ** sequence_length

        if sample_space_cardinality > threshold:
            raise ValueError(
                f"Sample space size {sample_space_cardinality} exceeds threshold of {threshold}. "
            )

        sequences = list(product(state_space, repeat=sequence_length))
        return cls(
            data=sequences,
            sample_prefix=sample_prefix,
            feature_rv_prefix=feature_rv_prefix,
            initial_sample_index=initial_sample_index,
            initial_feature_index=initial_feature_index,
        )
