from collections.abc import Hashable

import pandas as pd


class FeatureEmbedding:

    # --------------------- constructor --------------------- #

    def __init__(self, name: Hashable, features: pd.DataFrame):
        self._values = features.copy()
        self._name = name

    # --------------------- properties --------------------- #

    @property
    def values(self) -> pd.DataFrame:
        return self._values.copy()

    @property
    def name(self) -> str:
        return self._name

    # --------------------- iter methods --------------------- #

    def iter_sample_features(self):
        for sample_index in self.values.index:
            yield sample_index, self.values.loc[sample_index]

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"Feature embedding {self.name}:\n{self.values}"

    # --------------------- equality --------------------- #

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureEmbedding):
            return False
        return self.values.equals(other.values) and self.name == other.name
