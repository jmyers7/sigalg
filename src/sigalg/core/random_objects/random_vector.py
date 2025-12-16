from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd

from ..base.feature_index import FeatureIndex

if TYPE_CHECKING:
    from ..base.sample_space import SampleSpace
    from .random_variable import RandomVariable


class RandomVector:

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        outputs: dict[Hashable, tuple] | None = None,
        domain: SampleSpace | None = None,
        values: pd.DataFrame | None = None,
        name: Hashable = "X",
    ):
        from ..base.feature_index import FeatureIndex
        from ..base.sample_space import SampleSpace

        # TODO: validation

        if values is not None:
            self.values = values
            self._feature_index = FeatureIndex(
                values=self.values.columns, values_name=self.values.columns.name
            )
            self._components = None  # lazy evaluation
            self.domain = SampleSpace(
                indices=list(self.values.index),
                name="Omega",
                values_name=self.values.index.name,
            )
            self._outputs = None  # lazy evaluation
        elif outputs is not None:
            self.values = pd.DataFrame.from_dict(outputs, orient="index")
            self.values.index.name = domain.values_name
            num_components = self.values.shape[1]
            if num_components == 1:
                self._feature_index = FeatureIndex(
                    indices=[name],
                    values_name="feature",
                )
            else:
                self._feature_index = FeatureIndex(
                    indices=[f"{name}{i}" for i in range(self.values.shape[1])],
                    values_name="feature",
                )
            self.values.columns = self._feature_index.values
            self._components = None  # lazy evaluation
            self.domain = domain
            self._outputs = outputs

        self._name = name
        self.dimension = self.values.shape[1]

        # caches for properties
        self._range_counts: pd.Series | None = None

    # --------------------- properties --------------------- #

    @property
    def outputs(self):
        if self._outputs is None:
            self._outputs = {}
            for idx in self.values.index:
                series = self.values.loc[idx]
                if len(series) == 1:
                    self._outputs[idx] = series.iloc[0]
                else:
                    self._outputs[idx] = tuple(series)
        return self._outputs

    @property
    def components(self):
        if self._components is None:
            self._components = []
            for col in self.values:
                values = self.values[col].to_frame()
                values.columns.name = self.values.columns.name
                self._components.append(RandomVector(values=values, name=col))
        return self._components

    @property
    def feature_index(self):
        return self._feature_index

    @feature_index.setter
    def feature_index(self, feature_index: FeatureIndex):

        # TODO: validation

        for i, rv in enumerate(self.components):
            rv.name = feature_index[i]
        self.values.columns = feature_index.values
        self._feature_index = feature_index

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: Hashable):
        if not isinstance(name, Hashable):
            raise TypeError("name must be a hashable type.")
        self._name = name

    @property
    def range(self):
        from ..base import SampleSpace

        range_df = self.values.value_counts().reset_index(name="count")
        range_sample_space = SampleSpace.generate_default(
            size=len(range_df),
            prefix=self.name.lower(),
            values_name="output",
        )
        range_df.index = range_sample_space.values
        self._range_counts = range_df["count"]
        range_df.drop(columns=["count"], inplace=True)
        range_df.columns = self.values.columns
        return RandomVector(values=range_df, name=f"range({self.name})")

    @property
    def range_counts(self) -> pd.Series:
        if self._range_counts is None:
            _ = self.range  # triggers computation of range and counts
        return self._range_counts

    # --------------------- data access --------------------- #

    def __call__(self, key):
        from ..base.event import Event
        from ..featurized_spaces.sample_point_features import SamplePointFeatures

        # TODO: validation

        if isinstance(key, Hashable):
            return SamplePointFeatures(values=self.values.loc[key], name=key)
        if isinstance(key, list):
            return RandomVector(values=self.values.loc[key], name=f"{self.name}|event")
        if isinstance(key, Event):
            return RandomVector(
                values=self.values.loc[key.indices],
                name=f"{self.name}|{key.name}",
            )
        else:
            raise TypeError("key must be a Hashable, list, or Event.")

    def __getitem__(self, key):

        # TODO: validation

        if isinstance(key, int):
            sample_index = self.domain[key]
            return self(sample_index)
        if isinstance(key, slice):
            event = self.domain[key]
            event.name = "event"
            return self(event)
        if isinstance(key, list):
            event = self.domain[key]
            event.name = "event"
            return self(event)
        else:
            raise TypeError("key must be an int, slice, or list.")

    def get_component(self, index: Hashable) -> RandomVector:

        if self._components is not None:
            pos = self.feature_index.values.to_list().index(index)
            return self._components[pos]
        else:
            values = self.values[[index]]
            return RandomVector(values=values, name=index)

    # --------------------- equality --------------------- #

    def __eq__(self, other) -> bool:

        if not isinstance(other, RandomVector):
            return False
        if not self.domain == other.domain:
            return False
        if not self.feature_index == other.feature_index:
            return False
        return self.values.equals(other.values)

    # --------------------- arithmetic operations --------------------- #

    def __add__(self, other: RandomVector | Real) -> RandomVector:
        if isinstance(other, Real):
            new_values = self.values + other
            new_name = f"({self.name}+{other})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError("Cannot add RandomVectors with different domains.")
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.values.columns = pd.RangeIndex(self.dimension)
            other.values.columns = pd.RangeIndex(other.dimension)
            new_values = self.values + other.values
            new_name = f"({self.name}+{other.name})"
        else:
            raise TypeError("Can only add RandomVector or scalar to RandomVector.")

        new_feature_index = FeatureIndex.generate_default(
            size=self.dimension, prefix=new_name, values_name="feature"
        )
        result = RandomVector(values=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result

    def __radd__(self, other: RandomVector | Real) -> RandomVector:
        return self.__add__(other)

    def __sub__(self, other: RandomVector | Real) -> RandomVector:
        if isinstance(other, Real):
            new_values = self.values - other
            new_name = f"({self.name}-{other})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot subtract RandomVectors with different domains."
                )
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.values.columns = pd.RangeIndex(self.dimension)
            other.values.columns = pd.RangeIndex(other.dimension)
            new_values = self.values - other.values
            new_name = f"({self.name}-{other.name})"
        else:
            raise TypeError(
                "Can only subtract RandomVector or scalar from RandomVector."
            )

        new_feature_index = FeatureIndex.generate_default(
            size=self.dimension, prefix=new_name, values_name="feature"
        )
        result = RandomVector(values=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result

    def __rsub__(self, other: RandomVector | Real) -> RandomVector:
        if isinstance(other, Real):
            new_values = other - self.values
            new_name = f"({other}-{self.name})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot subtract RandomVectors with different domains."
                )
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.values.columns = pd.RangeIndex(self.dimension)
            other.values.columns = pd.RangeIndex(other.dimension)
            new_values = other.values - self.values
            new_name = f"({other.name}-{self.name})"
        else:
            raise TypeError(
                "Can only subtract RandomVector or scalar from RandomVector."
            )

        new_feature_index = FeatureIndex.generate_default(
            size=self.dimension, prefix=new_name, values_name="feature"
        )
        result = RandomVector(values=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result

    def __mul__(self, other: RandomVector | Real) -> RandomVector:
        if isinstance(other, Real):
            new_values = self.values * other
            new_name = f"({self.name}*{other})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot multiply RandomVectors with different domains."
                )
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.values.columns = pd.RangeIndex(self.dimension)
            other.values.columns = pd.RangeIndex(other.dimension)
            new_values = self.values * other.values
            new_name = f"({self.name}*{other.name})"
        else:
            raise TypeError(
                "Can only multiply RandomVector or scalar with RandomVector."
            )

        new_feature_index = FeatureIndex.generate_default(
            size=self.dimension, prefix=new_name, values_name="feature"
        )
        result = RandomVector(values=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result

    def __rmul__(self, other: RandomVector | Real) -> RandomVector:
        return self.__mul__(other)

    def __truediv__(self, other: RandomVector | Real) -> RandomVector:
        if isinstance(other, Real):
            new_values = self.values / other
            new_name = f"({self.name}/{other})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError("Cannot divide RandomVectors with different domains.")
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.values.columns = pd.RangeIndex(self.dimension)
            other.values.columns = pd.RangeIndex(other.dimension)
            new_values = self.values / other.values
            new_name = f"({self.name}/{other.name})"
        else:
            raise TypeError("Can only divide RandomVector or scalar with RandomVector.")

        new_feature_index = FeatureIndex.generate_default(
            size=self.dimension, prefix=new_name, values_name="feature"
        )
        result = RandomVector(values=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result

    def __rtruediv__(self, other: RandomVector | Real) -> RandomVector:
        if isinstance(other, Real):
            new_values = other / self.values
            new_name = f"({other}/{self.name})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError("Cannot divide RandomVectors with different domains.")
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.values.columns = pd.RangeIndex(self.dimension)
            other.values.columns = pd.RangeIndex(other.dimension)
            new_values = other.values / self.values
            new_name = f"({other.name}/{self.name})"
        else:
            raise TypeError("Can only divide RandomVector or scalar with RandomVector.")

        new_feature_index = FeatureIndex.generate_default(
            size=self.dimension, prefix=new_name, values_name="feature"
        )
        result = RandomVector(values=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result

    def __pow__(self, other: RandomVector | Real) -> RandomVector:
        if isinstance(other, Real):
            new_values = self.values**other
            new_name = f"({self.name}**{other})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot exponentiate RandomVectors with different domains."
                )
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.values.columns = pd.RangeIndex(self.dimension)
            other.values.columns = pd.RangeIndex(other.dimension)
            new_values = self.values**other.values
            new_name = f"({self.name}**{other.name})"
        else:
            raise TypeError(
                "Can only exponentiate RandomVector or scalar with RandomVector."
            )

        new_feature_index = FeatureIndex.generate_default(
            size=self.dimension, prefix=new_name, values_name="feature"
        )
        result = RandomVector(values=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result

    def __rpow__(self, other: RandomVector | Real) -> RandomVector:
        if isinstance(other, Real):
            new_values = other**self.values
            new_name = f"({other}**{self.name})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot exponentiate RandomVectors with different domains."
                )
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.values.columns = pd.RangeIndex(self.dimension)
            other.values.columns = pd.RangeIndex(other.dimension)
            new_values = other.values**self.values
            new_name = f"({other.name}**{self.name})"
        else:
            raise TypeError(
                "Can only exponentiate RandomVector or scalar with RandomVector."
            )

        new_feature_index = FeatureIndex.generate_default(
            size=self.dimension, prefix=new_name, values_name="feature"
        )
        result = RandomVector(values=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result
