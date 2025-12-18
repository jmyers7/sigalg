from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..base.feature_index import FeatureIndex

if TYPE_CHECKING:
    from ..base.event import Event
    from ..base.sample_space import SampleSpace
    from ..featurized_spaces.sample_point_features import SamplePointFeatures
    from .random_variable import RandomVariable


class RandomVector:

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        outputs: dict[Hashable, Any],
        domain: SampleSpace,
        name: Hashable = "X",
    ) -> None:

        self._validate_parameters(outputs=outputs, domain=domain, name=name)

        self.outputs = outputs
        self.domain = domain
        self._name = name

        # caches for properties
        self._range_counts: pd.Series | None = None
        self._values: pd.DataFrame | None = None

    # --------------------- properties --------------------- #

    @property
    def values(self) -> pd.DataFrame:
        if self._values is None:
            df = pd.DataFrame.from_dict(self.outputs, orient="index")
            df.index.name = self.domain.values_name
            num_components = df.shape[1]
            if num_components == 1:
                feature_index = FeatureIndex(
                    indices=[self._name],
                    values_name="feature",
                )
            else:
                feature_index = FeatureIndex(
                    indices=[f"{self._name}{i}" for i in range(df.shape[1])],
                    values_name="feature",
                )
            df.columns = feature_index.data
            self._values = df
        return self._values

    @values.setter
    def values(self, values: pd.DataFrame) -> None:
        self._values = values

    @classmethod
    def from_values(cls, values: pd.DataFrame, name: Hashable = "X") -> RandomVector:
        from ..base.sample_space import SampleSpace

        if not isinstance(values, pd.DataFrame):
            raise TypeError("values must be a pd.DataFrame.")

        n_components = values.shape[1]
        if n_components == 1:
            outputs = values.iloc[:, 0].to_dict()
        else:
            outputs = values.apply(lambda row: tuple(row), axis=1).to_dict()
        domain = SampleSpace(values=values.index)
        rv = cls(outputs=outputs, domain=domain, name=name)
        rv.values = values
        return rv

    @property
    def name(self) -> Hashable:
        return self._name

    @property
    def feature_index(self) -> FeatureIndex:
        if not hasattr(self, "_feature_index"):
            self._feature_index = FeatureIndex(values=self.values.columns)
        return self._feature_index

    @feature_index.setter
    def feature_index(self, feature_index: FeatureIndex) -> None:
        self._feature_index = feature_index
        self.values.columns = feature_index.data

    @property
    def range(self) -> RandomVector:
        from ..base import SampleSpace

        range_df = self.values.value_counts().reset_index(name="count")
        range_sample_space = SampleSpace.generate_default(
            size=len(range_df),
            prefix=self.name.lower(),
            values_name="output",
        )
        range_df.index = range_sample_space.data
        self._range_counts = range_df["count"]
        range_df.drop(columns=["count"], inplace=True)
        range_df.columns = self.values.columns
        return RandomVector.from_values(values=range_df, name=f"range({self.name})")

    @property
    def range_counts(self) -> pd.Series:
        if self._range_counts is None:
            _ = self.range  # triggers computation of range and counts
        return self._range_counts

    @property
    def dimension(self) -> int:
        return self.values.shape[1]

    # --------------------- data access --------------------- #

    def __call__(
        self, key: Hashable | list[Hashable] | Event
    ) -> SamplePointFeatures | RandomVector:
        from ..base.event import Event
        from ..featurized_spaces.sample_point_features import SamplePointFeatures

        if not isinstance(key, (Hashable, list, Event)):
            raise TypeError("key must be a Hashable, list, or Event.")
        if isinstance(key, Hashable) and not isinstance(key, (list, Event)):
            if key not in self.domain:
                raise KeyError(f"Sample '{key}' not found in domain.")
            return SamplePointFeatures(values=self.values.loc[key], name=key)
        if isinstance(key, list):
            invalid_indices = [k for k in key if k not in self.domain.data]
            if invalid_indices:
                raise KeyError(f"Samples {invalid_indices} not found in domain.")
            return RandomVector.from_values(
                values=self.values.loc[key], name=f"{self.name}|event"
            )
        if isinstance(key, Event):
            if key.sample_space != self.domain:
                raise ValueError(
                    "Event's sample_space must match RandomVector's domain."
                )
            return RandomVector.from_values(
                values=self.values.loc[key.indices],
                name=f"{self.name}|{key.name}",
            )

    def __getitem__(
        self, key: int | slice | list[int]
    ) -> SamplePointFeatures | RandomVector:

        if not isinstance(key, (int, slice, list)):
            raise TypeError("key must be an int, slice, or list of ints.")
        if isinstance(key, int):
            if key < 0 or key >= len(self.domain):
                raise IndexError(
                    f"Index {key} out of range for domain of size {len(self.domain)}."
                )
            sample_index = self.domain[key]
            return self(sample_index)
        if isinstance(key, list):
            if not all(isinstance(k, int) for k in key):
                raise TypeError("All elements in list must be integers.")
            invalid_indices = [k for k in key if k < 0 or k >= len(self.domain)]
            if invalid_indices:
                raise IndexError(
                    f"Indices {invalid_indices} out of range for domain of size {len(self.domain)}."
                )
            event = self.domain[key]
            event.name = "event"
            return self(event)
        if isinstance(key, slice):
            event = self.domain[key]
            event.name = "event"
            return self(event)

    def get_components(
        self, key: Hashable | list[Hashable]
    ) -> RandomVariable | RandomVector:
        from .random_variable import RandomVariable

        if isinstance(key, list):
            for k in key:
                if not isinstance(k, Hashable):
                    raise TypeError("All elements in list must be Hashable.")
                if k not in self.feature_index:
                    raise KeyError(f"Feature '{k}' not found in feature index.")
            positions = [self.feature_index.data.to_list().index(k) for k in key]
            values = self.values.iloc[:, positions]
            return RandomVector.from_values(values=values, name=f"{self.name}_sub")
        elif isinstance(key, Hashable):
            if key not in self.feature_index:
                raise KeyError(f"Feature '{key}' not found in feature index.")
            position = self.feature_index.data.to_list().index(key)
            values = self.values.iloc[:, position]
            return RandomVariable.from_values(values=values, name=key)
        else:
            raise TypeError("key must be a Hashable or list of Hashables.")

    # --------------------- equality --------------------- #

    def __eq__(self, other: RandomVector) -> bool:

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
        result = RandomVector.from_values(values=new_values, name=new_name)
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
        result = RandomVector.from_values(values=new_values, name=new_name)
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
        result = RandomVector.from_values(values=new_values, name=new_name)
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
        result = RandomVector.from_values(values=new_values, name=new_name)
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
        result = RandomVector.from_values(values=new_values, name=new_name)
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
        result = RandomVector.from_values(values=new_values, name=new_name)
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
        result = RandomVector.from_values(values=new_values, name=new_name)
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
        result = RandomVector.from_values(values=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        outputs: dict[Hashable, Any],
        domain: SampleSpace,
        name: Hashable,
    ):
        from ..base.sample_space import SampleSpace

        if not isinstance(outputs, dict):
            raise TypeError("outputs must be a dictionary.")
        if not isinstance(domain, SampleSpace):
            raise TypeError("domain must be a SampleSpace.")
        if not all(idx in domain.data for idx in outputs.keys()):
            raise ValueError(
                "All output keys must be in the domain SampleSpace values."
            )
        if not isinstance(name, Hashable):
            raise TypeError("name must be a hashable type.")


class RandomVectorMethods:
    """Mixin class providing RandomVector methods."""

    def get_components(
        self, key: Hashable | list[Hashable]
    ) -> RandomVariable | RandomVector:
        return self.random_vector.get_components(key)
