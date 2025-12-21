from __future__ import annotations

from collections.abc import Hashable, Mapping
from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd

from ...validation.sample_space_mapping_in import SampleSpaceMappingIn

if TYPE_CHECKING:
    from ..base.event import Event
    from ..base.index import Index
    from ..base.sample_space import SampleSpace
    from ..featurized_spaces.sample_point_features import SamplePointFeatures
    from .random_variable import RandomVariable


class RandomVector:
    """A random vector.

    An instance of `RandomVector` represents a mapping `X: Omega -> S` from a sample space `Omega` to a feature space `S`. Given a sample point `omega` in `Omega`, we conceptualize `X(omega)` as a vector of features associated with that sample point.

    Parameters
    ----------
    outputs : Mapping[Hashable, Hashable]
        A mapping from sample points in the domain to their corresponding output vectors (e.g., tuples of feature values).
    domain : SampleSpace
        The sample space over which the random vector is defined.
    name : Hashable | None, default="X"
        The name of the random vector.

    Raises
    ------
    TypeError
        If `outputs` is not a mapping from hashable types to hashable types, or if `name` is not hashable.
    ValueError
        If `outputs` does not contain an entry for every sample ID in `domain`.

    Examples
    --------
    >>> from sigalg.core import SampleSpace, RandomVector
    >>> domain = SampleSpace.generate_default(size=3, prefix="s", name="S")
    >>> outputs = {"s0": (0.1, 0.2), "s1": (0.3, 0.4), "s2": (0.5, 0.6)}
    >>> # Generate a 2-dimensional random vector
    >>> X = RandomVector(outputs=outputs, domain=domain, name="X")
    >>> tuple(X("s0"))
    (0.1, 0.2)
    >>> X.dimension
    2
    >>> # Generate a 1-dimensional random vector from a pd.DataFrame
    >>> import pandas as pd
    >>> data = pd.DataFrame(
    ...     [10, 20, 30],
    ...     index=pd.Index([0, 1, 2], name="numbers"),
    ... )
    >>> V = RandomVector.from_pandas(data, name="V")
    >>> int(V(1))
    20
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        outputs: Mapping[Hashable, Hashable],
        domain: SampleSpace,
        name: Hashable | None = "X",
    ) -> None:
        v = SampleSpaceMappingIn(mapping=outputs, sample_space=domain, name=name)

        self.outputs = v.mapping
        self.domain = v.sample_space
        self._name = v.name

        # caches for properties
        self._range_counts: pd.Series | None = None
        self._data: pd.DataFrame | None = None
        self._feature_index: Index | None = None

    # --------------------- properties --------------------- #

    @property
    def data(self) -> pd.DataFrame:
        """Get the underlying `pd.DataFrame`.

        Returns
        -------
        data: pd.DataFrame
            The underlying `pd.DataFrame` representing the random vector.
        """
        from ..base.index import Index

        if self._data is None:
            data = pd.DataFrame.from_dict(self.outputs, orient="index")
            dimension = data.shape[1]
            feature_index = Index.generate_default(
                size=dimension,
                prefix=self.name,
                name="feature_index",
                data_name="feature",
            )
            data.columns = feature_index.data
            data.index.name = self.domain.data.name
            self._data = data
            self._feature_index = feature_index
        return self._data

    @data.setter
    def data(self, data: pd.DataFrame) -> None:
        """Set the underlying `pd.DataFrame`.

        The `data` property is not meant to be set directly by the user. This setter is provided so that the `from_pandas` factory method can set the property.

        Parameters
        ----------
        data : pd.DataFrame
            New `pd.DataFrame` to set.

        Raises
        ------
        TypeError
            If `data` is not a `pd.DataFrame`.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pd.DataFrame.")
        self._data = data

    @classmethod
    def from_pandas(
        cls, data: pd.DataFrame, name: Hashable | None = "X"
    ) -> RandomVector:
        """Create a `RandomVector` from a `pd.DataFrame`.

        A domain `SampleSpace` is automatically generated from the index of the provided `pd.DataFrame`. Its name defaults to `Omega`, which may be reset through the `domain.name` property after construction. A feature index (i.e., an instance of `Index`) is also automatically generated based on the columns of the `pd.DataFrame`.

        Parameters
        ----------
        data : pd.DataFrame
            A `pd.DataFrame` where each row corresponds to a sample point and each column corresponds to a feature.
        name : Hashable | None, default="X"
            The name of the random vector.

        Raises
        ------
        TypeError
            If `data` is not a `pd.DataFrame`.

        Returns
        -------
        rv : RandomVector
            The constructed `RandomVector` instance.

        """
        from ..base.sample_space import SampleSpace

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pd.DataFrame.")

        dimension = data.shape[1]
        if dimension == 1:
            outputs = data.iloc[:, 0].to_dict()
        else:
            outputs = data.apply(lambda row: tuple(row), axis=1).to_dict()
        domain = SampleSpace.from_pandas(data=data.index)
        rv = cls(outputs=outputs, domain=domain, name=name)
        rv.data = data
        return rv

    @property
    def name(self) -> Hashable:
        """Get the name of the random vector.

        Returns
        -------
        name : Hashable
            The name of the random vector.
        """
        return self._name

    @name.setter
    def name(self, name: Hashable) -> None:
        if not isinstance(name, Hashable):
            raise TypeError("name must be a Hashable.")
        self._name = name

    @property
    def feature_index(self) -> Index:
        """Get the feature index.

        Returns
        -------
        feature_index : Index
            The feature index of the random vector.
        """
        from ..base.index import Index

        if self._feature_index is None:
            self._feature_index = Index.from_pandas(
                data=self.data.columns, name="feature_index"
            )
        return self._feature_index

    @feature_index.setter
    def feature_index(self, feature_index: Index) -> None:
        from ..base.index import Index

        if not isinstance(feature_index, Index):
            raise TypeError("feature_index must be an Index.")
        if feature_index.size != self.dimension:
            raise ValueError(
                "feature_index size must match the dimension of the RandomVector."
            )
        self._feature_index = feature_index
        self.data.columns = feature_index.data

    @property
    def range(self) -> RandomVector:
        """Get the range of the random vector.

        Mathematically, the range of a random vector `X:Omega -> S` is the set of all vectors `X(omega)`, as `omega` varies over the sample space `Omega`. In this implementation, the range is represented as another `RandomVector`, where the domain is a `SampleSpace` that indexes the unique output vectors of the original random vector, and the outputs are these unique vectors themselves.

        If the random vector has a string name (e.g., `X`), the range random vector is named `range(X)`, the domain of `range(X)` has indices `x0`, `x1`, etc., and the feature indices of `range(X)` match those of `X` itself. Otherwise, numerical indices are used.

        """
        from ..base import SampleSpace

        range_data = self.data.value_counts().reset_index(name="count")
        range_name = f"range({self.name})" if isinstance(self.name, str) else None
        prefix = self.name.lower() if isinstance(self.name, str) else None
        range_sample_space = SampleSpace.generate_default(
            size=len(range_data),
            prefix=prefix,
            name=range_name,
            data_name="output",
        )
        range_data.index = range_sample_space.data
        self._range_counts = range_data["count"]
        range_data.drop(columns=["count"], inplace=True)
        range_data.columns = self.data.columns
        return RandomVector.from_pandas(data=range_data, name=range_name)

    # TODO: docstring for range_counts, along with unit tests
    @property
    def range_counts(self) -> pd.Series:
        if self._range_counts is None:
            _ = self.range  # triggers computation of range and counts
        return self._range_counts

    @property
    def dimension(self) -> int:
        return self.data.shape[1]

    # --------------------- data access --------------------- #

    def __call__(
        self, key: Hashable | list[Hashable] | Event
    ) -> Hashable | SamplePointFeatures | RandomVector:
        from ..base.event import Event
        from ..featurized_spaces.sample_point_features import SamplePointFeatures

        if not isinstance(key, (Hashable, list, Event)):
            raise TypeError("key must be a Hashable, list, or Event.")
        if isinstance(key, Hashable) and not isinstance(key, (list, Event)):
            if key not in self.domain:
                raise KeyError(f"Sample '{key}' not found in domain.")
            result = SamplePointFeatures(values=self.data.loc[key], name=key)
            if len(result) == 1:
                return result.values[0]
            else:
                return result
        if isinstance(key, list):
            invalid_indices = [k for k in key if k not in self.domain.data]
            if invalid_indices:
                raise KeyError(f"Samples {invalid_indices} not found in domain.")
            return RandomVector.from_pandas(
                data=self.data.loc[key], name=f"{self.name}|event"
            )
        if isinstance(key, Event):
            if key.sample_space != self.domain:
                raise ValueError(
                    "Event's sample_space must match RandomVector's domain."
                )
            return RandomVector.from_pandas(
                data=self.data.loc[key.indices],
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
            values = self.data.iloc[:, positions]
            return RandomVector.from_pandas(data=values, name=f"{self.name}_sub")
        elif isinstance(key, Hashable):
            if key not in self.feature_index:
                raise KeyError(f"Feature '{key}' not found in feature index.")
            position = self.feature_index.data.to_list().index(key)
            values = self.data.iloc[:, position]
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
        return self.data.equals(other.data)

    # --------------------- arithmetic operations --------------------- #

    def __add__(self, other: RandomVector | Real) -> RandomVector:
        if isinstance(other, Real):
            new_values = self.data + other
            new_name = f"({self.name}+{other})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError("Cannot add RandomVectors with different domains.")
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.data.columns = pd.RangeIndex(self.dimension)
            other.data.columns = pd.RangeIndex(other.dimension)
            new_values = self.data + other.data
            new_name = f"({self.name}+{other.name})"
        else:
            raise TypeError("Can only add RandomVector or scalar to RandomVector.")

        new_feature_index = Index.generate_default(
            size=self.dimension, prefix=new_name, data_name="feature"
        )
        result = RandomVector.from_pandas(data=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result

    def __radd__(self, other: RandomVector | Real) -> RandomVector:
        return self.__add__(other)

    def __sub__(self, other: RandomVector | Real) -> RandomVector:
        if isinstance(other, Real):
            new_values = self.data - other
            new_name = f"({self.name}-{other})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot subtract RandomVectors with different domains."
                )
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.data.columns = pd.RangeIndex(self.dimension)
            other.data.columns = pd.RangeIndex(other.dimension)
            new_values = self.data - other.data
            new_name = f"({self.name}-{other.name})"
        else:
            raise TypeError(
                "Can only subtract RandomVector or scalar from RandomVector."
            )

        new_feature_index = Index.generate_default(
            size=self.dimension, prefix=new_name, data_name="feature"
        )
        result = RandomVector.from_pandas(data=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result

    def __rsub__(self, other: RandomVector | Real) -> RandomVector:
        if isinstance(other, Real):
            new_values = other - self.data
            new_name = f"({other}-{self.name})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot subtract RandomVectors with different domains."
                )
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.data.columns = pd.RangeIndex(self.dimension)
            other.data.columns = pd.RangeIndex(other.dimension)
            new_values = other.data - self.data
            new_name = f"({other.name}-{self.name})"
        else:
            raise TypeError(
                "Can only subtract RandomVector or scalar from RandomVector."
            )

        new_feature_index = Index.generate_default(
            size=self.dimension, prefix=new_name, data_name="feature"
        )
        result = RandomVector.from_pandas(data=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result

    def __mul__(self, other: RandomVector | Real) -> RandomVector:
        if isinstance(other, Real):
            new_values = self.data * other
            new_name = f"({self.name}*{other})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot multiply RandomVectors with different domains."
                )
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.data.columns = pd.RangeIndex(self.dimension)
            other.data.columns = pd.RangeIndex(other.dimension)
            new_values = self.data * other.data
            new_name = f"({self.name}*{other.name})"
        else:
            raise TypeError(
                "Can only multiply RandomVector or scalar with RandomVector."
            )

        new_feature_index = Index.generate_default(
            size=self.dimension, prefix=new_name, data_name="feature"
        )
        result = RandomVector.from_pandas(data=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result

    def __rmul__(self, other: RandomVector | Real) -> RandomVector:
        return self.__mul__(other)

    def __truediv__(self, other: RandomVector | Real) -> RandomVector:
        if isinstance(other, Real):
            new_values = self.data / other
            new_name = f"({self.name}/{other})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError("Cannot divide RandomVectors with different domains.")
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.data.columns = pd.RangeIndex(self.dimension)
            other.data.columns = pd.RangeIndex(other.dimension)
            new_values = self.data / other.data
            new_name = f"({self.name}/{other.name})"
        else:
            raise TypeError("Can only divide RandomVector or scalar with RandomVector.")

        new_feature_index = Index.generate_default(
            size=self.dimension, prefix=new_name, data_name="feature"
        )
        result = RandomVector.from_pandas(data=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result

    def __rtruediv__(self, other: RandomVector | Real) -> RandomVector:
        if isinstance(other, Real):
            new_values = other / self.data
            new_name = f"({other}/{self.name})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError("Cannot divide RandomVectors with different domains.")
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.data.columns = pd.RangeIndex(self.dimension)
            other.data.columns = pd.RangeIndex(other.dimension)
            new_values = other.data / self.data
            new_name = f"({other.name}/{self.name})"
        else:
            raise TypeError("Can only divide RandomVector or scalar with RandomVector.")

        new_feature_index = Index.generate_default(
            size=self.dimension, prefix=new_name, data_name="feature"
        )
        result = RandomVector.from_pandas(data=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result

    def __pow__(self, other: RandomVector | Real) -> RandomVector:
        if isinstance(other, Real):
            new_values = self.data**other
            new_name = f"({self.name}**{other})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot exponentiate RandomVectors with different domains."
                )
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.data.columns = pd.RangeIndex(self.dimension)
            other.data.columns = pd.RangeIndex(other.dimension)
            new_values = self.data**other.data
            new_name = f"({self.name}**{other.name})"
        else:
            raise TypeError(
                "Can only exponentiate RandomVector or scalar with RandomVector."
            )

        new_feature_index = Index.generate_default(
            size=self.dimension, prefix=new_name, data_name="feature"
        )
        result = RandomVector.from_pandas(data=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result

    def __rpow__(self, other: RandomVector | Real) -> RandomVector:
        if isinstance(other, Real):
            new_values = other**self.data
            new_name = f"({other}**{self.name})"
        elif isinstance(other, RandomVector):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot exponentiate RandomVectors with different domains."
                )
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")
            self.data.columns = pd.RangeIndex(self.dimension)
            other.data.columns = pd.RangeIndex(other.dimension)
            new_values = other.data**self.data
            new_name = f"({other.name}**{self.name})"
        else:
            raise TypeError(
                "Can only exponentiate RandomVector or scalar with RandomVector."
            )

        new_feature_index = Index.generate_default(
            size=self.dimension, prefix=new_name, data_name="feature"
        )
        result = RandomVector.from_pandas(data=new_values, name=new_name)
        result.feature_index = new_feature_index
        return result


class RandomVectorMethods:
    """Mixin class providing RandomVector methods."""

    def get_components(
        self, key: Hashable | list[Hashable]
    ) -> RandomVariable | RandomVector:
        return self.random_vector.get_components(key)
