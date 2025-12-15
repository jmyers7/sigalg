from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..base.feature_index import FeatureIndex
    from .random_variable import RandomVariable


class RandomVector:

    def __init__(
        self,
        components: list[RandomVariable] | None = None,
        values: pd.DataFrame | None = None,
        name: Hashable = "X",
    ):
        if values is not None:
            self._values = values
            self._components = None  # lazy evaluation
            self._domain = None  # lazy evaluation
        elif components is not None:
            self._values = None  # lazy evaluation
            self._components = components
            self._domain = components[0].domain
        self._name = name

        # caches for properties
        self._feature_index: FeatureIndex | None = None

    @property
    def components(self):
        from .random_variable import RandomVariable

        # only true if `values` was passed to constructor
        if self._components is None:
            domain = self.domain
            components = []
            for col in self._values:
                outputs = self._values[col].to_dict()
                rv = RandomVariable(outputs=outputs, domain=domain, name=col)
                components.append(rv)
            self._components = components
        return self._components

    @property
    def values(self):

        # only true if `components` was passed to constructor
        if self._values is None:
            rv_values = [rv.values for rv in self._components]
            df = pd.concat(rv_values, axis=1)
            df.columns.name = "feature"
            self._values = df
        return self._values

    @property
    def domain(self):
        from ..base.sample_space import SampleSpace

        # only true if `values` was passed to consturctor
        if self._domain is None:
            indices = list(self._values.index)
            values_name = self._values.index.name
            self._domain = SampleSpace(
                indices=indices, name="Omega", values_name=values_name
            )
        return self._domain

    @property
    def feature_index(self):
        from ..base import FeatureIndex

        if self._feature_index is None:
            self._feature_index = FeatureIndex(
                values=self.values.columns, values_name=self.values.columns.name
            )
        return self._feature_index

    @feature_index.setter
    def feature_index(self, feature_index: FeatureIndex):
        for i, rv in enumerate(self.components):
            rv.name = feature_index[i]
        self.values.columns = feature_index.values

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: Hashable):
        self._name = name

    @property
    def range(self):
        from ..base import SampleSpace

        unique_values = list(self.values.apply(lambda row: tuple(row), axis=1).unique())
        num_unique_values = len(unique_values)
        range_sample_space = SampleSpace.generate_default(
            size=num_unique_values,
            prefix="x",
            values_name="output",
        )
        result = pd.DataFrame(
            unique_values,
            index=range_sample_space.values,
            columns=self.values.columns,
        )
        return RandomVector(values=result, name=f"range({self.name})")
