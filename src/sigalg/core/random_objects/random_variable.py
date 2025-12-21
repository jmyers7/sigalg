# from __future__ import annotations

# from collections.abc import Hashable, Mapping
# from numbers import Real
# from typing import TYPE_CHECKING, Any

# import pandas as pd

# if TYPE_CHECKING:
#     from ..base.event import Event
#     from ..base.sample_space import SampleSpace
#     from ..featurized_spaces.sample_point_features import SamplePointFeatures
#     from .random_variable import RandomVariable


# class RandomVariable:

#     # --------------------- constructor --------------------- #

#     def __init__(
#         self,
#         outputs: Mapping[Hashable, Hashable],
#         domain: SampleSpace,
#         name: Hashable = "X",
#     ) -> None:

#         self._validate_parameters(outputs=outputs, domain=domain, name=name)

#         self.outputs = outputs
#         self.domain = domain
#         self._name = name

#         # caches for properties
#         self._range_counts: pd.Series | None = None
#         self._values: pd.Series | None = None

#     # --------------------- properties --------------------- #

#     @property
#     def values(self) -> pd.Series:
#         if self._values is None:
#             series = pd.Series(self.outputs, name=self._name)
#             series.index.name = self.domain.values_name
#             self._values = series
#         return self._values

#     @values.setter
#     def values(self, values: pd.Series) -> None:
#         self._values = values

#     @classmethod
#     def from_values(cls, values: pd.Series, name: Hashable = "X") -> RandomVariable:
#         from ..base.sample_space import SampleSpace

#         if not isinstance(values, pd.Series):
#             raise TypeError("values must be a pd.Series.")
#         outputs = values.to_dict()
#         domain = SampleSpace(values=values.index)
#         rv = cls(outputs=outputs, domain=domain, name=name)
#         rv.values = values
#         rv.values.name = name
#         return rv

#     @property
#     def name(self) -> Hashable:
#         return self._name

#     @property
#     def range(self) -> RandomVariable:
#         from ..base import SampleSpace

#         range_series = self.values.value_counts()
#         range_sample_space = SampleSpace.generate_default(
#             size=len(range_series),
#             prefix=self.name.lower(),
#             values_name="output",
#         )
#         self._range_counts = pd.Series(
#             range_series.values, index=range_sample_space.data, name="count"
#         )
#         range_series = pd.Series(
#             range_series.index, index=range_sample_space.data, name=self.name
#         )
#         return RandomVariable.from_values(
#             values=range_series, name=f"range({self.name})"
#         )

#     @property
#     def range_counts(self) -> pd.Series:
#         if self._range_counts is None:
#             _ = self.range  # triggers computation of range and counts
#         return self._range_counts

#     # --------------------- data access --------------------- #

#     def __call__(
#         self, key: Hashable | list[Hashable] | Event
#     ) -> SamplePointFeatures | RandomVariable:
#         from ..base.event import Event
#         from ..featurized_spaces.sample_point_features import SamplePointFeatures

#         if not isinstance(key, (Hashable, list, Event)):
#             raise TypeError("key must be a Hashable, list, or Event.")
#         if isinstance(key, Hashable) and not isinstance(key, (list, Event)):
#             if key not in self.domain:
#                 raise KeyError(f"Sample '{key}' not found in domain.")
#             return self.values.loc[key]
#         if isinstance(key, list):
#             invalid_indices = [k for k in key if k not in self.domain.data]
#             if invalid_indices:
#                 raise KeyError(f"Samples {invalid_indices} not found in domain.")
#             return RandomVariable.from_values(
#                 values=self.values.loc[key], name=f"{self.name}|event"
#             )
#         if isinstance(key, Event):
#             if key.sample_space != self.domain:
#                 raise ValueError(
#                     "Event's sample_space must match RandomVector's domain."
#                 )
#             return RandomVariable.from_values(
#                 values=self.values.loc[key.indices],
#                 name=f"{self.name}|{key.name}",
#             )

#     def __getitem__(
#         self, key: int | slice | list[int]
#     ) -> SamplePointFeatures | RandomVariable:

#         if not isinstance(key, (int, slice, list)):
#             raise TypeError("key must be an int, slice, or list of ints.")
#         if isinstance(key, int):
#             if key < 0 or key >= len(self.domain):
#                 raise IndexError(
#                     f"Index {key} out of range for domain of size {len(self.domain)}."
#                 )
#             sample_index = self.domain[key]
#             return self(sample_index)
#         if isinstance(key, list):
#             if not all(isinstance(k, int) for k in key):
#                 raise TypeError("All elements in list must be integers.")
#             invalid_indices = [k for k in key if k < 0 or k >= len(self.domain)]
#             if invalid_indices:
#                 raise IndexError(
#                     f"Indices {invalid_indices} out of range for domain of size {len(self.domain)}."
#                 )
#             event = self.domain[key]
#             event.name = "event"
#             return self(event)
#         if isinstance(key, slice):
#             event = self.domain[key]
#             event.name = "event"
#             return self(event)

#     # --------------------- equality --------------------- #

#     def __eq__(self, other: RandomVariable) -> bool:

#         if not isinstance(other, RandomVariable):
#             return False
#         if not self.domain == other.domain:
#             return False
#         return self.values.equals(other.values)

#     # --------------------- arithmetic operations --------------------- #

#     def __add__(self, other: RandomVariable | Real) -> RandomVariable:
#         if isinstance(other, Real):
#             new_name = f"({self.name}+{other})"
#             new_values = self.values + other
#         elif isinstance(other, RandomVariable):
#             if self.domain != other.domain:
#                 raise ValueError("Cannot add RandomVariables with different domains.")
#             new_name = f"({self.name}+{other.name})"
#             new_values = self.values + other.values
#         else:
#             raise TypeError("Can only add RandomVariable or scalar to RandomVariable.")
#         new_values.name = new_name
#         result = RandomVariable.from_values(values=new_values, name=new_name)
#         return result

#     def __radd__(self, other: RandomVariable | Real) -> RandomVariable:
#         return self.__add__(other)

#     def __sub__(self, other: RandomVariable | Real) -> RandomVariable:
#         if isinstance(other, Real):
#             new_name = f"({self.name}-{other})"
#             new_values = self.values - other
#         elif isinstance(other, RandomVariable):
#             if self.domain != other.domain:
#                 raise ValueError(
#                     "Cannot subtract RandomVariables with different domains."
#                 )
#             new_name = f"({self.name}-{other.name})"
#             new_values = self.values - other.values
#         else:
#             raise TypeError(
#                 "Can only subtract RandomVariable or scalar from RandomVariable."
#             )
#         new_values.name = new_name
#         result = RandomVariable.from_values(values=new_values, name=new_name)
#         return result

#     def __rsub__(self, other: RandomVariable | Real) -> RandomVariable:
#         if isinstance(other, Real):
#             new_name = f"{other}-({self.name})"
#             new_values = other - self.values
#         elif isinstance(other, RandomVariable):
#             if self.domain != other.domain:
#                 raise ValueError(
#                     "Cannot subtract RandomVariables with different domains."
#                 )
#             new_name = f"({other.name}-{self.name})"
#             new_values = other.values - self.values
#         else:
#             raise TypeError(
#                 "Can only subtract RandomVariable or scalar from RandomVariable."
#             )
#         new_values.name = new_name
#         result = RandomVariable.from_values(values=new_values, name=new_name)
#         return result

#     def __mul__(self, other: RandomVariable | Real) -> RandomVariable:
#         if isinstance(other, Real):
#             new_name = f"({self.name}*{other})"
#             new_values = self.values * other
#         elif isinstance(other, RandomVariable):
#             if self.domain != other.domain:
#                 raise ValueError(
#                     "Cannot multiply RandomVariables with different domains."
#                 )
#             new_name = f"({self.name}*{other.name})"
#             new_values = self.values * other.values
#         else:
#             raise TypeError(
#                 "Can only multiply RandomVariable or scalar with RandomVariable."
#             )
#         new_values.name = new_name
#         result = RandomVariable.from_values(values=new_values, name=new_name)
#         return result

#     def __rmul__(self, other: RandomVariable | Real) -> RandomVariable:
#         return self.__mul__(other)

#     def __truediv__(self, other: RandomVariable | Real) -> RandomVariable:
#         if isinstance(other, Real):
#             new_name = f"({self.name}/{other})"
#             new_values = self.values / other
#         elif isinstance(other, RandomVariable):
#             if self.domain != other.domain:
#                 raise ValueError(
#                     "Cannot divide RandomVariables with different domains."
#                 )
#             new_name = f"({self.name}/{other.name})"
#             new_values = self.values / other.values
#         else:
#             raise TypeError(
#                 "Can only divide RandomVariable or scalar with RandomVariable."
#             )
#         new_values.name = new_name
#         result = RandomVariable.from_values(values=new_values, name=new_name)
#         return result

#     def __rtruediv__(self, other: RandomVariable | Real) -> RandomVariable:
#         if isinstance(other, Real):
#             new_name = f"{other}/({self.name})"
#             new_values = other / self.values
#         elif isinstance(other, RandomVariable):
#             if self.domain != other.domain:
#                 raise ValueError(
#                     "Cannot divide RandomVariables with different domains."
#                 )
#             new_name = f"({other.name}/{self.name})"
#             new_values = other.values / self.values
#         else:
#             raise TypeError(
#                 "Can only divide RandomVariable or scalar with RandomVariable."
#             )
#         new_values.name = new_name
#         result = RandomVariable.from_values(values=new_values, name=new_name)
#         return result

#     def __pow__(self, other: RandomVariable | Real) -> RandomVariable:
#         if isinstance(other, Real):
#             new_name = f"({self.name}**{other})"
#             new_values = self.values**other
#         elif isinstance(other, RandomVariable):
#             if self.domain != other.domain:
#                 raise ValueError(
#                     "Cannot exponentiate RandomVariables with different domains."
#                 )
#             new_name = f"({self.name}**{other.name})"
#             new_values = self.values**other.values
#         else:
#             raise TypeError(
#                 "Can only exponentiate RandomVariable or scalar with RandomVariable."
#             )
#         new_values.name = new_name
#         result = RandomVariable.from_values(values=new_values, name=new_name)
#         return result

#     def __rpow__(self, other: RandomVariable | Real) -> RandomVariable:
#         if isinstance(other, Real):
#             new_name = f"{other}**({self.name})"
#             new_values = other**self.values
#         elif isinstance(other, RandomVariable):
#             if self.domain != other.domain:
#                 raise ValueError(
#                     "Cannot exponentiate RandomVariables with different domains."
#                 )
#             new_name = f"({other.name}**{self.name})"
#             new_values = other.values**self.values
#         else:
#             raise TypeError(
#                 "Can only exponentiate RandomVariable or scalar with RandomVariable."
#             )
#         new_values.name = new_name
#         result = RandomVariable.from_values(values=new_values, name=new_name)
#         return result

#     # --------------------- validation methods --------------------- #

#     @staticmethod
#     def _validate_parameters(
#         outputs: dict[Hashable, Any],
#         domain: SampleSpace,
#         name: Hashable,
#     ):
#         from ..base.sample_space import SampleSpace

#         if not isinstance(outputs, dict):
#             raise TypeError("outputs must be a dictionary.")
#         if not isinstance(domain, SampleSpace):
#             raise TypeError("domain must be a SampleSpace.")
#         if not all(idx in domain.data for idx in outputs.keys()):
#             raise ValueError(
#                 "All output keys must be in the domain SampleSpace values."
#             )
#         if not isinstance(name, Hashable):
#             raise TypeError("name must be a hashable type.")
