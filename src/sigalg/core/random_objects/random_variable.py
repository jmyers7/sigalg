from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..base.feature_index import FeatureIndex

# from .random_object import RandomObject

if TYPE_CHECKING:
    from ..base.event import Event
    from ..base.sample_space import SampleSpace
    from ..featurized_spaces.sample_point_features import SamplePointFeatures
    from .random_variable import RandomVariable


class RandomVariable:

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
        self._values: pd.Series | None = None

    # --------------------- properties --------------------- #

    @property
    def values(self) -> pd.Series:
        if self._values is None:
            series = pd.Series(self.outputs, name=self._name)
            series.index.name = self.domain.values_name
            self._values = series
        return self._values

    @values.setter
    def values(self, values: pd.Series) -> None:
        self._values = values

    @classmethod
    def from_values(cls, values: pd.Series, name: Hashable = "X") -> RandomVariable:
        from ..base.sample_space import SampleSpace

        if not isinstance(values, pd.Series):
            raise TypeError("values must be a pd.Series.")
        outputs = values.to_dict()
        domain = SampleSpace(values=values.index)
        rv = cls(outputs=outputs, domain=domain, name=name)
        rv.values = values
        rv.values.name = name
        return rv

    @property
    def name(self) -> Hashable:
        return self._name

    @property
    def range(self) -> RandomVariable:
        from ..base import SampleSpace

        range_series = self.values.value_counts()
        range_sample_space = SampleSpace.generate_default(
            size=len(range_series),
            prefix=self.name.lower(),
            values_name="output",
        )
        self._range_counts = pd.Series(
            range_series.values, index=range_sample_space.data, name="count"
        )
        range_series = pd.Series(
            range_series.index, index=range_sample_space.data, name=self.name
        )
        return RandomVariable.from_values(
            values=range_series, name=f"range({self.name})"
        )

    @property
    def range_counts(self) -> pd.Series:
        if self._range_counts is None:
            _ = self.range  # triggers computation of range and counts
        return self._range_counts

    # --------------------- data access --------------------- #

    def __call__(
        self, key: Hashable | list[Hashable] | Event
    ) -> SamplePointFeatures | RandomVariable:
        from ..base.event import Event
        from ..featurized_spaces.sample_point_features import SamplePointFeatures

        if not isinstance(key, (Hashable, list, Event)):
            raise TypeError("key must be a Hashable, list, or Event.")
        if isinstance(key, Hashable) and not isinstance(key, (list, Event)):
            if key not in self.domain:
                raise KeyError(f"Sample '{key}' not found in domain.")
            return self.values.loc[key]
        if isinstance(key, list):
            invalid_indices = [k for k in key if k not in self.domain.data]
            if invalid_indices:
                raise KeyError(f"Samples {invalid_indices} not found in domain.")
            return RandomVariable.from_values(
                values=self.values.loc[key], name=f"{self.name}|event"
            )
        if isinstance(key, Event):
            if key.sample_space != self.domain:
                raise ValueError(
                    "Event's sample_space must match RandomVector's domain."
                )
            return RandomVariable.from_values(
                values=self.values.loc[key.indices],
                name=f"{self.name}|{key.name}",
            )

    def __getitem__(
        self, key: int | slice | list[int]
    ) -> SamplePointFeatures | RandomVariable:

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

    # --------------------- equality --------------------- #

    def __eq__(self, other: RandomVariable) -> bool:

        if not isinstance(other, RandomVariable):
            return False
        if not self.domain == other.domain:
            return False
        return self.values.equals(other.values)

    # --------------------- arithmetic operations --------------------- #

    def __add__(self, other: RandomVariable | Real) -> RandomVariable:
        if isinstance(other, Real):
            new_name = f"({self.name}+{other})"
            new_values = self.values + other
        elif isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError("Cannot add RandomVariables with different domains.")
            new_name = f"({self.name}+{other.name})"
            new_values = self.values + other.values
        else:
            raise TypeError("Can only add RandomVariable or scalar to RandomVariable.")
        new_values.name = new_name
        result = RandomVariable.from_values(values=new_values, name=new_name)
        return result

    def __radd__(self, other: RandomVariable | Real) -> RandomVariable:
        return self.__add__(other)

    def __sub__(self, other: RandomVariable | Real) -> RandomVariable:
        if isinstance(other, Real):
            new_name = f"({self.name}-{other})"
            new_values = self.values - other
        elif isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot subtract RandomVariables with different domains."
                )
            new_name = f"({self.name}-{other.name})"
            new_values = self.values - other.values
        else:
            raise TypeError(
                "Can only subtract RandomVariable or scalar from RandomVariable."
            )
        new_values.name = new_name
        result = RandomVariable.from_values(values=new_values, name=new_name)
        return result

    def __rsub__(self, other: RandomVariable | Real) -> RandomVariable:
        if isinstance(other, Real):
            new_name = f"{other}-({self.name})"
            new_values = other - self.values
        elif isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot subtract RandomVariables with different domains."
                )
            new_name = f"({other.name}-{self.name})"
            new_values = other.values - self.values
        else:
            raise TypeError(
                "Can only subtract RandomVariable or scalar from RandomVariable."
            )
        new_values.name = new_name
        result = RandomVariable.from_values(values=new_values, name=new_name)
        return result

    def __mul__(self, other: RandomVariable | Real) -> RandomVariable:
        if isinstance(other, Real):
            new_name = f"({self.name}*{other})"
            new_values = self.values * other
        elif isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot multiply RandomVariables with different domains."
                )
            new_name = f"({self.name}*{other.name})"
            new_values = self.values * other.values
        else:
            raise TypeError(
                "Can only multiply RandomVariable or scalar with RandomVariable."
            )
        new_values.name = new_name
        result = RandomVariable.from_values(values=new_values, name=new_name)
        return result

    def __rmul__(self, other: RandomVariable | Real) -> RandomVariable:
        return self.__mul__(other)

    def __truediv__(self, other: RandomVariable | Real) -> RandomVariable:
        if isinstance(other, Real):
            new_name = f"({self.name}/{other})"
            new_values = self.values / other
        elif isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot divide RandomVariables with different domains."
                )
            new_name = f"({self.name}/{other.name})"
            new_values = self.values / other.values
        else:
            raise TypeError(
                "Can only divide RandomVariable or scalar with RandomVariable."
            )
        new_values.name = new_name
        result = RandomVariable.from_values(values=new_values, name=new_name)
        return result

    def __rtruediv__(self, other: RandomVariable | Real) -> RandomVariable:
        if isinstance(other, Real):
            new_name = f"{other}/({self.name})"
            new_values = other / self.values
        elif isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot divide RandomVariables with different domains."
                )
            new_name = f"({other.name}/{self.name})"
            new_values = other.values / self.values
        else:
            raise TypeError(
                "Can only divide RandomVariable or scalar with RandomVariable."
            )
        new_values.name = new_name
        result = RandomVariable.from_values(values=new_values, name=new_name)
        return result

    def __pow__(self, other: RandomVariable | Real) -> RandomVariable:
        if isinstance(other, Real):
            new_name = f"({self.name}**{other})"
            new_values = self.values**other
        elif isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot exponentiate RandomVariables with different domains."
                )
            new_name = f"({self.name}**{other.name})"
            new_values = self.values**other.values
        else:
            raise TypeError(
                "Can only exponentiate RandomVariable or scalar with RandomVariable."
            )
        new_values.name = new_name
        result = RandomVariable.from_values(values=new_values, name=new_name)
        return result

    def __rpow__(self, other: RandomVariable | Real) -> RandomVariable:
        if isinstance(other, Real):
            new_name = f"{other}**({self.name})"
            new_values = other**self.values
        elif isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot exponentiate RandomVariables with different domains."
                )
            new_name = f"({other.name}**{self.name})"
            new_values = other.values**self.values
        else:
            raise TypeError(
                "Can only exponentiate RandomVariable or scalar with RandomVariable."
            )
        new_values.name = new_name
        result = RandomVariable.from_values(values=new_values, name=new_name)
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


# from __future__ import annotations

# from collections.abc import Hashable
# from numbers import Real
# from typing import TYPE_CHECKING, Any, Callable

# import numpy as np
# import pandas as pd

# if TYPE_CHECKING:
#     from ..base.event import Event
#     from ..base.probability_space import ProbabilitySpace
#     from ..base.sample_space import SampleSpace
#     from ..featurized_spaces.feature_embedding import FeatureEmbedding
#     from ..featurized_spaces.sample_point_features import SamplePointFeatures
#     from ..probability_measures.probability_measure import ProbabilityMeasure
#     from ..sigma_algebras.sigma_algebra import SigmaAlgebra
#     from .random_variable_range import (
#         RandomVariableRange,
#         RandomVariableRangeWithProbability,
#     )


# class RandomVariable:

#     # --------------------- constructor --------------------- #

#     def __init__(
#         self,
#         outputs: dict[Hashable, Any] | None = None,
#         domain: SampleSpace | None = None,
#         values: pd.Series | None = None,
#         name: str = "X",
#     ):
#         self._validate_parameters(
#             outputs=outputs,
#             domain=domain,
#             values=values,
#             name=name,
#         )
#         from ..base.sample_space import SampleSpace
#         from ..sigma_algebras.sigma_algebra import SigmaAlgebra

#         if values is not None:
#             self.values = values
#             self.outputs = self.values.to_dict()
#             self.domain = SampleSpace(indices=self.values.index.to_list())
#             self._name = values.name if values.name is not None else name
#         elif outputs is not None:
#             self.values = pd.Series(data=outputs, name=name)
#             self.values.index.name = "sample"
#             self.outputs = outputs
#             self.domain = domain
#             self._name = name

#         self.sigma_algebra = SigmaAlgebra(
#             sample_id_to_atom_id=self.outputs,
#             sample_space=self.domain,
#             name=f"sigma({self._name})",
#         )

#         # caches for properties
#         self._function: Callable[[SamplePointFeatures], Any] | None = None
#         self._probability_space: ProbabilitySpace | None = None
#         self._range: RandomVariableRange | RandomVariableRangeWithProbability | None = (
#             None
#         )
#         self._probability_measure: ProbabilityMeasure | None = None
#         self._unique_values: np.ndarray | None = None
#         self._rv_value_to_range_id: dict | None = None

#     # --------------------- properties --------------------- #

#     @property
#     def name(self) -> str:
#         return self._name

#     @name.setter
#     def name(self, new_name: Hashable) -> None:
#         if not isinstance(new_name, Hashable):
#             raise TypeError("name must be hashable.")
#         self._name = new_name
#         self.values.name = new_name

#     @property
#     def function(self) -> Callable[[SamplePointFeatures], Any] | None:
#         return self._function

#     @function.setter
#     def function(self, new_function: Callable[[SamplePointFeatures], Any]) -> None:
#         if not isinstance(new_function, Callable):
#             raise TypeError("function must be callable.")
#         self._function = new_function

#     @property
#     def probability_space(self) -> ProbabilitySpace | None:
#         return self._probability_space

#     @probability_space.setter
#     def probability_space(self, new_probability_space: ProbabilitySpace) -> None:
#         from ..base.probability_space import ProbabilitySpace

#         if not isinstance(new_probability_space, ProbabilitySpace):
#             raise TypeError("probability_space must be a ProbabilitySpace.")
#         if new_probability_space.sample_space != self.domain:
#             raise ValueError(
#                 "The sample space of the provided ProbabilitySpace does not match the domain of this RandomVariable."
#             )
#         self._probability_space = new_probability_space

#     @property
#     def probability_measure(self) -> ProbabilityMeasure | None:
#         if self._probability_measure is None:
#             _ = self.range  # trigger computation of the range
#         return self._probability_measure

#     @property
#     def rv_value_to_range_id(self) -> dict:
#         if self._rv_value_to_range_id is None:
#             _ = self.range  # trigger computation of the range
#         return self._rv_value_to_range_id

#     @property
#     def unique_values(self) -> np.ndarray:
#         if self._unique_values is None:
#             self._unique_values = self.values.unique()
#         return self._unique_values

#     @property
#     def range(self) -> RandomVariableRange | RandomVariableRangeWithProbability:
#         if self._range is None:
#             from ..base.probability_space import ProbabilitySpace
#             from ..base.sample_space import SampleSpace
#             from ..probability_measures import ProbabilityMeasure
#             from .random_variable_range import (
#                 RandomVariableRange,
#                 RandomVariableRangeWithProbability,
#             )

#             range_sample_space = SampleSpace.generate_default(
#                 prefix=self._name.lower(),
#                 size=len(self.unique_values),
#                 values_name="outputs",
#                 name=f"{self._name}_range",
#             )

#             self._rv_value_to_range_id = dict(
#                 zip(self.unique_values, range_sample_space)
#             )

#             range_values = self.unique_values.reshape(-1, 1)
#             range_df = pd.DataFrame(
#                 data=range_values, index=range_sample_space.values, columns=[self.name]
#             )
#             rv_range = RandomVariableRange(values=range_df, name=self.name)

#             if self.probability_space is not None:
#                 level_sets = self.sigma_algebra.atom_id_to_event
#                 range_probabilities = {
#                     range_value: self.probability_space.P(level_set)
#                     for range_value, level_set in zip(
#                         range_sample_space, level_sets.values()
#                     )
#                 }
#                 range_probability_measure = ProbabilityMeasure(
#                     sample_space=range_sample_space,
#                     probabilities=range_probabilities,
#                     name=f"P_{self._name}",
#                 )
#                 range_probability_space = ProbabilitySpace(
#                     sample_space=range_sample_space,
#                     probability_measure=range_probability_measure,
#                 )
#                 self._range = RandomVariableRangeWithProbability(
#                     sample_space=range_sample_space,
#                     feature_embedding=rv_range,
#                     probability_measure=range_probability_measure,
#                 )
#                 self._probability_measure = range_probability_space.probability_measure
#             else:
#                 self._range = rv_range
#                 self._probability_measure = None
#         return self._range

#     # --------------------- methods --------------------- #

#     def is_measurable(self, sigma_algebra: SigmaAlgebra = None) -> bool:
#         if sigma_algebra is None and self.probability_space is None:
#             raise ValueError(
#                 "Either sigma_algebra or probability_space must be provided."
#             )
#         elif sigma_algebra is None and self.probability_space is not None:
#             sigma_algebra = self.probability_space.sigma_algebra

#         return self.sigma_algebra <= sigma_algebra

#     # --------------------- probability methods --------------------- #

#     def P(self, key) -> Real:
#         if self.probability_measure is None:
#             raise ValueError(
#                 "This RandomVariable does not have an associated ProbabilityMeasure."
#             )
#         if key not in self.rv_value_to_range_id:
#             raise ValueError(
#                 f"Value {key} is not in the range of this random variable."
#             )
#         idx = self.rv_value_to_range_id[key]
#         return self.probability_measure.P(idx)

#     def add_probability_measure_to_domain(
#         self, probability_measure: ProbabilityMeasure
#     ) -> None:
#         from ..base.probability_space import ProbabilitySpace

#         if self.probability_space is not None:
#             raise ValueError(
#                 "This RandomVariable already has an associated ProbabilitySpace."
#             )
#         if probability_measure.sample_space != self.domain:
#             raise ValueError(
#                 "The sample space of the provided ProbabilityMeasure does not match the domain of this RandomVariable."
#             )
#         self._probability_measure = probability_measure
#         self.probability_space = ProbabilitySpace(
#             sample_space=self.domain,
#             sigma_algebra=self.sigma_algebra,
#             probability_measure=probability_measure,
#         )
#         self._range = None  # reset range to trigger new computation on next call

#     # --------------------- factory methods --------------------- #

#     @classmethod
#     def on_probability_space(
#         cls,
#         outputs: dict[Hashable, Any],
#         probability_space: ProbabilitySpace,
#         name: str = "X",
#     ):
#         domain = probability_space.sample_space
#         rv = cls(outputs=outputs, domain=domain, name=name)
#         rv.probability_space = probability_space
#         rv._probability_measure = probability_space.probability_measure
#         return rv

#     @classmethod
#     def from_features(
#         cls,
#         function: Callable[[SamplePointFeatures], Any],
#         feature_embedding: FeatureEmbedding,
#         name: str = "X",
#     ):
#         data = feature_embedding.apply_to_features(function)
#         domain = feature_embedding.domain
#         outputs = data.to_dict()
#         rv = cls(outputs=outputs, domain=domain, name=name)
#         rv.function = function
#         return rv

#     # --------------------- call methods --------------------- #

#     def __call__(
#         self, key: SamplePointFeatures | Hashable | Event | ProbabilitySpace
#     ) -> Any:
#         from ..base.event import Event
#         from ..base.probability_space import ProbabilitySpace
#         from ..featurized_spaces.sample_point_features import SamplePointFeatures

#         if isinstance(key, SamplePointFeatures):
#             if self.function is None:
#                 raise ValueError("This RandomVariable was not defined with a function.")
#             return self.function(key)
#         elif isinstance(key, Event):
#             outputs = {idx: self.values[idx] for idx in key.values}
#             return RandomVariable(
#                 domain=key.to_sample_space(), outputs=outputs, name=self._name
#             )
#         elif isinstance(key, ProbabilitySpace):
#             outputs = {idx: self.values[idx] for idx in key.sample_space.values}
#             return RandomVariable.on_probability_space(
#                 outputs=outputs, probability_space=key, name=self._name
#             )
#         else:
#             if key not in self.outputs:
#                 raise KeyError(f"Key '{key}' not found in domain.")
#             return self.outputs[key]

#     # --------------------- equality --------------------- #

#     def __eq__(self, other: object) -> bool:
#         if not isinstance(other, RandomVariable):
#             return False
#         if self.domain != other.domain:
#             return False
#         return self.values.equals(other.values)

#     # --------------------- representation --------------------- #

#     def __repr__(self) -> str:
#         return f"RandomVariable(name={self.name}, domain={self.domain.name})"

#     def __str__(self) -> str:
#         header = f"Random variable '{self.name}' on sample space '{self.domain.name}'"
#         separator = "=" * len(header)
#         return (
#             header
#             + "\n"
#             + separator
#             + "\n\n* "
#             + repr(self.domain)
#             + "\n\n* Values:\n"
#             + f"{self.values.to_frame()}"
#             + "\n\n* "
#             + repr(self.range)
#         )

#     # --------------------- validation methods --------------------- #

#     @staticmethod
#     def _validate_parameters(
#         outputs: dict[Hashable, Any] | None,
#         domain: SampleSpace | None,
#         values: pd.Series | None,
#         name: Hashable,
#     ) -> None:
#         from ..base.sample_space import SampleSpace

#         if (outputs is not None or domain is not None) and values is not None:
#             raise ValueError("Cannot provide both outputs/domain and values.")
#         if outputs is None and domain is None and values is None:
#             raise ValueError("Must provide either outputs/domain or values.")
#         if outputs is not None and not isinstance(outputs, dict):
#             raise TypeError("outputs must be a dict.")
#         if domain is not None and not isinstance(domain, SampleSpace):
#             raise TypeError("domain must be a SampleSpace instance.")
#         if (
#             outputs is not None
#             and domain is not None
#             and any(key not in domain for key in outputs)
#         ):
#             raise ValueError("All keys in outputs must be in the domain.")
#         if values is not None and not isinstance(values, pd.Series):
#             raise TypeError("values must be a pandas Series instance.")
#         if name is None:
#             raise ValueError("name cannot be None.")
#         if not isinstance(name, Hashable):
#             raise TypeError("name must be hashable.")

#     # --------------------- arithmetic operations --------------------- #

#     def _constructor_wrapper_with_other_rv(
#         self, other, outputs: dict[Hashable, Any], name: str
#     ) -> RandomVariable:
#         if self.probability_space is not None and other.probability_space is not None:
#             return RandomVariable.on_probability_space(
#                 outputs=outputs,
#                 probability_space=self.probability_space,
#                 name=name,
#             )
#         else:
#             return RandomVariable(
#                 domain=self.domain,
#                 outputs=outputs,
#                 name=name,
#             )

#     def _constructor_wrapper_with_other_scalar(
#         self, outputs: dict[Hashable, Any], name: str
#     ) -> RandomVariable:
#         if self.probability_space is not None:
#             return RandomVariable.on_probability_space(
#                 outputs=outputs,
#                 probability_space=self.probability_space,
#                 name=name,
#             )
#         else:
#             return RandomVariable(
#                 domain=self.domain,
#                 outputs=outputs,
#                 name=name,
#             )

#     def __add__(self, other):
#         if isinstance(other, RandomVariable):
#             if self.domain != other.domain:
#                 raise ValueError("Cannot add RandomVariables with different domains.")
#             new_values = (self.values + other.values).to_dict()
#             name = f"({self.name}+{other.name})"
#             return self._constructor_wrapper_with_other_rv(
#                 other=other,
#                 outputs=new_values,
#                 name=name,
#             )
#         else:
#             new_values = (self.values + other).to_dict()
#             name = f"({self.name}+{other})"
#             return self._constructor_wrapper_with_other_scalar(
#                 outputs=new_values,
#                 name=name,
#             )

#     def __radd__(self, other):
#         return self.__add__(other)

#     def __mul__(self, other):
#         if isinstance(other, RandomVariable):
#             if self.domain != other.domain:
#                 raise ValueError(
#                     "Cannot multiply RandomVariables with different domains."
#                 )
#             new_values = (self.values * other.values).to_dict()
#             name = f"({self.name}*{other.name})"
#             return self._constructor_wrapper_with_other_rv(
#                 other=other,
#                 outputs=new_values,
#                 name=name,
#             )
#         else:
#             new_values = (self.values * other).to_dict()
#             name = f"({self.name}*{other})"
#             return self._constructor_wrapper_with_other_scalar(
#                 outputs=new_values,
#                 name=name,
#             )

#     def __rmul__(self, other):
#         return self.__mul__(other)

#     def __sub__(self, other):
#         if isinstance(other, RandomVariable):
#             if self.domain != other.domain:
#                 raise ValueError(
#                     "Cannot subtract RandomVariables with different domains."
#                 )
#             new_values = (self.values - other.values).to_dict()
#             name = f"({self.name}-{other.name})"
#             return self._constructor_wrapper_with_other_rv(
#                 other=other,
#                 outputs=new_values,
#                 name=name,
#             )
#         else:
#             new_values = (self.values - other).to_dict()
#             name = f"({self.name}-{other})"
#             return self._constructor_wrapper_with_other_scalar(
#                 outputs=new_values,
#                 name=name,
#             )

#     def __rsub__(self, other):
#         if isinstance(other, RandomVariable):
#             if self.domain != other.domain:
#                 raise ValueError(
#                     "Cannot subtract RandomVariables with different domains."
#                 )
#             new_values = (other.values - self.values).to_dict()
#             name = f"({other.name}-{self.name})"
#             return self._constructor_wrapper_with_other_rv(
#                 other=other,
#                 outputs=new_values,
#                 name=name,
#             )
#         else:
#             new_values = (other - self.values).to_dict()
#             name = f"({other}-{self.name})"
#             return self._constructor_wrapper_with_other_scalar(
#                 outputs=new_values,
#                 name=name,
#             )

#     def __truediv__(self, other):
#         if isinstance(other, RandomVariable):
#             if self.domain != other.domain:
#                 raise ValueError(
#                     "Cannot divide RandomVariables with different domains."
#                 )
#             new_values = (self.values / other.values).to_dict()
#             name = f"({self.name}/{other.name})"
#             return self._constructor_wrapper_with_other_rv(
#                 other=other,
#                 outputs=new_values,
#                 name=name,
#             )
#         else:
#             new_values = (self.values / other).to_dict()
#             name = f"({self.name}/{other})"
#             return self._constructor_wrapper_with_other_scalar(
#                 outputs=new_values,
#                 name=name,
#             )

#     def __rtruediv__(self, other):
#         if isinstance(other, RandomVariable):
#             if self.domain != other.domain:
#                 raise ValueError(
#                     "Cannot divide RandomVariables with different domains."
#                 )
#             new_values = (other.values / self.values).to_dict()
#             name = f"({other.name}/{self.name})"
#             return self._constructor_wrapper_with_other_rv(
#                 other=other,
#                 outputs=new_values,
#                 name=name,
#             )
#         else:
#             new_values = (other / self.values).to_dict()
#             name = f"({other}/{self.name})"
#             return self._constructor_wrapper_with_other_scalar(
#                 outputs=new_values,
#                 name=name,
#             )

#     def __pow__(self, power):
#         if isinstance(power, RandomVariable):
#             if self.domain != power.domain:
#                 raise ValueError(
#                     "Cannot exponentiate RandomVariables with different domains."
#                 )
#             new_values = (self.values**power.values).to_dict()
#             name = f"({self.name}^{power.name})"
#             return self._constructor_wrapper_with_other_rv(
#                 other=power,
#                 outputs=new_values,
#                 name=name,
#             )
#         else:
#             new_values = (self.values**power).to_dict()
#             name = f"({self.name}^{power})"
#             return self._constructor_wrapper_with_other_scalar(
#                 outputs=new_values,
#                 name=name,
#             )

#     def __rpow__(self, other):
#         if isinstance(other, RandomVariable):
#             if self.domain != other.domain:
#                 raise ValueError(
#                     "Cannot exponentiate RandomVariables with different domains."
#                 )
#             new_values = (other.values**self.values).to_dict()
#             name = f"({other.name}^{self.name})"
#             return self._constructor_wrapper_with_other_rv(
#                 other=other,
#                 outputs=new_values,
#                 name=name,
#             )
#         else:
#             new_values = (other**self.values).to_dict()
#             name = f"({other}^{self.name})"
#             return self._constructor_wrapper_with_other_scalar(
#                 outputs=new_values,
#                 name=name,
#             )
