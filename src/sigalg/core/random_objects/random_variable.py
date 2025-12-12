from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..base.event import Event
    from ..base.probability_space import ProbabilitySpace
    from ..base.sample_space import SampleSpace
    from ..featurized_spaces.feature_embedding import FeatureEmbedding
    from ..featurized_spaces.sample_point_features import SamplePointFeatures
    from ..probability_measures.probability_measure import ProbabilityMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from .random_variable_range import (
        RandomVariableRange,
        RandomVariableRangeWithProbability,
    )


class RandomVariable:

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        outputs: dict[Hashable, Any] | None = None,
        domain: SampleSpace | None = None,
        values: pd.Series | None = None,
        name: str = "X",
    ):
        self._validate_parameters(
            outputs=outputs,
            domain=domain,
            values=values,
            name=name,
        )

        if values is not None:
            self.values = values.copy()
            self._outputs = None
            self._domain = None
            self._name = values.name if values.name is not None else name
        elif outputs is not None:
            self.values = pd.Series(data=outputs, name=name)
            self.values.index.name = "sample"
            self._outputs = outputs
            self._domain = domain
            self._name = name

        self.probability_space: ProbabilitySpace | None = None
        self.function: Callable[[SamplePointFeatures], Any] | None = None

        self._sigma_algebra: SigmaAlgebra = None
        self._range: RandomVariableRange | RandomVariableRangeWithProbability = None
        self._probability_measure: ProbabilityMeasure | None = None
        self._unique_values: np.ndarray = None
        self._rv_value_to_range_id: dict = None

    # --------------------- properties --------------------- #

    @property
    def domain(self) -> SampleSpace:
        from ..base.sample_space import SampleSpace

        if self._domain is None:
            self._domain = SampleSpace(indices=self.values.index.to_list())
        return self._domain

    @property
    def outputs(self) -> dict[Hashable, Any]:
        if self._outputs is None:
            self._outputs = self.values.to_dict()
        return self._outputs

    @property
    def sigma_algebra(self) -> SigmaAlgebra:
        from ..sigma_algebras import SigmaAlgebra

        if self._sigma_algebra is None:
            self._sigma_algebra = SigmaAlgebra(
                sample_id_to_atom_id=self.outputs,
                sample_space=self.domain,
                name=f"sigma({self._name})",
            )
        return self._sigma_algebra

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        if not isinstance(new_name, str):
            raise TypeError("name must be a string.")
        self._name = new_name
        self.values.name = new_name

    @property
    def range(self):
        if self._range is None:
            from ..base.probability_space import ProbabilitySpace
            from ..base.sample_space import SampleSpace
            from ..probability_measures import ProbabilityMeasure
            from .random_variable_range import (
                RandomVariableRange,
                RandomVariableRangeWithProbability,
            )

            range_sample_space = SampleSpace.generate_default(
                prefix=self._name.lower(),
                size=len(self.unique_values),
                values_name="outputs",
                name=f"{self._name}_range",
            )

            self._rv_value_to_range_id = dict(
                zip(self.unique_values, range_sample_space)
            )

            range_values = self.unique_values.reshape(-1, 1)
            range_df = pd.DataFrame(
                data=range_values, index=range_sample_space.values, columns=[self.name]
            )
            rv_range = RandomVariableRange.from_df(df=range_df, name=self.name)

            if self.probability_space is not None:
                level_sets = self.sigma_algebra.atom_id_to_event
                range_probabilities = {
                    range_value: self.probability_space.P(level_set)
                    for range_value, level_set in zip(
                        range_sample_space, level_sets.values()
                    )
                }
                range_probability_measure = ProbabilityMeasure(
                    sample_space=range_sample_space,
                    probabilities=range_probabilities,
                    name=f"P_{self._name}",
                )
                range_probability_space = ProbabilitySpace(
                    sample_space=range_sample_space,
                    probability_measure=range_probability_measure,
                )
                self._range = RandomVariableRangeWithProbability(
                    sample_space=range_sample_space,
                    feature_embedding=rv_range,
                    probability_measure=range_probability_measure,
                )
                self._probability_measure = range_probability_space.probability_measure
            else:
                self._range = rv_range
                self._probability_measure = None
        return self._range

    @property
    def probability_measure(self):
        if self._probability_measure is None:
            _ = self.range  # trigger computation of the range
        return self._probability_measure

    @property
    def rv_value_to_range_id(self):
        if self._rv_value_to_range_id is None:
            _ = self.range  # trigger computation of the range
        return self._rv_value_to_range_id

    @property
    def unique_values(self):
        if self._unique_values is None:
            self._unique_values = self.values.unique()
        return self._unique_values

    # --------------------- methods --------------------- #

    def is_measurable(self, sigma_algebra: SigmaAlgebra = None) -> bool:
        if sigma_algebra is None and self.probability_space is None:
            raise ValueError(
                "Either sigma_algebra or probability_space must be provided."
            )
        elif sigma_algebra is None and self.probability_space is not None:
            sigma_algebra = self.probability_space.sigma_algebra

        return self.sigma_algebra <= sigma_algebra

    # --------------------- probability methods --------------------- #

    def P(self, key) -> Real:
        if self.probability_measure is None:
            raise ValueError(
                "This RandomVariable does not have an associated ProbabilityMeasure."
            )
        if key not in self.rv_value_to_range_id:
            raise ValueError(
                f"Value {key} is not in the range of this random variable."
            )
        idx = self.rv_value_to_range_id[key]
        return self.probability_measure.P(idx)

    def add_probability_measure_to_domain(
        self, probability_measure: ProbabilityMeasure
    ) -> None:
        from ..base.probability_space import ProbabilitySpace

        if self.probability_space is not None:
            raise ValueError(
                "This RandomVariable already has an associated ProbabilitySpace."
            )
        if probability_measure.sample_space != self.domain:
            raise ValueError(
                "The sample space of the provided ProbabilityMeasure does not match the domain of this RandomVariable."
            )
        self._probability_measure = probability_measure
        self.probability_space = ProbabilitySpace(
            sample_space=self.domain,
            sigma_algebra=self.sigma_algebra,
            probability_measure=probability_measure,
        )
        self._range = None  # reset range to trigger new computation on next call

    # --------------------- factory methods --------------------- #

    @classmethod
    def on_probability_space(
        cls,
        outputs: dict[Hashable, Any],
        probability_space: ProbabilitySpace,
        name: str = "X",
    ):
        domain = probability_space.sample_space
        rv = cls(outputs=outputs, domain=domain, name=name)
        rv.probability_space = probability_space
        rv._probability_measure = probability_space.probability_measure
        return rv

    @classmethod
    def from_features(
        cls,
        function: Callable[[SamplePointFeatures], Any],
        feature_embedding: FeatureEmbedding,
        name: str = "X",
    ):
        data = feature_embedding.apply_to_features(function)
        domain = feature_embedding.domain
        outputs = data.to_dict()
        rv = cls(outputs=outputs, domain=domain, name=name)
        rv.function = function
        return rv

    # --------------------- call methods --------------------- #

    def __call__(
        self, key: SamplePointFeatures | Hashable | Event | ProbabilitySpace
    ) -> Any:
        from ..base.event import Event
        from ..base.probability_space import ProbabilitySpace
        from ..featurized_spaces.sample_point_features import SamplePointFeatures

        if isinstance(key, SamplePointFeatures):
            if self.function is None:
                raise ValueError("This RandomVariable was not defined with a function.")
            return self.function(key)
        elif isinstance(key, Event):
            outputs = {idx: self.values[idx] for idx in key.values}
            return RandomVariable(
                domain=key.to_sample_space(), outputs=outputs, name=self._name
            )
        elif isinstance(key, ProbabilitySpace):
            outputs = {idx: self.values[idx] for idx in key.sample_space.values}
            return RandomVariable.on_probability_space(
                outputs=outputs, probability_space=key, name=self._name
            )
        else:
            if key not in self.outputs:
                raise KeyError(f"Key '{key}' not found in domain.")
            return self.outputs[key]

    # --------------------- equality --------------------- #

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RandomVariable):
            return False
        if self.domain != other.domain:
            return False
        return self.values.equals(other.values)

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"RandomVariable(name={self.name}, domain={self.domain.name})"

    def __str__(self) -> str:
        header = f"Random variable '{self.name}' on sample space '{self.domain.name}'"
        separator = "=" * len(header)
        return (
            header
            + "\n"
            + separator
            + "\n\n* "
            + repr(self.domain)
            + "\n\n* Values:\n"
            + f"{self.values.to_frame()}"
            + "\n\n* "
            + repr(self.range)
        )

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        outputs: dict[Hashable, Any] | None,
        domain: SampleSpace | None,
        values: pd.Series | None,
        name: str,
    ) -> None:
        from ..base.sample_space import SampleSpace

        if (outputs is not None or domain is not None) and values is not None:
            raise ValueError("Cannot provide both outputs/domain and values.")
        if outputs is None and domain is None and values is None:
            raise ValueError("Must provide either outputs/domain or values.")
        if outputs is not None and not isinstance(outputs, dict):
            raise TypeError("outputs must be a dict.")
        if domain is not None and not isinstance(domain, SampleSpace):
            raise TypeError("domain must be a SampleSpace instance.")
        if (
            outputs is not None
            and domain is not None
            and any(key not in domain for key in outputs)
        ):
            raise ValueError("All keys in outputs must be in the domain.")
        if values is not None and not isinstance(values, pd.Series):
            raise TypeError("values must be a pandas Series instance.")
        if not isinstance(name, str):
            raise TypeError("name must be a string.")

    # --------------------- arithmetic operations --------------------- #

    def _constructor_wrapper_with_other_rv(
        self, other, outputs: dict[Hashable, Any], name: str
    ) -> RandomVariable:
        if self.probability_space is not None and other.probability_space is not None:
            return RandomVariable.on_probability_space(
                outputs=outputs,
                probability_space=self.probability_space,
                name=name,
            )
        else:
            return RandomVariable(
                domain=self.domain,
                outputs=outputs,
                name=name,
            )

    def _constructor_wrapper_with_other_scalar(
        self, outputs: dict[Hashable, Any], name: str
    ) -> RandomVariable:
        if self.probability_space is not None:
            return RandomVariable.on_probability_space(
                outputs=outputs,
                probability_space=self.probability_space,
                name=name,
            )
        else:
            return RandomVariable(
                domain=self.domain,
                outputs=outputs,
                name=name,
            )

    def __add__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError("Cannot add RandomVariables with different domains.")
            new_values = (self.values + other.values).to_dict()
            name = f"({self.name}+{other.name})"
            return self._constructor_wrapper_with_other_rv(
                other=other,
                outputs=new_values,
                name=name,
            )
        else:
            new_values = (self.values + other).to_dict()
            name = f"({self.name}+{other})"
            return self._constructor_wrapper_with_other_scalar(
                outputs=new_values,
                name=name,
            )

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot multiply RandomVariables with different domains."
                )
            new_values = (self.values * other.values).to_dict()
            name = f"({self.name}*{other.name})"
            return self._constructor_wrapper_with_other_rv(
                other=other,
                outputs=new_values,
                name=name,
            )
        else:
            new_values = (self.values * other).to_dict()
            name = f"({self.name}*{other})"
            return self._constructor_wrapper_with_other_scalar(
                outputs=new_values,
                name=name,
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot subtract RandomVariables with different domains."
                )
            new_values = (self.values - other.values).to_dict()
            name = f"({self.name}-{other.name})"
            return self._constructor_wrapper_with_other_rv(
                other=other,
                outputs=new_values,
                name=name,
            )
        else:
            new_values = (self.values - other).to_dict()
            name = f"({self.name}-{other})"
            return self._constructor_wrapper_with_other_scalar(
                outputs=new_values,
                name=name,
            )

    def __rsub__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot subtract RandomVariables with different domains."
                )
            new_values = (other.values - self.values).to_dict()
            name = f"({other.name}-{self.name})"
            return self._constructor_wrapper_with_other_rv(
                other=other,
                outputs=new_values,
                name=name,
            )
        else:
            new_values = (other - self.values).to_dict()
            name = f"({other}-{self.name})"
            return self._constructor_wrapper_with_other_scalar(
                outputs=new_values,
                name=name,
            )

    def __truediv__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot divide RandomVariables with different domains."
                )
            new_values = (self.values / other.values).to_dict()
            name = f"({self.name}/{other.name})"
            return self._constructor_wrapper_with_other_rv(
                other=other,
                outputs=new_values,
                name=name,
            )
        else:
            new_values = (self.values / other).to_dict()
            name = f"({self.name}/{other})"
            return self._constructor_wrapper_with_other_scalar(
                outputs=new_values,
                name=name,
            )

    def __rtruediv__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot divide RandomVariables with different domains."
                )
            new_values = (other.values / self.values).to_dict()
            name = f"({other.name}/{self.name})"
            return self._constructor_wrapper_with_other_rv(
                other=other,
                outputs=new_values,
                name=name,
            )
        else:
            new_values = (other / self.values).to_dict()
            name = f"({other}/{self.name})"
            return self._constructor_wrapper_with_other_scalar(
                outputs=new_values,
                name=name,
            )

    def __pow__(self, power):
        if isinstance(power, RandomVariable):
            if self.domain != power.domain:
                raise ValueError(
                    "Cannot exponentiate RandomVariables with different domains."
                )
            new_values = (self.values**power.values).to_dict()
            name = f"({self.name}^{power.name})"
            return self._constructor_wrapper_with_other_rv(
                other=power,
                outputs=new_values,
                name=name,
            )
        else:
            new_values = (self.values**power).to_dict()
            name = f"({self.name}^{power})"
            return self._constructor_wrapper_with_other_scalar(
                outputs=new_values,
                name=name,
            )

    def __rpow__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot exponentiate RandomVariables with different domains."
                )
            new_values = (other.values**self.values).to_dict()
            name = f"({other.name}^{self.name})"
            return self._constructor_wrapper_with_other_rv(
                other=other,
                outputs=new_values,
                name=name,
            )
        else:
            new_values = (other**self.values).to_dict()
            name = f"({other}^{self.name})"
            return self._constructor_wrapper_with_other_scalar(
                outputs=new_values,
                name=name,
            )
