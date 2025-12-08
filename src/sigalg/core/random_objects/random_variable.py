from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..featurized_spaces.feature_embedding import FeatureEmbedding
    from ..featurized_spaces.featurized_probability_space import (
        FeaturizedProbabilitySpace,
    )
    from ..featurized_spaces.sample_point_features import SamplePointFeatures
    from ..probability_measures.probability_measure import ProbabilityMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.event import Event
    from ..spaces.probability_space import ProbabilitySpace
    from ..spaces.sample_space import SampleSpace


class RandomVariable:

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        outputs: dict[Hashable, Any],
        domain: SampleSpace | None = None,
        probability_space: ProbabilitySpace | None = None,
        function: Callable[[SamplePointFeatures], Any] | None = None,
        name: str = "X",
    ):
        from ..sigma_algebras import SigmaAlgebra

        self._validate_parameters(outputs, domain, probability_space, function, name)
        if probability_space is not None:
            domain = probability_space.sample_space
            self._probability_space = probability_space
        else:
            self._probability_space = None
        self._domain = domain
        self._outputs = outputs
        self._values = pd.Series(outputs, name=name)
        self._values.index.name = domain.name
        self._function = function
        self._name = name
        self._unique_values: np.ndarray = self._values.unique()
        self._sigma_algebra = SigmaAlgebra(
            sample_id_to_atom_id=outputs,
            sample_space=domain,
            probability_space=probability_space,
            name=f"sigma({name})",
        )
        self._generate_range()

    # --------------------- properties --------------------- #

    @property
    def domain(self) -> SampleSpace:
        return self._domain

    @property
    def probability_space(self) -> ProbabilitySpace | None:
        return self._probability_space

    @property
    def values(self) -> pd.Series:
        return self._values.copy()

    @property
    def function(self) -> Callable | None:
        return self._function

    @property
    def sigma_algebra(self) -> SigmaAlgebra:
        return self._sigma_algebra

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        if not isinstance(new_name, str):
            raise TypeError("name must be a string.")
        self._name = new_name
        self._values.name = new_name

    @property
    def outputs(self) -> dict[Hashable, Any]:
        return self._outputs.copy()

    @property
    def probability_measure(self) -> ProbabilityMeasure | None:
        return self._probability_measure

    @property
    def range(self):
        return self._range

    # --------------------- methods --------------------- #

    def is_measurable(self, sigma_algebra: SigmaAlgebra = None) -> bool:
        if sigma_algebra is None and self.probability_space is None:
            raise ValueError(
                "Either sigma_algebra or probability_space must be provided."
            )
        elif sigma_algebra is None and self.probability_space is not None:
            sigma_algebra = self.probability_space.sigma_algebra

        return self.sigma_algebra <= sigma_algebra

    def add_probability_measure_to_domain(
        self, probability_measure: ProbabilityMeasure
    ) -> None:
        from ..spaces.probability_space import ProbabilitySpace

        if self.probability_space is not None:
            raise ValueError(
                "This RandomVariable already has an associated ProbabilitySpace."
            )
        if probability_measure.sample_space != self.domain:
            raise ValueError(
                "The sample space of the provided ProbabilityMeasure does not match the domain of this RandomVariable."
            )
        self._probability_measure = probability_measure
        self._probability_space = ProbabilitySpace(
            sample_space=self.domain,
            sigma_algebra=self.sigma_algebra,
            probability_measure=probability_measure,
        )
        self._generate_range()

    def _generate_range(self) -> None:
        from ..probability_measures import ProbabilityMeasure
        from ..spaces.probability_space import ProbabilitySpace
        from ..spaces.sample_space import SampleSpace
        from .random_variable_range import (
            RandomVariableRange,
            RandomVariableRangeWithProbability,
        )

        range_ids = [
            f"{self._name.lower()}{i}" for i in range(len(self._unique_values))
        ]
        range_sample_space = SampleSpace(range_ids, values_name="outputs")

        self._range_id_to_rv_value = dict(zip(range_ids, self._unique_values))
        self._rv_value_to_range_id = dict(zip(self._unique_values, range_ids))

        range_values = self._unique_values.reshape(-1, 1)
        range_df = pd.DataFrame(
            data=range_values, index=range_sample_space.values, columns=[self.name]
        )

        from ..featurized_spaces.feature_index import FeatureIndex

        range_feature_index = FeatureIndex([self.name], values_name=self.name)

        rv_range = RandomVariableRange(
            sample_space=range_sample_space,
            feature_index=range_feature_index,
            values=range_df,
            name=self.name,
        )

        if self.probability_space is not None:
            level_sets = self._sigma_algebra.atom_id_to_event
            range_probabilities = {
                range_value: self.probability_space.P(level_set)
                for range_value, level_set in zip(
                    range_sample_space, level_sets.values()
                )
            }
            range_probability_measure = ProbabilityMeasure(
                sample_space=range_sample_space,
                probabilities=range_probabilities,
                name="P_X",
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

    # --------------------- probability methods --------------------- #

    def P(self, key) -> Real:
        if self._probability_measure is None:
            raise ValueError(
                "This RandomVariable does not have an associated ProbabilityMeasure."
            )
        if key not in self._rv_value_to_range_id:
            raise ValueError(
                f"Value {key} is not in the range of this random variable."
            )
        idx = self._rv_value_to_range_id[key]
        return self._probability_measure.P(idx)

    # --------------------- class methods --------------------- #

    @classmethod
    def from_features(
        cls,
        function: Callable[[SamplePointFeatures], Any],
        feature_embedding: FeatureEmbedding | None = None,
        fps: FeaturizedProbabilitySpace | None = None,
        name: str = "X",
    ):
        if fps is not None:
            feature_embedding = fps.feature_embedding
        data = feature_embedding.apply_to_features(function)
        domain = feature_embedding.sample_space
        probability_space = fps.probability_space if fps is not None else None
        outputs = data.to_dict()
        return cls(
            outputs=outputs,
            domain=domain,
            probability_space=probability_space,
            function=function,
            name=name,
        )

    @classmethod
    def from_values(
        cls,
        values: pd.Series,
        domain: SampleSpace | None = None,
        probability_space: ProbabilitySpace | None = None,
        name: str = "X",
    ):
        if domain is None and probability_space is not None:
            domain = probability_space.sample_space
        outputs = values.to_dict()
        return cls(
            domain=domain,
            probability_space=probability_space,
            outputs=outputs,
            name=name,
        )

    # --------------------- call methods --------------------- #

    def __call__(
        self, key: SamplePointFeatures | Hashable | Event | ProbabilitySpace
    ) -> Any:
        from ..featurized_spaces.sample_point_features import SamplePointFeatures
        from ..spaces.event import Event
        from ..spaces.probability_space import ProbabilitySpace

        if isinstance(key, SamplePointFeatures):
            if self._function is None:
                raise ValueError("This RandomVariable was not defined with a function.")
            return self._function(key)
        elif isinstance(key, Event):
            outputs = {idx: self._values[idx] for idx in key.values}
            return RandomVariable(
                domain=key.to_sample_space(), outputs=outputs, name=self._name
            )
        elif isinstance(key, ProbabilitySpace):
            outputs = {idx: self._values[idx] for idx in key.sample_space.values}
            return RandomVariable(
                probability_space=key, outputs=outputs, name=self._name
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
        return self._values.equals(other._values)

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"RandomVariable(name={self.name}, domain={self.domain.name})"

    def __str__(self) -> str:
        header = f"Random variable {self.name} on sample space '{self.domain.name}'"
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
        outputs: dict[Hashable, Any],
        domain: SampleSpace | None,
        probability_space: ProbabilitySpace | None,
        function: Callable | None,
        name: str,
    ) -> None:
        from ..spaces.probability_space import ProbabilitySpace
        from ..spaces.sample_space import SampleSpace

        if domain is None and probability_space is None:
            raise ValueError("Either domain or probability_space must be provided.")
        if domain is not None and not isinstance(domain, SampleSpace):
            raise TypeError("domain must be a SampleSpace instance.")
        if probability_space is not None and not isinstance(
            probability_space, ProbabilitySpace
        ):
            raise TypeError("probability_space must be a ProbabilitySpace instance.")
        if domain is not None and probability_space is not None:
            if domain != probability_space.sample_space:
                raise ValueError(
                    "domain and probability_space.sample_space must be the same."
                )
        if not isinstance(outputs, dict):
            raise TypeError("outputs must be a dict.")
        if not outputs:
            raise ValueError("outputs dictionary cannot be empty.")
        if function is not None and not callable(function):
            raise TypeError("function must be callable.")
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if not name:
            raise ValueError("name cannot be an empty string.")
        actual_domain = (
            probability_space.sample_space if probability_space is not None else domain
        )
        if set(outputs.keys()) != set(actual_domain.values):
            raise ValueError("outputs keys must match domain indices.")

    # --------------------- arithmetic operations --------------------- #

    def __add__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError("Cannot add RandomVariables with different domains.")
            new_values = self._values + other._values
            return RandomVariable(
                domain=self.domain,
                probability_space=self.probability_space,
                outputs=new_values.to_dict(),
                name=f"({self.name}+{other.name})",
            )
        else:
            new_values = self._values + other
            return RandomVariable(
                domain=self.domain,
                probability_space=self.probability_space,
                outputs=new_values.to_dict(),
                name=f"({self.name}+{other})",
            )

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot multiply RandomVariables with different domains."
                )
            new_values = self._values * other._values
            return RandomVariable(
                domain=self.domain,
                probability_space=self.probability_space,
                outputs=new_values.to_dict(),
                name=f"({self.name}*{other.name})",
            )
        else:
            new_values = self._values * other
            return RandomVariable(
                domain=self.domain,
                probability_space=self.probability_space,
                outputs=new_values.to_dict(),
                name=f"({self.name}*{other})",
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot subtract RandomVariables with different domains."
                )
            new_values = self._values - other._values
            return RandomVariable(
                domain=self.domain,
                probability_space=self.probability_space,
                outputs=new_values.to_dict(),
                name=f"({self.name}-{other.name})",
            )
        else:
            new_values = self._values - other
            return RandomVariable(
                domain=self.domain,
                probability_space=self.probability_space,
                outputs=new_values.to_dict(),
                name=f"({self.name}-{other})",
            )

    def __rsub__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot subtract RandomVariables with different domains."
                )
            new_values = other._values - self._values
            return RandomVariable(
                domain=self.domain,
                probability_space=self.probability_space,
                outputs=new_values.to_dict(),
                name=f"({other.name}-{self.name})",
            )
        else:
            new_values = other - self._values
            return RandomVariable(
                domain=self.domain,
                probability_space=self.probability_space,
                outputs=new_values.to_dict(),
                name=f"({other}-{self.name})",
            )

    def __truediv__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot divide RandomVariables with different domains."
                )
            new_values = self._values / other._values
            return RandomVariable(
                domain=self.domain,
                probability_space=self.probability_space,
                outputs=new_values.to_dict(),
                name=f"({self.name}/{other.name})",
            )
        else:
            new_values = self._values / other
            return RandomVariable(
                domain=self.domain,
                probability_space=self.probability_space,
                outputs=new_values.to_dict(),
                name=f"({self.name}/{other})",
            )

    def __rtruediv__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot divide RandomVariables with different domains."
                )
            new_values = other._values / self._values
            return RandomVariable(
                domain=self.domain,
                probability_space=self.probability_space,
                outputs=new_values.to_dict(),
                name=f"({other.name}/{self.name})",
            )
        else:
            new_values = other / self._values
            return RandomVariable(
                domain=self.domain,
                probability_space=self.probability_space,
                outputs=new_values.to_dict(),
                name=f"({other}/{self.name})",
            )

    def __pow__(self, power):
        if isinstance(power, RandomVariable):
            if self.domain != power.domain:
                raise ValueError(
                    "Cannot exponentiate RandomVariables with different domains."
                )
            new_values = self._values**power._values
            return RandomVariable(
                domain=self.domain,
                probability_space=self.probability_space,
                outputs=new_values.to_dict(),
                name=f"({self.name}^{power.name})",
            )
        else:
            new_values = self._values**power
            return RandomVariable(
                domain=self.domain,
                probability_space=self.probability_space,
                outputs=new_values.to_dict(),
                name=f"({self.name}^{power})",
            )

    def __rpow__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot exponentiate RandomVariables with different domains."
                )
            new_values = other._values**self._values
            return RandomVariable(
                domain=self.domain,
                probability_space=self.probability_space,
                outputs=new_values.to_dict(),
                name=f"({other.name}^{self.name})",
            )
        else:
            new_values = other**self._values
            return RandomVariable(
                domain=self.domain,
                probability_space=self.probability_space,
                outputs=new_values.to_dict(),
                name=f"({other}^{self.name})",
            )
