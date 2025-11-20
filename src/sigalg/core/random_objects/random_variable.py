from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from ..sigma_algebras import SigmaAlgebra
from ..spaces import Event, ProbabilitySpace, SampleSpace

if TYPE_CHECKING:
    from ..featurized_spaces import (
        FeaturizedProbabilitySpace,
        FeaturizedSampleSpace,
        SamplePointFeatures,
    )


class RandomVariable:

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        domain: SampleSpace | ProbabilitySpace,
        outputs: dict[Hashable, Any],
        function: Callable[[SamplePointFeatures], Any] | None = None,
        name: str = "X",
    ):
        self._validate_parameters(domain, outputs, name)
        self._domain = domain
        self._outputs = outputs
        self._values: pd.Series = pd.Series(outputs, name=name)
        self._function = function
        self._name = name
        self._unique_values: np.ndarray = self._values.unique()
        if isinstance(domain, SampleSpace):
            self._sigma_algebra = SigmaAlgebra(space=domain, atom_ids=outputs)
        else:
            self._sigma_algebra = SigmaAlgebra(
                space=domain.sample_space, atom_ids=outputs
            )
        self._generate_range()

    # --------------------- properties --------------------- #

    @property
    def domain(self) -> SampleSpace | ProbabilitySpace:
        return self._domain

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

    @property
    def outputs(self):
        return self._outputs

    @property
    def probability_measure(self):
        return self._probability_measure

    @property
    def range(self) -> FeaturizedSampleSpace | FeaturizedProbabilitySpace:
        return self._range

    # --------------------- methods --------------------- #

    def set_name(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        self._name = name
        self._values.name = name

    def _generate_range(self) -> None:
        from ..featurized_spaces import (
            FeaturizedProbabilitySpace,
            FeaturizedSampleSpace,
        )

        range_values = self._unique_values.reshape(-1, 1)
        range_indices = [
            f"{self._name.lower()}{i}" for i in range(len(self._unique_values))
        ]
        self._range_idx_to_value = dict(zip(range_indices, self._unique_values))
        self._range_value_to_idx = dict(zip(self._unique_values, range_indices))
        range_sample_space = SampleSpace(range_indices)
        fss = FeaturizedSampleSpace(
            features=range_values,
            sample_space=range_sample_space,
            feature_index=[self._name],
        )
        if isinstance(self._domain, ProbabilitySpace):
            events = self._sigma_algebra.to_events()
            probabilities = {
                range_idx: self._domain.P(event)
                for range_idx, event in zip(range_sample_space, events.values())
            }
            range_probability_space = ProbabilitySpace(
                sample_space=range_sample_space, probabilities=probabilities
            )
            self._range = FeaturizedProbabilitySpace(
                probability_space=range_probability_space, featurized_sample_space=fss
            )
            self._probability_measure = range_probability_space.probability_measure
        else:
            self._range = fss
            self._probability_measure = None

    # --------------------- probability methods --------------------- #

    def P(self, key: Hashable | Event) -> Real:
        if self._probability_measure is None:
            raise ValueError(
                "This RandomVariable does not have an associated ProbabilityMeasure."
            )
        else:
            idx = self._range_value_to_idx[key]
            return self._probability_measure.P(idx)

    # --------------------- conversion methods --------------------- #

    def to_pandas(self) -> pd.Series:
        return self._values.copy()

    def to_dict(self) -> dict[Hashable, Any]:
        return dict(self._values)

    # --------------------- class methods --------------------- #

    @classmethod
    def from_features(
        cls,
        domain_features: FeaturizedSampleSpace | FeaturizedProbabilitySpace,
        function: Callable[[SamplePointFeatures], Any],
        name: str = "X",
    ):
        from ..featurized_spaces import FeaturizedProbabilitySpace

        data = domain_features.apply_to_features(function)
        if isinstance(domain_features, FeaturizedProbabilitySpace):
            domain = domain_features.probability_space
        else:
            domain = domain_features.sample_space
        outputs = data.to_dict()
        return cls(domain=domain, outputs=outputs, function=function, name=name)

    @classmethod
    def from_values(
        cls,
        domain: SampleSpace,
        values: pd.Series,
        name: str = "X",
    ):
        outputs = values.to_dict()
        return cls(domain=domain, outputs=outputs, name=name)

    # --------------------- call methods --------------------- #

    def __call__(self, key: SamplePointFeatures | Hashable | Event) -> Any:
        from ..featurized_spaces import SamplePointFeatures

        if isinstance(key, SamplePointFeatures):
            if self._function is None:
                raise ValueError("This RandomVariable was not defined with a function.")
            return self._function(key)
        elif isinstance(key, Event):
            outputs = {idx: self._values[idx] for idx in key}
            return RandomVariable(
                domain=key.to_sample_space(), outputs=outputs, name=self._name
            )
        elif isinstance(key, ProbabilitySpace):
            outputs = {idx: self._values[idx] for idx in key.sample_space}
            return RandomVariable(domain=key, outputs=outputs, name=self._name)
        else:
            values_dict = dict(self._values)
            if key not in values_dict:
                raise KeyError(f"Key '{key}' not found in domain.")
            return values_dict[key]

    # --------------------- equality and hashing --------------------- #

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RandomVariable):
            return False
        if self.domain != other.domain:
            return False
        return self._values.equals(other._values)

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"RandomVariable(name='{self.name}',\n{self._values})"

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        domain: SampleSpace,
        values: dict[Hashable, Any],
        name: str,
    ) -> None:
        if not isinstance(domain, SampleSpace | ProbabilitySpace):
            raise TypeError(
                "domain must be a SampleSpace or ProbabilitySpace instance."
            )

        if not isinstance(values, dict):
            raise TypeError("values must be a dict.")

        if not isinstance(name, str):
            raise TypeError("name must be a string.")

        if set(values.keys()) != set(domain.values):
            raise ValueError("values keys must match domain indices.")

    # --------------------- arithmetic operations --------------------- #

    def __add__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError("Cannot add RandomVariables with different domains.")
            new_values = self._values + other._values
            return RandomVariable(
                domain=self.domain,
                outputs=new_values.to_dict(),
                name=f"({self.name}+{other.name})",
            )
        else:
            new_values = self._values + other
            return RandomVariable(
                domain=self.domain,
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
                outputs=new_values.to_dict(),
                name=f"({self.name}*{other.name})",
            )
        else:
            new_values = self._values * other
            return RandomVariable(
                domain=self.domain,
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
                outputs=new_values.to_dict(),
                name=f"({self.name}-{other.name})",
            )
        else:
            new_values = self._values - other
            return RandomVariable(
                domain=self.domain,
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
                outputs=new_values.to_dict(),
                name=f"({other.name}-{self.name})",
            )
        else:
            new_values = other - self._values
            return RandomVariable(
                domain=self.domain,
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
                outputs=new_values.to_dict(),
                name=f"({self.name}/{other.name})",
            )
        else:
            new_values = self._values / other
            return RandomVariable(
                domain=self.domain,
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
                outputs=new_values.to_dict(),
                name=f"({other.name}/{self.name})",
            )
        else:
            new_values = other / self._values
            return RandomVariable(
                domain=self.domain,
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
                outputs=new_values.to_dict(),
                name=f"({self.name}^{power.name})",
            )
        else:
            new_values = self._values**power
            return RandomVariable(
                domain=self.domain,
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
                outputs=new_values.to_dict(),
                name=f"({other.name}^{self.name})",
            )
        else:
            new_values = other**self._values
            return RandomVariable(
                domain=self.domain,
                outputs=new_values.to_dict(),
                name=f"({other}^{self.name})",
            )
