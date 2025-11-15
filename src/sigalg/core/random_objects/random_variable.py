from typing import Any, Callable

import pandas as pd

from ..feature_representations import SampleFeatures
from ..sigma_algebras import SigmaAlgebra
from ..spaces import SampleSpace


class RandomVariable:
    def __init__(
        self,
        domain: SampleSpace,
        values,
        function=None,
        name: str = "X",
    ):
        from ..spaces import ProbabilitySpace

        self._domain = domain
        self._values = pd.Series(values, name=name)
        self._function = function
        self._name = name
        self._unique_values = self._values.unique()

        atom_ids = self._values.to_dict()
        self._sigma_algebra = SigmaAlgebra(sample_space=domain, atom_ids=atom_ids)

        if isinstance(domain, ProbabilitySpace):
            probabilities = {}
            for val in self._unique_values:
                preimage_indices = self._values[self._values == val].index.tolist()
                prob = sum(self._domain.P(idx) for idx in preimage_indices)
                probabilities[val] = prob
            self._probabilities = probabilities
        else:
            self._probabilities = None

    @classmethod
    def from_features(cls, domain_features, function, name="X"):
        data = domain_features.apply_to_row(function)
        domain_features = domain_features.sample_space
        values = data.to_dict()
        return cls(domain=domain_features, values=values, function=function, name=name)

    @property
    def domain(self) -> SampleSpace:
        return self._domain

    @property
    def values(self) -> pd.Series:
        return self._values

    @property
    def function(self) -> Callable:
        return self._function

    @property
    def sigma_algebra(self) -> SigmaAlgebra:
        return self._sigma_algebra

    @property
    def name(self) -> str:
        return self._name

    @property
    def range(self):
        from ..spaces import ProbabilitySpace, SampleSpace

        if not isinstance(self.domain, ProbabilitySpace):
            return SampleSpace(self._unique_values)
        else:
            return ProbabilitySpace(
                list(self._unique_values), probabilities=self._probabilities
            )

    @property
    def probability_measure(self):
        if self._probabilities is None:
            raise ValueError(
                "The probability measure is only defined for RandomVariables "
                "with a ProbabilitySpace as their domain."
            )
        else:
            return self.range.probability_measure

    def set_name(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        self._values.name = name
        self._name = name

    def __call__(self, key) -> Any:
        if isinstance(key, SampleFeatures):
            if self._function is None:
                raise ValueError("This RandomVariable was not defined with a function.")
            else:
                return self._function(key)
        elif isinstance(key, str):
            return self._values[key]

    def __eq__(self, other) -> bool:
        if not isinstance(other, RandomVariable):
            return False
        elif self.domain != other.domain:
            return False
        elif not self.values.equals(other.values):
            return False
        else:
            return True

    def __add__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError("Cannot add RandomVariables with different domains.")
            new_values = self.values + other.values
            return RandomVariable(
                domain=self.domain,
                values=new_values.to_dict(),
                name=f"{self.name}+{other.name}",
            )
        else:
            new_values = self.values + other
            return RandomVariable(
                domain=self.domain,
                values=new_values.to_dict(),
                name=f"{self.name}+{other}",
            )

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot multiply RandomVariables with different domains."
                )
            new_values = self.values * other.values
            return RandomVariable(
                domain=self.domain,
                values=new_values.to_dict(),
                name=f"{self.name}*{other.name}",
            )
        else:
            new_values = self.values * other
            return RandomVariable(
                domain=self.domain,
                values=new_values.to_dict(),
                name=f"{self.name}*{other}",
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot subtract RandomVariables with different domains."
                )
            new_values = self.values - other.values
            return RandomVariable(
                domain=self.domain,
                values=new_values.to_dict(),
                name=f"{self.name}-{other.name}",
            )
        else:
            new_values = self.values - other
            return RandomVariable(
                domain=self.domain,
                values=new_values.to_dict(),
                name=f"{self.name}-{other}",
            )

    def __rsub__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot subtract RandomVariables with different domains."
                )
            new_values = other.values - self.values
            return RandomVariable(
                domain=self.domain,
                values=new_values.to_dict(),
                name=f"{other.name}-{self.name}",
            )
        else:
            new_values = other - self.values
            return RandomVariable(
                domain=self.domain,
                values=new_values.to_dict(),
                name=f"{other}-{self.name}",
            )

    def __truediv__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot divide RandomVariables with different domains."
                )
            new_values = self.values / other.values
            return RandomVariable(
                domain=self.domain,
                values=new_values.to_dict(),
                name=f"{self.name}/{other.name}",
            )
        else:
            new_values = self.values / other
            return RandomVariable(
                domain=self.domain,
                values=new_values.to_dict(),
                name=f"{self.name}/{other}",
            )

    def __rtruediv__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot divide RandomVariables with different domains."
                )
            new_values = other.values / self.values
            return RandomVariable(
                domain=self.domain,
                values=new_values.to_dict(),
                name=f"{other.name}/{self.name}",
            )
        else:
            new_values = other / self.values
            return RandomVariable(
                domain=self.domain,
                values=new_values.to_dict(),
                name=f"{other}/{self.name}",
            )

    def __pow__(self, power):
        if isinstance(power, RandomVariable):
            if self.domain != power.domain:
                raise ValueError(
                    "Cannot exponentiate RandomVariables with different domains."
                )
            new_values = self.values**power.values
            return RandomVariable(
                domain=self.domain,
                values=new_values.to_dict(),
                name=f"{self.name}^{power.name}",
            )
        else:
            new_values = self.values**power
            return RandomVariable(
                domain=self.domain,
                values=new_values.to_dict(),
                name=f"{self.name}^{power}",
            )
