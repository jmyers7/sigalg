from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Callable

import pandas as pd

from ..feature_representations import SampleFeatures, SampleSpaceFeatures
from ..sigma_algebras import SigmaAlgebra
from ..spaces import SampleSpace


class RandomVariable:

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        domain: SampleSpace,
        values: dict[Hashable, Any],
        function: Callable[[SampleFeatures], Any] | None = None,
        name: str = "X",
    ):
        self._validate_parameters(domain, values, name)
        self._domain = domain
        self._values = tuple(sorted(values.items()))
        self._function = function
        self._name = name
        self._series = pd.Series(dict(self._values), name=self._name)
        self._unique_values = tuple(self._series.unique())
        atom_ids = dict(self._values)
        self._sigma_algebra = SigmaAlgebra(sample_space=domain, atom_ids=atom_ids)
        self._hash = None

    # --------------------- properties --------------------- #

    @property
    def domain(self) -> SampleSpace:
        return self._domain

    @property
    def values(self) -> dict[Hashable, Any]:
        return dict(self._values)

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
    def range(self) -> SampleSpace:
        return SampleSpace(list(self._unique_values))

    # --------------------- conversion methods --------------------- #

    def to_pandas(self) -> pd.Series:
        return self._series.copy()

    def to_dict(self) -> dict[Hashable, Any]:
        return dict(self._values)

    # --------------------- class methods --------------------- #

    @classmethod
    def from_features(
        cls,
        domain_features: SampleSpaceFeatures,
        function: Callable[[SampleFeatures], Any],
        name: str = "X",
    ):
        data = domain_features.apply_to_row(function)
        domain = domain_features.sample_space
        values = data.to_dict()
        return cls(domain=domain, values=values, function=function, name=name)

    # --------------------- call methods --------------------- #

    def __call__(self, key: SampleFeatures | Hashable) -> Any:
        if isinstance(key, SampleFeatures):
            if self._function is None:
                raise ValueError("This RandomVariable was not defined with a function.")
            return self._function(key)
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
        if self._values != other._values:
            return False
        return True

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self._domain, self._values, self._name))
        return self._hash

    # --------------------- arithmetic operations --------------------- #

    def __add__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError("Cannot add RandomVariables with different domains.")
            new_series = self._series + other._series
            return RandomVariable(
                domain=self.domain,
                values=new_series.to_dict(),
                name=f"({self.name}+{other.name})",
            )
        else:
            new_series = self._series + other
            return RandomVariable(
                domain=self.domain,
                values=new_series.to_dict(),
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
            new_series = self._series * other._series
            return RandomVariable(
                domain=self.domain,
                values=new_series.to_dict(),
                name=f"({self.name}*{other.name})",
            )
        else:
            new_series = self._series * other
            return RandomVariable(
                domain=self.domain,
                values=new_series.to_dict(),
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
            new_series = self._series - other._series
            return RandomVariable(
                domain=self.domain,
                values=new_series.to_dict(),
                name=f"({self.name}-{other.name})",
            )
        else:
            new_series = self._series - other
            return RandomVariable(
                domain=self.domain,
                values=new_series.to_dict(),
                name=f"({self.name}-{other})",
            )

    def __rsub__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot subtract RandomVariables with different domains."
                )
            new_series = other._series - self._series
            return RandomVariable(
                domain=self.domain,
                values=new_series.to_dict(),
                name=f"({other.name}-{self.name})",
            )
        else:
            new_series = other - self._series
            return RandomVariable(
                domain=self.domain,
                values=new_series.to_dict(),
                name=f"({other}-{self.name})",
            )

    def __truediv__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot divide RandomVariables with different domains."
                )
            new_series = self._series / other._series
            return RandomVariable(
                domain=self.domain,
                values=new_series.to_dict(),
                name=f"({self.name}/{other.name})",
            )
        else:
            new_series = self._series / other
            return RandomVariable(
                domain=self.domain,
                values=new_series.to_dict(),
                name=f"({self.name}/{other})",
            )

    def __rtruediv__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot divide RandomVariables with different domains."
                )
            new_series = other._series / self._series
            return RandomVariable(
                domain=self.domain,
                values=new_series.to_dict(),
                name=f"({other.name}/{self.name})",
            )
        else:
            new_series = other / self._series
            return RandomVariable(
                domain=self.domain,
                values=new_series.to_dict(),
                name=f"({other}/{self.name})",
            )

    def __pow__(self, power):
        if isinstance(power, RandomVariable):
            if self.domain != power.domain:
                raise ValueError(
                    "Cannot exponentiate RandomVariables with different domains."
                )
            new_series = self._series**power._series
            return RandomVariable(
                domain=self.domain,
                values=new_series.to_dict(),
                name=f"({self.name}^{power.name})",
            )
        else:
            new_series = self._series**power
            return RandomVariable(
                domain=self.domain,
                values=new_series.to_dict(),
                name=f"({self.name}^{power})",
            )

    def __rpow__(self, other):
        if isinstance(other, RandomVariable):
            if self.domain != other.domain:
                raise ValueError(
                    "Cannot exponentiate RandomVariables with different domains."
                )
            new_series = other._series**self._series
            return RandomVariable(
                domain=self.domain,
                values=new_series.to_dict(),
                name=f"({other.name}^{self.name})",
            )
        else:
            new_series = other**self._series
            return RandomVariable(
                domain=self.domain,
                values=new_series.to_dict(),
                name=f"({other}^{self.name})",
            )

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        return f"RandomVariable(name='{self.name}',\n{self._series})"

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        domain: SampleSpace,
        values: dict[Hashable, Any],
        name: str,
    ) -> None:
        if not isinstance(domain, SampleSpace):
            raise TypeError("domain must be a SampleSpace instance.")

        if not isinstance(values, dict):
            raise TypeError("values must be a dict.")

        if not isinstance(name, str):
            raise TypeError("name must be a string.")

        if set(values.keys()) != set(domain.index):
            raise ValueError("values keys must match domain indices.")
