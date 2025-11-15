from ..feature_representations import SampleSpaceFeatures, SampleFeatures
from ..sigma_algebras import SigmaAlgebra
from typing import List, Callable, Any, Dict
import pandas as pd


class RandomVariable:
    def __init__(
        self,
        domain_features: SampleSpaceFeatures = None,
        function: Callable[[SampleFeatures], Any] = None,
        values: Dict[str, Any] = None,
        name: str = "X",
    ):
        self._validate_parameters(domain_features, function, values, name)
        self._domain = domain_features

        if function is not None:
            self._function = function
            self._from_function(domain_features, function, name)
        elif values is not None:
            self._from_values(domain_features, values, name)

        self._values = pd.Series(data=self._data, index=domain_features.sample_index, name=name)
        atom_ids = dict(zip(domain_features.sample_index, self._values))
        self._sigma_algebra = SigmaAlgebra(sample_space=domain_features, atom_ids=atom_ids)
        range_values = self._values.unique()
        range_sample_index = [name.lower() + f"{i}" for i in range(len(range_values))]
        self._range = SampleSpaceFeatures(
            features=range_values,
            sample_index=range_sample_index,
            feature_index=[name],
        )

    def _from_function(self, domain, function, name):
        self._data = domain.apply_to_row(function)
        self._data.name = name
        self._function = function

        def idx_function(key):
            sample_features = domain[key]
            return function(sample_features)

        self._idx_function = idx_function

    def _from_values(self, domain, values, name):
        def idx_function(sample_features_index):
            return values[sample_features_index]

        self._data = domain.apply_to_index(idx_function)
        self._data.name = name
        self._idx_function = idx_function

        def function(sample_features):
            return idx_function(sample_features.sample_index)

        self._function = function

    def __call__(self, key) -> Any:
        if isinstance(key, SampleFeatures):
            return self._function(key)
        elif isinstance(key, str):
            return self._idx_function(key)

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
                domain_features=self.domain,
                values=new_values.to_dict(),
                name=f"{self.name}+{other.name}",
            )
        else:
            new_values = self.values + other
            return RandomVariable(
                domain_features=self.domain,
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
                domain_features=self.domain,
                values=new_values.to_dict(),
                name=f"{self.name}*{other.name}",
            )
        else:
            new_values = self.values * other
            return RandomVariable(
                domain_features=self.domain,
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
                domain_features=self.domain,
                values=new_values.to_dict(),
                name=f"{self.name}-{other.name}",
            )
        else:
            new_values = self.values - other
            return RandomVariable(
                domain_features=self.domain,
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
                domain_features=self.domain,
                values=new_values.to_dict(),
                name=f"{other.name}-{self.name}",
            )
        else:
            new_values = other - self.values
            return RandomVariable(
                domain_features=self.domain,
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
                domain_features=self.domain,
                values=new_values.to_dict(),
                name=f"{self.name}/{other.name}",
            )
        else:
            new_values = self.values / other
            return RandomVariable(
                domain_features=self.domain,
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
                domain_features=self.domain,
                values=new_values.to_dict(),
                name=f"{other.name}/{self.name}",
            )
        else:
            new_values = other / self.values
            return RandomVariable(
                domain_features=self.domain,
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
                domain_features=self.domain,
                values=new_values.to_dict(),
                name=f"{self.name}^{power.name}",
            )
        else:
            new_values = self.values**power
            return RandomVariable(
                domain_features=self.domain,
                values=new_values.to_dict(),
                name=f"{self.name}^{power}",
            )

    @property
    def domain(self) -> SampleSpaceFeatures:
        return self._domain

    @property
    def values(self) -> pd.Series:
        return self._values

    @property
    def range(self) -> SampleSpaceFeatures:
        return self._range

    @property
    def level_sets(self) -> Dict[Any, List[str]]:
        return {key: list(value) for key, value in self._level_sets.items()}

    @property
    def sigma_algebra(self) -> SigmaAlgebra:
        return self._sigma_algebra

    @property
    def function(self) -> Callable[[SampleFeatures], Any]:
        return self._function

    @property
    def idx_function(self) -> Callable[[str], Any]:
        return self._idx_function

    @property
    def name(self) -> str:
        return self._values.name

    def set_name(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        self._values.name = name

    @staticmethod
    def _validate_parameters(sample_space, function, values, name) -> None:
        if not isinstance(sample_space, SampleSpaceFeatures):
            raise TypeError("sample_space must be an instance of SampleSpaceFeatures.")
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        if function is not None and values is not None:
            raise ValueError("Provide only one of function or values, not both.")
        if function is None and values is None:
            raise ValueError("Must provide one of function or values.")
        if function is not None and not callable(function):
            raise TypeError("function must be callable.")
        if values is not None and not isinstance(values, dict):
            raise TypeError("values must be a dictionary.")
        if values is not None:
            for key in values.keys():
                if key not in sample_space.sample_index:
                    raise ValueError(
                        f"Key '{key}' in values is not a valid sample point index in the sample space."
                    )
