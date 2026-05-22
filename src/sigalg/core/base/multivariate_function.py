from __future__ import annotations

import inspect
from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .domain import Domain


class MultivariateFunction:
    """Pass."""

    _properties = [
        "_function",
        "_data",
        "_dict",
        "_parameter_list",
        "_signature",
        "_num_parameters",
        "_output_name",
    ]

    # --------------------- constructors --------------------- #

    def __init__(self, domain: Domain = None, name: Hashable = "f", **kwargs):
        self._domain = domain
        self._name = name
        self._initialize_property_caches()

    def _initialize_property_caches(self) -> None:
        for property in self._properties:
            setattr(self, property, None)

    def from_callable(
        self,
        function: Callable,
        parameter_names: list | None = None,
        output_name: Hashable = "output",
    ) -> MultivariateFunction:
        """Pass."""
        self._function = function
        self._signature = inspect.signature(function)

        if parameter_names is None:
            parameter_names = list(self._signature.parameters.keys())

        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("Duplicate parameter names are not allowed.")

        self._parameter_list = parameter_names
        self._num_parameters = len(parameter_names)
        self._output_name = output_name

        return self

    def from_pandas(self, data: pd.Series) -> MultivariateFunction:
        """Pass."""
        self._data = data
        self._num_parameters = data.index.nlevels
        self._parameter_list = list(data.index.names)
        return self

    # --------------------- properties --------------------- #

    @property
    def function(self):
        """Pass."""
        if self._function is None and self._data is not None:

            def make_function(series):
                names = series.index.names
                parameters = [
                    inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    for name in names
                ]
                sig = inspect.Signature(parameters)

                def function(*args, **kwargs):
                    bound = sig.bind(*args, **kwargs)
                    key = tuple(bound.arguments[name] for name in names)
                    return series[key[0] if len(key) == 1 else key]

                function.__signature__ = sig

                return function

            self._function = make_function(self.data)

        return self._function

    @property
    def data(self):
        """Pass."""
        if (
            self._data is None
            and self._function is not None
            and self._domain is not None
        ):
            if isinstance(self._domain.data, pd.MultiIndex):
                self._data = self._domain.data.map(
                    lambda parameter: self.function(*parameter)
                ).to_series()
            else:
                self._data = self._domain.data.map(
                    lambda parameter: self.function(parameter)
                ).to_series()

            self._data.index = self._domain.data
            self._data.name = self._output_name

        return self._data

    @property
    def dict(self):
        """Pass."""
        if self._dict is None and self.data is not None:
            self._dict = self.data.to_dict()
        return self._dict

    @property
    def parameter_list(self):
        """Pass."""
        return self._parameter_list

    @property
    def signature(self) -> inspect.Signature | None:
        """Pass."""
        if self._signature is None and self.function is not None:
            self._signature = inspect.signature(self.function)
        return self._signature

    @property
    def num_parameters(self) -> int | None:
        """Pass."""
        return self._num_parameters

    @property
    def domain(self):
        """Pass."""
        return self._domain

    @property
    def name(self):
        """Pass."""
        return self._name

    # --------------------- data access methods --------------------- #

    def __call__(self, *args, **kwargs):
        """Pass."""
        from .domain import Domain

        specified_parameters = self.signature.bind_partial(*args, **kwargs)
        unspecified_parameters = [
            inspect.Parameter(parameter, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for parameter in self.parameter_list
            if parameter not in specified_parameters.arguments.keys()
        ]

        if len(unspecified_parameters) == 0:
            return self.function(**specified_parameters.arguments)
        else:
            partial_signature = inspect.Signature(unspecified_parameters)

            def partial_function(*args, **kwargs):
                partial_parameters = partial_signature.bind(*args, **kwargs)
                all_args = {
                    **specified_parameters.arguments,
                    **partial_parameters.arguments,
                }
                return self._function(**all_args)

            partial_function.__signature__ = partial_signature

            name = f"{self.name}({', '.join(f'{p}={specified_parameters.arguments[p]}' for p in self.parameter_list if p in specified_parameters.arguments)})"

            if self.data is not None:
                try:
                    data = self.data.xs(
                        key=tuple(specified_parameters.arguments.values()),
                        level=tuple(specified_parameters.arguments.keys()),
                    ).index
                    domain_name = f"{self.domain.name}({', '.join(f'{p}={specified_parameters.arguments[p]}' for p in self.parameter_list if p in specified_parameters.arguments)})"
                    partial_domain = Domain(name=domain_name).from_pandas(data)
                except KeyError:
                    partial_domain = None
            else:
                partial_domain = None

            return MultivariateFunction(domain=partial_domain, name=name).from_callable(
                function=partial_function
            )

    # --------------------- representation --------------------- #

    def __repr__(self):
        """Pass."""
        if self.function is None:
            return f"Function '{self.name}': empty"
        else:
            if self.data is not None:
                return f"Function '{self.name}':\n{self.data.to_frame()}"

            else:
                return f"Function '{self.name}{self.signature}'"

    # --------------------- equality --------------------- #

    def __eq__(self, other):
        """Pass."""
        if not isinstance(other, MultivariateFunction):
            return False

        if self.function is None or other.function is None:
            raise ValueError("Cannot compare empty functions.")
        if self.domain is None or other.domain is None:
            raise ValueError(
                "Cannot compare functions when one (or both) domains are not defined."
            )
        if not self.domain.data.equals(other.domain.data):
            raise ValueError("Cannot compare functions with different domains.")

        return self.data.equals(other.data)

    # --------------------- arithmetic operations --------------------- #

    def __add__(self, other):
        """Pass."""
        if isinstance(other, MultivariateFunction):
            parameter_names = list(
                dict.fromkeys(self.parameter_list + other.parameter_list)
            )
            return MultivariateFunction(
                name=f"({self.name} + {other.name})"
            ).from_callable(
                lambda **parameters: self(**parameters) + other(**parameters),
                parameter_names=parameter_names,
            )
        else:
            return MultivariateFunction(name=f"({self.name} + {other})").from_callable(
                lambda **parameters: self(**parameters) + other,
                parameter_names=self.parameter_list,
            )

    def __sub__(self, other):
        """Pass."""
        if isinstance(other, MultivariateFunction):
            parameter_names = list(
                dict.fromkeys(self.parameter_list + other.parameter_list)
            )
            return MultivariateFunction(
                name=f"({self.name} - {other.name})"
            ).from_callable(
                lambda **parameters: self(**parameters) - other(**parameters),
                parameter_names=parameter_names,
            )
        else:
            return MultivariateFunction(name=f"({self.name} - {other})").from_callable(
                lambda **parameters: self(**parameters) - other,
                parameter_names=self.parameter_list,
            )

    def __mul__(self, other):
        """Pass."""
        if isinstance(other, MultivariateFunction):
            parameter_names = list(
                dict.fromkeys(self.parameter_list + other.parameter_list)
            )
            return MultivariateFunction(
                name=f"({self.name} * {other.name})"
            ).from_callable(
                lambda **parameters: self(**parameters) * other(**parameters),
                parameter_names=parameter_names,
            )
        else:
            return MultivariateFunction(name=f"({self.name} * {other})").from_callable(
                lambda **parameters: self(**parameters) * other,
                parameter_names=self.parameter_list,
            )

    def __truediv__(self, other):
        """Pass."""
        if isinstance(other, MultivariateFunction):
            parameter_names = list(
                dict.fromkeys(self.parameter_list + other.parameter_list)
            )
            return MultivariateFunction(
                name=f"({self.name} / {other.name})"
            ).from_callable(
                lambda **parameters: self(**parameters) / other(**parameters),
                parameter_names=parameter_names,
            )
        else:
            return MultivariateFunction(name=f"({self.name} / {other})").from_callable(
                lambda **parameters: self(**parameters) / other,
                parameter_names=self.parameter_list,
            )

    def __pow__(self, other):
        """Pass."""
        if isinstance(other, MultivariateFunction):
            parameter_names = list(
                dict.fromkeys(self.parameter_list + other.parameter_list)
            )
            return MultivariateFunction(
                name=f"({self.name} ** {other.name})"
            ).from_callable(
                lambda **parameters: self(**parameters) ** other(**parameters),
                parameter_names=parameter_names,
            )
        else:
            return MultivariateFunction(name=f"({self.name} ** {other})").from_callable(
                lambda **parameters: self(**parameters) ** other,
                parameter_names=self.parameter_list,
            )

    def __neg__(self):
        """Pass."""
        return MultivariateFunction(name=f"(-{self.name})").from_callable(
            lambda **parameters: -self(**parameters),
            parameter_names=self.parameter_list,
        )

    def __radd__(self, other):
        """Pass."""
        if isinstance(other, MultivariateFunction):
            return other.__add__(self)
        return MultivariateFunction(name=f"({other} + {self.name})").from_callable(
            lambda **parameters: other + self(**parameters),
            parameter_names=self.parameter_list,
        )

    def __rsub__(self, other):
        """Pass."""
        if isinstance(other, MultivariateFunction):
            return other.__sub__(self)
        return MultivariateFunction(name=f"({other} - {self.name})").from_callable(
            lambda **parameters: other - self(**parameters),
            parameter_names=self.parameter_list,
        )

    def __rmul__(self, other):
        """Pass."""
        if isinstance(other, MultivariateFunction):
            return other.__mul__(self)
        return MultivariateFunction(name=f"({other} * {self.name})").from_callable(
            lambda **parameters: other * self(**parameters),
            parameter_names=self.parameter_list,
        )

    def __rtruediv__(self, other):
        """Pass."""
        if isinstance(other, MultivariateFunction):
            return other.__truediv__(self)
        return MultivariateFunction(name=f"({other} / {self.name})").from_callable(
            lambda **parameters: other / self(**parameters),
            parameter_names=self.parameter_list,
        )

    def __rpow__(self, other):
        """Pass."""
        if isinstance(other, MultivariateFunction):
            return other.__pow__(self)
        return MultivariateFunction(name=f"({other} ** {self.name})").from_callable(
            lambda **parameters: other ** self(**parameters),
            parameter_names=self.parameter_list,
        )
