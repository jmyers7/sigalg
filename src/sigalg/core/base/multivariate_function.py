import inspect
from collections.abc import Callable, Hashable

import pandas as pd


class MultivariateFunction:
    """Pass."""

    # --------------------- constructors --------------------- #

    def __init__(self, domain=None, name: Hashable | None = "f", **kwargs):
        self._domain = domain
        self._name = name

        # caches
        self._function = None
        self._parameter_names = None
        self._data = None

    def from_callable(
        self,
        function: Callable,
        parameter_names: list | None = None,
    ):
        """Pass."""
        if not callable(function):
            raise TypeError("function must be a callable object.")

        self._function = function

        if parameter_names is None:
            try:
                parameter_names = list(inspect.signature(function).parameters.keys())
            except (ValueError, TypeError) as e:
                raise ValueError(
                    "Could not determine parameter names from the function signature. Please provide parameter names explicitly."
                ) from e
        if not parameter_names:
            raise ValueError(
                "Parameter names could not be determined from the function signature. Please provide parameter names explicitly."
            )

        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("Duplicate parameter names are not allowed.")

        self._parameter_names = parameter_names

        if self.domain is not None:
            if set(self.parameter_names) - set(self.domain.columns):
                raise ValueError("Domain must contain columns for all parameter names.")

        return self

    # --------------------- properties --------------------- #

    @property
    def domain(self):
        """Pass."""
        return self._domain

    @property
    def function(self):
        """Pass."""
        return self._function

    @property
    def parameter_names(self):
        """Pass."""
        return self._parameter_names

    @property
    def data(self):
        """Pass."""
        if self._data is None:
            if self.function is None:
                raise ValueError("Function is not defined.")
            if self.domain is None:
                raise ValueError("Domain is not defined.")
            self._data = self.evaluate_on_domain()
        return self._data

    @property
    def name(self):
        """Pass."""
        return self._name

    # --------------------- data access methods --------------------- #

    def __call__(self, **parameters):
        """Pass."""
        matched_parameters = {
            parameter_name: parameter
            for parameter_name, parameter in parameters.items()
            if parameter_name in self.parameter_names
        }

        if set(matched_parameters.keys()) == set(self.parameter_names):
            return self._function(**matched_parameters)
        else:

            def partial_function(**remaining_parameters):
                return self._function(**{**matched_parameters, **remaining_parameters})

            parameter_names = [
                p for p in self.parameter_names if p not in matched_parameters
            ]
            name = f"{self.name}({', '.join(f'{p}={matched_parameters[p]}' for p in self.parameter_names if p in matched_parameters)})"
            return MultivariateFunction(name=name).from_callable(
                partial_function,
                parameter_names=parameter_names,
            )

    def evaluate_on_domain(self):
        """Evaluate function on domain, returning Series with MultiIndex of parameter values."""
        if self.function is None:
            raise ValueError("Function is not defined.")
        if self.domain is None:
            raise ValueError("Domain is not defined.")
        if set(self.parameter_names) - set(self.domain.columns):
            raise ValueError("Domain must contain columns for all parameter names.")

        param_cols = [col for col in self.parameter_names if col in self.domain.columns]
        result = self.domain.apply(lambda row: self(**row), axis=1)

        if len(param_cols) > 1:
            result.index = pd.MultiIndex.from_frame(self.domain[param_cols])
        elif len(param_cols) == 1:
            result.index = self.domain[param_cols[0]]

        return result

    # --------------------- representation --------------------- #

    def __repr__(self):
        """Pass."""
        if self.function is None:
            return f"{self.name}(empty)"
        else:
            return f"{self.name}({', '.join(f'{param}' for param in self.parameter_names)})"

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
        if not self.domain.equals(other.domain):
            raise ValueError("Cannot compare functions with different domains.")

        return self.data.equals(other.data)

    # --------------------- arithmetic operations --------------------- #

    def __add__(self, other):
        """Pass."""
        if isinstance(other, MultivariateFunction):
            parameter_names = list(
                dict.fromkeys(self.parameter_names + other.parameter_names)
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
                parameter_names=self.parameter_names,
            )

    def __sub__(self, other):
        """Pass."""
        if isinstance(other, MultivariateFunction):
            parameter_names = list(
                dict.fromkeys(self.parameter_names + other.parameter_names)
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
                parameter_names=self.parameter_names,
            )

    def __mul__(self, other):
        """Pass."""
        if isinstance(other, MultivariateFunction):
            parameter_names = list(
                dict.fromkeys(self.parameter_names + other.parameter_names)
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
                parameter_names=self.parameter_names,
            )

    def __truediv__(self, other):
        """Pass."""
        if isinstance(other, MultivariateFunction):
            parameter_names = list(
                dict.fromkeys(self.parameter_names + other.parameter_names)
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
                parameter_names=self.parameter_names,
            )

    def __pow__(self, other):
        """Pass."""
        if isinstance(other, MultivariateFunction):
            parameter_names = list(
                dict.fromkeys(self.parameter_names + other.parameter_names)
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
                parameter_names=self.parameter_names,
            )

    def __neg__(self):
        """Pass."""
        return MultivariateFunction(name=f"(-{self.name})").from_callable(
            lambda **parameters: -self(**parameters),
            parameter_names=self.parameter_names,
        )

    def __radd__(self, other):
        """Pass."""
        if isinstance(other, MultivariateFunction):
            return other.__add__(self)
        return MultivariateFunction(name=f"({other} + {self.name})").from_callable(
            lambda **parameters: other + self(**parameters),
            parameter_names=self.parameter_names,
        )

    def __rsub__(self, other):
        """Pass."""
        if isinstance(other, MultivariateFunction):
            return other.__sub__(self)
        return MultivariateFunction(name=f"({other} - {self.name})").from_callable(
            lambda **parameters: other - self(**parameters),
            parameter_names=self.parameter_names,
        )

    def __rmul__(self, other):
        """Pass."""
        if isinstance(other, MultivariateFunction):
            return other.__mul__(self)
        return MultivariateFunction(name=f"({other} * {self.name})").from_callable(
            lambda **parameters: other * self(**parameters),
            parameter_names=self.parameter_names,
        )

    def __rtruediv__(self, other):
        """Pass."""
        if isinstance(other, MultivariateFunction):
            return other.__truediv__(self)
        return MultivariateFunction(name=f"({other} / {self.name})").from_callable(
            lambda **parameters: other / self(**parameters),
            parameter_names=self.parameter_names,
        )

    def __rpow__(self, other):
        """Pass."""
        if isinstance(other, MultivariateFunction):
            return other.__pow__(self)
        return MultivariateFunction(name=f"({other} ** {self.name})").from_callable(
            lambda **parameters: other ** self(**parameters),
            parameter_names=self.parameter_names,
        )
