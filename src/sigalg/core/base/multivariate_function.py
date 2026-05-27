"""A class representing a multivariate function."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Hashable
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .domain import Domain


class MultivariateFunction:
    """A class representing a multivariate function.

    Parameters
    ----------
    domain : Domain | None, default=None
        The domain of the function.
    name : Hashable, default="f"
        The name of the function.
    """

    _properties = [
        "_function",
        "_data",
        "_dict",
        "_parameter_names",
        "_signature",
        "_num_parameters",
        "_output_name",
    ]

    # --------------------- constructors --------------------- #

    def __init__(self, domain: Domain = None, name: Hashable = "f", **kwargs) -> None:
        self._domain = domain
        self._name = name
        self._initialize_property_caches()

    def _initialize_property_caches(self) -> None:
        for property in self._properties:
            setattr(self, property, None)

    # TODO: validate that parameter names of the callable match the data names of the domain, when the latter is provided
    def from_callable(
        self,
        function: Callable,
        output_name: Hashable = "output",
    ) -> MultivariateFunction:
        """Initialize the function from a callable.

        Parameters
        ----------
        function : Callable
            A callable that takes keyword-only arguments corresponding to the function's parameters.
        output_name : Hashable, default="output"
            The name of the output variable for the function.

        Raises
        ------
        TypeError
            If `function` is not callable or if `output_name` is not hashable.
        ValueError
            If `function` does not have all parameters as keyword-only parameters.

        Returns
        -------
        self : MultivariateFunction
            The instance of the `MultivariateFunction` initialized with the provided callable.

        Examples
        --------
        >>> from sigalg.core import Domain, MultivariateFunction
        >>> D = Domain().from_list([(1, 2), (2, 3), (1, 4)], data_name=["x", "y"])
        >>> f = MultivariateFunction(domain=D).from_callable(lambda *, x, y: 2 * x + y**2)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
             output
        x y
        1 2       6
        2 3      13
        1 4      18
        """
        if not callable(function):
            raise TypeError("The provided function must be callable.")
        if not isinstance(output_name, Hashable):
            raise TypeError("The output_name must be a hashable type.")

        self._signature = inspect.signature(function)

        if not all(
            (
                param.kind
                in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD)
            )
            for param in self._signature.parameters.values()
        ):
            raise ValueError(
                "Multivariate functions must have all parameters as keyword-only parameters."
            )

        self._function = function
        self._parameter_names = list(self._signature.parameters.keys())
        self._num_parameters = len(self._parameter_names)
        self._output_name = output_name
        return self

    def from_pandas(self, data: pd.Series) -> MultivariateFunction:
        """Initialize the function from a `pd.Series` object.

        If a domain is provided at initialization, the index of the provided data must match the domain. If no domain is provided, a new domain will be created based on the index of the provided data.

        Parameters
        ----------
        data : pd.Series
            A pandas Series where the index represents the function's parameters and the values represent the function's output.

        Raises
        ------
        TypeError
            If `data` is not a `pd.Series` object.
        ValueError
            If the provided data's index does not match the domain (when the latter is provided).

        Returns
        -------
        self : MultivariateFunction
            The instance of the `MultivariateFunction` initialized with the provided data.

        Examples
        --------
        >>> import pandas as pd
        >>> from sigalg.core import MultivariateFunction
        >>> data = pd.Series(
        ...     [6, 13, 18],
        ...     index=pd.MultiIndex.from_tuples([(1, 2), (2, 3), (1, 4)], names=["x", "y"]),
        ...     name="output",
        ... )
        >>> f = MultivariateFunction().from_pandas(data)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
             output
        x y
        1 2       6
        2 3      13
        1 4      18
        """
        from .domain import Domain

        if not isinstance(data, pd.Series):
            raise TypeError("The provided data must be a `pd.Series` object.")

        self._data = data
        self._parameter_names = list(data.index.names)
        self._num_parameters = data.index.nlevels

        if self.domain is not None and (
            not self.domain.data.equals(data.index)
            or self.domain.data_name != data.index.names
        ):
            print(self.domain.data)
            print(data.index)
            raise ValueError("The provided data's index does not match the domain.")
        if self.domain is None:
            self._domain = Domain().from_pandas(data.index)

        return self

    # --------------------- properties --------------------- #

    @property
    def function(self) -> Callable | None:
        """Get the underlying callable function.

        Returns
        -------
        function : Callable | None
            The underlying callable function if defined, otherwise None.

        Examples
        --------
        >>> import pandas as pd
        >>> from sigalg.core import MultivariateFunction
        >>> data = pd.Series(
        ...     [6, 13, 18],
        ...     index=pd.MultiIndex.from_tuples([(1, 2), (2, 3), (1, 4)], names=["x", "y"]),
        ...     name="output",
        ... )
        >>> f = MultivariateFunction().from_pandas(data)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
             output
        x y
        1 2       6
        2 3      13
        1 4      18
        >>> print(f.function(x=1, y=2))
        6
        """
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
            self._output_name = self.data.name

        return self._function

    @property
    def data(self) -> pd.Series | None:
        """Get the underlying data as a `pd.Series` object.

        The data is computed from either a domain provided at initialization and a callable function from the `from_callable` method, or from a `pd.Series` object provided via the `from_pandas` method.

        Returns
        -------
        data : pd.Series | None
            The underlying data as a `pd.Series` object if defined, otherwise None.

        Examples
        --------
        >>> from sigalg.core import Domain, MultivariateFunction
        >>> D = Domain().from_list([(1, 2), (2, 3), (1, 4)], data_name=["x", "y"])
        >>> f = MultivariateFunction(domain=D).from_callable(lambda *, x, y: 2 * x + y**2)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
             output
        x y
        1 2       6
        2 3      13
        1 4      18
        >>> print(f.data)  # doctest: +NORMALIZE_WHITESPACE
        x  y
        1  2     6
        2  3    13
        1  4    18
        Name: output, dtype: int64
        """
        if (
            self._data is None
            and self._function is not None
            and self._domain is not None
        ):
            if isinstance(self._domain.data, pd.MultiIndex):
                self._data = self._domain.data.map(
                    lambda parameter: self.function(
                        **dict(zip(self._domain.data.names, parameter))
                    )
                ).to_series()
            else:
                self._data = self._domain.data.map(
                    lambda parameter: self.function(
                        **{self.parameter_names[0]: parameter}
                    )
                ).to_series()

            self._data.index = self._domain.data
            self._data.name = self._output_name

        return self._data

    @property
    def dict(self) -> dict | None:
        """Get the underlying data as a dictionary.

        The dictionary is computed from either a domain provided at initialization and a callable function from the `from_callable` method, or from a `pd.Series` object provided via the `from_pandas` method.

        Returns
        -------
        result_dict : dict | None
            The underlying data as a dictionary if defined, otherwise None.

        Examples
        --------
        >>> from sigalg.core import Domain, MultivariateFunction
        >>> D = Domain().from_list([(1, 2), (2, 3), (1, 4)], data_name=["x", "y"])
        >>> f = MultivariateFunction(domain=D).from_callable(lambda *, x, y: 2 * x + y**2)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
             output
        x y
        1 2       6
        2 3      13
        1 4      18
        >>> print(f.dict)
        {(1, 2): 6, (2, 3): 13, (1, 4): 18}
        """
        if self._dict is None and self.data is not None:
            self._dict = self.data.to_dict()
        return self._dict

    @property
    def parameter_names(self):
        """Pass."""
        return self._parameter_names

    @property
    def signature(self) -> inspect.Signature | None:
        """Get the signature of the underlying callable function.

        Returns
        -------
        signature : inspect.Signature | None
            The signature of the underlying callable function if defined, otherwise None.

        Examples
        --------
        >>> from sigalg.core import Domain, MultivariateFunction
        >>> D = Domain().from_list([(1, 2), (2, 3), (1, 4)], data_name=["x", "y"])
        >>> f = MultivariateFunction(domain=D).from_callable(lambda *, x, y: 2 * x + y**2)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
             output
        x y
        1 2       6
        2 3      13
        1 4      18
        >>> print(f.signature)
        (*, x, y)
        """
        if self._signature is None and self.function is not None:
            self._signature = inspect.signature(self.function)
        return self._signature

    @property
    def num_parameters(self) -> int | None:
        """Get the number of parameters of the function.

        Returns
        -------
        num_parameters : int | None
            The number of parameters of the function if defined, otherwise None.

        Examples
        --------
        >>> from sigalg.core import Domain, MultivariateFunction
        >>> D = Domain().from_list([(1, 2), (2, 3), (1, 4)], data_name=["x", "y"])
        >>> f = MultivariateFunction(domain=D).from_callable(lambda *, x, y: 2 * x + y**2)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
             output
        x y
        1 2       6
        2 3      13
        1 4      18
        >>> print(f.num_parameters)
        2
        """
        return self._num_parameters

    @property
    def domain(self) -> Domain | None:
        """Get the domain of the function.

        Returns
        -------
        domain : Domain | None
            The domain of the function if defined, otherwise None.

        Examples
        --------
        >>> from sigalg.core import Domain, MultivariateFunction
        >>> D = Domain().from_list([(1, 2), (2, 3), (1, 4)], data_name=["x", "y"])
        >>> f = MultivariateFunction(domain=D).from_callable(lambda *, x, y: 2 * x + y**2)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
             output
        x y
        1 2       6
        2 3      13
        1 4      18
        >>> print(f.domain)
        Domain 'D':
        [(1, 2), (2, 3), (1, 4)]
        """
        return self._domain

    @property
    def name(self) -> Hashable:
        """Get the name of the function.

        Returns
        -------
        name : Hashable
            The name of the function.

        Examples
        --------
        >>> from sigalg.core import Domain, MultivariateFunction
        >>> D = Domain().from_list([(1, 2), (2, 3), (1, 4)], data_name=["x", "y"])
        >>> g = MultivariateFunction(domain=D, name="g").from_callable(lambda *, x, y: 2 * x + y**2)
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
             output
        x y
        1 2       6
        2 3      13
        1 4      18
        >>> print(g.name)
        g
        """
        return self._name

    @property
    def output_name(self) -> Hashable | None:
        """Pass."""
        return self._output_name

    # --------------------- data access methods --------------------- #

    def __call__(self, **kwargs) -> Real | MultivariateFunction:
        """Call the function with the provided arguments.

        If a complete set of arguments is provided, the function is evaluated and the result is returned. If a partial set of arguments is provided, a new `MultivariateFunction` instance is returned, representing the partially applied function.

        Parameters
        ----------
        **kwargs : keyword arguments
            Keyword arguments for the function.

        Returns
        -------
        result : Real or MultivariateFunction
            The result of evaluating the function with the provided arguments, or a new `MultivariateFunction` instance representing the partially applied function.

        Examples
        --------
        >>> from sigalg.core import Domain, MultivariateFunction
        >>> D = Domain().from_list([(1, 2), (2, 3), (1, 4)], data_name=["x", "y"])
        >>> f = MultivariateFunction(domain=D).from_callable(lambda *, x, y: 2 * x + y**2)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
             output
        x y
        1 2       6
        2 3      13
        1 4      18
        >>> print(f(x=2, y=3))
        13
        >>> print(f(x=1))  # doctest: +NORMALIZE_WHITESPACE
        Function 'f(x=1)':
           output
        y
        2       6
        4      18
        >>> print(f(x=1)(y=4))
        18
        """
        from .domain import Domain

        specified_parameters = self.signature.bind_partial(**kwargs)
        unspecified_parameters = [
            inspect.Parameter(parameter, inspect.Parameter.KEYWORD_ONLY)
            for parameter in self.parameter_names
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
                return self.function(**all_args)

            partial_function.__signature__ = partial_signature

            name = f"{self.name}({', '.join(f'{p}={specified_parameters.arguments[p]}' for p in self.parameter_names if p in specified_parameters.arguments)})"

            if self.data is not None:
                try:
                    data = self.data.xs(
                        key=tuple(specified_parameters.arguments.values()),
                        level=tuple(specified_parameters.arguments.keys()),
                    ).index
                    domain_name = f"{self.domain.name}({', '.join(f'{p}={specified_parameters.arguments[p]}' for p in self.parameter_names if p in specified_parameters.arguments)})"
                    partial_domain = Domain(name=domain_name).from_pandas(data)

                except KeyError:
                    partial_domain = None

            else:
                partial_domain = None

            return MultivariateFunction(partial_domain, name).from_callable(
                partial_function, output_name=self._output_name
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
                parameter_list = ", ".join(self.parameter_names)
                return f"Function '{self.name}({parameter_list})'"

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
        if (
            not self.domain.data.equals(other.domain.data)
            or self.domain.data.names != other.domain.data.names
        ):
            raise ValueError("Cannot compare functions with different domains.")

        return np.allclose(self.data.values, other.data.values)

    # --------------------- arithmetic operations --------------------- #

    def _apply_binary_operation(
        self,
        other,
        operation: Callable,
        op_symbol: str,
        reverse: bool = False,
    ) -> MultivariateFunction:
        """Apply a binary operation to this function.

        Parameters
        ----------
        other : MultivariateFunction or scalar
            The other operand.
        operation : Callable
            The operation to apply (e.g., lambda a, b: a + b).
        op_symbol : str
            Symbol representing the operation (e.g., '+', '-', '*').
        reverse : bool, default=False
            Whether this is a reverse operation (e.g., __radd__ vs __add__).

        Returns
        -------
        MultivariateFunction
            A new function representing the result of the operation.

        Raises
        ------
        TypeError
            If `other` is not a `MultivariateFunction` or a scalar.
        """
        from .domain import Domain

        if isinstance(other, MultivariateFunction):
            if self.domain is not None and other.domain is not None:
                if not isinstance(self.domain.data, pd.MultiIndex):
                    self_domain_data = pd.MultiIndex.from_arrays(
                        [self.domain.data], names=[self.domain.data.name]
                    )
                else:
                    self_domain_data = self.domain.data
                if not isinstance(other.domain.data, pd.MultiIndex):
                    other_domain_data = pd.MultiIndex.from_arrays(
                        [other.domain.data], names=[other.domain.data.name]
                    )
                else:
                    other_domain_data = other.domain.data

                dummy_series1 = pd.Series(
                    [0] * len(self.domain), index=self_domain_data
                )
                dummy_series2 = pd.Series(
                    [0] * len(other.domain), index=other_domain_data
                )
                data = (dummy_series1 + dummy_series2).dropna().index
                if len(data.names) == 1:
                    data = data.to_flat_index().map(lambda x: x[0])

                domain_name = f"({self.domain.name} {op_symbol} {other.domain.name})"
                domain = Domain(name=domain_name).from_pandas(data)

            else:
                domain = None

            parameter_names = list(
                dict.fromkeys(self.parameter_names + other.parameter_names)
            )
            parameters = [
                inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
                for name in parameter_names
            ]
            sig = inspect.Signature(parameters)

            def binary_function(**kwargs):
                bound = sig.bind(**kwargs)
                self_parameters = {
                    name: bound.arguments[name]
                    for name in self.parameter_names
                    if name in bound.arguments
                }
                other_parameters = {
                    name: bound.arguments[name]
                    for name in other.parameter_names
                    if name in bound.arguments
                }
                if reverse:
                    return operation(other(**other_parameters), self(**self_parameters))
                else:
                    return operation(self(**self_parameters), other(**other_parameters))

            binary_function.__signature__ = sig

            if reverse:
                function_name = f"({other.name} {op_symbol} {self.name})"
            else:
                function_name = f"({self.name} {op_symbol} {other.name})"

            return MultivariateFunction(domain, function_name).from_callable(
                binary_function
            )

        elif isinstance(other, Real):

            def scalar_function(**kwargs):
                bound = self.signature.bind(**kwargs)
                if reverse:
                    return operation(other, self(**bound.arguments))
                else:
                    return operation(self(**bound.arguments), other)

            scalar_function.__signature__ = self.signature

            if reverse:
                function_name = f"({other} {op_symbol} {self.name})"
            else:
                function_name = f"({self.name} {op_symbol} {other})"

            return MultivariateFunction(self.domain, function_name).from_callable(
                scalar_function
            )

        else:
            raise TypeError(
                f"Unsupported operand type(s) for {op_symbol}: 'MultivariateFunction' and '{type(other).__name__}'"
            )

    def __add__(self, other):
        """Add two multivariate functions or a multivariate function and a scalar.

        Parameters
        ----------
        other : MultivariateFunction or scalar
            The other function or scalar to add to this function.

        Raises
        ------
        TypeError
            If `other` is not a `MultivariateFunction` or a scalar.
        """
        return self._apply_binary_operation(other, lambda a, b: a + b, "+")

    def __sub__(self, other):
        """Subtract another multivariate function or a scalar from this function."""
        return self._apply_binary_operation(other, lambda a, b: a - b, "-")

    def __mul__(self, other):
        """Multiply this function by another multivariate function or a scalar."""
        return self._apply_binary_operation(other, lambda a, b: a * b, "*")

    def __truediv__(self, other):
        """Divide this function by another multivariate function or a scalar."""
        return self._apply_binary_operation(other, lambda a, b: a / b, "/")

    def __pow__(self, other):
        """Raise this function to the power of another multivariate function or a scalar."""
        return self._apply_binary_operation(other, lambda a, b: a**b, "**")

    def __neg__(self):
        """Negate this function."""

        def neg_function(**kwargs):
            bound = self.signature.bind(**kwargs)
            return -self(**bound.arguments)

        neg_function.__signature__ = self.signature

        function_name = f"(-{self.name})"

        return MultivariateFunction(self.domain, function_name).from_callable(
            neg_function
        )

    def __radd__(self, other):
        """Add this function to another multivariate function or a scalar (right-hand side)."""
        if isinstance(other, MultivariateFunction):
            return other.__add__(self)
        return self._apply_binary_operation(
            other, lambda a, b: a + b, "+", reverse=True
        )

    def __rsub__(self, other):
        """Subtract this function from another multivariate function or a scalar (right-hand side)."""
        if isinstance(other, MultivariateFunction):
            return other.__sub__(self)
        return self._apply_binary_operation(
            other, lambda a, b: a - b, "-", reverse=True
        )

    def __rmul__(self, other):
        """Multiply this function by another multivariate function or a scalar (right-hand side)."""
        if isinstance(other, MultivariateFunction):
            return other.__mul__(self)
        return self._apply_binary_operation(
            other, lambda a, b: a * b, "*", reverse=True
        )

    def __rtruediv__(self, other):
        """Divide another multivariate function or a scalar by this function (right-hand side)."""
        if isinstance(other, MultivariateFunction):
            return other.__truediv__(self)
        return self._apply_binary_operation(
            other, lambda a, b: a / b, "/", reverse=True
        )

    def __rpow__(self, other):
        """Raise another multivariate function or a scalar to the power of this function (right-hand side)."""
        if isinstance(other, MultivariateFunction):
            return other.__pow__(self)
        return self._apply_binary_operation(
            other, lambda a, b: a**b, "**", reverse=True
        )
