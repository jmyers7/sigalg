"""A class representing a multivariate function."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Hashable
from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ...typing.index_like import IndexLike
    from ...typing.mapping_like import MappingLike
    from ..measures.measure import Measure
    from ..measures.parametrized_measure import ParametrizedMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.domain import Domain
    from .measurable_function import MeasurableFunction
    from .parametrized_measurable_function import ParametrizedMeasurableFunction


class MultivariateFunction:
    """A class representing a multivariate function.

    Mathematically, a function requires three items: A domain set, a codomain set, and a rule defining the function. For instances of `MultivariateFunction`:

    * The domain of the function is passed as the parameter `domain`, but this parameter is *not* required. This allows for the creation of functions whose domains are supposed to be continuous.
    * The codomain of an instance of `MultivariateFunction` is always assumed to be the set of real numbers.
    * The rule defining the function may be passed into the constructor as the parameter `mapping`. If `mapping` is a callable, its parameters **must** be keyword-only.

    Parameters
    ----------
    domain : IndexLike | None, default=None
        The domain of the function.
    mapping : MappingLike | None, default=None
        The underlying rule defining the function. If a `Callable`, its parameters **must** be keyword-only.
    kind : Literal["any", "measure", "probability"], default="any"
        The kind of outputs of the function. The options `measure` and `probability` are meant to be used by measures.
    output_name: Hashable, default="output"
        The name of the outputs of the function.
    name : Hashable | None, default=None
        The name of the function. If `None`, a default name of `f` will be used.
    **kwargs
        Additional keyword arguments passed to subclasses.

    Examples
    --------
    Define a `MultivariateFunction` with an explicit `domain` and a `mapping` expressed as a lambda function. Note that the parameters to the lambda function are keyword-only.

    >>> import pandas as pd
    >>> from sigalg.core import Domain, MultivariateFunction
    >>> X = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
    >>> f = MultivariateFunction(domain=X, mapping=lambda *, x, y: 2 * x + y**2)
    >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
    Function 'f':
          output
    x y
    1 2        6
    2 3       13
    1 4       18

    Define a function from a `pd.Series` object and an explicit `domain`.

    >>> mapping = pd.Series([6, 13, 18], index=X.data)
    >>> g = MultivariateFunction(domain=X, mapping=mapping, name="g")
    >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
    Function 'g':
          output
    x y
    1 2        6
    2 3       13
    1 4       18

    Define a function from a dictionary and an explicit `domain`.

    >>> mapping = {(1, 2): 6, (2, 3): 13, (1, 4): 18}
    >>> h = MultivariateFunction(domain=X, mapping=mapping, name="h")
    >>> print(h)  # doctest: +NORMALIZE_WHITESPACE
    Function 'h':
          output
    x y
    1 2        6
    2 3       13
    1 4       18

    Define a function from a lambda function without an explicit `domain`. We no longer can print the range of the function, but we can evaluate the function.

    >>> k = MultivariateFunction(mapping=lambda *, x, y: x * 2 + y, name="k")
    >>> print(k)
    Function 'k(x, y)'
    >>> print(k(x=2, y=1))
    5
    """

    _default_name = "f"
    _properties = ["_dict"]
    _repr_name = "Function"
    _str_name = "Function"

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        domain: IndexLike | None = None,
        mapping: MappingLike | None = None,
        kind: Literal["any", "measure", "probability"] = "any",
        output_name: Hashable = "output",
        name: Hashable | None = None,
        **kwargs,
    ) -> None:
        from ...validation.mapping_validator import MappingValidator
        from ..spaces.domain import Domain

        if name is None:
            name = type(self)._default_name

        if not isinstance(domain, Domain):
            domain = Domain(domain) if domain is not None else None

        v = MappingValidator(
            mapping=mapping,
            domain=domain,
            output_name=output_name,
            name=name,
            kind=kind,
        )

        self._data = v.data
        self._domain = v.domain
        self._output_name = v.output_name
        self._name = v.name
        self._variable_names = v.argument_names
        self._num_variables = v.num_arguments
        self._kind = kind

        try:
            self._function = v.fun
            self._signature = v.signature
        except (TypeError, ValueError) as e:
            raise ValueError(  # noqa: B904
                "Error when constructing callable multivariate function. Perhaps an invalid variable name?"
            ) from e

        self._initialize_property_caches()

    def _initialize_property_caches(self) -> None:
        for property in self._properties:
            setattr(self, property, None)

    @classmethod
    def from_numpy(
        cls,
        arr: ArrayLike,
        output_name: Hashable = "output",
        variable_names: list[Hashable] | None = None,
        variable_name_prefix: str | None = None,
        name: Hashable | None = None,
    ) -> MultivariateFunction:
        """Create a multivariate function from a NumPy array.

        The function is generated in such a way that `f(i, j, ...)` corresponds to the element at position `(i, j, ...)` in the NumPy array.

        Parameters
        ----------
        arr : ArrayLike
            The array-like object representing the function values.
        output_name : Hashable, default="output"
            The name of the outputs of the function.
        variable_names : list[Hashable] | None, default=None
            The names of the variables. If `None`, either `variable_name_prefix` will be used to generate names or default names will be generated.
        variable_name_prefix : str | None, default=None
            The prefix for generating variable names. If `None`, either default names will be generated or `variable_names` must be provided.
        name : Hashable | None, default=None
            The name of the function. If `None`, a default name will be used.

        Raises
        ------
        TypeError
            If `arr` is not a NumPy array, or if `variable_names` is not a list of hashable items or `None`, or if `variable_name_prefix` is not a string or `None`.
        ValueError
            If the length of `variable_names` does not match the number of dimensions of `arr`, or if both `variable_names` and `variable_name_prefix` are specified.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import MultivariateFunction
        >>> arr = np.array([[1, 2], [3, 4]])
        >>> f = MultivariateFunction.from_numpy(arr=arr, variable_name_prefix="x", name="f")
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
                output
        x_0 x_1
        0   0         1
            1         2
        1   0         3
            1         4
        """
        try:
            arr = np.array(arr)
        except Exception as e:
            raise TypeError("Failed to convert `arr` to a NumPy array.") from e
        if (
            variable_names is not None
            and not isinstance(variable_names, list)
            and not all(isinstance(name, Hashable) for name in variable_names)
        ):
            raise TypeError(
                "`variable_names` must be a list of hashable items or None."
            )
        if variable_names is not None and len(variable_names) != arr.ndim:
            raise ValueError(
                "The length of `variable_names` must match the number of dimensions of `arr`."
            )
        if variable_name_prefix is not None and not isinstance(
            variable_name_prefix, str
        ):
            raise TypeError("`variable_name_prefix` must be a string or None.")
        if variable_names is not None and variable_name_prefix is not None:
            raise ValueError(
                "Cannot specify both `variable_names` and `variable_name_prefix`."
            )
        if variable_names is None and variable_name_prefix is not None:
            variable_names = [f"{variable_name_prefix}_{i}" for i in range(arr.ndim)]

        if arr.ndim == 1:
            idx = pd.Index(range(arr.shape[0]))
        else:
            idx = pd.MultiIndex.from_product([range(dim) for dim in arr.shape])
        data = pd.Series(arr.ravel(), index=idx)

        if variable_names is not None:
            data.index.names = variable_names

        return MultivariateFunction(mapping=data, output_name=output_name, name=name)

    @classmethod
    def from_rand(
        cls,
        domain_dims: tuple[int],
        output_name: Hashable = "output",
        variable_names: list[Hashable] | None = None,
        variable_name_prefix: str | None = None,
        distribution: Literal["uniform", "normal"] = "uniform",
        min_value: int = 1,
        max_value: int = 10,
        loc: float = 0.0,
        scale: float = 1.0,
        name: Hashable | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> MultivariateFunction:
        """Generate a random multivariate function.

        Parameters
        ----------
        domain_dims : tuple[int]
            The dimensions of the domain of the function.
        output_name : Hashable, default="output"
            The name of the outputs of the function.
        variable_names : list[Hashable] | None, default=None
            The names of the variables. If `None`, either `variable_name_prefix` will be used to generate names or default names will be generated.
        variable_name_prefix : str | None, default=None
            The prefix for generating variable names. If `None`, either default names will be generated or `variable_names` must be provided.
        distribution : Literal["uniform", "normal"], default="uniform"
            The distribution to use for generating random values.
        min_value : int, default=1
            The minimum value for the uniform distribution.
        max_value : int, default=10
            The maximum value for the uniform distribution.
        loc : float, default=0.0
            The mean for the normal distribution.
        scale : float, default=1.0
            The standard deviation for the normal distribution.
        name : Hashable | None, default=None
            The name of the function. If `None`, a default name will be used.
        random_state : int | np.random.Generator | None, default=None
            The random state for reproducibility.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import MultivariateFunction
        >>> rng = np.random.default_rng(42)

        Generate a random multivariate function with values drawn from a standard normal distribution.

        >>> f = MultivariateFunction.from_rand(
        ...     domain_dims=(2, 3),
        ...     variable_name_prefix="x",
        ...     distribution="normal",
        ...     random_state=rng,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
                   output
        x_0 x_1
        0   0    0.304717
            1   -1.039984
            2    0.750451
        1   0    0.940565
            1   -1.951035
            2   -1.302180

        Generate a random multivariate function with values drawn from a uniform distribution on the integers `[-10, 10)`.

        >>> g = MultivariateFunction.from_rand(
        ...     domain_dims=(2, 3),
        ...     variable_name_prefix="x",
        ...     distribution="uniform",
        ...     min_value=-10,
        ...     max_value=10,
        ...     name="g",
        ...     random_state=rng,
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
                 output
        x_0 x_1
        0   0         4
            1         5
            2         4
        1   0         5
            1         0
            2        -8
        """
        if (
            not isinstance(domain_dims, tuple)
            or not all(isinstance(dim, int) for dim in domain_dims)
            or len(domain_dims) == 0
        ):
            raise TypeError("`domain_dims` must be a non-empty tuple of integers.")
        if not all(dim > 0 for dim in domain_dims):
            raise ValueError(
                "All dimensions in `domain_dims` must be positive integers."
            )
        if not isinstance(output_name, Hashable):
            raise TypeError("`output_name` must be hashable.")
        if variable_names is not None and not all(
            isinstance(name, Hashable) for name in variable_names
        ):
            raise TypeError("All elements of `variable_names` must be hashable.")
        if variable_names is not None and len(variable_names) != len(domain_dims):
            raise ValueError(
                "The length of `variable_names` must match the number of dimensions in `domain_dims`."
            )
        if variable_name_prefix is not None and not isinstance(
            variable_name_prefix, str
        ):
            raise TypeError("`variable_name_prefix` must be a string or None.")
        if distribution not in ("uniform", "normal"):
            raise ValueError(f"Unsupported distribution: {distribution}")
        if not isinstance(min_value, int):
            raise TypeError("`min_value` must be an integer.")
        if not isinstance(max_value, int):
            raise TypeError("`max_value` must be an integer.")
        if min_value > max_value:
            raise ValueError("`min_value` cannot be greater than `max_value`.")
        if not isinstance(loc, (int, float)):
            raise TypeError("`loc` must be a number.")
        if not isinstance(scale, (int, float)):
            raise TypeError("`scale` must be a number.")
        if scale <= 0:
            raise ValueError("`scale` must be positive.")
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("`name` must be hashable or None.")
        if random_state is not None and not isinstance(
            random_state, (int, np.random.Generator)
        ):
            raise TypeError(
                "`random_state` must be an integer, a NumPy random Generator, or None."
            )

        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        if distribution == "normal":
            arr = rng.normal(loc=loc, scale=scale, size=domain_dims)
        elif distribution == "uniform":
            arr = rng.integers(low=min_value, high=max_value, size=domain_dims)

        return MultivariateFunction.from_numpy(
            arr=arr,
            output_name=output_name,
            variable_names=variable_names,
            variable_name_prefix=variable_name_prefix,
            name=name,
        )

    @classmethod
    def tensor_product(
        cls,
        factors: list[MultivariateFunction],
        variable_names: list[Hashable] | None = None,
        output_name: Hashable | None = None,
        name: Hashable | None = None,
    ) -> MultivariateFunction:
        r"""Compute the tensor product of a list of functions.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        factors : list[MultivariateFunction]
            The factors of the tensor product.
        variable_names : list[Hashable] | None, default=None
            The variable names of the resulting function. If `None`, the variable names will be inferred from the input functions.
        output_name : Hashable | None, default=None
            The output name of the resulting function. If `None`, a default name of `output` will be used.
        name : Hashable | None, default=None
            The name of the resulting function. If `None`, a default name will be generated from the names of the input functions.

        Raises
        ------
        TypeError
            If any element of `factors` is not a `MultivariateFunction`, or if `variable_names` is not a list or `None`, or if any element of `variable_names` is not hashable (if given), or if `output_name` is not hashable (if given), or if `name` is not hashable (if given).
        ValueError
            If the length of `variable_names` does not match the total number of arguments in `factors`.

        Returns
        -------
        tensor_prod : MultivariateFunction
            The tensor product of the input functions.

        Examples
        --------
        Define two functions.

        >>> from sigalg.core import Domain, MultivariateFunction
        >>> X = Domain.from_sequence(size=2, variable_name="x")
        >>> Y = Domain.from_sequence(size=2, variable_name="y", name="Y")
        >>> f = MultivariateFunction(
        ...     domain=X @ Y,
        ...     mapping=lambda *, x, y: x**2 + y + 2,
        ... )
        >>> g = MultivariateFunction(
        ...     domain=Y,
        ...     mapping=lambda *, y: y + 5,
        ...     name="g",
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
             output
        x y
        0 0       2
          1       3
        1 0       3
          1       4
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
           output
        y
        0       5
        1       6

        Compute their tensor product using the `tensor_product` method.

        >>> prod = MultivariateFunction.tensor_product([f, g])
        >>> print(prod)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f x g':
                   output
        x y_0 y_1
        0 0   0        10
              1        12
          1   0        15
              1        18
        1 0   0        15
              1        18
          1   0        20
              1        24

        Compute the same tensor product using the `@` operator.

        >>> prod = f @ g
        >>> print(prod)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f x g':
                    output
        x y_0 y_1
        0 0   0        10
              1        12
          1   0        15
              1        18
        1 0   0        15
              1        18
          1   0        20
              1        24

        Notes
        -----
        Let $f:X \to \mathbb{R}$ and $g: Y \to \mathbb{R}$ be two functions. Their *tensor product*, denoted $f\otimes g$, is the function defined by

        $$
        f \otimes g: X \times Y \to \mathbb{R}, \quad (f \otimes g)(x,y) = f(x)g(y).
        $$
        """
        from ..indices.index import Index
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from ..spaces.domain import Domain

        if not all(isinstance(function, MultivariateFunction) for function in factors):
            raise TypeError(
                "All elements of `factors` must be instances of MultivariateFunction."
            )
        if variable_names is not None and not isinstance(variable_names, list):
            raise TypeError("`variable_names` must be a list or None.")
        if isinstance(variable_names, list) and not all(
            isinstance(name, Hashable) for name in variable_names
        ):
            raise TypeError("All elements of `variable_names` must be hashable.")
        if variable_names is not None and len(variable_names) != sum(
            function.num_variables for function in factors
        ):
            raise ValueError(
                "The length of `variable_names` must match the total number of arguments in `factors`."
            )
        if output_name is not None and not isinstance(output_name, Hashable):
            raise TypeError("`output_name` must be hashable or None.")
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("`name` must be hashable or None.")

        prod_arg_names = Index._subscript_var_names(
            [function.variable_names for function in factors],
            grouped=True,
        )

        function_data = []

        for k, (arg_names, function) in enumerate(zip(prod_arg_names, factors)):
            new_func_data = function.data.rename(f"{function.output_name}_{k}")
            new_func_data.index.names = arg_names
            function_data.append(new_func_data)

        product_data = function_data[0].reset_index()

        for next_data in function_data[1:]:
            product_data = pd.merge(
                left=product_data,
                right=next_data.reset_index(),
                how="cross",
            )

        mapping = product_data.set_index(
            [name for lst in prod_arg_names for name in lst]
        ).prod(axis=1)

        if output_name is None:
            output_name = "output"
        if variable_names is None:
            variable_names = mapping.index.names
        else:
            mapping.index.names = variable_names
        if name is None:
            name = " x ".join([function.name for function in factors])

        domain = Domain(
            indices=mapping.index,
            name=" x ".join([function.domain.name for function in factors]),
            variable_names=variable_names,
            bypass_validation=True,
        )

        if cls.__name__ == "Measure":
            all_probs = len(
                [
                    function.kind
                    for function in factors
                    if function.kind == "probability"
                ]
            ) == len(factors)

            return cls(
                domain=SigmaAlgebra.cartesian_product(
                    [function.sig_alg for function in factors]
                ),
                mapping=mapping,
                kind="probability" if all_probs else "measure",
                name=name,
            )
        else:
            return cls(
                domain=domain,
                mapping=mapping,
                output_name=output_name,
                name=name,
            )

    def __matmul__(self, other: MultivariateFunction) -> MultivariateFunction:
        """Form the tensor product of this instance of `MultivariateFunction` with another.

        Internally calls the `tensor_product` method.

        Parameters
        ----------
        other : MultivariateFunction
            The other function to form the tensor product with.

        Returns
        -------
        tensor_product : MultivariateFunction
            The tensor product.
        """
        return type(self).tensor_product(factors=[self, other])

    @classmethod
    def tensor_power(
        cls, function: MultivariateFunction, n: int
    ) -> MultivariateFunction:
        r"""Get the tensor power of the function.

        Parameters
        ----------
        function : MultivariateFunction
            The base of the tensor power.
        n : int
            The power of the tensor power.

        Raises
        ------
        TypeError
            If `n` is not an integer or `function` is not a `MultivariateFunction`.
        ValueError
            If `n` is not positive.

        Returns
        -------
        tensor_power : MultivariateFunction
            The tensor power.

        Examples
        --------
        Define a function.

        >>> from sigalg.core import Domain, MultivariateFunction
        >>> X = Domain([1, 2, 3], variable_names=["x"])
        >>> f = MultivariateFunction(
        ...     domain=X,
        ...     mapping=lambda *, x: x**2,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
           output
        x
        1       1
        2       4
        3       9

        Compute the second tensor power using the `tensor_power` class method.

        >>> print(MultivariateFunction.tensor_power(f, 2))  # doctest: +NORMALIZE_WHITESPACE
        Function 'f ^ 2':
                 output
        x_0 x_1
        1   1         1
            2         4
            3         9
        2   1         4
            2        16
            3        36
        3   1         9
            2        36
            3        81

        Define a second function.

        >>> E = Domain([(1, 2), (3, 4)], variable_names=["x", "y"], name="E")
        >>> g = MultivariateFunction(
        ...     domain=E,
        ...     mapping=lambda *, x, y: x + y,
        ...     name="g",
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
             output
        x y
        1 2       3
        3 4       7

        Compute the third tensor power using the `^` operator notation.

        >>> print(g ^ 3)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g ^ 3':
                                 output
        x_0 y_0 x_1 y_1 x_2 y_2
        1   2   1   2   1   2        27
                        3   4        63
                3   4   1   2        63
                        3   4       147
        3   4   1   2   1   2        63
                        3   4       147
                3   4   1   2       147
                        3   4       343
        """
        name = f"{function.name} ^ {n}"
        return cls.tensor_product(factors=[function] * n, name=name)

    def __xor__(self, n: int) -> MultivariateFunction:
        """Form the tensor power of this instance of `MultivariateFunction`.

        Internally calls the `tensor_power` method.

        Parameters
        ----------
        n : int
            The power of the tensor power.

        Returns
        -------
        tensor_power : MultivariateFunction
            The tensor power.
        """
        return type(self).tensor_power(function=self, n=n)

    # --------------------- properties --------------------- #

    @property
    def function(self) -> Callable | None:
        """Get the underlying callable function.

        Returns
        -------
        function : Callable | None
            The underlying callable function if defined, otherwise `None`.

        Examples
        --------
        >>> import pandas as pd
        >>> from sigalg.core import Domain, MultivariateFunction
        >>> X = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> mapping = pd.Series([6, 13, 18], index=X.data)
        >>> f = MultivariateFunction(domain=X, mapping=mapping)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
              output
        x y
        1 2        6
        2 3       13
        1 4       18
        >>> print(f.function(x=1, y=2))
        6
        """
        return self._function

    @property
    def data(self) -> pd.Series | None:
        """Get the underlying data as a `pd.Series` object.

        In order for the `data` to be computed, a `domain` must be provided at initialization.

        Returns
        -------
        data : pd.Series | None
            The underlying data as a `pd.Series` object if defined, otherwise `None`.

        Examples
        --------
        >>> from sigalg.core import Domain, MultivariateFunction
        >>> X = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> f = MultivariateFunction(domain=X, mapping=lambda *, x, y: 2 * x + y**2)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
              output
        x y
        1 2        6
        2 3       13
        1 4       18
        >>> print(f.data)  # doctest: +NORMALIZE_WHITESPACE
        x  y
        1  2     6
        2  3    13
        1  4    18
        Name: output, dtype: int64
        """
        return self._data

    @property
    def dict(self) -> dict | None:
        """Get the underlying data as a dictionary.

        In order for the `dict` to be computed, a `domain` must be provided at initialization.

        Returns
        -------
        result_dict : dict | None
            The underlying data as a dictionary if defined, otherwise `None`.

        Examples
        --------
        >>> from sigalg.core import Domain, MultivariateFunction
        >>> X = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> f = MultivariateFunction(domain=X, mapping=lambda *, x, y: 2 * x + y**2)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
              output
        x y
        1 2        6
        2 3       13
        1 4       18
        >>> print(f.dict)
        {(1, 2): 6, (2, 3): 13, (1, 4): 18}
        """
        if self._dict is None and self.data is not None:
            self._dict = self.data.to_dict()
        return self._dict

    @property
    def variable_names(self) -> list[Hashable] | None:
        """Get the variable names of the function.

        Returns
        -------
        variable_names : list[Hashable] | None
            The variable names of the function if defined, otherwise `None`.

        Examples
        --------
        >>> from sigalg.core import Domain, MultivariateFunction
        >>> X = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> f = MultivariateFunction(domain=X, mapping=lambda *, x, y: 2 * x + y**2)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
              output
        x y
        1 2        6
        2 3       13
        1 4       18
        >>> print(f.variable_names)
        ['x', 'y']
        """
        return self._variable_names

    @property
    def signature(self) -> inspect.Signature | None:
        """Get the signature of the underlying callable function.

        Returns
        -------
        signature : inspect.Signature | None
            The signature of the underlying callable function if defined, otherwise `None`.

        Examples
        --------
        >>> from sigalg.core import Domain, MultivariateFunction
        >>> X = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> f = MultivariateFunction(domain=X, mapping=lambda *, x, y: 2 * x + y**2)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
              output
        x y
        1 2        6
        2 3       13
        1 4       18
        >>> print(f.signature)
        (*, x, y)
        """
        return self._signature

    @property
    def num_variables(self) -> int | None:
        """Get the number of variables of the function.

        Returns
        -------
        num_variables : int | None
            The number of variables of the function if defined, otherwise `None`.

        Examples
        --------
        >>> from sigalg.core import Domain, MultivariateFunction
        >>> X = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> f = MultivariateFunction(domain=X, mapping=lambda *, x, y: 2 * x + y**2)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
              output
        x y
        1 2        6
        2 3       13
        1 4       18
        >>> print(f.num_variables)
        2
        """
        return self._num_variables

    @property
    def domain(self) -> Domain | None:
        """Get the domain of the function.

        Returns
        -------
        domain : Domain | None
            The domain of the function if defined, otherwise `None`.

        Examples
        --------
        >>> from sigalg.core import Domain, MultivariateFunction
        >>> X = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> f = MultivariateFunction(domain=X, mapping=lambda *, x, y: 2 * x + y**2)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
              output
        x y
        1 2        6
        2 3       13
        1 4       18
        >>> print(f.domain)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'X':
         x  y
         1  2
         2  3
         1  4
        """
        return self._domain

    @property
    def name(self) -> Hashable:
        """Get the name of the function.

        The `name` property is settable.

        Returns
        -------
        name : Hashable
            The name of the function.

        Examples
        --------
        >>> from sigalg.core import Domain, MultivariateFunction
        >>> X = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> g = MultivariateFunction(domain=X, mapping=lambda *, x, y: 2 * x + y**2, name="g")
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
              output
        x y
        1 2        6
        2 3       13
        1 4       18
        >>> print(g.name)
        g
        >>> g.name = "fun"
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'fun':
              output
        x y
        1 2        6
        2 3       13
        1 4       18
        """
        return self._name

    @name.setter
    def name(self, name: Hashable) -> None:
        """Set the name of the function.

        Parameters
        ----------
        name : Hashable
            The new name for the function.

        Raises
        ------
        TypeError
            If `name` is not a hashable type.
        """
        if not isinstance(name, Hashable):
            raise TypeError("The name must be a hashable type.")
        self._name = name

    def with_name(self, name: Hashable) -> MultivariateFunction:
        """Set the name of the function and return self for chaining.

        Parameters
        ----------
        name : Hashable
            The new name for the function.

        Returns
        -------
        self : MultivariateFunction
            The instance of the function with the updated name.
        """
        self.name = name
        return self

    @property
    def output_name(self) -> Hashable | None:
        """Get the output name of the function.

        Returns
        -------
        output_name : Hashable | None
            The output name of the function if defined, otherwise `None`.

        Examples
        --------
        >>> from sigalg.core import Domain, MultivariateFunction
        >>> X = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> f = MultivariateFunction(domain=X, mapping=lambda *, x, y: 2 * x + y**2)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
              output
        x y
        1 2        6
        2 3       13
        1 4       18
        >>> print(f.output_name)
        output
        """
        return self._output_name

    # --------------------- data methods --------------------- #

    def __call__(self, **kwargs) -> Real | MultivariateFunction:
        """Call the function with the provided arguments.

        If a complete set of arguments is provided, the function is evaluated and the result is returned. If a partial set of arguments is provided, a new `MultivariateFunction` instance is returned, representing the partially applied function.

        Parameters
        ----------
        **kwargs : keyword arguments
            Keyword arguments for the function.

        Returns
        -------
        result : Real | MultivariateFunction
            The result of evaluating the function with the provided arguments, or a new `MultivariateFunction` instance representing the partially applied function.

        Examples
        --------
        >>> from sigalg.core import Domain, MultivariateFunction
        >>> X = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> f = MultivariateFunction(domain=X, mapping=lambda *, x, y: 2 * x + y**2)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
              output
        x y
        1 2        6
        2 3       13
        1 4       18
        >>> print(f(x=2, y=3))
        13
        >>> print(f(x=1))  # doctest: +NORMALIZE_WHITESPACE
        Function 'f(x=1)':
              output
        y
        2          6
        4         18
        >>> print(f(x=1)(y=4))
        18
        """
        from ..spaces.domain import Domain

        specified_arguments = self.signature.bind_partial(**kwargs)
        unspecified_arguments = [
            inspect.Parameter(parameter, inspect.Parameter.KEYWORD_ONLY)
            for parameter in self.variable_names
            if parameter not in specified_arguments.arguments.keys()
        ]

        if len(unspecified_arguments) == 0:
            return self.function(**specified_arguments.arguments)
        else:
            partial_signature = inspect.Signature(unspecified_arguments)

            def partial_function(*args, **kwargs):
                partial_parameters = partial_signature.bind(*args, **kwargs)
                all_args = {
                    **specified_arguments.arguments,
                    **partial_parameters.arguments,
                }
                return self.function(**all_args)

            partial_function.__signature__ = partial_signature

            name = f"{self.name}({', '.join(f'{p}={specified_arguments.arguments[p]}' for p in self.variable_names if p in specified_arguments.arguments)})"

            if self.data is not None:
                try:
                    data = self.data.xs(
                        key=tuple(specified_arguments.arguments.values()),
                        level=tuple(specified_arguments.arguments.keys()),
                    ).index
                    parameter_string = ", ".join(
                        f"{name}={value}"
                        for name, value in specified_arguments.arguments.items()
                    )
                    domain_name = f"{self.domain.name}|{{{parameter_string}}}"
                    partial_domain = Domain(indices=data, name=domain_name)

                except KeyError as e:
                    raise ValueError(
                        "The specified arguments do not correspond to any entries in the function's data."
                    ) from e

            else:
                partial_domain = None

            return MultivariateFunction(
                domain=partial_domain,
                name=name,
                mapping=partial_function,
                output_name=self.output_name,
            )

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Return the function's data as a NumPy array.

        Parameters
        ----------
        dtype : data-type | None, default=None
            The desired data-type for the array. If `None`, the data-type of the underlying data is used.
        copy : bool | None, default=None
            Whether to return a copy of the data. If `None`, the default behavior is used.

        Returns
        -------
        np.ndarray
            The function's data as a NumPy array.
        """
        arr = self.data.values
        if dtype is not None:
            arr = np.asarray(arr, dtype=dtype)
        if copy:
            arr = arr.copy()

        return arr

    def to_numpy(self, multi_dim: bool = False, dtype=None, copy=None) -> np.ndarray:
        """Return the function's data as a NumPy array.

        Parameters
        ----------
        dtype : data-type | None, default=None
            The desired data-type for the array. If `None`, the data-type of the underlying data is used.
        copy : bool | None, default=None
            Whether to return a copy of the data. If `None`, the default behavior is used.

        Returns
        -------
        np.ndarray
            The function's data as a NumPy array.
        """
        if multi_dim:
            arr = self.data.to_xarray().values
            if dtype is not None:
                arr = np.asarray(arr, dtype=dtype)
            if copy:
                arr = arr.copy()
            return arr
        else:
            return self.__array__(dtype=dtype, copy=copy)

    # --------------------- conversion methods --------------------- #

    def to_measure(
        self,
        measure_domain: SigmaAlgebra | IndexLike,
        kind: Literal["measure", "probability"] = "measure",
        name: Hashable | None = None,
        in_place: bool = False,
    ) -> Measure | ParametrizedMeasure:
        """Generate a parametrized probability measure or measure from the multivariate function.

        This method does not validate whether the resulting parametrized measure (or measure) actually *is* a measure. It is the user's responsibility to ensure that the function satisfies the necessary properties of a measure.

        Parameters
        ----------
        measure_domain : SigmaAlgebra | IndexLike
            The domain of the measure. Must be a `SigmaAlgebra` or an `IndexLike` object that can be converted to a `Domain`. In the latter case, the sigma-algebra will be the power-set sigma-algebra of the domain.
        kind : Literal["measure", "probability"], default="measure"
            The kind of measure to create. Must be either "measure" or "probability".
        name : Hashable | None, default=None
            The name of the resulting measure. If `None`, the name will be inherited from the function. If the function's name is also `None`, a default name will be generated.
        in_place : bool, default=False
            If `True`, the current instance will be converted to a measure in place. If `False`, a new measure instance will be returned.

        Examples
        --------
        Define a multivariate function on a Cartesian product of a 2-dimensional parameter space and a 1-dimensional measure domain.

        >>> from sigalg.core import Domain, MultivariateFunction
        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> X = Domain.from_sequence(size=3, variable_name="x")
        >>> f = MultivariateFunction(
        ...     domain=(Theta ^ 2) @ X,
        ...     mapping=lambda *, theta_0, theta_1, x: theta_0 + 2 * theta_1 + x,
        ...     output_name="measure",
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
                           measure
        theta_0 theta_1 x
        0       0       0        0
                        1        1
                        2        2
                1       0        2
                        1        3
                        2        4
        1       0       0        1
                        1        2
                        2        3
                1       0        3
                        1        4
                        2        5

        Convert the function to a parametrized measure by specifying the measure's domain.

        >>> parametrized_measure = f.to_measure(X)
        >>> print(parametrized_measure)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'f':
                           measure
        theta_0 theta_1 x
        0       0       0        0
                        1        1
                        2        2
                1       0        2
                        1        3
                        2        4
        1       0       0        1
                        1        2
                        2        3
                1       0        3
                        1        4
                        2        5

        Create a partial function by fixing one of the parameters and convert it to a parametrized measure.

        >>> partial_function = f(theta_0=0).to_measure(X)
        >>> print(partial_function)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'f(theta_0=0)':
                   measure
        theta_1 x
        0       0        0
                1        1
                2        2
        1       0        2
                1        3
                2        4

        Fix all parameters and convert the resulting function to a measure.

        >>> measure = f(theta_0=0, theta_1=1).to_measure(X)
        >>> print(measure)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'f(theta_0=0, theta_1=1)':
           measure
        x
        0        2
        1        3
        2        4
        """
        from ...validation.measure_domain_validator import MeasureDomainValidator
        from ..measures.measure import Measure
        from ..measures.parametrized_measure import ParametrizedMeasure
        from ..measures.parametrized_probability_measure import (
            ParametrizedProbabilityMeasure,
        )

        if self.domain is not None:
            if name is not None and not isinstance(name, Hashable):
                raise TypeError("If provided, name must be a hashable type.")

            v = MeasureDomainValidator(measure_domain=measure_domain, kind=kind)

            if name is None:
                name = self.name

            is_measure = v.sig_alg.variable_names == self.variable_names

            if is_measure:
                measure = Measure(
                    domain=v.sig_alg,
                    mapping=self.data,
                    kind=kind,
                    output_name=self.output_name,
                    name=name,
                )
            else:
                measure = ParametrizedMeasure(
                    domain=self.domain,
                    mapping=self.data,
                    # kind=kind,
                    output_name=self.output_name,
                    name=name,
                )
                # HACK: the call to the ParametrizedMeasure constructor uses input validation to screen out probability measures, so we manually change the class
                if kind == "probability":
                    measure.__class__ = ParametrizedProbabilityMeasure

                measure._init_measure_attrs(
                    sig_alg=v.sig_alg,
                    kind=kind,
                )

            if in_place:
                self.__class__ = type(measure)
                self.__dict__.update(measure.__dict__)
                return self
            else:
                return measure

        else:
            return NotImplementedError(
                "The to_measure method is not implemented yet for functions without an explicit domain."
            )

    def to_measurable_function(
        self, sig_alg: SigmaAlgebra
    ) -> MeasurableFunction | ParametrizedMeasurableFunction:
        """Pass."""

    def with_variable_names(
        self, variable_names: list[Hashable]
    ) -> MultivariateFunction:
        """Return a new instance of the multivariate function with updated variable names."""
        from ..measures.measure import Measure

        constructor_sig = inspect.signature(MultivariateFunction)
        params = {
            name.strip("_"): value
            for name, value in self.__dict__.items()
            if name.strip("_") in constructor_sig.parameters
        }

        if self.domain is not None:
            domain = copy.deepcopy(self.domain)
            mapping = copy.deepcopy(self.data)
            domain.name = f"{self.domain.name}_new"
            domain.variable_names = variable_names
            mapping.index.names = variable_names
            params["mapping"] = mapping
            params["domain"] = domain
            params["name"] = f"{self.name}_new"

        if self.domain is None:
            mapping = copy.deepcopy(self.function)
            new_params = [
                inspect.Parameter(name=name, kind=inspect.Parameter.KEYWORD_ONLY)
                for name in variable_names
            ]
            new_sig = inspect.Signature(new_params)

            def mapping(**kwargs):  # noqa: D103
                new_to_old = {
                    old: kwargs[new]
                    for old, new in zip(self.variable_names, variable_names)
                }
                return self.function(**new_to_old)

            mapping.__signature__ = new_sig
            params["mapping"] = mapping

        params["kind"] = "any"

        if isinstance(self, Measure):
            return type(self)(**params)
        else:
            return MultivariateFunction(**params)

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a concise string representation of the function.

        Returns
        -------
        repr_str : str
            The string representation of the function.
        """
        if self.variable_names is not None:
            parameter_list = ", ".join(self.variable_names)
            return (
                f"{type(self)._repr_name}(parameters=({parameter_list}), "
                f"domain={self.domain.name}, "
                f"output_name={self.output_name}, "
                f"name={self.name})"
            )
        else:
            return type(self)._repr_name + "(empty)"

    def __str__(self) -> str:
        """Return a detailed string representation of the function.

        Returns
        -------
        repr_str : str
            The string representation of the function.
        """
        if self.data is not None:
            return f"{type(self)._str_name} '{self.name}':\n{self.data.to_frame()}"
        elif self.variable_names is not None:
            parameter_list = ", ".join(self.variable_names)
            return f"{type(self)._str_name} '{self.name}({parameter_list})'"
        else:
            return f"{type(self)._str_name} '{self.name}': empty"

    # --------------------- equality --------------------- #

    # TODO: add an `equal_as_measures` method
    def __eq__(self, other: MultivariateFunction | Real) -> bool:
        """Check if two multivariate functions are equal.

        Equality may only be checked if both functions have defined data and domains. If the arguments of the two functions are the same but in a different order, the method will attempt to reorder the levels of the other function's data to match the order of this function's arguments before comparing the values.

        Parameters
        ----------
        other : MultivariateFunction | Real
            The other multivariate function to compare with.

        Raises
        ------
        ValueError
            If either function has undefined data or domain, if the sigma-algebras of the two functions are different, or if the argument names of the two functions are different and cannot be reconciled.

        Returns
        -------
        are_equal : bool
            True if the two functions are equal, False otherwise.

        Examples
        --------
        >>> from sigalg.core import Domain, MultivariateFunction
        >>> D_f = Domain([(0, 1), (1, 2)], variable_names=["x", "y"], name="D_f")
        >>> D_g = Domain([(1, 0), (2, 1)], variable_names=["y", "x"], name="D_g")
        >>> f = MultivariateFunction(domain=D_f, mapping=lambda *, x, y: x**2 + y**2)
        >>> g = MultivariateFunction(domain=D_g, mapping=lambda *, y, x: x**2 + y**2, name="g")
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
              output
        x y
        0 1        1
        1 2        5
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
              output
        y x
        1 0        1
        2 1        5
        >>> print(f == g)
        True
        """
        if isinstance(other, MultivariateFunction):
            if self.function is None or other.function is None:
                raise ValueError("Cannot compare empty functions.")
            if self.domain is None or other.domain is None:
                raise ValueError(
                    "Cannot compare functions when one (or both) domains are not defined."
                )
            if self.num_variables != other.num_variables or set(
                self.variable_names
            ) != set(other.variable_names):
                raise ValueError(
                    "Cannot compare functions with different numbers of arguments or argument names."
                )

            if self.num_variables > 1:
                return np.allclose(
                    self.data.values,
                    other.data.reorder_levels(self.data.index.names)
                    .reindex(self.data.index)
                    .values,
                )
            else:
                return np.allclose(self.data.values, other.data.values)

        elif isinstance(other, Real):
            if self.function is None:
                raise ValueError("Cannot compare empty functions.")
            if self.domain is None:
                raise ValueError(
                    "Cannot compare functions when the domain is not defined."
                )
            return np.allclose(self.data.values, other)

        else:
            raise TypeError(
                "Can only compare with another MultivariateFunction or a scalar."
            )

    # --------------------- arithmetic operations --------------------- #

    # TODO: the domain names are really weird, like f + g. find something better, bro
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
        from ..spaces.domain import Domain

        if isinstance(other, MultivariateFunction):
            if reverse:
                function_name = f"({other.name} {op_symbol} {self.name})"
            else:
                function_name = f"({self.name} {op_symbol} {other.name})"

            argument_names = list(
                dict.fromkeys(self.variable_names + other.variable_names)
            )
            output_name = (
                self.output_name
                if self.output_name == other.output_name
                else f"({self.output_name} {op_symbol} {other.output_name})"
            )

            if self.domain is not None and other.domain is not None:
                if len(argument_names) < len(self.variable_names) + len(
                    other.variable_names
                ):
                    merged = pd.merge(
                        self.data,
                        other.data,
                        how="inner",
                        left_index=True,
                        right_index=True,
                        suffixes=("_self", "_other"),
                    )
                    data = operation(
                        merged[f"{output_name}_self"],
                        merged[f"{output_name}_other"],
                    ).rename(output_name)

                else:
                    merged = pd.merge(
                        self.data.reset_index(),
                        other.data.reset_index(),
                        how="cross",
                        suffixes=("_self", "_other"),
                    )
                    merged.set_index(
                        self.variable_names + other.variable_names, inplace=True
                    )
                    data = operation(
                        merged[f"{output_name}_self"],
                        merged[f"{output_name}_other"],
                    ).rename(output_name)

                domain_data = data.index
                domain_name = f"({self.domain.name} {op_symbol} {other.domain.name})"
                domain = Domain(indices=domain_data, name=domain_name)

                return MultivariateFunction(
                    domain=domain,
                    mapping=data,
                    name=function_name,
                    output_name=data.name,
                )

            else:
                arguments = [
                    inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
                    for name in argument_names
                ]
                sig = inspect.Signature(arguments)

                def binary_function(**kwargs):
                    bound = sig.bind(**kwargs)
                    self_arguments = {
                        name: bound.arguments[name]
                        for name in self.variable_names
                        if name in bound.arguments
                    }
                    other_arguments = {
                        name: bound.arguments[name]
                        for name in other.variable_names
                        if name in bound.arguments
                    }
                    if reverse:
                        return operation(
                            other(**other_arguments), self(**self_arguments)
                        )
                    else:
                        return operation(
                            self(**self_arguments), other(**other_arguments)
                        )

                binary_function.__signature__ = sig

                return MultivariateFunction(name=function_name, mapping=binary_function)

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

            return MultivariateFunction(
                domain=self.domain,
                name=function_name,
                mapping=scalar_function,
                output_name=self.output_name,
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

    @staticmethod
    def _to_float(x):
        return x.astype(float) if hasattr(x, "astype") else float(x)

    def __pow__(self, other):
        """Raise this function to the power of another multivariate function or a scalar."""
        return self._apply_binary_operation(
            other, lambda a, b: self._to_float(a) ** self._to_float(b), "**"
        )

    def __neg__(self):
        """Negate this function."""

        def neg_function(**kwargs):
            bound = self.signature.bind(**kwargs)
            return -self(**bound.arguments)

        neg_function.__signature__ = self.signature

        function_name = f"(-{self.name})"

        return MultivariateFunction(
            domain=self.domain, name=function_name, mapping=neg_function
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
            other,
            lambda a, b: self._to_float(a) ** self._to_float(b),
            "**",
            reverse=True,
        )
