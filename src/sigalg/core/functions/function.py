"""A class representing a function."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Hashable, Iterator
from functools import cached_property
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from numbers import Real

    import numpy as np
    import pandas as pd
    from numpy.typing import ArrayLike

    from ...typing.index_like import IndexLike
    from ...typing.mapping_like import MappingLike
    from ..indices.index import Index
    from ..measures.measure import Measure
    from ..measures.parametrized_measure import ParametrizedMeasure
    from ..sigma_algebras.lattice import Lattice
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.domain import Domain
    from ..spaces.set import Set
    from .measurable_function import MeasurableFunction
    from .parametrized_measurable_function import ParametrizedMeasurableFunction

    PandasLike = pd.Series | pd.DataFrame


class Function:
    """A class representing a function.

    Mathematically, a function requires three items: A domain set, a codomain set, and a rule defining the function. For instances of `Function`:

    * The domain of the function is passed as the parameter `domain`, but this parameter is *not* required. This allows for the creation of functions whose domains are supposed to be continuous.
    * The codomain of an instance of `Function` is always assumed to be the set of real numbers.
    * The rule defining the function may be passed into the constructor as the parameter `mapping`. If `mapping` is a callable, its parameters **must** be keyword-only.

    Parameters
    ----------
    domain : IndexLike | None, default=None
        The domain of the function.
    mapping : MappingLike | None, default=None
        The underlying rule defining the function. If a `Callable`, its parameters must either all be keyword-only, or all positional only.
    kind : Literal["any", "measure", "probability"], default="any"
        The kind of outputs of the function. The options `measure` and `probability` are meant to be used by measures.
    name : Hashable | None, default=None
        The name of the function. If `None`, a default name will be generated.
    **kwargs
        Additional keyword arguments passed to subclasses.

    Examples
    --------
    Define a `Function` with an explicit `domain` and a `mapping` expressed as a lambda function. Note that the parameters to the lambda function are keyword-only.

    >>> import pandas as pd
    >>> from sigalg.core import Domain, Function
    >>> X = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
    >>> f = Function(domain=X, mapping=lambda *, x, y: 2 * x + y**2)
    >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
    Function 'f':
          f
    x y
    1 2   6
    2 3  13
    1 4  18

    Define a function from a `pd.Series` object and an explicit `domain`.

    >>> mapping = pd.Series([6, 13, 18], index=X.data)
    >>> g = Function(domain=X, mapping=mapping, name="g")
    >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
    Function 'g':
          g
    x y
    1 2   6
    2 3  13
    1 4  18

    Define a function from a dictionary and an explicit `domain`.

    >>> mapping = {(1, 2): 6, (2, 3): 13, (1, 4): 18}
    >>> h = Function(domain=X, mapping=mapping, name="h")
    >>> print(h)  # doctest: +NORMALIZE_WHITESPACE
    Function 'h':
          h
    x y
    1 2   6
    2 3  13
    1 4  18

    Define a function from a lambda function without an explicit `domain`. We no longer can print the range of the function, but we can evaluate the function.

    >>> k = Function(mapping=lambda *, x, y: x * 2 + y, name="k")
    >>> print(k)
    Function(parameters=(x, y), name=k)
    >>> print(k(x=2, y=1))
    5
    """

    _properties = []
    _default_name = "f"
    _repr_name = "Function"
    _str_name = "Function"

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        domain: IndexLike | None = None,
        mapping: MappingLike | None = None,
        kind: Literal[
            "any",
            "measure",
            "probability",
            "param_measure",
            "param_probability",
        ] = "any",
        domain_kind: Literal["Domain", "SampleSpace"] = "Domain",
        domain_name: Hashable | None = None,
        multi_dim_outputs: bool = True,
        output_name: Hashable | None = None,
        index: IndexLike | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
    ) -> None:
        from ...validation.domain_index_validator import DomainIndexValidator
        from ...validation.mapping_validator import MappingValidator

        u = DomainIndexValidator(
            domain=domain,
            domain_kind=domain_kind,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
        )

        domain = u.domain
        index = u.index
        self.domain_kind = u.domain_kind
        self.domain_name = u.domain_name
        self.index_kind = u.index_kind
        self.index_name = u.index_name

        if name is None:
            name = type(self)._default_name
        if output_name is None:
            output_name = name

        v = MappingValidator(
            domain=domain,
            mapping=mapping,
            kind=kind,
            domain_kind=domain_kind,
            multi_dim_outputs=multi_dim_outputs,
            output_name=output_name,
            index=index,
            index_kind=index_kind,
            name=name,
        )

        self.data = v.data
        self.name = v.name
        self.kind = v.kind

    @classmethod
    def _from_validated(
        cls,
        *,
        data: pd.Series | Callable,
        kind: Literal[
            "any",
            "measure",
            "probability",
            "param_measure",
            "param_probability",
        ],
        name: Hashable,
        domain_kind: Literal["Domain", "SampleSpace"],
        domain_name: Hashable | None,
        index_kind: Literal["Index", "Time"],
        index_name: Hashable | None,
    ) -> Function:
        function = object.__new__(cls)
        function.data = data
        function.name = name
        function.kind = kind
        function.domain_kind = domain_kind
        function.domain_name = domain_name
        function.index_kind = index_kind
        function.index_name = index_name
        return function

    @classmethod
    def from_numpy(
        cls,
        arr: ArrayLike,
        variable_names: list[Hashable] | None = None,
        kind: Literal["any", "measure", "probability"] = "any",
        name: Hashable | None = None,
        domain_kind: Literal["Domain", "SampleSpace"] = "Domain",
        domain_name: Hashable | None = None,
    ) -> Function:
        """Create a function from a NumPy array.

        The function is generated in such a way that `f(i, j, ...)` corresponds to the element at position `(i, j, ...)` in the NumPy array.

        Parameters
        ----------
        arr : ArrayLike
            The array-like object representing the function values.
        variable_names : list[Hashable] | None, default=None
            The names of the variables. If `None`, defaults will be generated.
        kind : Literal["any", "measure", "probability"], default="any"
            The kind of outputs of the function. The options `measure` and `probability` are meant to be used by measures.
        name : Hashable | None, default=None
            The name of the function. If `None`, a default name will be generated.
        domain_class: Literal["Domain", "SampleSpace"], default="Domain
            The class of the underlying domain.
        domain_name: Hashble | None, default=None
            The name of the domain. If `None`, a default will be generated.

        Raises
        ------
        TypeError
            If `arr` is not a NumPy array or if `variable_names` is not a list of hashable items or `None`.
        ValueError
            If the length of `variable_names` does not match the number of dimensions of `arr`.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import Function
        >>> arr = np.array([[1, 2], [3, 4]])
        >>> f = Function.from_numpy(arr=arr, name="f")
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
                  f
        x_0 x_1
        0   0     1
            1     2
        1   0     3
            1     4
        >>> g = Function.from_numpy(arr=arr, name="g", domain_kind="SampleSpace")
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
                  g
        s_0 s_1
        0   0     1
            1     2
        1   0     3
            1     4
        """
        import numpy as np
        import pandas as pd

        from ..spaces.domain import Domain
        from ..spaces.sample_space import SampleSpace

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

        if name is None:
            name = cls._default_name
        domain_class = Domain if domain_kind == "Domain" else SampleSpace
        if domain_name is None:
            domain_name = domain_class._default_name

        if arr.ndim == 1:
            if variable_names is None:
                variable_names = [domain_class._variable_names_prefix]
            idx = pd.Index(range(arr.shape[0]), name=variable_names[0])
        else:
            if variable_names is None:
                variable_names = [
                    f"{domain_class._variable_names_prefix}_{i}"
                    for i in range(arr.ndim)
                ]
            idx = pd.MultiIndex.from_product(
                [range(dim) for dim in arr.shape], names=variable_names
            )
        data = pd.Series(arr.ravel(), index=idx, name=name)

        return cls._from_validated(
            data=data,
            name=name,
            kind=kind,
            domain_kind=domain_kind,
            domain_name=domain_name,
            index_kind=None,
            index_name=None,
        )

    @classmethod
    def from_rand(
        cls,
        domain_dims: tuple[int] | int,
        variable_names: list[Hashable] | None = None,
        distribution: Literal["uniform", "normal"] = "uniform",
        low: int = 0,
        high: int = 10,
        loc: float = 0.0,
        scale: float = 1.0,
        name: Hashable | None = None,
        domain_kind: Literal["Domain", "SampleSpace"] = "Domain",
        domain_name: Hashable | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> Function:
        """Generate a random function.

        Parameters
        ----------
        domain_dims : tuple[int] | int
            The dimensions of the domain of the function.
        variable_names : list[Hashable] | None, default=None
            The names of the variables. If `None`, defaults will be generated.
        variable_name_prefix : str | None, default=None
            The prefix for generating variable names. If `None`, either default names will be generated or `variable_names` must be provided.
        distribution : Literal["uniform", "normal"], default="uniform"
            The distribution to use for generating random values.
        min_value : int, default=0
            The minimum value for the uniform distribution.
        max_value : int, default=10
            The maximum value for the uniform distribution.
        loc : float, default=0.0
            The mean for the normal distribution.
        scale : float, default=1.0
            The standard deviation for the normal distribution.
        name : Hashable | None, default=None
            The name of the function. If `None`, a default name will be used.
        domain_class: Literal["Domain", "SampleSpace"], default="Domain
            The class of the underlying domain.
        domain_name: Hashble | None, default=None
            The name of the domain. If `None`, a default will be generated.
        random_state : int | np.random.Generator | None, default=None
            The random state for reproducibility.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import Function
        >>> rng = np.random.default_rng(42)

        Generate a random function with values drawn from a standard normal distribution.

        >>> f = Function.from_rand(
        ...     domain_dims=(2, 3),
        ...     distribution="normal",
        ...     random_state=rng,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
                        f
        x_0 x_1
        0   0    0.304717
            1   -1.039984
            2    0.750451
        1   0    0.940565
            1   -1.951035
            2   -1.302180

        Generate a random function with values drawn from a uniform distribution on the integers `[-10, 10)`.

        >>> g = Function.from_rand(
        ...     domain_dims=(2, 3),
        ...     distribution="uniform",
        ...     low=-10,
        ...     high=10,
        ...     name="g",
        ...     random_state=rng,
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
                   g
        x_0 x_1
        0   0      4
            1      5
            2      4
        1   0      5
            1      0
            2     -8
        """
        import numpy as np

        if isinstance(domain_dims, int):
            if domain_dims <= 0:
                raise ValueError("If domain_dims is an integer, it must be positive.")
            domain_dims = (domain_dims,)
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

        if variable_names is not None and not all(
            isinstance(name, Hashable) for name in variable_names
        ):
            raise TypeError("All elements of `variable_names` must be hashable.")
        if variable_names is not None and len(variable_names) != len(domain_dims):
            raise ValueError(
                "The length of `variable_names` must match the number of dimensions in `domain_dims`."
            )
        if distribution not in ("uniform", "normal"):
            raise ValueError(f"Unsupported distribution: {distribution}")
        if not isinstance(low, int):
            raise TypeError("`min_value` must be an integer.")
        if not isinstance(high, int):
            raise TypeError("`max_value` must be an integer.")
        if low > high:
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
            arr = rng.integers(low=low, high=high, size=domain_dims)

        return cls.from_numpy(
            arr=arr,
            variable_names=variable_names,
            kind="any",
            name=name,
            domain_kind=domain_kind,
            domain_name=domain_name,
        )

    @classmethod
    def tensor_product(
        cls,
        factors: list[Function],
        variable_names: list[Hashable] | None = None,
        name: Hashable | None = None,
        domain_name: Hashable | None = None,
    ) -> Function:
        r"""Compute the tensor product of a list of functions.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        factors : list[Function]
            The factors of the tensor product.
        variable_names : list[Hashable] | None, default=None
            The variable names of the resulting function. If `None`, the variable names will be inferred from the input functions.
        name : Hashable | None, default=None
            The name of the resulting function. If `None`, a default name will be generated from the names of the input functions.
        domain_name: Hashble | None, default=None
            The name of the domain. If `None`, a default will be generated.

        Raises
        ------
        TypeError
            If any element of `factors` is not a `Function`, or if `variable_names` is not a list or `None`, or if any element of `variable_names` is not hashable (if given), or if `name` is not hashable (if given).
        ValueError
            If the length of `variable_names` does not match the total number of arguments in `factors`.

        Returns
        -------
        tensor_prod : Function
            The tensor product of the input functions.

        Examples
        --------
        Define two functions.

        >>> from sigalg.core import Domain, Function
        >>> X = Domain.from_sequence(size=2, variable_name="x")
        >>> Y = Domain.from_sequence(size=2, variable_name="y", name="Y")
        >>> f = Function(
        ...     domain=X @ Y,
        ...     mapping=lambda *, x, y: x**2 + y + 2,
        ... )
        >>> g = Function(
        ...     domain=Y,
        ...     mapping=lambda *, y: y + 5,
        ...     name="g",
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
             f
        x y
        0 0  2
          1  3
        1 0  3
          1  4
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
           g
        y
        0  5
        1  6

        Compute their tensor product using the `tensor_product` method.

        >>> prod = Function.tensor_product([f, g])
        >>> print(prod)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f x g':
                    f x g
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
                    f x g
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
        import pandas as pd

        from .._utils.utils import subscript_var_names
        from ..measures.measure import Measure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not all(isinstance(function, Function) for function in factors):
            raise TypeError("All elements of `factors` must be instances of Function.")
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
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("`name` must be hashable or None.")

        prod_arg_names = subscript_var_names(
            [function.variable_names for function in factors],
            grouped=True,
        )

        function_data = []

        for k, (arg_names, function) in enumerate(zip(prod_arg_names, factors)):
            new_func_data = function.data.rename(f"{function.name}_{k}")
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

        if variable_names is None:
            variable_names = mapping.index.names
        else:
            mapping.index.names = variable_names
        if name is None:
            name = " x ".join([function.name for function in factors])
        if domain_name is None:
            domain_name = (" x ".join([function.domain.name for function in factors]),)

        mapping = mapping.rename(name)

        if cls.__name__ == "Measure":
            all_probs = len(
                [
                    function.kind
                    for function in factors
                    if function.kind == "probability"
                ]
            ) == len(factors)

            sig_alg = SigmaAlgebra.cartesian_product(
                [function.sig_alg for function in factors]
            )

            return Measure._from_validated(
                measure_data=mapping,
                measure_kind="probability" if all_probs else "measure",
                measure_name=name,
                sig_alg=sig_alg,
            )

        else:
            return cls._from_validated(
                data=mapping,
                kind="any",
                name=name,
                domain_kind="Domain",
                domain_name=domain_name,
                index_kind=None,
                index_name=None,
            )

    @classmethod
    def tensor_power(cls, function: Function, n: int) -> Function:
        r"""Get the tensor power of the function.

        Parameters
        ----------
        function : Function
            The base of the tensor power.
        n : int
            The power of the tensor power.

        Raises
        ------
        TypeError
            If `n` is not an integer or `function` is not a `Function`.
        ValueError
            If `n` is not positive.

        Returns
        -------
        tensor_power : Function
            The tensor power.

        Examples
        --------
        Define a function.

        >>> from sigalg.core import Domain, Function
        >>> X = Domain([1, 2, 3], variable_names=["x"])
        >>> f = Function(
        ...     domain=X,
        ...     mapping=lambda *, x: x**2,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
           f
        x
        1  1
        2  4
        3  9

        Compute the second tensor power using the `tensor_power` class method.

        >>> print(Function.tensor_power(f, 2))  # doctest: +NORMALIZE_WHITESPACE
        Function 'f ^ 2':
                  f ^ 2
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
        >>> g = Function(
        ...     domain=E,
        ...     mapping=lambda *, x, y: x + y,
        ...     name="g",
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
              g
        x y
        1 2   3
        3 4   7

        Compute the third tensor power using the `^` operator notation.

        >>> print(g ^ 3)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g ^ 3':
                                  g ^ 3
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

    def __matmul__(self, other: Function) -> Function:
        """Form the tensor product of this instance of `Function` with another.

        Internally calls the `tensor_product` method.

        Parameters
        ----------
        other : Function
            The other function to form the tensor product with.

        Returns
        -------
        tensor_product : Function
            The tensor product.
        """
        return type(self).tensor_product(factors=[self, other])

    def __xor__(self, n: int) -> Function:
        """Form the tensor power of this instance of `Function`.

        Internally calls the `tensor_power` method.

        Parameters
        ----------
        n : int
            The power of the tensor power.

        Returns
        -------
        tensor_power : Function
            The tensor power.
        """
        return type(self).tensor_power(function=self, n=n)

    # --------------------- properties --------------------- #

    @property
    def variable_names(self) -> list[Hashable] | None:
        """Get the variable names of the function.

        Returns
        -------
        variable_names : list[Hashable] | None
            The variable names of the function if defined, otherwise `None`.

        Examples
        --------
        >>> from sigalg.core import Domain, Function
        >>> X = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> f = Function(domain=X, mapping=lambda *, x, y: 2 * x + y**2)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
              f
        x y
        1 2   6
        2 3  13
        1 4  18
        >>> print(f.variable_names)
        ['x', 'y']
        >>> g = Function(mapping=lambda *, x: x**2, name="g")
        >>> print(g.variable_names)
        ['x']
        """
        import pandas as pd

        PandasLike = pd.Series | pd.DataFrame

        if isinstance(self.data, PandasLike):
            return list(self.data.index.names)
        elif isinstance(self.data, Callable):
            sig = inspect.signature(self.data)
            return list(sig.parameters.keys())
        else:
            return None

    @property
    def num_variables(self) -> int | None:
        """Get the number of variables of the function.

        Returns
        -------
        num_variables : int | None
            The number of variables of the function if defined, otherwise `None`.

        Examples
        --------
        >>> from sigalg.core import Domain, Function
        >>> X = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> f = Function(domain=X, mapping=lambda *, x, y: 2 * x + y**2)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
              f
        x y
        1 2   6
        2 3  13
        1 4  18
        >>> print(f.num_variables)
        2
        """
        return len(self.variable_names)

    @property
    def signature(self) -> inspect.Signature | None:
        """Get the signature of the function.

        Returns
        -------
        signature : inspect.Signature
            The signature of the function.

        Examples
        --------
        >>> from sigalg.core import Domain, Function
        >>> X = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> f = Function(domain=X, mapping=lambda *, x, y: 2 * x + y**2)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
                f
        x y
        1 2   6
        2 3  13
        1 4  18
        >>> print(f.signature)
        (*, x, y)
        """
        import pandas as pd

        PandasLike = pd.Series | pd.DataFrame

        if isinstance(self.data, PandasLike):
            parameters = [
                inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
                for name in self.variable_names
            ]
            return inspect.Signature(parameters)

        elif isinstance(self.data, Callable):
            return inspect.signature(self.data)

        else:
            return None

    @property
    def domain(self) -> Domain | None:
        """Get the domain of the function.

        Returns
        -------
        domain : Domain | None
            The domain of the function if defined, otherwise `None`.

        Examples
        --------
        >>> from sigalg.core import Domain, Function
        >>> X = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> f = Function(domain=X, mapping=lambda *, x, y: 2 * x + y**2)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
              f
        x y
        1 2   6
        2 3  13
        1 4  18
        >>> print(f.domain)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'X':
         x  y
         1  2
         2  3
         1  4
        """
        import pandas as pd

        from ..spaces.domain import Domain
        from ..spaces.sample_space import SampleSpace

        PandasLike = pd.Series | pd.DataFrame

        if isinstance(self.data, PandasLike):
            domain_class = Domain if self.domain_kind == "Domain" else SampleSpace
            return domain_class._from_validated(
                data=self.data.index, name=self.domain_name
            )
        else:
            return None

    @property
    def index(self) -> Index | None:
        """Get the index of the function.

        Returns
        -------
        domain : Domain | None
            The domain of the function if defined, otherwise `None`.

        Examples
        --------
        >>> from sigalg.core import Domain, Function, Index
        >>> J = Index([1, 2], variable_names=["j"], name="J")
        >>> X = Domain.from_sequence(size=2)
        >>> f = Function(
        ...     domain=X,
        ...     mapping=lambda x: (x, x**2),
        ...     index=J,
        ...     multi_dim_outputs=True,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
        j  1  2
        x
        0  0  0
        1  1  1
        >>> print(f.index)  # doctest: +NORMALIZE_WHITESPACE
        Index 'J':
         j
         1
         2
        """
        import pandas as pd

        from ..indices.index import Index
        from ..indices.time import Time

        if isinstance(self.data, pd.DataFrame):
            index_class = Index if self.index_kind == "Index" else Time
            index = index_class._from_validated(
                data=self.data.columns, name=self.index_name
            )
            return index
        else:
            return None

    @property
    def dimension(self) -> int | None:
        """Get the dimension of the outputs of the function.

        Returns
        -------
        dim : int | None
            The dimension of the outputs of the function, or `None` if the underlying data of the function is not a `pd.Series` or `pd.DataFrame`.

        Examples
        --------
        >>> from sigalg.core import Domain, Function, Index
        >>> J = Index([1, 2], variable_names=["j"], name="J")
        >>> X = Domain.from_sequence(size=2)
        >>> f = Function(
        ...     domain=X,
        ...     mapping=lambda x: (x, x**2),
        ...     index=J,
        ...     multi_dim_outputs=True,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
        j  1  2
        x
        0  0  0
        1  1  1
        >>> f.dimension
        2
        >>> g = Function(domain=X, mapping=lambda x: x**2)
        >>> g.dimension
        1
        """
        import pandas as pd

        if isinstance(self.data, pd.DataFrame):
            return self.data.shape[1]
        elif isinstance(self.data, pd.Series):
            return 1
        else:
            return None

    @property
    def component_names(self) -> dict[Hashable, Hashable] | None:
        """Get the names of the component functions of the function.

        Returns
        -------
        component_names : list[Hashable] | None
            A list of the names of the component functions of the function if it has multi-dimensional outputs, or `None`.

        Examples
        --------
        >>> from sigalg.core import Domain, Function, Index
        >>> J = Index([1, 2], variable_names=["j"], name="J")
        >>> X = Domain.from_sequence(size=2)
        >>> f = Function(
        ...     domain=X,
        ...     mapping=lambda x: (x, x**2),
        ...     index=J,
        ...     multi_dim_outputs=True,
        ... )
        >>> f.component_names
        {1: 'f_1', 2: 'f_2'}
        """
        import pandas as pd

        if isinstance(self.data, pd.DataFrame):
            if not hasattr(self, "_component_names"):
                return {
                    idx: f"{self.name}_{idx}".replace(".", "_") for idx in self.index
                }
            else:
                return self._component_names
        else:
            return None

    @component_names.setter
    def component_names(self, names: dict[Hashable, Hashable]) -> None:
        """Pass."""
        self._component_names = names

    @cached_property
    def components(self) -> list[Function] | None:
        r"""Get the component measurable functions of the measurable vector.

        See the Notes section below for the mathematical details.

        Raises
        ------
        ValueError
            If `self` has an empty `data` attribute.

        Returns
        -------
        components : list[MeasurableFunction] | None
            A list of the component measurable functions of the measurable vector.

        Examples
        --------
        Extract the component functions of a 2-dimensional measurable vector.

        >>> from sigalg.core import Domain, MeasurableVector
        >>> X = Domain.from_sequence(size=3)
        >>> f = MeasurableVector.from_rand(
        ...     domain=X,
        ...     low=0,
        ...     high=3,
        ...     dim=2,
        ...     random_state=42,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        i   0  1
        x
        0   0  2
        1   1  1
        2   1  2
        >>> for component in f.components:
        ...     print(component)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'f_0':
                f_0
        x
        0         0
        1         1
        2         1
        Measurable function 'f_1':
                f_1
        x
        0         2
        1         1
        2         2
        >>> g = MeasurableVector.from_rand(
        ...     domain=X,
        ...     low=0,
        ...     high=3,
        ...     dim=1,
        ...     random_state=42,
        ...     name="g",
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'g':
                g
        x
        0       0
        1       2
        2       1
        >>> for component in g.components:
        ...     print(component)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'g':
                g
        x
        0       0
        1       2
        2       1

        Notes
        -----
        If $f: X \to \mathbb{R}^d$ is a measurable vector, then for each $x \in X$ we may write

        $$
        f(x) = (f_1(x),f_2(x),\ldots, f_d(x))
        $$

        where $f_j: X \to \mathbb{R}$ is the *$j$-th component measurable function* of $f$.
        """
        import pandas as pd

        if isinstance(self.data, pd.DataFrame):
            if self.dimension == 1:
                return [self]
            else:
                return [self.get_component(idx) for idx in self.index]
        elif isinstance(self.data, pd.Series):
            return [self]
        else:
            return None

    @cached_property
    def range(self) -> Domain | None:
        """Return the range of the function if it is defined on an explicit domain.

        Returns
        -------
        range : Domain | None
            The range of the function as an instance of `Domain`, or `None` if the underlying data is not a `pd.Series` or `pd.DataFrame`.

        Examples
        --------
        >>> from sigalg.core import Domain, Index, Function
        >>> J = Index([1, 2], variable_names=["j"], name="J")
        >>> X = Domain.from_sequence(size=3)
        >>> f = Function(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (2, 3),
        ...     },
        ...     index=J,
        ...     multi_dim_outputs=True,
        ... )
        >>> print(f.range)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'f_range':
         f_1  f_2
           1    2
           2    3
        >>> Y = Domain([-2, -1, 0, 1, 2], variable_names=["y"], name="Y")
        >>> g = Function(domain=Y, mapping=lambda y: y**2, name="g")
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
            g
        y
        -2  4
        -1  1
         0  0
         1  1
         2  4
        >>> print(g.range)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'g_range':
         g
         4
         1
         0
        """
        import pandas as pd

        from ..spaces.domain import Domain

        if hasattr(self, "is_identity") and self.is_identity:
            return self.domain

        name = f"{self.name}_range"

        if isinstance(self.data, pd.Series):
            range_list = list(self.data.drop_duplicates())
            data = pd.Index(range_list, name=self.name)
            return Domain._from_validated(data=data, name=name)

        elif isinstance(self.data, pd.DataFrame):
            range_list = list(self.data.drop_duplicates().apply(tuple, axis=1))
            data = pd.MultiIndex.from_tuples(
                range_list, names=self.component_names.values()
            )
            return Domain._from_validated(data=data, name=name)

        else:
            return None

    @cached_property
    def generated_sig_alg(self) -> SigmaAlgebra | None:
        r"""Get the sigma-algebra generated by the function.

        See the Notes section below for the mathematical details.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra induced by the measurable vector.

        Examples
        --------
        Extract the generated sigma-algebra from a 2-dimensional measurable vector. Note that the atom identifiers are exactly the values of the vector.

        >>> from sigalg.core import (
        ...     Domain,
        ...     MeasurableVector,
        ...     SigmaAlgebra,
        ... )
        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...     },
        ... )
        >>> f = MeasurableVector(
        ...     domain=X,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...     },
        ... )
        >>> sig_f = f.generated_sig_alg
        >>> print(sig_f)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'sigma(f)':
        i      0    1
        x
        0      1    2
        1      3    4
        2      3    4
        3      3    4
        >>> print(sig_f <= F)
        True

        Notes
        -----
        A measurable vector $f: X \to \mathbb{R}^d$ on a measure space $(X, \mathcal{F},\mu)$ generates a $\sigma$-algebra denoted $\sigma(f)$. On a finite domain $X$, this $\sigma$-algebra is determined by its atoms, which are the nonempty preimages

        $$
        \{ x \in X : f(x) = y\},
        $$

        for $y\in \mathbb{R}^d$. The atom identifiers may thus be taken as the vectors $y\in \mathbb{R}^d$ in the range of $f$.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        return SigmaAlgebra.from_function(self)

    @cached_property
    def lattice(self) -> Lattice:
        r"""Get the (upward) lattice of sigma-algebras containing this function.

        See the Notes section below for the mathematical details.

        Examples
        --------
        >>> from sigalg.core import Domain, Function, SigmaAlgebra

        Define three sigma-algebras on a domain.

        >>> X = Domain.from_sequence(size=4, variable_name="x")
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ... )
        >>> G = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 0,
        ...     },
        ...     name="G",
        ...     variable_names=["v"],
        ... )
        >>> H = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 0,
        ...         3: 0,
        ...     },
        ...     name="H",
        ...     variable_names=["w"],
        ... )

        Define a function with 2-dimensional outputs.

        >>> f = Function(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...         3: (1, 2),
        ...     },
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
        i  0  1
        x
        0  1  2
        1  3  4
        2  3  4
        3  1  2

        We may test whether the function is measurable with respect to a sigma-algebra by using the `in` operator with the `lattice` attribute.

        >>> F in f.lattice
        True

        We may get the unique values of the function on the atoms of the sigma-algebra by calling the `get_atom_data` method.

        >>> print(f.lattice.get_atom_data(F))  # doctest: +NORMALIZE_WHITESPACE
        i    0    1
        u
        0    1    2
        1    3    4
        2    1    2

        Whenever a measurability check is executed, and the result is `True`, the sigma-algebra is added to the internal `lattice`.

        >>> f.lattice
        Lattice(base=sigma(f), type=upward, num_sig_algs=1)

        Perform another measurability check, inspect the `lattice` attribute to see the updated list of contents, and print the atom data.

        >>> G in f.lattice
        True
        >>> f.lattice
        Lattice(base=sigma(f), type=upward, num_sig_algs=2)
        >>> print(f.lattice.get_atom_data(G))  # doctest: +NORMALIZE_WHITESPACE
        i    0    1
        v
        0    1    2
        1    3    4

        Notice that the function is not measurable with respect to the third sigma-algebra. The measurability check accordingly returns `False`, and the contents of `lattice` is not changed.

        >>> H in f.lattice
        False
        >>> f.lattice
        Lattice(base=sigma(f), type=upward, num_sig_algs=2)

        Notes
        -----
        Let $f:X \to \mathbb{R}^d$ be a function on a set $X$. We shall say that a $\sigma$-algebra $\mathcal{F}$ on $X$ *contains* $f$ provided that $\sigma(f) \subset \mathcal{F}$, where $\sigma(f)$ is the $\sigma$-algebra generated by $f$. In other words, $\mathcal{F}$ contains $f$ if and only if $f$ is $\mathcal{F}$-measurable. There is thus an entire (upward) lattice of $\sigma$-algebras on $X$ that contain $f$.
        """
        from ..sigma_algebras.lattice import Lattice

        return Lattice(base=self.generated_sig_alg, type="upward")

    # --------------------- function methods --------------------- #

    def __call__(self, *args, **kwargs) -> Real | Function:
        """Call the function with the provided arguments.

        The `__call__` method is very flexible. See the Examples section below.

        Parameters
        ----------
        *args : positional arguments
            Positional arguments for the function.
        **kwargs : keyword arguments
            Keyword arguments for the function.

        Returns
        -------
        result : Real | Function
            The result of evaluating the function with the provided arguments.

        Examples
        --------
        >>> from sigalg.core import Domain, Function, MeasurableVector, SigmaAlgebra

        Define a function on a 2-dimensional domain with 2-dimensional outputs.

        >>> X = Domain([(1, 2), (2, 3), (1, 4), (4, 5)], variable_names=["x_0", "x_1"])
        >>> f = Function(domain=X, mapping=lambda *, x_0, x_1: (2 * x_0, x_1**2))
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
        i        0   1
        x_0 x_1
        1   2    2   4
        2   3    4   9
        1   4    2  16
        4   5    8  25

        Call the function on a complete set of arguments.

        >>> f(x_0=1, x_1=2)
        (2, 4)

        Call the function on a partial set of arguments to obtain another function.

        >>> print(f(x_0=1))  # doctest: +NORMALIZE_WHITESPACE
        Function 'f(x_0=1)':
        i    0   1
        x_1
        2    2   4
        4    2  16

        The partial function is an instance of `Function` as well, so it too may be called.

        >>> f(x_0=1)(x_1=2)
        (2, 4)

        Define another function with 3-dimensional outputs.

        >>> Y = Domain(
        ...     [(3, 5), (2, 16), (2, 4), (8, 25), (4, 9)],
        ...     variable_names=["y_0", "y_1"],
        ...     name="Y",
        ... )
        >>> g = Function(
        ...     domain=Y,
        ...     mapping=lambda *, y_0, y_1: (y_0, +2 * y_0 * y_1, y_1**2),
        ...     name="g",
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
        i        0    1    2
        y_0 y_1
        3   5    3   30   25
        2   16   2   64  256
            4    2   16   16
        8   25   8  400  625
        4   9    4   72   81

        Notice that the domain `Y` of `g` contains the range of `f`. This means we can compose the functions, by calling `g` on `f`.

        >>> print(g(f))  # doctest: +NORMALIZE_WHITESPACE
        Function 'g(f)':
                0    1    2
        x_0 x_1
        1   2    2   16   16
        2   3    4   72   81
        1   4    2   64  256
        4   5    8  400  625

        Finally, if an instance of `Function` is actually an instance of `MeasurableVector`, it may be called on atoms of the underlying sigma-algebra (since it is constant on each atom). Define such a function.

        >>> F = SigmaAlgebra(domain=X, mapping=dict(zip(X, [0, 0, 1, 2])))
        >>> print(F)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                 F
        x_0 x_1
        1   2    0
        2   3    0
        1   4    1
        4   5    2
        >>> V = MeasurableVector(
        ...     X, F, mapping=dict(zip(X, [(2, 1), (2, 1), (4, 0), (3, 2)])), name="V"
        ... )
        >>> print(V)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'V':
        i        0  1
        x_0 x_1
        1   2    2  1
        2   3    2  1
        1   4    4  0
        4   5    3  2

        Extract the three atoms of the sigma-algebra and call the vector on them.

        >>> A_0, A_1, A_2 = F
        >>> V(A_0)
        (2, 1)
        >>> V(A_1)
        (4, 0)
        >>> V(A_2)
        (3, 2)
        """
        from numbers import Real

        import pandas as pd

        from .._utils.function_helpers import compose_funcs
        from ..measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.lattice import Lattice
        from ..spaces.set import Set
        from .measurable_function import MeasurableFunction
        from .measurable_vector import MeasurableVector
        from .random_variable import RandomVariable
        from .random_vector import RandomVector

        PandasLike = pd.Series | pd.DataFrame

        if len(args) == 1 and len(kwargs) == 0:
            if isinstance(args[0], Function):
                inner_func = args[0]
                data = compose_funcs(inner_data=inner_func.data, outer_data=self.data)
                name = f"{self.name}({inner_func.name})"

                if len(data) != len(inner_func.domain):
                    raise ValueError(
                        "The outer function is not defined on the entire domain of the inner function."
                    )

                result = Function._from_validated(
                    data=data if not data.empty else None,
                    kind="any",
                    name=name,
                    domain_kind=type(inner_func.domain).__name__,
                    domain_name=inner_func.domain.name,
                    index_kind=type(inner_func.index).__name__
                    if inner_func.index
                    else None,
                    index_name=inner_func.index.name if inner_func.index else None,
                )

                if hasattr(inner_func, "measure") and inner_func.measure:
                    result.measure = inner_func.measure
                    result.sig_alg = inner_func.measure.sig_alg
                    if isinstance(result.measure, ProbabilityMeasure):
                        if result.dimension == 1:
                            result.__class__ = RandomVariable
                        else:
                            result.__class__ = RandomVector
                    else:
                        if result.dimension == 1:
                            result.__class__ = MeasurableFunction
                        else:
                            result.__class__ = MeasurableVector

                elif hasattr(inner_func, "sig_alg") and inner_func.sig_alg:
                    result.sig_alg = inner_func.sig_alg
                    if result.dimension == 1:
                        result.__class__ = MeasurableFunction
                    else:
                        result.__class__ = MeasurableVector

                return result

            elif isinstance(args[0], Set):
                subset = args[0]

                if not self.is_constant_on(subset):
                    raise ValueError(
                        "Cannot call an instance of Function on a set on which it is not constant."
                    )

                join = Lattice.join([subset.generated_sig_alg, self.generated_sig_alg])
                atom_ID = subset.atom_id(sig_alg=join)
                self.lattice.add(join)
                self_atom_data = self.lattice.get_atom_data(join)

                if self.dimension > 1:
                    return tuple(self_atom_data.loc[atom_ID])
                else:
                    return self_atom_data.loc[atom_ID].astype(Real)

            elif len(self.variable_names) == 1:
                if isinstance(self.data, pd.DataFrame):
                    return tuple(self.data.loc[args[0]])
                elif isinstance(self.data, pd.Series):
                    return self.data.loc[args[0]].astype(Real)
                else:
                    return self.data(**{self.variable_names[0]: args[0]})

            else:
                raise ValueError(
                    "If a single positional argument is passed, the function must be defined on a 1-dimensional domain."
                )

        elif len(args) == 0 and len(kwargs) > 0:
            if isinstance(self.data, PandasLike):
                return self._call_from_pandas(**kwargs)
            else:
                return self._call_from_callable(**kwargs)

        else:
            raise ValueError()

    def _call_from_callable(self, **kwargs):
        specified_arguments = self.signature.bind_partial(**kwargs)
        unspecified_arguments = [
            inspect.Parameter(parameter, inspect.Parameter.KEYWORD_ONLY)
            for parameter in self.variable_names
            if parameter not in specified_arguments.arguments.keys()
        ]

        if len(unspecified_arguments) == 0:
            return self.data(**specified_arguments.arguments)
        else:
            partial_signature = inspect.Signature(unspecified_arguments)

            def data(**kwargs):
                partial_parameters = partial_signature.bind(**kwargs)
                all_args = {
                    **specified_arguments.arguments,
                    **partial_parameters.arguments,
                }
                return self.data(**all_args)

            data.__signature__ = partial_signature

            parameter_string = (
                f"{', '.join(f'{name}={value}' for name, value in kwargs.items())}"
            )
            name = f"{self.name}({parameter_string})"

            return Function._from_validated(
                data=data,
                kind="any",
                name=name,
                domain_kind=None,
                domain_name=None,
                index_kind=None,
                index_name=None,
            )

    def _call_from_pandas(self, **kwargs) -> Real | Function:
        from numbers import Real

        import pandas as pd

        try:
            if isinstance(self.data.index, pd.MultiIndex):
                data = self.data.xs(
                    key=tuple(kwargs.values()), level=tuple(kwargs.keys())
                )
            else:
                data = self.data.loc[kwargs.values()]
        except Exception as e:
            raise ValueError("There is an error in evaluating the function.") from e

        kwargs = {name: kwargs[name] for name in self.variable_names if name in kwargs}

        if len(data) == 1:
            if isinstance(data, pd.DataFrame):
                return tuple(data.iloc[0])
            else:
                return data.iloc[0].astype(Real)

        parameter_string = (
            f"{', '.join(f'{name}={value}' for name, value in kwargs.items())}"
        )
        name = f"{self.name}({parameter_string})"
        domain_name = f"{self.domain.name}|{{{parameter_string}}}"

        if isinstance(data, pd.Series):
            data = data.rename(name)

        return Function._from_validated(
            data=data,
            kind="any",
            name=name,
            domain_kind=self.domain_kind,
            domain_name=domain_name,
            index_kind=None,
            index_name=None,
        )

    def get_inverse_image(
        self, value: Hashable | tuple[Hashable] | pd.Series
    ) -> list[Hashable] | Set:
        """Get the inverse image of a value under the measurable vector.

        Parameters
        ----------
        value : Hashable | tuple[Hashable] | pd.Series
            The value to find the inverse image of. If the measurable vector is 1-dimensional, `value` should be a Hashable. If the measurable vector is multi-dimensional, `value` should be a tuple of hashables or a `pd.Series` with an index matching the variable names of the measurable vector.

        Raises
        ------
        ValueError
            If `value` is not in the range of the measurable vector.

        Returns
        -------
        event : MeasurableSet
            The event in the sigma-algebra corresponding to the inverse image of `value` under the measurable vector.

        Examples
        --------
        Generate a 2-dimensional measurable vector.

        >>> import numpy as np
        >>> import pandas as pd
        >>> from sigalg.core import (
        ...     Domain,
        ...     Measure,
        ...     MeasurableVector,
        ...     SigmaAlgebra,
        ... )
        >>> rng = np.random.default_rng(101)
        >>> X = Domain.from_sequence(size=10)
        >>> F = SigmaAlgebra.from_rand(
        ...     domain=X,
        ...     num_atoms=3,
        ...     random_state=rng,
        ... )
        >>> f = MeasurableVector.from_rand(
        ...     domain=X, sig_alg=F, high=2, dim=2, random_state=rng
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        i      0  1
        x
        0      1  1
        1      0  1
        2      0  1
        3      0  1
        4      0  0
        5      1  1
        6      0  0
        7      0  1
        8      0  1
        9      0  1

        Get an inverse image using the `get_inverse_image` method.

        >>> inv_1 = f.get_inverse_image((1, 1))
        >>> print(inv_1)  # doctest: +NORMALIZE_WHITESPACE
        Set '{f = (1, 1)}':
            x
            0
            5

        Get an inverse image using the overloaded operator `==`.

        >>> inv_2 = f == (0, 1)
        >>> print(inv_2)  # doctest: +NORMALIZE_WHITESPACE
        Set '{f = (0, 1)}':
            x
            1
            2
            3
            7
            8
            9

        Get an inverse image using the overloaded operator `==` and a `pd.Series`.

        >>> s = pd.Series([0, 0], index=f.index)
        >>> inv_3 = f == s
        >>> print(inv_3)  # doctest: +NORMALIZE_WHITESPACE
        Set '{f = (0, 0)}':
            x
            4
            6
        """
        import pandas as pd

        if not isinstance(value, (Hashable, tuple, pd.Series)):
            raise TypeError(
                "value must be a Hashable, tuple, or pd.Series corresponding to the output of the measurable vector."
            )

        if self.data is None:
            raise ValueError(
                "Cannot get inverse image of a measurable vector without outputs."
            )

        if isinstance(value, pd.Series):
            if not isinstance(self.data, pd.DataFrame):
                raise ValueError(
                    "The measurable vector is 1-dimensional, but the provided value is a pd.Series."
                )
            if not value.index.equals(self.index.data):
                raise ValueError(
                    "The index of the provided value does not match the index of the measurable vector."
                )
            value = tuple(value)
        if isinstance(value, tuple) and len(value) != self.dimension:
            raise ValueError(
                "The dimension of the provided value does not match the dimension of the measurable vector."
            )

        mask = (
            (self.data == value).all(axis=1)
            if isinstance(value, tuple)
            else self.data == value
        )

        inv_image = list(self.data.index[mask])

        if hasattr(self, "sig_alg"):
            name = f"{{{self.name} = {value}}}"
            return self.sig_alg.get_set(inv_image, name=name)
        else:
            return inv_image

    def is_measurable(self, sig_alg: SigmaAlgebra | None = None) -> bool:
        r"""Check if the function is measurable with respect to a given sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The sigma-algebra to check measurability against.

        Returns
        -------
        is_measurable : bool
            `True` if the measurable vector is measurable with respect to the given sigma-algebra, `False` otherwise.

        Examples
        --------
        >>> from sigalg.core import Domain, Function, SigmaAlgebra

        Define two functions with 2-dimensional outputs and a sigma-algebra. The first is constant on the atoms of the sigma-algebra and hence measurable, while the second is not.

        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ... )
        >>> f = Function(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...     },
        ... )
        >>> g = Function(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (5, 6),
        ...         3: (7, 8),
        ...     },
        ...     name="g",
        ... )
        >>> print(f.is_measurable(F))
        True
        >>> print(g.is_measurable(F))
        False

        Notes
        -----
        Let $f: X \to \mathbb{R}^d$ be a function on a set $X$. In the case that $X$ is finite (as in SigAlg), the function $f$ is *measurable* with respect to a $\sigma$-algebra $\mathcal{F}$ on $X$ if $f$ is constant on the atoms of $\mathcal{F}$. When the identity of the $\sigma$-algebra needs to made explict, we shall say that $f$ is *$\mathcal{F}$-measurable*.
        """
        import pandas as pd

        PandasLike = pd.Series | pd.DataFrame

        if isinstance(self.data, PandasLike):
            return sig_alg in self.lattice
        else:
            return None

    def is_constant_on(self, subset: Set) -> bool | None:
        """Determine whether the function is constant on a given subtset of its domain.

        Parameters
        ----------
        subset : Set
            The subset.

        Returns
        -------
        is_constant : bool | None
            Either `True` if the function is constant on the set, `False` if not, or `None` if the data of the function is not a `pd.Series` or `pd.DataFrame`.

        Examples
        --------
        >>> from sigalg.core import Domain, Function, Set
        >>> X = Domain.from_sequence(size=4, variable_name="x")
        >>> f = Function(domain=X, mapping=dict(zip(X, [0, 1, 1, 2])))
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
           f
        x
        0  0
        1  1
        2  1
        3  2
        >>> U = Set([1, 2], domain=X, name="U")
        >>> f.is_constant_on(U)
        True
        >>> V = Set([1, 2, 3], domain=X, name="V")
        >>> f.is_constant_on(V)
        False
        """
        import pandas as pd

        from ..sigma_algebras.lattice import Lattice

        if isinstance(self.data, pd.Series | pd.DataFrame):
            return subset.is_atom(
                Lattice.join([subset.generated_sig_alg, self.generated_sig_alg])
            )

        else:
            return None

    # --------------------- data access methods --------------------- #

    def get_sub_vector(self, indices: list[Hashable]) -> Function:
        r"""Get a sub-vector of the measurable vector by selecting a collection of component functions.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        indices : list[Hashable]
            List of indices to select for the sub-vector.

        Returns
        -------
        sub_vector : MeasurableVector
            A new `MeasurableVector` containing only the specified component functions.

        Raises
        ------
        ValueError
            If any index is not found or if the measurable vector is 1-dimensional.

        Examples
        --------
        Define a 3-dimensional measurable vector.

        >>> from sigalg.core import Domain, MeasurableVector
        >>> X = Domain.from_sequence(size=2)
        >>> f = MeasurableVector(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 2, 3),
        ...         1: (4, 5, 6),
        ...     },
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        i  0  1  2
        x
        0  1  2  3
        1  4  5  6

        Get a sub-vector by using the `get_sub_vector` method.

        >>> f_sub = f.get_sub_vector([1, 2])
        >>> print(f_sub)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector '(f_1, f_2)':
        i  1  2
        x
        0  2  3
        1  5  6

        Get a sub-vector by using subscript notation.

        >>> f_sub = f[0, 1]
        >>> print(f_sub)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector '(f_0, f_1)':
        i  0  1
        x
        0  1  2
        1  4  5

        Notes
        -----
        Given a measurable vector $f: X \to \mathbb{R}^d$ on a measure space $(X, \mathcal{F}, \mu)$, for each $x\in X$ we may write

        $$
        f(x) = (f_1(x), f_2(x), \ldots, f_d(x)),
        $$

        where $f_j: X \to \mathbb{R}$ are the component functions of $f$. We may create a *sub-vector* by choosing a collection of the component functions to get a measurable vector of smaller dimension. For example, we may select the first and last components to create the $2$-dimensional measurable vector

        $$
        x \mapsto (f_1 (x), f_d(x)).
        $$
        """
        from ..measures.probability_measure import ProbabilityMeasure
        from .measurable_function import MeasurableFunction
        from .measurable_vector import MeasurableVector
        from .random_variable import RandomVariable
        from .random_vector import RandomVector

        if self.dimension == 1:
            raise ValueError(
                "Cannot get sub-vector of a function with 1-dimensional outputs."
            )
        invalid_features = [
            invalid_feature
            for invalid_feature in indices
            if invalid_feature not in self.index
        ]
        if invalid_features:
            raise ValueError(
                f"Indices {invalid_features} not found when forming the sub-vector"
            )

        sub_data = self.data[indices]

        if len(indices) == 1:
            name = self.component_names[indices[0]]

            result = Function._from_validated(
                data=sub_data.squeeze(axis=1).rename(name),
                kind="any",
                name=name,
                domain_kind=type(self.domain).__name__,
                domain_name=self.domain.name,
                index_kind=None,
                index_name=None,
            )

            if hasattr(self, "measure") and self.measure:
                result.measure = self.measure
                result.sig_alg = self.measure.sig_alg
                if isinstance(self.measure, ProbabilityMeasure):
                    result.__class__ = RandomVariable
                else:
                    result.__class__ = MeasurableFunction
            elif hasattr(self, "sig_alg"):
                result.sig_alg = self.sig_alg
                result.measure = None
                result.__class__ = MeasurableFunction

        else:
            name = (
                "("
                + ", ".join([f"{self.name}_{idx}".replace(".", "_") for idx in indices])
                + ")"
            )

            result = Function._from_validated(
                data=sub_data,
                kind="any",
                name=name,
                domain_kind=type(self.domain).__name__,
                domain_name=self.domain.name,
                index_kind=type(self.index).__name__,
                index_name=self.index.name,
            )

            if hasattr(self, "measure") and self.measure:
                result.measure = self.measure
                result.sig_alg = self.measure.sig_alg
                if isinstance(self.measure, ProbabilityMeasure):
                    result.__class__ = RandomVector
                else:
                    result.__class__ = MeasurableVector
            elif hasattr(self, "sig_alg"):
                result.sig_alg = self.sig_alg
                result.measure = None
                result.__class__ = MeasurableVector

            # result.component_names = [f"{self.name}_{idx}" for idx in indices]
            result.component_names = {
                idx: f"{self.name}_{idx}".replace(".", "_") for idx in indices
            }

        return result

    def get_component(self, index: Hashable) -> Function:
        r"""Get a component function of the measurable vector.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        index : Hashable
            The index for which to get the component function.

        Returns
        -------
        component : MeasurableFunction
            The desired component function.

        Examples
        --------
        Define a 3-dimensional measurable vector.

        >>> from sigalg.core import Domain, MeasurableVector
        >>> X = Domain.from_sequence(size=2)
        >>> f = MeasurableVector(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 2, 3),
        ...         1: (4, 5, 6),
        ...     },
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        i   0  1  2
        x
        0   1  2  3
        1   4  5  6

        Get a component function using the `get_component` method.

        >>> f_1 = f.get_component(1)
        >>> print(f_1)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'f_1':
           f_1
        x
        0    2
        1    5

        Get a component function using subscript notation.

        >>> f_0 = f[0]
        >>> print(f_0)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'f_0':
           f_0
        x
        0    1
        1    4

        Notes
        -----
        Given a measurable vector $f: X \to \mathbb{R}^d$ on a measurable space $(X, \mathcal{F})$, for each $x \in X$ we may write

        $$
        f(x) = (f_1(x), f_2(x), \ldots, f_d(x)),
        $$

        where $f_j: X \to \mathbb{R}$ are the component functions of $f$.
        """
        return self.get_sub_vector([index])

    def __getitem__(self, *args) -> Function:
        """Get a sub-vector of the measurable vector by selecting a collection of component functions, or a single component function if only one index is provided.

        Calls `get_sub_vector` with the provided indices. See the documentation of that method for details.

        Parameters
        ----------
        *args : Hashable | tuple[Hashable]
            The indices of the component functions to select for the sub-vector.

        Returns
        -------
        sub_vector : MeasurableVector
            A new `MeasurableVector` containing only the specified component functions.
        """
        indices = list(*args) if isinstance(args[0], tuple) else list(args)
        return self.get_sub_vector(indices=indices)

    def __iter__(self) -> Iterator[Function]:
        """Iterate over the components of the measurable vector.

        Returns
        -------
        iterator : Iterator[MeasurableFunction]
            An iterator over the components of the measurable vector.
        """
        return iter(self.components)

    # --------------------- conversion methods --------------------- #

    def to_measure(
        self,
        measure_domain: SigmaAlgebra | IndexLike,
        kind: Literal["measure", "probability"] = "measure",
        name: Hashable | None = None,
        in_place: bool = False,
    ) -> Measure | ParametrizedMeasure:
        """Generate a parametrized probability measure or measure from the function.

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
        Define a function on a Cartesian product of a 2-dimensional parameter space and a 1-dimensional measure domain.

        >>> from sigalg.core import Domain, Function
        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> X = Domain.from_sequence(size=3, variable_name="x")
        >>> f = Function(
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
        from ...validation.measure_domain_normalizer import MeasureDomainNormalizer
        from ..measures.measure import Measure
        from ..measures.parametrized_measure import ParametrizedMeasure
        from ..measures.parametrized_probability_measure import (
            ParametrizedProbabilityMeasure,
        )

        if self.domain is not None:
            if name is not None and not isinstance(name, Hashable):
                raise TypeError("If provided, name must be a hashable type.")

            v = MeasureDomainNormalizer(measure_domain=measure_domain, kind=kind)

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
                    output_name=self.data.name,
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
        self,
        sig_alg: SigmaAlgebra,
        measure: Measure | None = None,
        name: Hashable | None = None,
    ) -> MeasurableFunction | ParametrizedMeasurableFunction:
        """Pass."""
        from .._utils.function_helpers import sig_alg_func_to_measurable_func
        from ..measures.measure import Measure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .measurable_function import MeasurableFunction
        from .parametrized_measurable_function import ParametrizedMeasurableFunction

        if self.data is None:
            raise ValueError(
                "Cannot convert a function to a measurable function if the data attribute is empty."
            )
        if not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be an instance of SigmaAlgebra.")
        if measure is not None:
            if not isinstance(measure, Measure):
                raise TypeError("If given, measure must be an instance of Measure.")
            if measure.sig_alg != sig_alg:
                raise ValueError(
                    "If given, the sigma-algebra of measure must be the same as the sig_alg parameter."
                )

        domain = sig_alg.domain

        if name is None:
            name = self.name

        if not set(sig_alg.variable_names) <= set(self.variable_names):
            raise ValueError(
                "The variable names of the sigma-algebra are not contained in the variable names of the function."
            )

        parameter_names = [
            name for name in self.variable_names if name not in sig_alg.variable_names
        ]

        if set(domain.variable_names) & set(parameter_names):
            raise ValueError(
                "There is an overlap between the domain variable names and the parameter names."
            )

        mapping = sig_alg_func_to_measurable_func(
            self_data=self.data,
            sig_alg_data=sig_alg.data,
            parameter_names=parameter_names,
            output_name=self.output_name,
        ).rename(name)

        if not parameter_names:
            return MeasurableFunction(
                domain=domain,
                sig_alg=sig_alg,
                measure=measure,
                mapping=mapping,
                name=name,
            )
        else:
            return ParametrizedMeasurableFunction.from_domains(
                mapping=mapping,
                name=name,
                measurable_domain=domain,
                sig_alg=sig_alg,
                measure=measure,
            )

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
        import numpy as np

        if multi_dim:
            arr = self.data.to_xarray().values
            if dtype is not None:
                arr = np.asarray(arr, dtype=dtype)
            if copy:
                arr = arr.copy()
            return arr
        else:
            return self.__array__(dtype=dtype, copy=copy)

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
        import numpy as np

        arr = self.data.values
        if dtype is not None:
            arr = np.asarray(arr, dtype=dtype)
        if copy:
            arr = arr.copy()

        return arr

    def with_variable_names(self, variable_names: list[Hashable]) -> Function:
        """Return a new instance of the function with updated variable names."""
        from ..measures.measure import Measure

        constructor_sig = inspect.signature(Function)
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
            return Function(**params)

    def with_name(self, name: Hashable) -> Function:
        """Set the name of the function and return self for chaining.

        Parameters
        ----------
        name : Hashable
            The new name for the function.

        Returns
        -------
        self : Function
            The instance of the function with the updated name.
        """
        self.name = name
        return self

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a concise string representation of the function.

        Returns
        -------
        repr_str : str
            The string representation of the function.
        """
        import pandas as pd

        PandasLike = pd.Series | pd.DataFrame

        if isinstance(self.data, PandasLike):
            parameter_list = ", ".join(self.variable_names)
            return (
                f"{type(self)._repr_name}(parameters=({parameter_list}), "
                f"domain={self.domain.name}, "
                f"name={self.name})"
            )
        elif isinstance(self.data, Callable):
            parameter_list = ", ".join(self.variable_names)
            return (
                f"{type(self)._repr_name}(parameters=({parameter_list}), "
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
        import pandas as pd

        if isinstance(self.data, pd.Series):
            return f"{type(self)._str_name} '{self.name}':\n{self.data.to_frame()}"
        elif isinstance(self.data, pd.DataFrame):
            return f"{type(self)._str_name} '{self.name}':\n{self.data}"
        elif isinstance(self.data, Callable):
            return self.__repr__()
        else:
            return f"{type(self)._str_name} '{self.name}': empty"

    # --------------------- equality --------------------- #

    # TODO: add an `equal_as_measures` method
    def __eq__(self, other: Function | Real) -> bool:
        """Check if two functions are equal.

        Equality may only be checked if both functions have domains. If the arguments of the two functions are the same but in a different order, the method will attempt to reorder the levels of the other function's data to match the order of this function's arguments before comparing the values.

        Parameters
        ----------
        other : Function | Real
            The other function to compare with.

        Returns
        -------
        are_equal : bool
            True if the two functions are equal, False otherwise.

        Examples
        --------
        Define two functions whose domains are the same up to order and variable order.
        >>> from sigalg.core import Domain, Function
        >>> D_f = Domain([(0, 1), (1, 2)], variable_names=["x", "y"], name="D_f")
        >>> D_g = Domain([(2, 1), (1, 0)], variable_names=["y", "x"], name="D_g")
        >>> f = Function(domain=D_f, mapping=lambda *, x, y: x**2 + y**2)
        >>> g = Function(domain=D_g, mapping=lambda *, y, x: x**2 + y**2, name="g")
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
              f
        x y
        0 1   1
        1 2   5
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
              g
        y x
        2 1   5
        1 0   1

        These functions are equal.

        >>> f == g
        True
        """
        import pandas as pd

        from .._utils import align_index, pandas_all_equal

        if isinstance(other, Function):
            if self.domain is None or other.domain is None:
                raise ValueError(
                    "Cannot compare functions when one (or both) domains are not defined."
                )

            try:
                other_data = align_index(other.data, by=self.data.index)
            except ValueError:
                return False

            return pandas_all_equal(self.data, other_data, check_series_names=False)

        elif isinstance(other, Hashable | tuple | pd.Series):
            return self.get_inverse_image(value=other)

        else:
            raise TypeError("Can only compare with another Function or a scalar.")

    # --------------------- arithmetic operations --------------------- #

    def _apply_binary_operation(
        self,
        other,
        operation: Callable,
        op_symbol: str,
        reverse: bool = False,
    ) -> Function:
        """Apply a binary operation to this function.

        Parameters
        ----------
        other : Function or scalar
            The other operand.
        operation : Callable
            The operation to apply (e.g., lambda a, b: a + b).
        op_symbol : str
            Symbol representing the operation (e.g., '+', '-', '*').
        reverse : bool, default=False
            Whether this is a reverse operation (e.g., __radd__ vs __add__).

        Returns
        -------
        Function
            A new function representing the result of the operation.

        Raises
        ------
        TypeError
            If `other` is not a `Function` or a scalar.
        """
        from numbers import Real

        import pandas as pd

        if isinstance(other, Function):
            if reverse:
                name = f"({other.name} {op_symbol} {self.name})"
            else:
                name = f"({self.name} {op_symbol} {other.name})"

            if isinstance(self.data, pd.Series) and isinstance(other.data, pd.Series):
                if set(self.variable_names) & set(other.variable_names):
                    if not reverse:
                        data = operation(self.data, other.data).dropna().rename(name)
                        if set(self.variable_names) != set(other.variable_names):
                            domain_name = (
                                f"({self.domain_name} int {other.domain_name})"
                            )
                        else:
                            domain_name = self.domain.name

                    else:
                        data = operation(other.data, self.data).dropna().rename(name)

                        if set(self.variable_names) != set(other.variable_names):
                            domain_name = (
                                f"({other.domain_name} int {self.domain_name})"
                            )
                        else:
                            domain_name = self.domain.name

                else:
                    data = pd.merge(
                        left=self.data.reset_index(),
                        right=other.data.reset_index(),
                        how="cross",
                    ).set_index(self.variable_names + other.variable_names)

                    if not reverse:
                        data = operation(data[self.name], data[other.name]).rename(name)
                        domain_name = f"{self.domain_name} x {other.domain_name}"
                    else:
                        data = operation(data[other.name], data[self.name]).rename(name)
                        domain_name = f"{other.domain_name} x {self.domain_name}"

                return Function._from_validated(
                    data=data,
                    kind="any",
                    name=name,
                    domain_kind="Domain",
                    domain_name=domain_name,
                    index_kind=None,
                    index_name=None,
                )

            elif isinstance(self.data, Callable) and isinstance(other.data, Callable):
                variable_names = list(
                    dict.fromkeys(self.variable_names + other.variable_names)
                )

                self_sig = inspect.signature(self.data)
                other_sig = inspect.signature(other.data)

                def data(**kwargs):
                    self_arguments = {
                        name: value
                        for name, value in kwargs.items()
                        if name in self.variable_names
                    }
                    other_arguments = {
                        name: value
                        for name, value in kwargs.items()
                        if name in other.variable_names
                    }

                    self_arguments = self_sig.bind(**self_arguments)
                    other_arguments = other_sig.bind(**other_arguments)

                    return operation(
                        self.data(**self_arguments.arguments),
                        other.data(**other_arguments.arguments),
                    )

                parameters = [
                    inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
                    for name in variable_names
                ]
                sig = inspect.Signature(parameters)
                data.__signature__ = sig

                return Function._from_validated(
                    data=data,
                    kind="any",
                    name=name,
                    domain_kind=None,
                    domain_name=None,
                    index_kind=None,
                    index_name=None,
                )

        elif isinstance(other, Real):
            if isinstance(self.data, pd.Series):
                data = pd.Series(other, index=self.domain.data, name=str(other))
                other = Function._from_validated(
                    data=data,
                    kind="any",
                    name=str(other),
                    domain_kind=self.domain_kind,
                    domain_name=self.domain.name,
                    index_kind=None,
                    index_name=None,
                )
                return self._apply_binary_operation(
                    other, operation, op_symbol, reverse
                )

        else:
            raise TypeError(
                f"Unsupported operand type(s) for {op_symbol}: 'Function' and '{type(other).__name__}'"
            )

    def __add__(self, other):
        """Add two functions or a function and a scalar.

        Parameters
        ----------
        other : Function or scalar
            The other function or scalar to add to this function.

        Raises
        ------
        TypeError
            If `other` is not a `Function` or a scalar.
        """
        return self._apply_binary_operation(other, lambda a, b: a + b, "+")

    def __sub__(self, other):
        """Subtract another function or a scalar from this function."""
        return self._apply_binary_operation(other, lambda a, b: a - b, "-")

    def __mul__(self, other):
        """Multiply this function by another function or a scalar."""
        return self._apply_binary_operation(other, lambda a, b: a * b, "*")

    def __truediv__(self, other):
        """Divide this function by another function or a scalar."""
        return self._apply_binary_operation(other, lambda a, b: a / b, "/")

    def __pow__(self, other):
        """Raise this function to the power of another function or a scalar."""
        return self._apply_binary_operation(
            other, lambda a, b: self._to_float(a) ** self._to_float(b), "**"
        )

    def __neg__(self):
        """Negate this function."""
        import pandas as pd

        name = f"(-{self.name})"

        if isinstance(self.data, pd.Series):
            return Function._from_validated(
                data=-self.data,
                kind=self.kind,
                name=name,
                domain_kind=self.domain_kind,
                domain_name=self.domain.name,
                index_kind=None,
                index_name=None,
            )

        elif isinstance(self.data, Callable):
            sig = inspect.signature(self.data)

            def data(**kwargs):
                bound = sig.bind(**kwargs)
                return -self.data(**bound.arguments)

            data.__signature__ = sig

            return Function._from_validated(
                data=data,
                kind="any",
                name=name,
                domain_kind=None,
                domain_name=None,
                index_kind=None,
                index_name=None,
            )

    def __radd__(self, other):
        """Add this function to another function or a scalar (right-hand side)."""
        if isinstance(other, Function):
            return other.__add__(self)
        return self._apply_binary_operation(
            other, lambda a, b: a + b, "+", reverse=True
        )

    def __rsub__(self, other):
        """Subtract this function from another function or a scalar (right-hand side)."""
        if isinstance(other, Function):
            return other.__sub__(self)
        return self._apply_binary_operation(
            other, lambda a, b: a - b, "-", reverse=True
        )

    def __rmul__(self, other):
        """Multiply this function by another function or a scalar (right-hand side)."""
        if isinstance(other, Function):
            return other.__mul__(self)
        return self._apply_binary_operation(
            other, lambda a, b: a * b, "*", reverse=True
        )

    def __rtruediv__(self, other):
        """Divide another function or a scalar by this function (right-hand side)."""
        if isinstance(other, Function):
            return other.__truediv__(self)
        return self._apply_binary_operation(
            other, lambda a, b: a / b, "/", reverse=True
        )

    def __rpow__(self, other):
        """Raise another function or a scalar to the power of this function (right-hand side)."""
        if isinstance(other, Function):
            return other.__pow__(self)
        return self._apply_binary_operation(
            other,
            lambda a, b: self._to_float(a) ** self._to_float(b),
            "**",
            reverse=True,
        )

    @staticmethod
    def _to_float(x):
        return x.astype(float) if hasattr(x, "astype") else float(x)
