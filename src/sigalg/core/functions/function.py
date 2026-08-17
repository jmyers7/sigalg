"""A class representing a function."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Hashable, Iterator
from functools import cached_property
from itertools import combinations
from numbers import Real
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
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
        parameter_names: list[Hashable]
        | None = None,  # TODO: remove this and use kwargs for subclasses
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
            index=index,
            index_kind=index_kind,
            multi_dim_outputs=multi_dim_outputs,
            output_name=output_name,
            parameter_names=parameter_names,
            name=name,
        )

        self.data = v.data
        self.name = v.name
        self.kind = v.kind

    @classmethod
    def _from_validated(
        cls,
        *,
        data: pd.Series | pd.DataFrame | Callable,
        kind: Literal[
            "any",
            "measure",
            "probability",
            "param_measure",
            "param_probability",
        ],
        domain_kind: Literal["Domain", "SampleSpace"],
        domain_name: Hashable,
        index_kind: Literal["Index", "Time"],
        index_name: Hashable | None,
        name: Hashable,
        **kwargs,
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
    def from_constant(
        cls,
        domain: IndexLike,
        constant: Hashable | None,
        domain_kind: Literal["Domain", "SampleSpace"] = "Domain",
        domain_name: Hashable | None = None,
        output_name: Hashable | None = None,
        index: IndexLike | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        """Create a function that maps every point in the domain to the same constant output value.

        Returns
        -------
        const_func : Function
            The function that maps every point in its domain to the given constant value.

        Examples
        --------
        >>> from sigalg.core import Domain, Function

        Create a constant function with 2-dimensional outputs.

        >>> X = Domain.from_sequence(size=3)
        >>> f = Function.from_constant(domain=X, constant=(1, 2), index=[1, 2])
        >>> print(f) # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
        i  1  2
        x
        0  1  2
        1  1  2
        2  1  2

        Create a constant function.

        >>> g = Function.from_constant(domain=X, constant=2, name="g")
        >>> print(g) # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
                g
        x
        0       2
        1       2
        2       2
        """
        import pandas as pd

        from ...validation.domain_index_validator import DomainIndexValidator
        from ..indices.index import Index
        from ..indices.time import Time

        if not isinstance(constant, Hashable):
            raise TypeError("constant must be a Hashable.")
        if (
            index is not None
            and isinstance(constant, tuple)
            and len(constant) != len(index)
        ):
            raise ValueError(
                "Length of constant tuple must match the length of the index."
            )

        v = DomainIndexValidator(
            domain=domain,
            domain_kind=domain_kind,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
        )

        domain = v.domain
        domain_kind = v.domain_kind
        domain_name = v.domain_name
        index = v.index
        index_kind = v.index_kind
        index_name = v.index_name

        if name is None:
            name = cls._default_name
        if output_name is None:
            output_name = name

        if index is not None and not isinstance(constant, tuple):
            constant = (constant,) * len(index)

        if index is None and isinstance(constant, tuple):
            index_class = Index if index_kind == "Index" else Time
            index = index_class.from_sequence(size=len(constant), name=index_name)

        if isinstance(constant, tuple):
            data = pd.DataFrame([constant], index=domain.data, columns=index.data)

        else:
            data = pd.Series(constant, index=domain.data, name=output_name)

        return cls._from_validated(
            data=data,
            kind="any",
            domain_kind=domain_kind,
            domain_name=domain_name,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            **kwargs,
        )

    @classmethod
    def from_identity(
        cls,
        domain: IndexLike,
        domain_kind: Literal["Domain", "SampleSpace"] = "Domain",
        domain_name: Hashable | None = None,
        output_name: Hashable | None = None,
        index: IndexLike | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        """Create a measurable vector that maps every point in the domain to itself.

        For this construction method, the sigma-algebra must be the power set.

        Parameters
        ----------
        domain: IndexLike
            The domain of the measurable vector.
        sig_alg: SigmaAlgebra | None, default=None
            The sigma-algebra of the underlying measurable space. The sigma-algebra must be the power-set. This parameter is here only for consistency with other constructors.
        measure: Measure | None, default=None
            An optional measure carried by the measurable vector.
        index : IndexLike | None, default=None
            The index of the measurable vector.
        name : Hashable | None, default=None
            The name of the measurable vector. If `None`, a default will be generated.

        Raises
        ------
        ValueError
            If the sigma-algebra is not the power set (if given), or if the length of the index (if given) does not match the dimension of the domain.

        Returns
        -------
        vector : MeasurableVector
            A measurable vector mapping every point in the domain to itself.

        Examples
        --------
        >>> from sigalg.core import Domain, MeasurableVector

        Create an identity function on a 2-dimensional domain.

        >>> X = Domain.cartesian_power(
        ...     [0, 1], n=2, name="X", variable_names=["x_0", "x_1"]
        ... )
        >>> f = Function.from_identity(domain=X)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
        i        0  1
        x_0 x_1
        0   0    0  0
            1    0  1
        1   0    1  0
            1    1  1

        Print its range.

        >>> print(f.range)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'X':
         x_0  x_1
           0    0
           0    1
           1    0
           1    1

        Now define an identity vector on a 1-dimensional domain and print its range.

        >>> S = Domain(indices=["a", "b"], name="S")
        >>> g = Function.from_identity(domain=S, name="g")
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
           g
        x
        a  a
        b  b
        >>> print(g.range)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'S':
         x
         a
         b
        """
        from ...validation.domain_index_validator import DomainIndexValidator
        from ..indices.index import Index
        from ..indices.time import Time

        if index is not None and len(index) != domain.dimension:
            raise ValueError(
                "The length of the index must match the dimension of the domain."
            )

        v = DomainIndexValidator(
            domain=domain,
            domain_kind=domain_kind,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
        )

        domain = v.domain
        domain_kind = v.domain_kind
        domain_name = v.domain_name
        index = v.index
        index_kind = v.index_kind
        index_name = v.index_name

        if name is None:
            name = cls._default_name
        if output_name is None:
            output_name = name

        data = domain.data.to_frame()

        if data.shape[1] == 1:
            data = data.squeeze(axis=1)
            data.name = output_name
            index = None

        else:
            if index is None:
                index_class = Index if index_kind == "Index" else Time
                index = index_class.from_sequence(
                    size=domain.dimension, name=index_name
                )
            data.columns = index.data

        function = cls._from_validated(
            data=data,
            kind="any",
            domain_kind=domain_kind,
            domain_name=domain_name,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            **kwargs,
        )

        function.is_identity = True

        return function

    @classmethod
    def from_rand(
        cls,
        domain: IndexLike | None = None,
        dim: int | None = None,
        distribution: Literal["uniform", "normal"] = "uniform",
        min_value: int = 0,
        max_value: int = 10,
        loc: float = 0.0,
        scale: float = 1.0,
        domain_kind: Literal["Domain", "SampleSpace"] = "Domain",
        domain_name: Hashable | None = None,
        output_name: Hashable | None = None,
        index: IndexLike | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        random_state: int | np.random.Generator | None = None,
        **kwargs,
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
        >>> from sigalg.core import Domain, Function
        >>> rng = np.random.default_rng(42)

        Generate a random function with 2-dimensional values drawn from a standard normal distribution.

        >>> X = Domain.from_sequence(size=3)
        >>> f = Function.from_rand(
        ...     domain=X,
        ...     dim=2,
        ...     distribution="normal",
        ...     random_state=rng,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
        i         0         1
        x
        0  0.304717 -1.039984
        1  0.750451  0.940565
        2 -1.951035 -1.302180

        Generate a function with random 1-dimensional outputs drawn from a uniform distribution on the integers `[-10, 10)`.

        >>> g = Function.from_rand(
        ...     domain=X,
        ...     dim=1,
        ...     distribution="uniform",
        ...     low=-10,
        ...     high=10,
        ...     name="g",
        ...     random_state=rng,
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
           g
        x
        0  7
        1  7
        2  7

        """
        import numpy as np
        import pandas as pd

        from ...validation.domain_index_validator import DomainIndexValidator
        from ..indices.index import Index
        from ..indices.time import Time

        if distribution not in ("uniform", "normal"):
            raise ValueError('distribution must be either "uniform" or "normal".')
        if dim is not None and (not isinstance(dim, int) or dim < 1):
            raise ValueError("If given, dim must be a positive integer.")
        if not isinstance(min_value, int):
            raise TypeError("min_value must be an integer.")
        if not isinstance(max_value, int):
            raise TypeError("max_value must be an integer.")
        if min_value > max_value:
            raise ValueError("min_value cannot be greater than max_value.")
        if not isinstance(loc, Real):
            raise TypeError("loc must be a number.")
        if not isinstance(scale, Real) or scale <= 0:
            raise TypeError("scale must be a positive number.")
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("If given, name must be hashable.")
        if random_state is not None and not isinstance(
            random_state, (int, np.random.Generator)
        ):
            raise TypeError(
                "random_state must be an integer, np.random.Generator, or None."
            )

        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        v = DomainIndexValidator(
            domain=domain,
            domain_kind=domain_kind,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
        )

        domain = v.domain
        domain_kind = v.domain_kind
        domain_name = v.domain_name
        index = v.index
        index_kind = v.index_kind
        index_name = v.index_name

        if index is not None:
            if dim is not None and len(index) != dim:
                raise ValueError(
                    "If both index and dim are given, the length of the former must equal the latter."
                )
            dim = len(index)

        else:
            if dim is None:
                raise ValueError("One or the other of dim or index must be given.")
            index_class = Index if index_kind == "Index" else Time
            index = index_class.from_sequence(size=dim, name=index_name)

        if distribution == "normal":
            arr = rng.normal(loc=loc, scale=scale, size=(len(domain), dim))
        elif distribution == "uniform":
            arr = rng.integers(low=min_value, high=max_value, size=(len(domain), dim))

        if name is None:
            name = cls._default_name
        if output_name is None:
            output_name = name

        if dim > 1:
            data = pd.DataFrame(arr, index=domain.data, columns=index.data)
        else:
            data = pd.Series(arr.squeeze(axis=1), index=domain.data, name=output_name)

        return cls._from_validated(
            data=data,
            kind="any",
            domain_kind=domain_kind,
            domain_name=domain_name,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            **kwargs,
        )

    @classmethod
    def concatenate(
        cls,
        factors: list[Function | Real],
        index: IndexLike | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        """Concatenate a list of measurable vectors or scalars into a single measurable vector.

        Parameters
        ----------
        factors : list[MeasurableFunction | MeasurableVector | Real]
            A list of measurable vectors or scalars to combine.
        index : IndexLike | None, default=None
            The index of the resulting measurable vector. If `None`, the index will be generated by concatenating the indices of the input measurable vectors, provided that they are disjoint; otherwise, a new default index will be generated.
        name : Hashable | None, default=None
            The name of the resulting measurable vector. If `None`, the name will be generated by concatenating the names of the input measurable vectors.

        Raises
        ------
        TypeError
            If `factors` is not a list, if any element of `factors` is not a `MeasurableFunction`, `MeasurableVector`, or scalar, or if `name` is not a `Hashable` or `None`.
        ValueError
            If there is not at least one `MeasurableVector` instance in `factors`, or if the measurable vectors in `factors` are not defined on the same measurable space.

        Returns
        -------
        concatenation : MeasurableVector
            A new measurable vector created by combining the input measurable vectors.

        Examples
        --------
        >>> from sigalg.core import Domain, Index, Function

        Generate two functions with disjoint indices.

        >>> X = Domain.from_sequence(size=4)
        >>> I = Index([0, 1, 2])
        >>> f = Function.from_rand(
        ...     domain=X,
        ...     max_value=2,
        ...     index=I,
        ...     random_state=42,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
        i       0  1  2
        x
        0       0  1  1
        1       0  0  1
        2       0  1  0
        3       0  1  1
        >>> J = Index([3, 4], name="J")
        >>> g = Function.from_rand(
        ...     domain=X,
        ...     index=J,
        ...     max_value=2,
        ...     random_state=42,
        ...     name="g",
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
        i       3  4
        x
        0       0  1
        1       1  0
        2       0  1
        3       0  1

        Concatenate the two functions.

        >>> fg = Function.concatenate([f, g])
        >>> print(fg)  # doctest: +NORMALIZE_WHITESPACE
        Function 'fg':
        i  0  1  2  3  4
        x
        0  0  1  1  0  1
        1  0  0  1  1  0
        2  0  1  0  0  1
        3  0  1  1  0  1

        Generate a measurable function with 1-dimensional outputs

        >>> h = Function.from_rand(
        ...     domain=X,
        ...     dim=1,
        ...     max_value=2,
        ...     random_state=42,
        ...     name="h",
        ... )
        >>> print(h)  # doctest: +NORMALIZE_WHITESPACE
        Function 'h':
                h
        x
        0       0
        1       1
        2       1
        3       0

        Concatenate all the functions, along with a scalar.

        >>> fh2g = f | h | 2 | g
        >>> print(fh2g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'fh2g':
        i  0  1  2  3  4  5  6
        x
        0  0  1  1  0  2  0  1
        1  0  0  1  1  2  1  0
        2  0  1  0  1  2  0  1
        3  0  1  1  0  2  0  1

        Form a concatenation with a custom index and name.

        >>> k = Function.concatenate([0, h, f], index=[0, 1, 2, 3, 4], name="k")
        >>> print(k)  # doctest: +NORMALIZE_WHITESPACE
        Function 'k':
        i  0  1  2  3  4
        x
        0  0  0  0  1  1
        1  0  1  0  0  1
        2  0  1  0  1  0
        3  0  0  0  1  1
        """
        import pandas as pd

        from ...validation.domain_index_validator import DomainIndexValidator
        from ..indices.index import Index
        from ..indices.time import Time

        if not isinstance(factors, list):
            raise TypeError(
                "factors must be a list of instances of Function and scalars."
            )
        actual_funcs = [func for func in factors if isinstance(func, Function)]
        if not actual_funcs:
            raise ValueError("There must be at least one function in factors.")
        domain = actual_funcs[0].domain
        if any(func.domain != domain for func in actual_funcs):
            raise ValueError(
                "All Function instances must be defined on the same domain."
            )

        v = DomainIndexValidator(
            domain=domain,
            domain_kind=type(domain).__name__,
            domain_name=domain.name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
        )

        index = v.index
        index_kind = v.index_kind
        index_name = v.index_name

        try:
            factor_data = [
                pd.Series(func, index=domain.data, name=func)
                if not isinstance(func, Function)
                else func.data
                for func in factors
            ]
        except TypeError as e:
            raise TypeError(
                "Cannot form constant functions from the non-Function factors."
            ) from e

        indices = [
            set(data.index) if isinstance(data, pd.DataFrame) else {data.name}
            for data in factor_data
        ]

        ignore_index = any(
            len(idx1 & idx2) >= 1 for idx1, idx2 in combinations(indices, 2)
        )

        if name is None:
            name = "".join(
                func.name if isinstance(func, Function) else str(func)
                for func in factors
            )

        data = pd.concat(factor_data, axis=1, ignore_index=ignore_index)

        if index is not None:
            data.columns = index.data
        else:
            index_class = Index if index_kind == "Index" else Time
            data.columns.name = index_class._variable_names_prefix

        return cls._from_validated(
            data=data,
            kind="any",
            domain_kind=type(domain).__name__,
            domain_name=domain.name,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            **kwargs,
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
                data=mapping,
                kind="probability" if all_probs else "measure",
                name=name,
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
        """
        name = f"{function.name} ^ {n}"
        return cls.tensor_product(factors=[function] * n, name=name)

    # TODO: add fast path if all factors are identities
    @classmethod
    def cartesian_product(
        cls,
        factors: list[Function],
        domain_name: Hashable | None = None,
        index: IndexLike | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        r"""Form the Cartesian product of a list of measurable vectors.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        factors : list[MeasurableVector]
            The factors of the Cartesian product.
        index : IndexLike | None, default=None
            The index of the Cartesian product. If `None`, a default index will be generated.
        name : Hashable | None, default=None
            The name of the Cartesian product. If `None`, a default will be generated.
        domain_name : Hashable | None, default=None
            The name of the domain of the Cartesian product. If `None`, a default will be generated.
        sig_alg_name : Hashable | None, default=None
            The name of the sigma-algebra of the Cartesian product. If `None`, a default will be generated.
        measure_name : Hashable | None, default=None
            The name of the measure of the Cartesian product. If `None`, a default will be generated.

        Returns
        -------
        product : MeasurableVector
            The Cartesian product of the measurable vectors.

        Examples
        --------
        >>> from sigalg.core import Domain, Function

        Define two functions on two 1-dimensional domains.

        >>> X = Domain.from_sequence(size=2)
        >>> Y = Domain.from_sequence(size=3, variable_name="y", name="Y")
        >>> f = Function(domain=X, mapping=dict(zip(X, [1, 2])))
        >>> g = Function(domain=Y, mapping=dict(zip(Y, [(3, 4), (5, 6), (7, 8)])), name="g")

        Form the Cartesian product of the two functions using the `cartesian_product` method.

        >>> product = Function.cartesian_product([f, g])
        >>> print(product)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f x g':
        i    0  1  2
        x y
        0 0  1  3  4
          1  1  5  6
          2  1  7  8
        1 0  2  3  4
          1  2  5  6
          2  2  7  8

        Form the same Cartesian product using the `@` operator.

        >>> print(f @ g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f x g':
        i    0  1  2
        x y
        0 0  1  3  4
          1  1  5  6
          2  1  7  8
        1 0  2  3  4
          1  2  5  6
          2  2  7  8

        Notes
        -----
        Given one measurable vector $f: X \to \mathbb{R}^d$ on a measurable space $(X,\mathcal{F})$, and a second measurable vector $g: Y \to \mathbb{R}^e$ on a measurable space $(Y,\mathcal{G})$, their *Cartesian product*, denoted $f \times g$, is the $(\mathcal{F} \times \mathcal{G})$-measurable measurable vector defined

        $$
        (f \times g) : X \times Y \to \mathbb{R}^{d+e}, \quad (f\times g)(x, y) = (f(x),g(y)).
        $$

        Here, $\mathcal{F} \times \mathcal{G}$ is the product $\sigma$-algebra.
        """
        import pandas as pd

        from ...validation.domain_index_validator import DomainIndexValidator
        from .._utils.utils import subscript_var_names
        from ..indices.index import Index
        from ..indices.time import Time

        if not isinstance(factors, list) or not all(
            isinstance(rv, Function) for rv in factors
        ):
            raise TypeError("factors must be a list of Function instances.")

        v = DomainIndexValidator(
            domain=None,
            domain_kind="Domain",
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
        )

        index = v.index
        index_kind = v.index_kind
        index_name = v.index_name

        domain_variable_names = subscript_var_names(
            [func.data.index.names for func in factors], grouped=True
        )

        data = factors[0].data.copy()
        data.index.names = domain_variable_names[0]
        data = data.reset_index()

        for var_names, factor in zip(domain_variable_names[1:], factors[1:]):
            factor_data = factor.data.copy()
            factor_data.index.names = var_names
            factor_data = factor_data.reset_index()
            data = pd.merge(
                left=data,
                right=factor_data,
                how="cross",
            )

        data = data.set_index([name for lst in domain_variable_names for name in lst])

        if index is None:
            index_class = Index if index_kind == "Index" else Time
            index = index_class.from_sequence(size=data.shape[1], name=index_name)

        data.columns = index.data

        if name is None:
            name = " x ".join([factor.name for factor in factors])
        if domain_name is None:
            domain_name = " x ".join([factor.domain.name for factor in factors])

        return cls._from_validated(
            data=data,
            kind="any",
            domain_kind="Domain",
            domain_name=domain_name,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            **kwargs,
        )

    @classmethod
    def cartesian_power(
        cls,
        vector: Function,
        n: int,
        index: IndexLike | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
    ) -> Function:
        """Form the Cartesian power of a measurable vector.

        Parameters
        ----------
        vector : MeasurableVector
            The base of the Cartesian power.
        n : int
            The power of the Cartesian power.
        index : IndexLike | None, default=None
            The index of the Cartesian power. If `None`, a default index will be generated.

        Raises
        ------
        TypeError
            If `vector` is not a `MeasurableVector` or if `n` is not an integer.
        ValueError
            If `n` is not positive.

        Examples
        --------
        >>> from sigalg.core import Domain, Function

        Define a function with 2-dimensional outputs.

        >>> X = Domain.from_sequence(size=4, variable_name="x")
        >>> f = Function(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...         3: (5, 6),
        ...     },
        ... )

        Compute the second Cartesian power of the function `f`.

        >>> cart_pow = Function.cartesian_power(f, 2)
        >>> print(cart_pow)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f ^ 2':
        i        0  1  2  3
        x_0 x_1
        0   0    1  2  1  2
            1    1  2  3  4
            2    1  2  3  4
            3    1  2  5  6
        1   0    3  4  1  2
            1    3  4  3  4
            2    3  4  3  4
            3    3  4  5  6
        2   0    3  4  1  2
            1    3  4  3  4
            2    3  4  3  4
            3    3  4  5  6
        3   0    5  6  1  2
            1    5  6  3  4
            2    5  6  3  4
            3    5  6  5  6

        Compute the third Cartesian power using the `^` operator.

        >>> print(f ^ 3)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f ^ 3':
        i            0  1  2  3  4  5
        x_0 x_1 x_2
        0   0   0    1  2  1  2  1  2
                1    1  2  1  2  3  4
                2    1  2  1  2  3  4
                3    1  2  1  2  5  6
            1   0    1  2  3  4  1  2
        ...         .. .. .. .. .. ..
        3   2   3    5  6  3  4  5  6
            3   0    5  6  5  6  1  2
                1    5  6  5  6  3  4
                2    5  6  5  6  3  4
                3    5  6  5  6  5  6
        <BLANKLINE>
        [64 rows x 6 columns]
        """
        name = f"{vector.name} ^ {n}"
        domain_name = f"{vector.domain.name} ^ {n}"

        return cls.cartesian_product(
            factors=[vector] * n,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
        )

    # --------------------- dunder operators --------------------- #

    def __or__(self, other: Function | Real | Set) -> Function:
        """Concatenate the current instance with a second measurable vector, a constant measurable function (represented as a `Real`), or restrict the measurable vector to a measurable subset.

        Calls `MeasurableVector.concatenate` if `other` is a `MeasurableVector`, `MeasurableFunction`, or scalar, or calls `MeasurableVector.restrict_to` if `other` is a `MeasurableSet`. See the documentation for those methods for more details.
        """
        from ..spaces.set import Set

        if isinstance(other, Set | list):
            return self.restrict_to(subset=other)
        else:
            return type(self).concatenate([self, other])

    def __ror__(self, other: Function | Real) -> Function:
        """Concatenate the current instance with a second measurable vector or a constant measurable function (represented as a `Real`).

        Calls `MeasurableVector.concatenate`.
        """
        return type(self).concatenate([other, self])

    def __matmul__(self, other: Function) -> Function:
        """Form the Cartesian product of a pair of measurable vectors.

        Calls the `MeasurableVector.cartesian_product` method. See the documentation of that method for details.
        """
        return type(self).cartesian_product([self, other])

    def __xor__(self, power: int) -> Function:
        """Form the Cartesian power of this instance of `MeasurableVector`.

        Calls the `MeasurableVector.cartesian_power` method. See the documentation of that method for details.
        """
        return type(self).cartesian_power(vector=self, n=power)

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

        >>> from sigalg.core import Domain, Function
        >>> X = Domain.from_sequence(size=3)
        >>> f = Function(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 4),
        ...         1: (2, 5),
        ...         2: (3, 6),
        ...     }
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
        i   0  1
        x
        0   1  4
        1   2  5
        2   3  6
        >>> for component in f.components:
        ...     print(component)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f_0':
           f_0
        x
        0    1
        1    2
        2    3
        Function 'f_1':
           f_1
        x
        0    4
        1    5
        2    6

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
         0
         1
         4
        """
        import pandas as pd

        from ..spaces.domain import Domain

        if hasattr(self, "is_identity") and self.is_identity:
            return self.domain

        name = f"{self.name}_range"

        if isinstance(self.data, pd.Series):
            range_list = list(self.data.drop_duplicates())
            data = pd.Index(range_list, name=self.name).sort_values()
            return Domain._from_validated(data=data, name=name)

        elif isinstance(self.data, pd.DataFrame):
            range_list = list(self.data.drop_duplicates().apply(tuple, axis=1))
            data = pd.MultiIndex.from_tuples(
                range_list, names=self.component_names.values()
            ).sort_values()
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
        F
        0    1    2
        1    3    4
        2    1    2

        Whenever a measurability check is executed, and the result is `True`, the sigma-algebra is added to the internal `lattice`.

        >>> f.lattice
        Lattice(base=sigma(f), type=upward, num_sig_algs=2)

        Perform another measurability check, inspect the `lattice` attribute to see the updated list of contents, and print the atom data.

        >>> G in f.lattice
        True
        >>> f.lattice
        Lattice(base=sigma(f), type=upward, num_sig_algs=3)
        >>> print(f.lattice.get_atom_data(G))  # doctest: +NORMALIZE_WHITESPACE
        i    0    1
        G
        0    1    2
        1    3    4

        Notice that the function is not measurable with respect to the third sigma-algebra. The measurability check accordingly returns `False`, and the contents of `lattice` is not changed.

        >>> H in f.lattice
        False
        >>> f.lattice
        Lattice(base=sigma(f), type=upward, num_sig_algs=3)

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
                atom_data = self.lattice.get_atom_data(join)
                ordered_atom_ID = tuple(atom_ID[name] for name in atom_data.index.names)

                if self.dimension > 1:
                    return tuple(atom_data.loc[ordered_atom_ID])
                else:
                    return atom_data.loc[ordered_atom_ID].astype(Real)

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
        >>> import pandas as pd
        >>> from sigalg.core import (
        ...     Domain,
        ...     Function,
        ... )
        >>> X = Domain.from_sequence(size=4)
        >>> f = Function(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 0),
        ...         1: (1, 0),
        ...         2: (0, 1),
        ...         3: (1, 0),
        ...     }
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
        i  0  1
        x
        0  1  0
        1  1  0
        2  0  1
        3  1  0

        Get an inverse image using the `get_inverse_image` method.

        >>> inv = f.get_inverse_image((1, 0))
        >>> print(inv)  # doctest: +NORMALIZE_WHITESPACE
        Set '{f = (1, 0)}':
         x
         0
         1
         3

        Get an inverse image using the overloaded operator `==`.

        >>> print(f == (1, 0))  # doctest: +NORMALIZE_WHITESPACE
        Set '{f = (1, 0)}':
         x
         0
         1
         3

        Get an inverse image using the overloaded operator `==` and a `pd.Series`.

        >>> s = pd.Series([1, 0], index=f.index)
        >>> print(f == s)  # doctest: +NORMALIZE_WHITESPACE
        Set '{f = (1, 0)}':
         x
         0
         1
         3
        """
        import pandas as pd

        from ..spaces.set import Set

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

        name = f"{{{self.name} = {value}}}"

        return Set(inv_image, domain=self.domain, name=name)

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

    def restrict_to(
        self,
        subset: Set | list,
        subset_name: Hashable | None = "A",
        **kwargs,
    ) -> Function:
        r"""Restrict the measurable vector to a measurable set.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        measurable_set : MeasurableSet | list
            The set to restrict the measurable vector to.
        set_name : Hashable | None, default="A"
            The name to use for the measurable set in the name of the resulting restricted measurable vector. This parameter is only used if `measurable_set` is a list of points, and is otherwise ignored if `measurable_set` is a `MeasurableSet` instance.

        Returns
        -------
        restricted_vec : MeasurableVector
            A new `MeasurableVector` representing the restriction of the original measurable vector to the given set.

        Examples
        --------
        >>> from sigalg.core import Domain, Function, Set

        Define a function with 2-dimensional outputs.

        >>> X = Domain.from_sequence(size=4)
        >>> f = Function(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...         3: (5, 6),
        ...     },
        ... )

        Restrict the function to a set using the `restrict_to` method.

        >>> A = Set([1, 2, 3], domain=X)
        >>> f_A = f.restrict_to(A)
        >>> print(f_A)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f|A':
        i  0  1
        x
        1  3  4
        2  3  4
        3  5  6

        Compute the same restriction using the overloaded `|` operator.

        >>> print(f | A)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f|A':
        i  0  1
        x
        1  3  4
        2  3  4
        3  5  6

        Test with a function with 1-dimensional outputs and a list of points, instance of a `Set`.

        >>> g = Function(domain=X, mapping=dict(zip(X, [3, 2, 4, 1])), name="g")
        >>> print(g | [1, 2, 3])  # doctest: +NORMALIZE_WHITESPACE
        Function 'g|A':
           g|A
        x
        1    2
        2    4
        3    1

        Notes
        -----
        Let $f: X \to \mathbb{R}^d$ be a measurable vector on a measure space $(X, \mathcal{F}, \mu)$. If $A\in \mathcal{F}$ is an measurable set, then we may restrict the measurable vector to obtain the function $f|_A : A \to \mathbb{R}^d$ on $A$.
        """
        from ..spaces.set import Set

        if not isinstance(subset, (Set, list)):
            raise TypeError("subtset must be a Set or a list of points.")
        if isinstance(subset, list):
            subset = Set(subset, domain=self.domain, name=subset_name)
        if not set(subset.data) <= set(self.domain.data):
            raise ValueError(
                "The subset must be a subset of the domain of the function."
            )

        data = self.data.loc[subset.data]
        data.index = subset.data
        name = f"{self.name}|{subset.name}"

        if self.dimension == 1:
            data.name = name

        return type(self)._from_validated(
            data=data,
            kind="any",
            domain_kind=type(self.domain).__name__,
            domain_name=subset.name,
            index_kind=type(self.index).__name__,
            index_name=self.index.name if self.index else None,
            name=name,
            **kwargs,
        )

    # TODO: stale docstring
    def atom_data(
        self, sig_alg: SigmaAlgebra | None = None
    ) -> pd.Series | pd.DataFrame | None:
        """Get the (parametrized) unique values of the function on the atoms of the underlying sigma-algebra.

        Returns
        -------
        atom_data : pd.Series | None
            A `pd.Series` with multi-index containing the unique values of the function on the atom identifiers of the sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedMeasurableFunction,
        ...     SigmaAlgebra,
        ... )

        Define a 1-dimensional parameter space, a 1-dimensional domain, and a sigma-algebra.

        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
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

        Define a parametrized measurable function.

        >>> mapping = {
        ...     (0, 0): 1,  # (theta, x) = (0, 0), etc ...
        ...     (0, 1): 2,
        ...     (0, 2): 2,
        ...     (0, 3): 2,
        ...     (1, 0): 0,
        ...     (1, 1): -3,
        ...     (1, 2): -3,
        ...     (1, 3): -3,
        ... }
        >>> f = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     mapping=mapping,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'f':
        theta  0  1
        x
        0      1  0
        1      2 -3
        2      2 -3
        3      2 -3

        By leaving the parameter to `atom_data` as its default `None`, it computes the unique values of the parametrized measurable function on each of the atoms of the underlying sigma-algebra (accessed through the `sig_alg` attribute).

        >>> print(f.atom_data())  # doctest: +NORMALIZE_WHITESPACE
        theta  0  1
        F
        0      1  0
        1      2 -3
        2      2 -3

        Note that the function is also measurable with respect to the following finer sigma-algebra.

        >>> G = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ...     name="G",
        ... )
        >>> f in G
        True

        We may thus pass `G` into the `atom_data` method to get the unique values of the function on each of the atoms of `G`.

        >>> print(f.atom_data(G))  # doctest: +NORMALIZE_WHITESPACE
        theta  0  1
        G
        0      1  0
        1      2 -3
        2      2 -3

        """
        if self.data is not None:
            if sig_alg is None:
                sig_alg = self.sig_alg
            self.lattice.add(sig_alg)
            return self.lattice.get_atom_data(sig_alg)
        else:
            return None

    # --------------------- util methods --------------------- #

    def item(self) -> Hashable | pd.Series:
        """Get the output value of a constant measurable vector.

        Returns
        -------
        output : Hashable | pd.Series
            The single output value of the measurable vector.

        Raises
        ------
        ValueError
            If the measurable vector is not constant.

        Examples
        --------
        >>> from sigalg.core import Domain, Function

        Get the output of a constant function with 2-dimensional outputs.

        >>> X = Domain.from_sequence(size=2)
        >>> f = Function(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...     },
        ... )
        >>> print(f.item())  # doctest: +NORMALIZE_WHITESPACE
        (1, 2)

        Get the output of a constant function with 1-dimensional outputs.

        >>> g = Function(domain=X, mapping=dict(zip(X, [1, 1])), name="g")
        >>> g.item()
        1
        """
        if self.data is None:
            raise ValueError("Cannot retrieve the item of an empty measurable vector.")

        if len(self.data.drop_duplicates()) != 1:
            raise ValueError(
                "Can only retrieve the item of a constant measurable vector."
            )

        return self(self.domain[0])

    def __round__(self, ndigits: int = None, **kwargs) -> Function:
        """Round the outputs of the measurable vector to a specified number of decimal places.

        Parameters
        ----------
        decimals : int, default=0
            The number of decimal places to round to. Must be a non-negative integer.

        Examples
        --------
        >>> from sigalg.core import Domain, Function
        >>> X = Domain.from_sequence(size=3)
        >>> f = Function(domain=X, mapping=dict(zip(X, [0.1, 0.45, 0.675])))
        >>> print(round(f))  # doctest: +NORMALIZE_WHITESPACE
        Function 'round(f)':
           round(f)
        x
        0         0
        1         0
        2         1
        >>> print(round(f, 1))  # doctest: +NORMALIZE_WHITESPACE
        Function 'round(f)':
           round(f)
        x
        0       0.1
        1       0.4
        2       0.7
        >>> print(round(f, 2))  # doctest: +NORMALIZE_WHITESPACE
        Function 'round(f)':
           round(f)
        x
        0      0.10
        1      0.45
        2      0.68
        >>> print(round(f, 3))  # doctest: +NORMALIZE_WHITESPACE
        Function 'round(f)':
           round(f)
        x
        0     0.100
        1     0.450
        2     0.675
        """
        import pandas as pd

        if self.data is not None:
            data = self.data.round(decimals=ndigits if ndigits else 0)

            if not ndigits:
                data = data.astype(int)

            name = f"round({self.name})"

            if isinstance(data, pd.Series):
                data.name = name

            return type(self)._from_validated(
                data=data,
                kind=self.kind,
                domain_kind=type(self.domain).__name__,
                domain_name=self.domain.name,
                index_kind=type(self.index).__name__ if self.index else "Index",
                index_name=getattr(self.index, "name", None),
                name=name,
                **kwargs,
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

            if hasattr(self, "measure") and self.measure is not None:
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
        sig_alg: SigmaAlgebra | None = None,
        kind: Literal["measure", "probability"] = "measure",
        parameter_names: list[Hashable] | None = None,
        parameter_domain_name: Hashable | None = "Theta",
        name: Hashable | None = None,
    ) -> Measure | ParametrizedMeasure:
        """Generate a parametrized measure from the function.

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
        >>> from sigalg.core import Domain, Function, SigmaAlgebra

        Define a 1-dimensional parameter domain and domain, and a sigma-algebra.

        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(domain=X, mapping=dict(zip(X, [0, 1, 1])))

        Define a function on a 2-dimensional domain.

        >>> mapping = {
        ...     (0, 0): 2,  # (theta, F) = (0, 0), ...
        ...     (0, 1): 1,
        ...     (1, 0): 0,
        ...     (1, 1): 3,
        ... }
        >>> f = Function(domain=Theta @ F.atom_space, mapping=mapping)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
                 f
        theta F
        0     0  2
              1  1
        1     0  0
              1  3

        Convert `f` to a parametrized measure.

        >>> mu = f.to_measure(sig_alg=F, parameter_names=["theta"])
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu':
        theta  0  1
        F
        0      2  0
        1      1  3

        Define a second function on a 2-dimensional domain.

        >>> mapping = {
        ...     (0, 0): 0.1,  # (theta, F) = (0, 0), ...
        ...     (0, 1): 0.9,
        ...     (1, 0): 0.0,
        ...     (1, 1): 1.0,
        ... }
        >>> g = Function(domain=Theta @ F.atom_space, mapping=mapping, name="g")
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
                   g
        theta F
        0     0  0.1
              1  0.9
        1     0  0.0
              1  1.0

        Convert `g` to a parametrized probability measure.

        >>> P = g.to_measure(sig_alg=F, kind="probability", parameter_names=["theta"])
        >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'P':
        theta    0    1
        F
        0      0.1  0.0
        1      0.9  1.0

        Define a function on a 1-dimensional domain and convert to a measure.

        >>> h = Function(domain=F.atom_space, mapping=dict(zip(F.atom_space, [1, 2])), name="h")
        >>> print(h)  # doctest: +NORMALIZE_WHITESPACE
        Function 'h':
           h
        F
        0  1
        1  2
        >>> nu = h.to_measure(sig_alg=F, name="nu")
        >>> print(nu)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'nu':
           nu
        F
        0   1
        1   2

        Define another function on a 1-dimensional domain and convert to a probability measure.

        >>> k = Function(
        ...     domain=F.atom_space, mapping=dict(zip(F.atom_space, [0.75, 0.25])), name="k"
        ... )
        >>> print(k)  # doctest: +NORMALIZE_WHITESPACE
        Function 'k':
              k
        F
        0  0.75
        1  0.25
        >>> Q = k.to_measure(sig_alg=F, kind="probability", name="Q")
        >>> print(Q)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'Q':
              Q
        F
        0  0.75
        1  0.25
        """
        from ...validation.mapping_validator import MappingValidator
        from ..measures.measure import Measure
        from ..measures.parametrized_measure import ParametrizedMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if self.domain is not None:
            if name is None:
                name = "mu" if kind == "measure" else "P"

            if parameter_names:
                kind = "param_measure" if kind == "measure" else "param_probability"

                _ = MappingValidator.validate_mapping_kind(
                    data=self.data, kind=kind, parameter_names=parameter_names
                )

                if (
                    self.domain.variable_names
                    != parameter_names + sig_alg.variable_names
                ):
                    raise ValueError(
                        "The variable names of the domain do not match the variable names of the given sigma-algebra."
                    )

                return ParametrizedMeasure._from_validated(
                    data=self.data.rename(name),
                    sig_alg=sig_alg,
                    kind=kind,
                    complete_domain_name=self.domain.name,
                    parameter_domain_name=parameter_domain_name,
                    parameter_names=parameter_names,
                    name=name,
                )

            else:
                if sig_alg is None:
                    sig_alg = SigmaAlgebra.power_set(self.domain)

                _ = MappingValidator.validate_mapping_kind(data=self.data, kind=kind)

                if self.domain.variable_names != sig_alg.variable_names:
                    raise ValueError(
                        "The variable names of the domain do not match the variable names of the given sigma-algebra."
                    )

                return Measure._from_validated(
                    data=self.data.rename(name),
                    kind=kind,
                    sig_alg=sig_alg,
                    name=name,
                )

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

    # --------------------- numpy methods --------------------- #

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> Function:
        """Override numpy ufuncs to operate on `Function` instances.

        Parameters
        ----------
        ufunc : numpy.ufunc
            The ufunc object that was called.
        method : str
            A string indicating which ufunc method was called.
        inputs : tuple
            A tuple of the input arguments to the ufunc.
        kwargs : dict
            A dictionary of keyword arguments passed to the ufunc.

        Returns
        -------
        result : Function
            A new instance of `Function` containing the result of applying the ufunc to the outputs of the function.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import Domain, Function
        >>> X = Domain.from_sequence(size=3)
        >>> f = Function(domain=X, mapping=dict(zip(X, [0, 1, 2])))
        >>> print(np.exp(f))  # doctest: +NORMALIZE_WHITESPACE
        Function 'exp(f)':
             exp(f)
        x
        0  1.000000
        1  2.718282
        2  7.389056
        """
        import pandas as pd

        if method != "__call__" or ufunc.nin != 1 or "out" in kwargs:
            return NotImplemented

        (func,) = inputs
        data = getattr(ufunc, method)(func.data, **kwargs)

        name = f"{ufunc.__name__}({self.name})"

        if isinstance(data, pd.Series):
            data.name = name

        return type(self)._from_validated(
            data=data,
            kind="any",
            domain_kind=type(self.domain).__name__,
            domain_name=self.domain.name,
            index_kind=type(self.index).__name__ if self.index else "Index",
            index_name=self.index.name if self.index else None,
            name=name,
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

    # --------------------- arithmetic operations --------------------- #

    def _apply_binary_operation(
        self,
        other: Function | Real,
        operation: Callable,
        op_symbol: str,
        reverse: bool = False,
        domain_name: Hashable | None = None,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
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
        func : Function
            A new function representing the result of the operation.
        """
        from numbers import Real

        import pandas as pd

        from .._utils.utils import pandas_all_equal, to_df
        from ..indices.index import Index
        from ..indices.time import Time
        from ..measures.measure import Measure

        if isinstance(other, Function):
            if reverse:
                name = f"({other.name} {op_symbol} {self.name})"
            else:
                name = f"({self.name} {op_symbol} {other.name})"

            if isinstance(self.data, pd.Series | pd.DataFrame) and isinstance(
                other.data, pd.Series | pd.DataFrame
            ):
                if self.dimension != other.dimension:
                    raise ValueError(
                        "Cannot add functions whose outputs have different dimensions."
                    )

                if index is None:
                    if self.dimension > 1 and pandas_all_equal(
                        self.index.data, other.index.data
                    ):
                        index = self.index
                    else:
                        index_class = Index if index_kind == "Index" else Time
                        index = index_class.from_sequence(
                            size=self.dimension, name=index_name
                        )

                elif len(index) != self.dimension:
                    raise ValueError(
                        "If given, the length of the index must match the dimension of the functions."
                    )

                index_name = index.name

                if set(self.variable_names) & set(other.variable_names):
                    self_data = to_df(self.data)
                    self_data.columns = index.data
                    other_data = to_df(other.data)
                    other_data.columns = index.data

                    if not reverse:
                        data = operation(self_data, other_data).dropna()
                        if set(self.variable_names) != set(other.variable_names):
                            domain_name = (
                                f"({self.domain_name} int {other.domain_name})"
                            )
                        else:
                            domain_name = self.domain.name

                    else:
                        data = operation(other_data, self_data).dropna()

                        if set(self.variable_names) != set(other.variable_names):
                            domain_name = (
                                domain_name
                                if domain_name
                                else (f"({other.domain_name} int {self.domain_name})")
                            )
                        else:
                            domain_name = (
                                domain_name if domain_name else self.domain.name
                            )

                else:
                    self_data = to_df(self.data, "_self")
                    self_cols = list(self_data.columns)
                    other_data = to_df(other.data, "_other")
                    other_cols = list(other_data.columns)

                    data = pd.merge(
                        left=self_data.reset_index(),
                        right=other_data.reset_index(),
                        how="cross",
                    ).set_index(self.variable_names + other.variable_names)

                    self_data = data[self_cols]
                    self_data.columns = index.data
                    other_data = data[other_cols]
                    other_data.columns = index.data

                    if not reverse:
                        data = operation(self_data, other_data)
                        domain_name = (
                            domain_name
                            if domain_name
                            else f"{self.domain_name} x {other.domain_name}"
                        )
                    else:
                        data = operation(other_data, self_data)
                        domain_name = (
                            domain_name
                            if domain_name
                            else f"{other.domain_name} x {self.domain_name}"
                        )

                if self.dimension == 1:
                    data = data.squeeze(axis=1).rename(name)
                    index_kind = "Index"
                    index_name = None
                else:
                    index_kind = type(index).__name__
                    index_name = index.name

                if isinstance(self, Measure):
                    result_class = Function
                else:
                    result_class = type(self)

                return result_class._from_validated(
                    data=data,
                    kind="any",
                    domain_kind="Domain",
                    domain_name=domain_name,
                    index_kind=index_kind,
                    index_name=index_name,
                    name=name,
                    **kwargs,
                )

            # TODO: check this branch
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

                if isinstance(self, Measure):
                    result_class = Function
                else:
                    result_class = type(self)

                return type(self)._from_validated(
                    data=data,
                    kind="any",
                    domain_kind="Domain",
                    domain_name=None,
                    index_kind="Index",
                    index_name=None,
                    name=name,
                    **kwargs,
                )

        elif isinstance(other, Real):
            if isinstance(self.data, pd.Series | pd.DataFrame):
                if reverse:
                    name = f"({other} {op_symbol} {self.name})"
                    data = operation(other, self.data)
                else:
                    name = f"({self.name} {op_symbol} {other})"
                    data = operation(self.data, other)

                if self.dimension == 1:
                    data = data.rename(name)

                return type(self)._from_validated(
                    data=data,
                    kind="any",
                    domain_kind=self.domain_kind,
                    domain_name=self.domain.name,
                    index_kind=type(self.index).__name__
                    if self.index is not None
                    else "Index",
                    index_name=self.index.name if self.index is not None else None,
                    name=name,
                    **kwargs,
                )

                # return self._apply_binary_operation(
                #     other=other,
                #     operation=operation,
                #     op_symbol=op_symbol,
                #     reverse=reverse,
                #     domain_name=domain_name,
                #     index=index,
                #     index_kind=index_kind,
                #     index_name=index_name,
                #     name=name,
                #     **kwargs,
                # )

            elif isinstance(self.data, Callable):
                raise NotImplementedError(
                    "Adding functions without data to scalars it not implemented yet."
                )

        else:
            raise TypeError(
                f"Unsupported operand type(s) for {op_symbol}: 'Function' and '{type(other).__name__}'"
            )

    def __add__(
        self,
        other: Function | Real,
        domain_name: Hashable | None = None,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        """Add another `Function` instance or scalar to this function.

        Parameters
        ----------
        other : Function | Real
            The object to add to the current function.
        domain_name : Hashable | None, default=None
            The name of the domain on which the sum is defined. If `None`, a default will be generated.
        index : Index | None, default=None
            An optional custom index for the sum, provided that the dimensions of the functions are > 1. If `None`, a default will be generated.
        index_kind : Literal["Index", "Time"], default="Index"
            The type of index of the sum, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        index_name : Hashable | None, default=None
            The name of the index of the sum, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        name : Hashable | None, default=None
            The name of the result. If `None`, a default will be generated.
        kwargs : dict
            Keyword arguments for subclasses.

        Returns
        -------
        sum : Function
            The sum of the function and the object.
        """
        return self._apply_binary_operation(
            other=other,
            operation=lambda a, b: a + b,
            op_symbol="+",
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            **kwargs,
        )

    def add(
        self,
        other: Function | Real,
        domain_name: Hashable | None = None,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        """Add another `Function` instance or scalar to this function.

        Parameters
        ----------
        other : Function | Real
            The object to add to the current function.
        domain_name : Hashable | None, default=None
            The name of the domain on which the sum is defined. If `None`, a default will be generated.
        index : Index | None, default=None
            An optional custom index for the sum, provided that the dimensions of the functions are > 1. If `None`, a default will be generated.
        index_kind : Literal["Index", "Time"], default="Index"
            The type of index of the sum, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        index_name : Hashable | None, default=None
            The name of the index of the sum, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        name : Hashable | None, default=None
            The name of the result. If `None`, a default will be generated.
        kwargs : dict
            Keyword arguments for subclasses.

        Returns
        -------
        sum : Function
            The sum of the function and the object.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import Domain, Function, Index, Time
        >>> rng = np.random.default_rng(42)

        Define a 2-dimensional domain, and two functions with 2-dimensional outputs and custom indices.

        >>> X = Domain([(1, 2), (3, 4), (5, 6)], variable_names=["u", "v"])
        >>> f = Function.from_rand(domain=X, dim=2, random_state=rng)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
        i    0  1
        u v
        1 2  0  7
        3 4  6  4
        5 6  4  8
        >>> g = Function.from_rand(domain=X, dim=2, name="g", random_state=rng)
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
        i    0  1
        u v
        1 2  0  6
        3 4  2  0
        5 6  5  9

        The domains and indices of the functions are fully aligned and the functions have the same dimension, so adding the functions produces the expected result.

        >>> print(f.add(g))  # doctest: +NORMALIZE_WHITESPACE
        Function '(f + g)':
        i    0   1
        u v
        1 2  0  13
        3 4  8   4
        5 6  9  17

        If the functions had different indices, but still the same dimension, it is still possible to add them. The index of the result will reset to a default.

        >>> J = Index([1, 2], variable_names=["j"], name="J")
        >>> f = Function.from_rand(domain=X, dim=2, index=J, random_state=rng)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
        j    1  2
        u v
        1 2  7  7
        3 4  7  7
        5 6  5  1
        >>> K = Index([3, 4], variable_names=["k"], name="K")
        >>> g = Function.from_rand(domain=X, dim=2, index=K, name="g", random_state=rng)
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
        k    3  4
        u v
        1 2  8  4
        3 4  5  3
        5 6  1  9
        >>> print(f.add(g))  # doctest: +NORMALIZE_WHITESPACE
        Function '(f + g)':
        i     0   1
        u v
        1 2  15  11
        3 4  12  10
        5 6   6  10

        One may also pass in a custom index.

        >>> T = Time.discrete(length=1, start=5)
        >>> print(f.add(g, index=T))  # doctest: +NORMALIZE_WHITESPACE
        Function '(f + g)':
        t     5   6
        u v
        1 2  15  11
        3 4  12  10
        5 6   6  10

        For functions whose domains are only partially aligned, the `add` method performs a merge on the common variable names and then adds. We demonstrate with functions with 1-dimensional outputs.

        >>> Y = Domain([(2, 7), (4, 8)], variable_names=["v", "w"], name="Y")
        >>> f = Function.from_rand(domain=X, dim=1, random_state=rng)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
             f
        u v
        1 2  7
        3 4  6
        5 6  4
        >>> g = Function.from_rand(domain=Y, dim=1, name="g", random_state=rng)
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
             g
        v w
        2 7  8
        4 8  5
        >>> print(f.add(g))  # doctest: +NORMALIZE_WHITESPACE
        Function '(f + g)':
               (f + g)
        u v w
        1 2 7     15.0
        3 4 8     11.0

        For functions whose domains have completely disjoint variable names, the `add` method forms the Cartesian product of the domains and then adds.

        >>> Y = Domain([(2, 7), (4, 8)], variable_names=["a", "b"], name="Y")
        >>> f = Function.from_rand(domain=X, dim=1, random_state=rng)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
             f
        u v
        1 2  4
        3 4  4
        5 6  2
        >>> g = Function.from_rand(domain=Y, dim=1, name="g", random_state=rng)
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
             g
        a b
        2 7  0
        4 8  5
        >>> print(f.add(g))  # doctest: +NORMALIZE_WHITESPACE
        Function '(f + g)':
                 (f + g)
        u v a b
        1 2 2 7        4
            4 8        9
        3 4 2 7        4
            4 8        9
        5 6 2 7        2
            4 8        7

        Finally, it is possible to add scalars to instances of `Function`.

        >>> print(f.add(4))  # doctest: +NORMALIZE_WHITESPACE
        Function '(f + 4)':
             (f + 4)
        u v
        1 2        8
        3 4        8
        5 6        6
        """
        return self.__add__(
            other=other,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            **kwargs,
        )

    def __sub__(
        self,
        other: Function | Real,
        domain_name: Hashable | None = None,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        """Subtract another `Function` instance or scalar from this function.

        Parameters
        ----------
        other : Function | Real
            The object to subtract from the current function.
        domain_name : Hashable | None, default=None
            The name of the domain on which the difference is defined. If `None`, a default will be generated.
        index : Index | None, default=None
            An optional custom index for the difference, provided that the dimensions of the functions are > 1. If `None`, a default will be generated.
        index_kind : Literal["Index", "Time"], default="Index"
            The type of index of the difference, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        index_name : Hashable | None, default=None
            The name of the index of the difference, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        name : Hashable | None, default=None
            The name of the result. If `None`, a default will be generated.
        kwargs : dict
            Keyword arguments for subclasses.

        Returns
        -------
        difference : Function
            The difference of the function and the object.
        """
        return self._apply_binary_operation(
            other=other,
            operation=lambda a, b: a - b,
            op_symbol="-",
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            **kwargs,
        )

    def subtract(
        self,
        other: Function | Real,
        domain_name: Hashable | None = None,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        """Subtract another `Function` instance or scalar from this function.

        Parameters
        ----------
        other : Function | Real
            The object to subtract from the current function.
        domain_name : Hashable | None, default=None
            The name of the domain on which the difference is defined. If `None`, a default will be generated.
        index : Index | None, default=None
            An optional custom index for the difference, provided that the dimensions of the functions are > 1. If `None`, a default will be generated.
        index_kind : Literal["Index", "Time"], default="Index"
            The type of index of the difference, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        index_name : Hashable | None, default=None
            The name of the index of the difference, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        name : Hashable | None, default=None
            The name of the result. If `None`, a default will be generated.
        kwargs : dict
            Keyword arguments for subclasses.

        Returns
        -------
        difference : Function
            The difference of the function and the object.
        """
        return self.__sub__(
            other=other,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            **kwargs,
        )

    def __mul__(
        self,
        other: Function | Real,
        domain_name: Hashable | None = None,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        """Multiply another `Function` instance or scalar with this function.

        Parameters
        ----------
        other : Function | Real
            The object to multiply with the current function.
        domain_name : Hashable | None, default=None
            The name of the domain on which the product is defined. If `None`, a default will be generated.
        index : Index | None, default=None
            An optional custom index for the product, provided that the dimensions of the functions are > 1. If `None`, a default will be generated.
        index_kind : Literal["Index", "Time"], default="Index"
            The type of index of the product, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        index_name : Hashable | None, default=None
            The name of the index of the product, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        name : Hashable | None, default=None
            The name of the result. If `None`, a default will be generated.
        kwargs : dict
            Keyword arguments for subclasses.

        Returns
        -------
        product : Function
            The product of the function and the object.
        """
        return self._apply_binary_operation(
            other=other,
            operation=lambda a, b: a * b,
            op_symbol="*",
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            **kwargs,
        )

    def multiply(
        self,
        other: Function | Real,
        domain_name: Hashable | None = None,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        """Multiply another `Function` instance or scalar with this function.

        Parameters
        ----------
        other : Function | Real
            The object to multiply with the current function.
        domain_name : Hashable | None, default=None
            The name of the domain on which the product is defined. If `None`, a default will be generated.
        index : Index | None, default=None
            An optional custom index for the product, provided that the dimensions of the functions are > 1. If `None`, a default will be generated.
        index_kind : Literal["Index", "Time"], default="Index"
            The type of index of the product, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        index_name : Hashable | None, default=None
            The name of the index of the product, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        name : Hashable | None, default=None
            The name of the result. If `None`, a default will be generated.
        kwargs : dict
            Keyword arguments for subclasses.

        Returns
        -------
        product : Function
            The product of the function and the object.
        """
        return self.__mul__(
            other=other,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            **kwargs,
        )

    def __truediv__(
        self,
        other: Function | Real,
        domain_name: Hashable | None = None,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        """Divide this function by another `Function` instance or scalar.

        Parameters
        ----------
        other : Function | Real
            The divisor.
        domain_name : Hashable | None, default=None
            The name of the domain on which the quotient is defined. If `None`, a default will be generated.
        index : Index | None, default=None
            An optional custom index for the quotient, provided that the dimensions of the functions are > 1. If `None`, a default will be generated.
        index_kind : Literal["Index", "Time"], default="Index"
            The type of index of the quotient, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        index_name : Hashable | None, default=None
            The name of the index of the quotient, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        name : Hashable | None, default=None
            The name of the result. If `None`, a default will be generated.
        kwargs : dict
            Keyword arguments for subclasses.

        Returns
        -------
        quotient : Function
            The quotient of the function and the object.
        """
        return self._apply_binary_operation(
            other=other,
            operation=lambda a, b: a / b,
            op_symbol="/",
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
        )

    def divide(
        self,
        other: Function | Real,
        domain_name: Hashable | None = None,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        """Divide this function by another `Function` instance or scalar.

        Parameters
        ----------
        other : Function | Real
            The divisor.
        domain_name : Hashable | None, default=None
            The name of the domain on which the quotient is defined. If `None`, a default will be generated.
        index : Index | None, default=None
            An optional custom index for the quotient, provided that the dimensions of the functions are > 1. If `None`, a default will be generated.
        index_kind : Literal["Index", "Time"], default="Index"
            The type of index of the quotient, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        index_name : Hashable | None, default=None
            The name of the index of the quotient, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        name : Hashable | None, default=None
            The name of the result. If `None`, a default will be generated.
        kwargs : dict
            Keyword arguments for subclasses.

        Returns
        -------
        quotient : Function
            The quotient of the function and the object.
        """
        return self.__truediv__(
            other=other,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            **kwargs,
        )

    def __pow__(
        self,
        other: Function | Real,
        domain_name: Hashable | None = None,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        """Raise this function to the power of another `Function` instance or a scalar.

        Parameters
        ----------
        other : Function | Real
            The power.
        domain_name : Hashable | None, default=None
            The name of the domain on which the power is defined. If `None`, a default will be generated.
        index : Index | None, default=None
            An optional custom index for the power, provided that the dimensions of the functions are > 1. If `None`, a default will be generated.
        index_kind : Literal["Index", "Time"], default="Index"
            The type of index of the power, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        index_name : Hashable | None, default=None
            The name of the index of the power, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        name : Hashable | None, default=None
            The name of the result. If `None`, a default will be generated.
        kwargs : dict
            Keyword arguments for subclasses.

        Returns
        -------
        power : Function
            The power of the function and the object.
        """
        return self._apply_binary_operation(
            other=other,
            operation=lambda a, b: self._to_float(a) ** self._to_float(b),
            op_symbol="**",
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
        )

    def expon(
        self,
        other: Function | Real,
        domain_name: Hashable | None = None,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        """Raise this function to the power of another `Function` instance or a scalar.

        Parameters
        ----------
        other : Function | Real
            The power.
        domain_name : Hashable | None, default=None
            The name of the domain on which the power is defined. If `None`, a default will be generated.
        index : Index | None, default=None
            An optional custom index for the power, provided that the dimensions of the functions are > 1. If `None`, a default will be generated.
        index_kind : Literal["Index", "Time"], default="Index"
            The type of index of the power, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        index_name : Hashable | None, default=None
            The name of the index of the power, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        name : Hashable | None, default=None
            The name of the result. If `None`, a default will be generated.
        kwargs : dict
            Keyword arguments for subclasses.

        Returns
        -------
        power : Function
            The power of the function and the object.
        """
        return self.__pow__(
            other=other,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            **kwargs,
        )

    def __neg__(self, name: Hashable | None = None, **kwargs) -> Function:
        """Negate this function.

        Parameters
        ----------
        name : Hashable | None, default=None
            The name of the result. If `None`, a default will be generated.
        kwargs : dict
            Keyword arguments for subclasses.

        Returns
        -------
        negation : Function
            The negation of the function.
        """
        import pandas as pd

        if name is None:
            name = f"(-{self.name})"

        if isinstance(self.data, pd.Series | pd.DataFrame):
            return type(self)._from_validated(
                data=-self.data.rename(name) if self.dimension == 1 else -self.data,
                kind=self.kind,
                domain_kind=self.domain_kind,
                domain_name=self.domain.name,
                index_kind=None,
                index_name=None,
                name=name,
                **kwargs,
            )

        elif isinstance(self.data, Callable):
            sig = inspect.signature(self.data)

            def data(**kwargs):
                bound = sig.bind(**kwargs)
                return -self.data(**bound.arguments)

            data.__signature__ = sig

            return type(self)._from_validated(
                data=data,
                kind="any",
                domain_kind=None,
                domain_name=None,
                index_kind=None,
                index_name=None,
                name=name,
                **kwargs,
            )

    def negate(self, name: Hashable | None = None, **kwargs) -> Function:
        """Negate this function.

        Parameters
        ----------
        name : Hashable | None, default=None
            The name of the result. If `None`, a default will be generated.
        kwargs : dict
            Keyword arguments for subclasses.

        Returns
        -------
        negation : Function
            The negation of the function.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import Domain, Function
        >>> rng = np.random.default_rng(42)
        >>> X = Domain([(1, 2), (3, 4), (5, 6)], variable_names=["u", "v"])
        >>> f = Function.from_rand(domain=X, dim=2, random_state=rng)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
        i    0  1
        u v
        1 2  0  7
        3 4  6  4
        5 6  4  8
        >>> print(-f)  # doctest: +NORMALIZE_WHITESPACE
        Function '(-f)':
        i    0  1
        u v
        1 2  0 -7
        3 4 -6 -4
        5 6 -4 -8
        >>> g = Function.from_rand(domain=X, dim=1, name="g", random_state=rng)
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
             g
        u v
        1 2  0
        3 4  6
        5 6  2
        >>> print(-g)  # doctest: +NORMALIZE_WHITESPACE
        Function '(-g)':
             (-g)
        u v
        1 2     0
        3 4    -6
        5 6    -2
        """
        return self.__neg__(name=name, **kwargs)

    def __radd__(  # noqa: D105
        self,
        other: Function | Real,
        domain_name: Hashable | None = None,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        return self._apply_binary_operation(
            other=other,
            operation=lambda a, b: a + b,
            op_symbol="+",
            reverse=True,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            **kwargs,
        )

    def __rsub__(  # noqa: D105
        self,
        other: Function | Real,
        domain_name: Hashable | None = None,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        return self._apply_binary_operation(
            other=other,
            operation=lambda a, b: a - b,
            op_symbol="-",
            reverse=True,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            **kwargs,
        )

    def __rmul__(  # noqa: D105
        self,
        other: Function | Real,
        domain_name: Hashable | None = None,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        return self._apply_binary_operation(
            other=other,
            operation=lambda a, b: a * b,
            op_symbol="*",
            reverse=True,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            **kwargs,
        )

    def __rtruediv__(  # noqa: D105
        self,
        other: Function | Real,
        domain_name: Hashable | None = None,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        return self._apply_binary_operation(
            other=other,
            operation=lambda a, b: a / b,
            op_symbol="/",
            reverse=True,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            **kwargs,
        )

    def __rpow__(  # noqa: D105
        self,
        other: Function | Real,
        domain_name: Hashable | None = None,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        return self._apply_binary_operation(
            other=other,
            operation=lambda a, b: self._to_float(a) ** self._to_float(b),
            op_symbol="**",
            reverse=True,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            **kwargs,
        )

    @staticmethod
    def _to_float(x):
        return x.astype(float) if hasattr(x, "astype") else float(x)

    # --------------------- comparison methods --------------------- #

    def __bool__(self) -> bool:
        """Prevent ambiguous boolean conversion of a function."""
        raise ValueError(
            "The truth value of a Function is ambiguous. Use 'all' or 'any' methods."
        )

    def all(self) -> bool:
        """Check if all outputs of the function are `True`.

        Returns
        -------
        all_true : bool
            `True` if all outputs are `True`.

        Examples
        --------
        >>> from sigalg.core import Domain, Function
        >>> X = Domain.from_sequence(size=2)
        >>> f = Function(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 1),
        ...         1: (1, 1),
        ...     },
        ... )
        >>> f.all()
        True
        >>> g = Function(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 0),
        ...         1: (0, 1),
        ...     },
        ...     name="g",
        ... )
        >>> g.all()
        False
        """
        return bool(self.data.all().all() if self.dimension > 1 else self.data.all())

    def any(self) -> bool:
        """Check if any output of the function is `True`.

        Returns
        -------
        any_true : bool
            `True` if any output is `True`.

        Examples
        --------
        >>> from sigalg.core import Domain, Function
        >>> X = Domain.from_sequence(size=2)
        >>> f = Function(
        ...     domain=X,
        ...     mapping={
        ...         0: (0, 1),
        ...         1: (1, 0),
        ...     },
        ... )
        >>> f.any()
        True
        >>> g = Function(
        ...     domain=X,
        ...     mapping={
        ...         0: (0, 0),
        ...         1: (0, 0),
        ...     },
        ...     name="g",
        ... )
        >>> g.any()
        False
        """
        return bool(self.data.any().any() if self.dimension > 1 else self.data.any())

    def _apply_comparison(
        self,
        other: Function | Real,
        operation: Callable,
        op_symbol: str,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        """Apply a comparison operation to this measurable vector.

        Parameters
        ----------
        other : MeasurableVector | Real
            The measurable vector or scalar to compare with.
        op : Callable
            The numpy comparison to apply (e.g., ``operator.lt``).
        op_symbol : str
            Symbol representing the comparison (e.g., '<', '<=', '>', '>=').

        Returns
        -------
        result : MeasurableVector
            A new measurable vector of booleans representing the comparison result.

        Raises
        ------
        TypeError
            If `other` is not a `MeasurableVector` or scalar.
        ValueError
            If the measurable vectors do not have the same domain or dimension.
        """
        import pandas as pd

        from .._utils.utils import pandas_all_equal
        from ..indices.index import Index
        from ..indices.time import Time

        if isinstance(self.data, pd.Series | pd.DataFrame) and isinstance(
            other.data, pd.Series | pd.DataFrame
        ):
            if self.dimension != other.dimension:
                raise ValueError(
                    "Cannot compare functions whose outputs have different dimensions."
                )
            if not pandas_all_equal(self.data.index, other.data.index):
                raise ValueError(
                    "Cannot compare two functions whose domains are not exactly aligned (same order, same variable names)."
                )

            if index is None:
                if self.dimension > 1 and pandas_all_equal(
                    self.index.data, other.index.data
                ):
                    index = self.index
                else:
                    index_class = Index if index_kind == "Index" else Time
                    index = index_class.from_sequence(
                        size=self.dimension, name=index_name
                    )

            elif len(index) != self.dimension:
                raise ValueError(
                    "If given, the length of the index must match the dimension of the functions."
                )

            index_name = index.name

            arr = operation(self.to_numpy(), other.to_numpy())

            if name is None:
                name = f"({self.name} {op_symbol} {other.name})"

            if self.dimension > 1:
                data = pd.DataFrame(
                    arr, index=self.domain.data, columns=index.data, dtype=int
                )
            else:
                data = pd.Series(
                    arr,
                    index=self.domain.data,
                    dtype=int,
                    name=name,
                )

            return type(self)._from_validated(
                data=data,
                kind=self.kind,
                domain_kind=type(self.domain).__name__,
                domain_name=self.domain.name,
                index_kind=type(index).__name__,
                index_name=index.name,
                name=name,
                **kwargs,
            )

        else:
            raise NotImplementedError(
                "Comparison of functions with empty data is not implemented."
            )

    def __lt__(
        self,
        other: Function | Real,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        """Compare this function through pointwise `<` with another `Function` or scalar.

        Parameters
        ----------
        other : Function | Real
            The function or scalar to compare against.
        index : Index | None, default=None
            An optional custom index for the result, provided that the dimensions of the functions are > 1. If `None`, a default will be generated.
        index_kind : Literal["Index", "Time"], default="Index"
            The type of index of the result, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        index_name : Hashable | None, default=None
            The name of the index of the result, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        name : Hashable | None, default=None
            The name of the result. If `None`, a default will be generated.
        kwargs : dict
            Keyword arguments for subclasses.

        Returns
        -------
        comparison : Function
            A new `Function` of `0`s and `1`s indicating where this function's outputs are less than the outputs of the other function or scalar.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import Domain, Function
        >>> rng = np.random.default_rng(42)
        >>> X = Domain([(1, 2), (3, 4), (5, 6)], variable_names=["u", "v"])
        >>> f = Function.from_rand(domain=X, dim=2, random_state=rng)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
        i    0  1
        u v
        1 2  0  7
        3 4  6  4
        5 6  4  8
        >>> g = Function.from_rand(domain=X, dim=2, name="g", random_state=rng)
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Function 'g':
        i    0  1
        u v
        1 2  0  6
        3 4  2  0
        5 6  5  9
        >>> print(f <= g)  # doctest: +NORMALIZE_WHITESPACE
        Function '(f <= g)':
        i    0  1
        u v
        1 2  1  0
        3 4  0  0
        5 6  1  1
        """
        import operator

        return self._apply_comparison(
            other=other,
            operation=operator.lt,
            op_symbol="<",
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            **kwargs,
        )

    def __le__(
        self,
        other: Function | Real,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        """Compare this function through pointwise `<=` with another `Function` or scalar.

        Parameters
        ----------
        other : Function | Real
            The function or scalar to compare against.
        index : Index | None, default=None
            An optional custom index for the result, provided that the dimensions of the functions are > 1. If `None`, a default will be generated.
        index_kind : Literal["Index", "Time"], default="Index"
            The type of index of the result, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        index_name : Hashable | None, default=None
            The name of the index of the result, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        name : Hashable | None, default=None
            The name of the result. If `None`, a default will be generated.
        kwargs : dict
            Keyword arguments for subclasses.

        Returns
        -------
        comparison : Function
            A new `Function` of `0`s and `1`s indicating where this function's outputs are less than or equal to the outputs of the other function or scalar.
        """
        import operator

        return self._apply_comparison(
            other=other,
            operation=operator.le,
            op_symbol="<=",
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            **kwargs,
        )

    def __gt__(
        self,
        other: Function | Real,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        """Compare this function through pointwise `>` with another `Function` or scalar.

        Parameters
        ----------
        other : Function | Real
            The function or scalar to compare against.
        index : Index | None, default=None
            An optional custom index for the result, provided that the dimensions of the functions are > 1. If `None`, a default will be generated.
        index_kind : Literal["Index", "Time"], default="Index"
            The type of index of the result, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        index_name : Hashable | None, default=None
            The name of the index of the result, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        name : Hashable | None, default=None
            The name of the result. If `None`, a default will be generated.
        kwargs : dict
            Keyword arguments for subclasses.

        Returns
        -------
        comparison : Function
            A new `Function` of `0`s and `1`s indicating where this function's outputs are greater than the outputs of the other function or scalar.
        """
        import operator

        return self._apply_comparison(
            other=other,
            operation=operator.gt,
            op_symbol=">",
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            **kwargs,
        )

    def __ge__(
        self,
        other: Function | Real,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
        **kwargs,
    ) -> Function:
        """Compare this function through pointwise `>=` with another `Function` or scalar.

        Parameters
        ----------
        other : Function | Real
            The function or scalar to compare against.
        index : Index | None, default=None
            An optional custom index for the result, provided that the dimensions of the functions are > 1. If `None`, a default will be generated.
        index_kind : Literal["Index", "Time"], default="Index"
            The type of index of the result, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        index_name : Hashable | None, default=None
            The name of the index of the result, provided that the dimensions of the functions are > 1. If `index` is not `None`, this parameter is ignored.
        name : Hashable | None, default=None
            The name of the result. If `None`, a default will be generated.
        kwargs : dict
            Keyword arguments for subclasses.

        Returns
        -------
        comparison : Function
            A new `Function` of `0`s and `1`s indicating where this function's outputs are greater than or equal to the outputs of the other function or scalar.
        """
        import operator

        return self._apply_comparison(
            other=other,
            operation=operator.ge,
            op_symbol=">=",
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            **kwargs,
        )
