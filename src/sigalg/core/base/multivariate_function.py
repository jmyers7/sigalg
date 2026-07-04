"""A class representing a multivariate function."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Hashable
from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ...validation.mapping_validator import MappingLike
    from ..base.sample_space import SampleSpace
    from ..probability_measures.parametrized_probability_measure import (
        ParametrizedProbabilityMeasure,
    )
    from ..probability_measures.probability_measure import ProbabilityMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from .domain import Domain


class MultivariateFunction:
    """A class representing a multivariate function.

    Mathematically, a function requires three items: A domain set, a codomain set, and a rule defining the function. For instances of `MultivariateFunction`:

    * The domain of the function is passed as the parameter `domain`, but this parameter is *not* required (e.g., cases in which the domain is continuous).
    * The codomain of an instance of `MultivariateFunction` is always assumed to be the set of real numbers.
    * The rule defining the function may be passed into the constructor as the parameter `mapping`. If `mapping` is a callable, its parameters *must* be keyword-only.

    Parameters
    ----------
    domain : Domain | None, default=None
        The domain of the function.
    mapping : MappingLike | Callable | None, default=None
        The underlying rule defining the function.
    output_name: Hashable, default="output"
        The name of the outputs of the function.
    name : Hashable | None, default=None
        The name of the function. If `None`, a default name of `f` will be used.
    kind : Literal["any", "probabilities"], default="any"
        The kind of outputs of the function. The parameter `probabilities` is meant to be used by probability measures.
    **kwargs
        Additional keyword arguments passed to subclasses.

    Examples
    --------
    Define a `MultivariateFunction` with an explict value for `domain` and a `mapping` expressed as a lambda function. Note that the parameters to the lambda function are keyword-only.

    >>> import pandas as pd
    >>> from sigalg.core import Domain, MultivariateFunction
    >>> D = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
    >>> f = MultivariateFunction(domain=D, mapping=lambda *, x, y: 2 * x + y**2)
    >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
    Function 'f':
          output
    x y
    1 2        6
    2 3       13
    1 4       18

    Define a function from a `pd.Series` object and an explict `domain`.

    >>> mapping = pd.Series([6, 13, 18], index=D.data)
    >>> g = MultivariateFunction(domain=D, mapping=mapping, name="g")
    >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
    Function 'g':
          output
    x y
    1 2        6
    2 3       13
    1 4       18

    Define a function from a dictionary and an explict `domain`.

    >>> mapping = {(1, 2): 6, (2, 3): 13, (1, 4): 18}
    >>> h = MultivariateFunction(domain=D, mapping=mapping, name="h")
    >>> print(h)  # doctest: +NORMALIZE_WHITESPACE
    Function 'h':
          output
    x y
    1 2        6
    2 3       13
    1 4       18

    Define a function from a lambda function without an explict `domain`. We no longer can print out the range of the function, but we can evaluate the function.

    >>> k = MultivariateFunction(mapping=lambda *, x, y: x * 2 + y, name="k")
    >>> print(k)
    Function 'k(x, y)'
    >>> print(k(x=2, y=1))
    5
    """

    _default_name = "f"
    _properties = ["_dict"]
    _repr_name = "Function"

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        domain: Domain | None = None,
        mapping: MappingLike | Callable | None = None,
        output_name: Hashable = "output",
        name: Hashable | None = None,
        kind: Literal["any", "probabilities"] = "any",
        **kwargs,
    ) -> None:
        from ...validation.mapping_validator import MappingValidator

        if name is None:
            name = type(self)._default_name

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
        self._argument_names = v.argument_names
        self._num_arguments = v.num_arguments

        try:
            self._fun = v.fun
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
    def cartesian_power(cls, fun: MultivariateFunction, n: int) -> MultivariateFunction:
        r"""Get the Cartesian power of the function.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        fun : MultivariateFunction
            The base of the Cartesian power.
        n : int
            The power of the Cartesian power.

        Raises
        ------
        TypeError
            If `n` is not an integer or `fun` is not a `MultivariateFunction`.
        ValueError
            If `n` is not positive.

        Returns
        -------
        cartesian_power : MultivariateFunction
            The Cartesian power.

        Examples
        --------
        Define a function.

        >>> from sigalg.core import Domain, MultivariateFunction
        >>> D = Domain([1, 2, 3], variable_names=["x"])
        >>> f = MultivariateFunction(
        ...     domain=D,
        ...     mapping=lambda *, x: x**2,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
           output
        x
        1       1
        2       4
        3       9

        Compute the second Cartesian power using the `cartesian_power` class method.

        >>> print(MultivariateFunction.cartesian_power(f, 2))  # doctest: +NORMALIZE_WHITESPACE
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

        Compute the third Cartesian power using the `^` operator notation.

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

        Notes
        -----
        Let $f:D \to \mathbb{R}$ be a function and let $n$ be a positive integer. Then *$n$-th Cartesian power$ of $f$ is the function

        $$
        f^n : D^n \to \mathbb{R}
        $$

        given by

        $$
        f(x_1,x_2,\ldots,x_n) = f(x_1)f(x_2) \cdots f(x_n),
        $$

        where $D^n$ denotes the $n$-th Cartesian power of the domain $D$.
        """
        from ..probability_measures.probability_measure import ProbabilityMeasure

        if not isinstance(fun, MultivariateFunction):
            raise TypeError("fun must be a MultivariateFunction")
        if not isinstance(n, int):
            raise TypeError("n must be an integer")
        if n <= 0:
            raise ValueError("n must be positive")

        if fun.data is not None:
            variable_names = list(fun.data.index.names)
            reset_data = []
            product_variable_names = []

            for k in range(n):
                reset_data.append(fun.data.reset_index().add_suffix(f"_{k}"))
                product_variable_names += [f"{name}_{k}" for name in variable_names]

            power_data = reset_data[0]

            for data in reset_data[1:]:
                power_data = pd.merge(
                    left=power_data,
                    right=data,
                    how="cross",
                )
            power_data = (
                power_data.set_index(product_variable_names)
                .prod(axis=1)
                .rename(fun.output_name)
            )

            result = type(fun)(
                domain=fun.domain ^ n,
                mapping=power_data,
                output_name=fun.output_name,
                name=f"{fun.name} ^ {n}",
            )

            if isinstance(fun, ProbabilityMeasure):
                result._sig_alg = fun.sig_alg ^ n

            return result

        else:
            return NotImplementedError(
                "The `cartesian_power` method is not yet implemneted for functions without explicit domains."
            )

    def __xor__(self, n: int) -> MultivariateFunction:
        """Form the Cartesian power of this instance of `MultivariateFunction`.

        Internally calls the `cartesian_power` method.

        Parameters
        ----------
        n : int
            The power of the Cartesian power.

        Returns
        -------
        cartesian_power : MultivariateFunction
            The Cartesian power.
        """
        return type(self).cartesian_power(fun=self, n=n)

    # --------------------- properties --------------------- #

    @property
    def fun(self) -> Callable | None:
        """Get the underlying callable function.

        Returns
        -------
        function : Callable | None
            The underlying callable function if defined, otherwise `None`.

        Examples
        --------
        >>> import pandas as pd
        >>> from sigalg.core import Domain, MultivariateFunction
        >>> D = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> mapping = pd.Series([6, 13, 18], index=D.data)
        >>> f = MultivariateFunction(domain=D, mapping=mapping)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
              output
        x y
        1 2        6
        2 3       13
        1 4       18
        >>> print(f.fun(x=1, y=2))
        6
        """
        return self._fun

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
        >>> D = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> f = MultivariateFunction(domain=D, mapping=lambda *, x, y: 2 * x + y**2)
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
        >>> D = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> f = MultivariateFunction(domain=D, mapping=lambda *, x, y: 2 * x + y**2)
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
    def argument_names(self) -> list[Hashable] | None:
        """Get the argument names of the function.

        Returns
        -------
        argument_names : list[Hashable] | None
            The argument names of the function if define, otherwise `None`.

        Examples
        --------
        >>> from sigalg.core import Domain, MultivariateFunction
        >>> D = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> f = MultivariateFunction(domain=D, mapping=lambda *, x, y: 2 * x + y**2)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
              output
        x y
        1 2        6
        2 3       13
        1 4       18
        >>> print(f.argument_names)
        ['x', 'y']
        """
        return self._argument_names

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
        >>> D = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> f = MultivariateFunction(domain=D, mapping=lambda *, x, y: 2 * x + y**2)
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
    def num_arguments(self) -> int | None:
        """Get the number of arguments of the function.

        Returns
        -------
        num_arguments : int | None
            The number of arguments of the function if defined, otherwise `None`.

        Examples
        --------
        >>> from sigalg.core import Domain, MultivariateFunction
        >>> D = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> f = MultivariateFunction(domain=D, mapping=lambda *, x, y: 2 * x + y**2)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
              output
        x y
        1 2        6
        2 3       13
        1 4       18
        >>> print(f.num_arguments)
        2
        """
        return self._num_arguments

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
        >>> D = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> f = MultivariateFunction(domain=D, mapping=lambda *, x, y: 2 * x + y**2)
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
              output
        x y
        1 2        6
        2 3       13
        1 4       18
        >>> print(f.domain)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'D':
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
        >>> D = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> g = MultivariateFunction(domain=D, mapping=lambda *, x, y: 2 * x + y**2, name="g")
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
        >>> D = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> f = MultivariateFunction(domain=D, mapping=lambda *, x, y: 2 * x + y**2)
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
        >>> D = Domain([(1, 2), (2, 3), (1, 4)], variable_names=["x", "y"])
        >>> f = MultivariateFunction(domain=D, mapping=lambda *, x, y: 2 * x + y**2)
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
        from .domain import Domain

        specified_arguments = self.signature.bind_partial(**kwargs)
        unspecified_arguments = [
            inspect.Parameter(parameter, inspect.Parameter.KEYWORD_ONLY)
            for parameter in self.argument_names
            if parameter not in specified_arguments.arguments.keys()
        ]

        if len(unspecified_arguments) == 0:
            return self.fun(**specified_arguments.arguments)
        else:
            partial_signature = inspect.Signature(unspecified_arguments)

            def partial_function(*args, **kwargs):
                partial_parameters = partial_signature.bind(*args, **kwargs)
                all_args = {
                    **specified_arguments.arguments,
                    **partial_parameters.arguments,
                }
                return self.fun(**all_args)

            partial_function.__signature__ = partial_signature

            name = f"{self.name}({', '.join(f'{p}={specified_arguments.arguments[p]}' for p in self.argument_names if p in specified_arguments.arguments)})"

            if self.data is not None:
                try:
                    data = self.data.xs(
                        key=tuple(specified_arguments.arguments.values()),
                        level=tuple(specified_arguments.arguments.keys()),
                    ).index
                    domain_name = f"{self.domain.name}({', '.join(f'{p}={specified_arguments.arguments[p]}' for p in self.argument_names if p in specified_arguments.arguments)})"
                    partial_domain = Domain(indices=data, name=domain_name)

                except KeyError:
                    partial_domain = None

            else:
                partial_domain = None

            return MultivariateFunction(
                domain=partial_domain,
                name=name,
                mapping=partial_function,
                output_name=self.output_name,
            )

    # --------------------- conversion methods --------------------- #

    def to_prob_measure(
        self,
        sig_alg: SigmaAlgebra | None = None,
        sample_space: SampleSpace | None = None,
        name: Hashable | None = None,
        in_place: bool = False,
    ) -> ParametrizedProbabilityMeasure | ProbabilityMeasure:
        """Generate a parametrized probability measure from the multivariate function.

        Examples
        --------
        Define the domain for the multivariate function as a Cartesian product. The variable `theta` will serve as the parameter of the subsequent parametrized probability measure, while `x` and `y` will serve as coordinates on a sample space.

        >>> from sigalg.core import (
        ...     Domain,
        ...     MultivariateFunction,
        ...     SampleSpace,
        ... )
        >>> D_theta = Domain([0.0, 0.25, 0.5, 0.75, 1.0], variable_names=["theta"],  name="D_theta")
        >>> D_x = Domain([0, 1], variable_names=["x"], name="D_x")
        >>> D_y = Domain([0, 1], variable_names=["y"], name="D_y")
        >>> D = Domain.cartesian_product([D_theta, D_x, D_y], name="D")

        Define a multivariate function.

        >>> def mapping(*, theta, x, y):
        ...     return theta ** (x + y) * (1 - theta) ** (2 - x - y)
        >>> f = MultivariateFunction(
        ...     domain=D,
        ...     mapping=mapping,
        ...     output_name="probability",
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f':
                   probability
        theta x y
        0.00  0 0       1.0000
                1       0.0000
              1 0       0.0000
                1       0.0000
        0.25  0 0       0.5625
                1       0.1875
              1 0       0.1875
                1       0.0625
        0.50  0 0       0.2500
                1       0.2500
              1 0       0.2500
                1       0.2500
        0.75  0 0       0.0625
                1       0.1875
              1 0       0.1875
                1       0.5625
        1.00  0 0       0.0000
                1       0.0000
              1 0       0.0000
                1       1.0000

        Define a sample space and promote the multivariate function to a parametrized probability measure.

        >>> Omega = SampleSpace.cartesian_product(indices=[D_x, D_y], variable_names=["x", "y"])
        >>> P = f.to_prob_measure(sample_space=Omega, name="P")
        >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'P':
                   probability
        theta x y
        0.00  0 0       1.0000
                1       0.0000
              1 0       0.0000
                1       0.0000
        0.25  0 0       0.5625
                1       0.1875
              1 0       0.1875
                1       0.0625
        0.50  0 0       0.2500
                1       0.2500
              1 0       0.2500
                1       0.2500
        0.75  0 0       0.0625
                1       0.1875
              1 0       0.1875
                1       0.5625
        1.00  0 0       0.0000
                1       0.0000
              1 0       0.0000
                1       1.0000

        Now methods belonging to `ParametrizedProbabilityMeasure` are available. For example, we can obtain an instance of `ProbabilityMeasure by (partially) evaluating at a parameter:

        >>> print(P(theta=0.25))  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P(theta=0.25)':
             probability
        x y
        0 0       0.5625
          1       0.1875
        1 0       0.1875
          1       0.0625
        """
        from ..base.sample_space import SampleSpace
        from ..probability_measures.parametrized_probability_measure import (
            ParametrizedProbabilityMeasure,
        )
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if self.domain is not None:
            if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
                raise TypeError("If provided, sig_alg must be a SigmaAlgebra.")
            if sample_space is not None and not isinstance(sample_space, SampleSpace):
                raise TypeError("If provided, sample_space must be a SampleSpace.")
            if name is not None and not isinstance(name, Hashable):
                raise TypeError("If provided, name must be a hashable type.")

            if name is None:
                name = self.name

            if sig_alg is None and sample_space is None:
                sample_space = SampleSpace.from_domain(self.domain)
                sig_alg = SigmaAlgebra.power_set(sample_space)

            in_variable_names = (
                sig_alg.variable_names
                if sig_alg is not None
                else sample_space.variable_names
            )

            is_prob_measure = in_variable_names == self.argument_names

            prob_measure = ParametrizedProbabilityMeasure(
                sig_alg=sig_alg,
                sample_space=sample_space,
                domain=self.domain,
                mapping=self.data,
                name=name,
            )

            if in_place:
                self.__class__ = (
                    ProbabilityMeasure
                    if is_prob_measure
                    else ParametrizedProbabilityMeasure
                )
                self.__dict__.update(prob_measure.__dict__)
                return self
            else:
                prob_measure.__class__ = (
                    ProbabilityMeasure
                    if is_prob_measure
                    else ParametrizedProbabilityMeasure
                )
                return prob_measure

        else:
            return NotImplementedError(
                "The to_parametrized_prob_measure method is not implemented yet for functions without an explicit domain."
            )

    # --------------------- representation --------------------- #

    def __repr__(self):
        """Pass."""
        if self.data is not None:
            return f"{type(self)._repr_name} '{self.name}':\n{self.data.to_frame()}"
        elif self.argument_names is not None:
            parameter_list = ", ".join(self.argument_names)
            return f"{type(self)._repr_name} '{self.name}({parameter_list})'"
        else:
            return f"{type(self)._repr_name} '{self.name}': empty"

    # --------------------- equality --------------------- #

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
        if not isinstance(other, MultivariateFunction) and not isinstance(other, Real):
            return False

        if isinstance(other, MultivariateFunction):
            if self.fun is None or other.fun is None:
                raise ValueError("Cannot compare empty functions.")
            if self.domain is None or other.domain is None:
                raise ValueError(
                    "Cannot compare functions when one (or both) domains are not defined."
                )

            if self.argument_names != other.argument_names:
                try:
                    new_argument_order = [
                        arg
                        for arg in other.argument_names
                        if arg in self.argument_names
                    ]
                    _ = self.domain.data.reorder_levels(new_argument_order)
                except (ValueError, AttributeError, AssertionError) as e:
                    raise ValueError(
                        "Cannot compare functions with different domains/argument names."
                    ) from e

            return np.allclose(self.data.values, other.data.values)

        else:
            if self.fun is None:
                raise ValueError("Cannot compare empty functions.")
            if self.domain is None:
                raise ValueError(
                    "Cannot compare functions when the domain is not defined."
                )
            return np.allclose(self.data.values, other)

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
            if reverse:
                function_name = f"({other.name} {op_symbol} {self.name})"
            else:
                function_name = f"({self.name} {op_symbol} {other.name})"

            argument_names = list(
                dict.fromkeys(self.argument_names + other.argument_names)
            )
            output_name = (
                self.output_name
                if self.output_name == other.output_name
                else f"({self.output_name} {op_symbol} {other.output_name})"
            )

            if self.domain is not None and other.domain is not None:
                if len(argument_names) < len(self.argument_names) + len(
                    other.argument_names
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
                        self.argument_names + other.argument_names, inplace=True
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
                        for name in self.argument_names
                        if name in bound.arguments
                    }
                    other_arguments = {
                        name: bound.arguments[name]
                        for name in other.argument_names
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
