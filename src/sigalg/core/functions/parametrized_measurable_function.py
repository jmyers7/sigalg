"""A class representing a parametrized measure."""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd

from .multivariate_function import MultivariateFunction

if TYPE_CHECKING:
    from ...typing.index_like import IndexLike
    from ...typing.mapping_like import MappingLike
    from ..measures.measure import Measure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.domain import Domain
    from ..spaces.measurable_space import MeasurableSpace
    from ..spaces.measure_space import MeasureSpace
    from .measurable_function import MeasurableFunction


class ParametrizedMeasurableFunction(MultivariateFunction):
    r"""A class representing a parametrized measurable function.

    The `__init__` constructor is not meant to be used directly. Instead, the user should use the `from_domains` class method.

    See the Notes section below for the mathematical details.

    Examples
    --------
    >>> from sigalg.core import (
    ...     Domain,
    ...     Measure,
    ...     ParametrizedMeasurableFunction,
    ...     SigmaAlgebra,
    ...     ProbabilityMeasure,
    ... )

    Define a 1-dimensional parameter domain and measurable domain.

    >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
    >>> X = Domain.from_sequence(size=3, variable_name="x")

    Define a sigma-algebra and a measure.

    >>> F = SigmaAlgebra(
    ...     domain=X,
    ...     mapping={
    ...         0: 0,
    ...         1: 1,
    ...         2: 1,
    ...     },
    ... )
    >>> mu = Measure(
    ...     domain=F,
    ...     mapping={
    ...         0: 1,
    ...         1: 2,
    ...     },
    ... )

    Define the mapping of a parametrized measurable function.

    >>> def mapping(*, theta, x):  # noqa: D103
    ...     if theta == 0:
    ...         if x == 0:
    ...             return 1
    ...         elif x == 1:
    ...             return 2
    ...         elif x == 2:
    ...             return 2
    ...     elif theta == 1:
    ...         if x == 0:
    ...             return 0
    ...         elif x == 1:
    ...             return -3
    ...         elif x == 2:
    ...             return -3

    Instantiate a parametrized measurable function and print it.

    >>> f = ParametrizedMeasurableFunction.from_domains(
    ...     measurable_domain=X,
    ...     parameter_domain=Theta,
    ...     sig_alg=F,
    ...     measure=mu,
    ...     mapping=mapping,
    ... )
    >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
    Parametrized measurable function 'f':
             output
    theta x
    0     0       1
          1       2
          2       2
    1     0       0
          1      -3
          2      -3

    Evaluate the function at a parameter to obtain a measurable function.

    >>> print(f(theta=0))  # doctest: +NORMALIZE_WHITESPACE
    Measurable function 'f(theta=0)':
       f(theta=0)
    x
    0           1
    1           2
    2           2

    Evaluate the function at a "measurable" variable to get an instance of `MultivariateFunction`.

    >>> print(f(x=0))  # doctest: +NORMALIZE_WHITESPACE
    Function 'f(x=0)':
           output
    theta
    0           1
    1           0

    Construct a parametrized measurable function with a probability measure and get an instance of `ParametrizedRandomVariable`.

    >>> P = ProbabilityMeasure(
    ...     domain=F,
    ...     mapping={
    ...         0: 0.2,
    ...         1: 0.8,
    ...     },
    ... )
    >>> rv = ParametrizedMeasurableFunction.from_domains(
    ...     measurable_domain=X,
    ...     parameter_domain=Theta,
    ...     sig_alg=F,
    ...     measure=P,
    ...     mapping=mapping,
    ...     name="rv",
    ... )
    >>> print(rv)  # doctest: +NORMALIZE_WHITESPACE
    Parametrized random variable 'rv':
             output
    theta x
    0     0       1
          1       2
          2       2
    1     0       0
          1      -3
          2      -3

    Evaluation at a parameter now returns a random variable.

    >>> print(rv(theta=0))  # doctest: +NORMALIZE_WHITESPACE
    Random variable 'rv(theta=0)':
       rv(theta=0)
    x
    0            1
    1            2
    2            2

    Notes
    -----
    Let $(X, \mathcal{F})$ be a measurable space and $\Theta$ a nonempty set. A *parametrized measurable function* is a function

    $$
    f : \Theta \times X \to \mathbb{R}
    $$

    such that, for each fixed $\theta \in \Theta$, the partial function

    $$
    f(\theta, -): X \to \mathbb{R}, \quad x \mapsto f(\theta,x),
    $$

    is a measurable function with respect to the $\sigma$-algebra $\mathcal{F}$. The set $\Theta$ is called the *parameter domain*, elements $\theta\in \Theta$ are called *parameters*, and $X$ is called the *measurable domain*.
    """

    _repr_name = "ParametrizedMeasurableFunction"
    _str_name = "Parametrized measurable function"
    _default_name = "f"
    _properties = MultivariateFunction._properties + [
        "_parameter_names",
        "_measurable_names",
    ]

    # --------------------- constructors --------------------- #

    # TODO: input validation
    @classmethod
    def from_domains(
        cls,
        measurable_domain: IndexLike,
        parameter_domain: IndexLike,
        sig_alg: SigmaAlgebra,
        mapping: MappingLike,
        measure: Measure | None = None,
        name: Hashable = "f",
    ) -> ParametrizedMeasurableFunction:
        r"""Construct a parametrized measurable function from a measurable domain and parameter domain.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        measurable_domain : IndexLike
            The measurable domain of the function.
        parameter_domain : IndexLike
            The parameter domain of the function.
        sig_alg : SigmaAlgebra
            The sigma-algebra of the underlying measurable space.
        mapping : MappingLike
            The mapping of the parametrized function.
        measure : Measure | None, default=None
            An optional measure on the underlying measurable space.
        name : Hashable, default="f"
            The name of the parametrized function.

        Returns
        -------
        param_func : ParametrizedMeasurableFunction
            The constructed parametrized measurable function.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     Measure,
        ...     ParametrizedMeasurableFunction,
        ...     SigmaAlgebra,
        ...     ProbabilityMeasure,
        ... )

        Define a 1-dimensional parameter domain and measurable domain.

        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> X = Domain.from_sequence(size=3, variable_name="x")

        Define a sigma-algebra and a measure.

        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     },
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...     },
        ... )

        Define the mapping of a parametrized measurable function.

        >>> def mapping(*, theta, x):  # noqa: D103
        ...     if theta == 0:
        ...         if x == 0:
        ...             return 1
        ...         elif x == 1:
        ...             return 2
        ...         elif x == 2:
        ...             return 2
        ...     elif theta == 1:
        ...         if x == 0:
        ...             return 0
        ...         elif x == 1:
        ...             return -3
        ...         elif x == 2:
        ...             return -3

        Instantiate a parametrized measurable function and print it.

        >>> f = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     measure=mu,
        ...     mapping=mapping,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'f':
                 output
        theta x
        0     0       1
              1       2
              2       2
        1     0       0
              1      -3
              2      -3

        Evaluate the function at a parameter to obtain a measurable function.

        >>> print(f(theta=0))  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'f(theta=0)':
           f(theta=0)
        x
        0           1
        1           2
        2           2

        Evaluate the function at a "measurable" variable to get an instance of `MultivariateFunction`.

        >>> print(f(x=0))  # doctest: +NORMALIZE_WHITESPACE
        Function 'f(x=0)':
               output
        theta
        0           1
        1           0

        Construct a parametrized measurable function with a probability measure and get an instance of `ParametrizedRandomVariable`.

        >>> P = ProbabilityMeasure(
        ...     domain=F,
        ...     mapping={
        ...         0: 0.2,
        ...         1: 0.8,
        ...     },
        ... )
        >>> rv = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     measure=P,
        ...     mapping=mapping,
        ...     name="rv",
        ... )
        >>> print(rv)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized random variable 'rv':
                 output
        theta x
        0     0       1
              1       2
              2       2
        1     0       0
              1      -3
              2      -3

        Evaluation at a parameter now returns a random variable.

        >>> print(rv(theta=0))  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'rv(theta=0)':
           rv(theta=0)
        x
        0            1
        1            2
        2            2

        Notes
        -----
        Let $(X, \mathcal{F})$ be a measurable space and $\Theta$ a nonempty set. A *parametrized measurable function* is a function

        $$
        f : \Theta \times X \to \mathbb{R}
        $$

        such that, for each fixed $\theta \in \Theta$, the partial function

        $$
        f(\theta, -): X \to \mathbb{R}, \quad x \mapsto f(\theta,x),
        $$

        is a measurable function with respect to the $\sigma$-algebra $\mathcal{F}$. The set $\Theta$ is called the *parameter domain*, elements $\theta\in \Theta$ are called *parameters*, and $X$ is called the *measurable domain*.
        """
        from ..measures.probability_measure import ProbabilityMeasure
        from ..spaces.domain import Domain
        from .parametrized_random_variable import ParametrizedRandomVariable

        domain = Domain.cartesian_product(factors=[parameter_domain, measurable_domain])

        function = cls(
            domain=domain,
            mapping=mapping,
            output_name="output",
            name=name,
        )

        function._initialize_attrs(
            measurable_domain=measurable_domain,
            parameter_domain=parameter_domain,
            sig_alg=sig_alg,
            measure=measure,
        )

        if not function._is_measurable():
            raise ValueError(
                "There are parameter values for which the function is not measurable."
            )

        if isinstance(measure, ProbabilityMeasure):
            function.__class__ = ParametrizedRandomVariable

        return function

    def _initialize_attrs(
        self,
        measurable_domain: IndexLike | None = None,
        parameter_domain: IndexLike | None = None,
        sig_alg: SigmaAlgebra | None = None,
        measure: Measure | None = None,
    ) -> None:
        from ..spaces.measurable_space import MeasurableSpace
        from ..spaces.measure_space import MeasureSpace

        self._measurable_domain = measurable_domain
        self._parameter_domain = parameter_domain
        self._sig_alg = sig_alg

        if measure is not None:
            self._measure_space = MeasureSpace(
                domain=measurable_domain,
                sig_alg=sig_alg,
                measure=measure,
            )
            self._measurable_space = self._measure_space.measurable_space
        else:
            self._measurable_space = MeasurableSpace(
                domain=measurable_domain,
                sig_alg=sig_alg,
            )
            self._measure_space = None

    def _is_measurable(self) -> bool:
        if self.sig_alg.is_power_set:
            return True

        sig_alg_data = self._to_df(self.sig_alg.data, "_alg")
        combined_data = pd.merge(
            left=self.data,
            right=sig_alg_data,
            left_index=True,
            right_index=True,
        )
        grouped = combined_data.groupby(
            self.parameter_names + list(sig_alg_data.columns)
        )

        return (grouped.nunique() == 1).all().all()

    @staticmethod
    def _to_df(
        data: pd.Series | pd.DataFrame, suffix: str | None = None
    ) -> pd.DataFrame:
        if suffix is None:
            suffix = ""
        if isinstance(data, pd.DataFrame):
            return data.add_suffix(suffix)
        else:
            return data.to_frame().add_suffix(suffix)

    # --------------------- properties --------------------- #

    @property
    def parameter_domain(self) -> Domain | None:
        """Get the parameter domain of the function.

        Returns
        -------
        parameter_domain : Domain | None
            The parameter domain of the function, or `None` if not set.
        """
        return self._parameter_domain

    @property
    def parameter_names(self) -> list[Hashable] | None:
        """Get the parameter names of the function.

        Returns
        -------
        parameter_names : list[Hashable] | None
            The names of the parameters of the function, or `None` if not set.
        """
        if self._parameter_names is None and self.parameter_domain is not None:
            self._parameter_names = self._parameter_domain.variable_names

        return self._parameter_names

    @property
    def measurable_domain(self) -> Domain | None:
        """Get the measurable domain of the function.

        Returns
        -------
        measurable_domain : Domain | None
            The measurable domain of the function, or `None` if not set.
        """
        return self._measurable_domain

    @property
    def measurable_names(self) -> list[Hashable] | None:
        """Get the measurable names of the function.

        Returns
        -------
        measurable_names : list[Hashable] | None
            The names of the measurable variables of the function, or `None` if not set.
        """
        return self.measurable_domain.variable_names

    @property
    def sig_alg(self) -> SigmaAlgebra | None:
        """Get the sigma-algebra of the function.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra of the function, or `None` if not set.
        """
        return self._sig_alg

    @property
    def measurable_space(self) -> MeasurableSpace | None:
        """Get the measurable space of the function.

        Returns
        -------
        measurable_space : MeasurableSpace | None
            The measurable space of the function, or `None` if not set.
        """
        return self._measurable_space

    @property
    def measure_space(self) -> MeasureSpace | None:
        """Get the measure space of the function.

        Returns
        -------
        measure_space : MeasureSpace | None
            The measure space of the function, or `None` if not set.
        """
        return self._measure_space

    @property
    def measure(self) -> Measure | None:
        """Get the measure of the function.

        Returns
        -------
        measure : Measure | None
            The measure of the function, or `None` if not set.
        """
        return self.measure_space.measure if self.measure_space is not None else None

    # --------------------- data methods --------------------- #

    def __call__(
        self, **kwargs
    ) -> (
        Real
        | MeasurableFunction
        | MultivariateFunction
        | ParametrizedMeasurableFunction
    ):
        """Call the function with the provided arguments.

        The return value is determined by the following rules:

        1. If all parameters and all measurable arguments are provided, a real number is returned.

        2. If all parameters are provided but no measurable arguments are provided, a measurable function is returned.

        3. If a partial set of parameters is provided and no measurable arguments are provided, a parametrized measurable function is provided.

        4. In all other cases, a multivariate function is returned.

        Parameters
        ----------
        **kwargs : keyword arguments
            Keyword arguments for the function.
        """
        from ..measures.probability_measure import ProbabilityMeasure
        from ..spaces.domain import Domain
        from .measurable_function import MeasurableFunction
        from .parametrized_random_variable import ParametrizedRandomVariable

        if self.data is None:
            raise NotImplementedError(
                "The __call__ method is not yet implemented for functions without data."
            )

        parameter_kwargs = {
            name: param
            for name, param in kwargs.items()
            if name in self.parameter_names
        }
        specified_parameters = self.signature.bind_partial(**parameter_kwargs).arguments
        unspecified_parameters = [
            parameter
            for parameter in self.parameter_names
            if parameter not in specified_parameters.keys()
        ]

        measurable_kwargs = {
            name: param
            for name, param in kwargs.items()
            if name in self.measurable_names
        }
        no_specified_measurables = len(measurable_kwargs) == 0

        if no_specified_measurables:
            parameter_string = ", ".join(
                f"{name}={value}" for name, value in specified_parameters.items()
            )
            name = f"{self.name}({parameter_string})"
            mapping = self.data.xs(
                key=tuple(specified_parameters.values()),
                level=tuple(specified_parameters.keys()),
            )

            if len(unspecified_parameters) == 0:
                return MeasurableFunction(
                    domain=self.measurable_domain,
                    sig_alg=self.sig_alg,
                    measure=self.measure,
                    mapping=mapping.rename(name),
                    name=name,
                )

            elif len(unspecified_parameters) != 0 and len(specified_parameters) != 0:
                parameter_domain_name = (
                    f"{self.parameter_domain.name}|{{{parameter_string}}}"
                )
                domain_name = f"{self.domain.name}|{{{parameter_string}}}"
                domain = Domain(mapping.index, name=domain_name)

                parameter_domain_data = (
                    mapping.index.to_frame()[unspecified_parameters]
                    .drop_duplicates()
                    .set_index(unspecified_parameters)
                    .index
                )
                parameter_domain = Domain(
                    parameter_domain_data, name=parameter_domain_name
                )

                function = ParametrizedMeasurableFunction(
                    domain=domain,
                    mapping=mapping,
                    output_name=self.output_name,
                    name=name,
                )

                function._parameter_names = unspecified_parameters
                function._initialize_attrs(
                    measurable_domain=self.measurable_domain,
                    parameter_domain=parameter_domain,
                    sig_alg=self.sig_alg,
                    measure=self.measure,
                )

                if isinstance(self.measure, ProbabilityMeasure):
                    function.__class__ = ParametrizedRandomVariable

                return function

        else:
            try:
                return super().__call__(**kwargs)
            except Exception as e:
                raise ValueError(
                    "Error while evaluating the parametrized measurable function on the given arguments."
                ) from e

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
                f"sig_alg={self.sig_alg.name}, "
                f"output_name={self.output_name}, "
                f"name={self.name})"
            )
        else:
            return type(self)._repr_name + "(empty)"
