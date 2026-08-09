"""A class representing a parametrized measure."""

from __future__ import annotations

from collections.abc import Hashable, Iterator
from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd

from .multivariate_function import MultivariateFunction

if TYPE_CHECKING:
    from ...typing.index_like import IndexLike
    from ...typing.mapping_like import MappingLike
    from ..measures.measure import Measure
    from ..measures.parametrized_measure import ParametrizedMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.domain import Domain
    from ..spaces.measurable_set import MeasurableSet
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
    ...     ParametrizedMeasurableFunction,
    ...     SigmaAlgebra,
    ...     ProbabilityMeasure,
    ... )

    Define a 1-dimensional parameter domain and measurable domain.

    >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
    >>> X = Domain.from_sequence(size=3, variable_name="x")

    Define a sigma-algebra.

    >>> F = SigmaAlgebra(
    ...     domain=X,
    ...     mapping={
    ...         0: 0,
    ...         1: 1,
    ...         2: 1,
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
    ...     mapping=mapping,
    ... )
    >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
    Parametrized measurable function 'f':
              f
    theta x
    0     0   1
          1   2
          2   2
    1     0   0
          1  -3
          2  -3

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
           f(x=0)
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
             rv
    theta x
    0     0   1
          1   2
          2   2
    1     0   0
          1  -3
          2  -3

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
        "_parameter_domain",
        "_parameter_names",
        "_measurable_names",
        "_atom_data",
        "_parameter_domain_name",
    ]

    # --------------------- constructors --------------------- #

    @classmethod
    def from_domains(
        cls,
        measurable_domain: Domain,
        parameter_domain: Domain | None = None,
        complete_domain: Domain | None = None,
        sig_alg: SigmaAlgebra | None = None,
        mapping: MappingLike | None = None,
        measure: Measure | None = None,
        parameter_domain_name: Hashable | None = None,
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
        ...     ParametrizedMeasurableFunction,
        ...     SigmaAlgebra,
        ...     ProbabilityMeasure,
        ... )

        Define a 1-dimensional parameter domain and measurable domain.

        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> X = Domain.from_sequence(size=3, variable_name="x")

        Define a sigma-algebra.

        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
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
        ...     mapping=mapping,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'f':
                 f
        theta x
        0     0  1
              1  2
              2  2
        1     0  0
              1 -3
              2 -3

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
               f(x=0)
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
                 rv
        theta x
        0     0   1
              1   2
              2   2
        1     0   0
              1  -3
              2  -3

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
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from ..spaces.domain import Domain
        from .parametrized_random_variable import ParametrizedRandomVariable

        if not isinstance(measurable_domain, Domain):
            raise TypeError("measurable_domain must be an instance of Domain.")
        if parameter_domain is not None and not isinstance(parameter_domain, Domain):
            raise TypeError("If given, parameter_domain must be an instance of Domain.")
        if complete_domain is not None and not isinstance(complete_domain, Domain):
            raise TypeError("If given, complete_domain must be an instance of Domain.")

        if sig_alg is None:
            sig_alg = SigmaAlgebra.power_set(measurable_domain)

        if parameter_domain is not None and complete_domain is None:
            domain = Domain.cartesian_product(
                factors=[parameter_domain, measurable_domain]
            )
            parameter_domain_name = parameter_domain.name
        elif parameter_domain is None and complete_domain is not None:
            domain = complete_domain
        elif parameter_domain is not None and complete_domain is not None:
            raise TypeError("Cannot pass both parameter_domain and complete_domain.")
        else:
            domain = None

        function = cls(
            domain=domain,
            mapping=mapping,
            output_name=name,
            name=name,
        )

        function._init_measurable_attrs(
            measurable_domain=measurable_domain,
            parameter_domain=parameter_domain,
            sig_alg=sig_alg,
            measure=measure,
            parameter_domain_name=parameter_domain_name,
        )

        if not function._is_mapping_consistent_with_measurable_domain():
            raise ValueError(
                "For each paramter value, the domain of the mapping must equal the measurable domain. This is not true."
            )
        if not function._is_measurable():
            raise ValueError(
                "There are parameter values for which the function is not measurable."
            )

        function._data = function._data.reorder_levels(
            function.parameter_names + function.measurable_names
        )

        if isinstance(measure, ProbabilityMeasure):
            function.__class__ = ParametrizedRandomVariable

        return function

    def _is_mapping_consistent_with_measurable_domain(self) -> bool:
        measurable_domain_data = self.measurable_domain.data.to_frame().reset_index(
            drop=True
        )
        measurable_domain_data["dummy"] = 0
        self_data = self.data.reset_index(self.measurable_domain.variable_names)
        df = pd.merge(left=measurable_domain_data, right=self_data, how="outer")

        return df.isna().sum().sum() == 0

    def _init_measurable_attrs(
        self,
        measurable_domain: IndexLike | None = None,
        parameter_domain: IndexLike | None = None,
        sig_alg: SigmaAlgebra | None = None,
        measure: Measure | None = None,
        parameter_domain_name: Hashable | None = None,
    ) -> None:
        from ..spaces.measurable_space import MeasurableSpace
        from ..spaces.measure_space import MeasureSpace

        self._measurable_domain = measurable_domain
        self._parameter_domain = parameter_domain
        self._parameter_domain_name = parameter_domain_name
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
        from ..utils.utils import _to_df

        if self.sig_alg.is_power_set:
            return True

        if set(self.measurable_domain.variable_names) & set(
            self.sig_alg.variable_names
        ):
            raise ValueError(
                "There is an overlap between the variable names of the measurable domain and the variable names of the sigma-algebra."
            )
        if set(self.parameter_names) & set(self.measurable_domain.variable_names):
            raise ValueError(
                "There is an overlap between the variable names of the measurable domain and the parameter names."
            )

        sig_alg_data = _to_df(self.sig_alg.data)

        combined_data = pd.merge(
            left=self.data.reset_index(),
            right=sig_alg_data.reset_index(),
        )

        grouped = combined_data.groupby(
            self.parameter_names + list(sig_alg_data.columns)
        )[self.output_name]

        return (grouped.nunique() == 1).all()

    # --------------------- properties --------------------- #

    @property
    def atom_data(self) -> pd.Series | None:
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
        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> X = Domain.from_sequence(size=3, variable_name="x")
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     },
        ... )
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
        >>> f = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     mapping=mapping,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'f':
                  f
        theta x
        0     0   1
              1   2
              2   2
        1     0   0
              1  -3
              2  -3
        >>> print(f.atom_data)  # doctest: +NORMALIZE_WHITESPACE
        theta  atom_ID
        0      0          1
               1          2
        1      0          0
               1         -3
        Name: f, dtype: int64
        """
        if (
            self._atom_data is None
            and self.data is not None
            and self.sig_alg is not None
        ):
            # TODO: check merge logic — possibly change to `on`?
            data = pd.merge(
                left=self.data,
                right=self.sig_alg.data,
                left_index=True,
                right_index=True,
            ).add_suffix("_func")

            sig_alg_subscripted_names = [
                f"{name}_func" for name in self.sig_alg.variable_names
            ]
            data = (
                data.reset_index()
                .drop_duplicates(self.parameter_names + sig_alg_subscripted_names)
                .set_index(self.parameter_names + sig_alg_subscripted_names)
                .drop(columns=self.measurable_domain.variable_names)
            )
            data.index.names = self.parameter_names + self.sig_alg.variable_names

            data = data.squeeze(axis=1).rename(self.output_name)

            self._atom_data = data

        return self._atom_data

    @property
    def parameter_domain(self) -> Domain | None:
        """Get the parameter domain of the function.

        Returns
        -------
        parameter_domain : Domain | None
            The parameter domain of the function, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedMeasurableFunction,
        ...     SigmaAlgebra,
        ... )
        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> X = Domain.from_sequence(size=3, variable_name="x")
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     },
        ... )
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
        >>> f = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     mapping=mapping,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'f':
                  f
        theta x
        0     0   1
              1   2
              2   2
        1     0   0
              1  -3
              2  -3
        >>> print(f.parameter_domain)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'Theta':
         theta
             0
             1
        """
        return self._parameter_domain

    @property
    def parameter_domain_name(self) -> Hashable | None:
        """Pass."""
        return self._parameter_domain_name

    @property
    def parameter_names(self) -> list[Hashable] | None:
        """Get the parameter names of the function.

        Returns
        -------
        parameter_names : list[Hashable] | None
            The names of the parameters of the function, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedMeasurableFunction,
        ...     SigmaAlgebra,
        ... )
        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> X = Domain.from_sequence(size=3, variable_name="x")
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     },
        ... )
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
        >>> f = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     mapping=mapping,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'f':
                  f
        theta x
        0     0   1
              1   2
              2   2
        1     0   0
              1  -3
              2  -3
        >>> print(f.parameter_names)
        ['theta']
        """
        if self._parameter_names is None and self.measurable_domain is not None:
            self._parameter_names = [
                name
                for name in self.variable_names
                if name not in self.measurable_domain.variable_names
            ]

        return self._parameter_names

    @property
    def measurable_domain(self) -> Domain | None:
        """Get the measurable domain of the function.

        Returns
        -------
        measurable_domain : Domain | None
            The measurable domain of the function, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedMeasurableFunction,
        ...     SigmaAlgebra,
        ... )
        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> X = Domain.from_sequence(size=3, variable_name="x")
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     },
        ... )
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
        >>> f = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     mapping=mapping,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'f':
                  f
        theta x
        0     0   1
              1   2
              2   2
        1     0   0
              1  -3
              2  -3
        >>> print(f.measurable_domain)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'X':
         x
         0
         1
         2
        """
        return self._measurable_domain

    @property
    def measurable_names(self) -> list[Hashable] | None:
        """Get the measurable names of the function.

        Returns
        -------
        measurable_names : list[Hashable] | None
            The names of the measurable variables of the function, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedMeasurableFunction,
        ...     SigmaAlgebra,
        ... )
        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> X = Domain.from_sequence(size=3, variable_name="x")
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     },
        ... )
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
        >>> f = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     mapping=mapping,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'f':
                  f
        theta x
        0     0   1
              1   2
              2   2
        1     0   0
              1  -3
              2  -3
        >>> print(f.measurable_names)
        ['x']
        """
        return self.measurable_domain.variable_names

    @property
    def sig_alg(self) -> SigmaAlgebra | None:
        """Get the sigma-algebra of the function.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra of the function, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedMeasurableFunction,
        ...     SigmaAlgebra,
        ... )
        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> X = Domain.from_sequence(size=3, variable_name="x")
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     },
        ... )
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
        >>> f = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     mapping=mapping,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'f':
                  f
        theta x
        0     0   1
              1   2
              2   2
        1     0   0
              1  -3
              2  -3
        >>> print(f.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
           atom_ID
        x
        0        0
        1        1
        2        1
        """
        return self._sig_alg

    @property
    def measurable_space(self) -> MeasurableSpace | None:
        """Get the measurable space of the function.

        Returns
        -------
        measurable_space : MeasurableSpace | None
            The measurable space of the function, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedMeasurableFunction,
        ...     SigmaAlgebra,
        ... )
        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> X = Domain.from_sequence(size=3, variable_name="x")
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     },
        ... )
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
        >>> f = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     mapping=mapping,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'f':
                  f
        theta x
        0     0   1
              1   2
              2   2
        1     0   0
              1  -3
              2  -3
        >>> repr(f.measurable_space)
        'MeasurableSpace(domain=X, sig_alg=F)'
        """
        return self._measurable_space

    @property
    def measure_space(self) -> MeasureSpace | None:
        """Get the measure space of the function.

        Returns
        -------
        measure_space : MeasureSpace | None
            The measure space of the function, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     Measure,
        ...     ParametrizedMeasurableFunction,
        ...     SigmaAlgebra,
        ... )
        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> X = Domain.from_sequence(size=3, variable_name="x")
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
        ...         1: 3,
        ...     },
        ... )
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
        >>> f = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     measure=mu,
        ...     mapping=mapping,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'f':
                  f
        theta x
        0     0   1
              1   2
              2   2
        1     0   0
              1  -3
              2  -3
        >>> repr(f.measure_space)
        'MeasureSpace(domain=X, sig_alg=F, measure=mu)'
        """
        return self._measure_space

    @property
    def measure(self) -> Measure | None:
        """Get the measure of the function.

        Returns
        -------
        measure : Measure | None
            The measure of the function, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     Measure,
        ...     ParametrizedMeasurableFunction,
        ...     SigmaAlgebra,
        ... )
        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> X = Domain.from_sequence(size=3, variable_name="x")
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
        ...         1: 3,
        ...     },
        ... )
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
        >>> f = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     measure=mu,
        ...     mapping=mapping,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'f':
                  f
        theta x
        0     0   1
              1   2
              2   2
        1     0   0
              1  -3
              2  -3
        >>> repr(f.measure)
        'Measure(domain=X, sig_alg=F, name=mu)'
        """
        return self.measure_space.measure if self.measure_space is not None else None

    # --------------------- measure-related methods --------------------- #

    def integrate(
        self,
        measurable_set: MeasurableSet | None = None,
        measure: Measure | ParametrizedMeasure | None = None,
    ) -> MultivariateFunction:
        r"""Compute the Lebesgue integral of a parametrized measurable function with respect to a measure over an (optional) set.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        measurable_set: MeasurableSet | None, default=None
            The optional set over which to integrate. If `None`, the integral will be taken over the entire domain of the measurable vector.
        measure : Measure | ParametrizedMeasure | None, default=None
            The measure or parametrized measure with respect to which to integrate. If `None`, the measure of the underlying measure space is used (if it exists) carried by the measurable vector or parametrized measurable function.

        Returns
        -------
        integral : Real | pd.Series | MultivariateFunction
            Returns the following:

            * If `measure` is a `Measure`, returns a `MultivariateFunction` representing the integral of the function with respect to the measure over the specified set for each parameter value.

            * If `measure` is a `ParametrizedMeasure`, returns a `MultivariateFunction` representing the integral of the function with respect to the measure over the specified set for each parameter value.

        Examples
        --------
        Define a measure space and a measurable function.

        >>> from sigalg.core import (
        ...     Domain,
        ...     MeasurableFunction,
        ...     Measure,
        ...     MeasureSpace,
        ...     Operators,
        ...     ParametrizedMeasurableFunction,
        ...     ParametrizedMeasure,
        ...     SigmaAlgebra,
        ... )

        Define a measure space, parametrized measurable function, and parametrized measure.

        >>> X = Domain.from_sequence(size=3, variable_name="x")
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: (0, 1),
        ...         1: (1, 1),
        ...         2: (1, 1),
        ...     },
        ...     variable_names=["u", "v"],
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         (0, 1): 2,
        ...         (1, 1): 3,
        ...     },
        ... )
        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> f = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     mapping={
        ...         (0, 0): 2,
        ...         (0, 1): 4,
        ...         (0, 2): 4,
        ...         (1, 0): 1,
        ...         (1, 1): -1,
        ...         (1, 2): -1,
        ...     },
        ... )
        >>> nu = ParametrizedMeasure.from_domains(
        ...     measure_domain=F,
        ...     parameter_domain=Theta,
        ...     mapping={
        ...         (0, 0, 1): 3,
        ...         (0, 1, 1): 4,
        ...         (1, 0, 1): 1,
        ...         (1, 1, 1): 2,
        ...     },
        ...     name="nu",
        ... )

        Extract a measurable set from the sigma-algebra.

        >>> U = F.get_set([1, 2], name="U")

        It is convenient to conceptualize a parametrized measurable function as a family of measurable functions. Then integration of a parametrized measurable function against a measure returns a function of the parameters whose values are the integrals of the functions against the measure. Iteration over the parametrized measurable function yields the functions, allowing us to check that these integrals match.

        >>> all(f.integrate(U, mu)(**param) == function.integrate(U, mu) for param, function in f)
        True

        It is possible to integrate a parametrized measurable function against a parametried measure as long as their parameter domains agree.

        >>> all(
        ...     f.integrate(U, nu)(**param) == function.integrate(U, measure)
        ...     for (param, function), (_, measure) in zip(f, nu)
        ... )
        True

        Notes
        -----
        Let $f: X \to \mathbb{R}$ be a measurable function on a measure space $(X, \mathcal{F}, \mu)$. Assuming $X$ is finite (as it always is, in SigAlg), the $\sigma$-algebra $\mathcal{F}$ is determined by its set $\alpha(\mathcal{F})$ of atoms. Let $U$ be a measurable set in $\mathcal{F}$, and write $I_U$ for its indicator function. Since both $f$ and $I_U$ are $\mathcal{F}$-measurable, they take constant values on each atom $A\in \alpha(\mathcal{F})$ that we write as $f(A)$ and $I_U(A)$, respectively. Then the *Lebesgue integral* of $f$ over $U$ is the number

        $$
        \int_U f \, d\mu = \sum_{A\in \alpha(\mathcal{F})} I_U(A)f(A) \mu(A).
        $$

        If $f:X \to \mathbb{R}^d$ is instead a measurable vector of dimension $d>1$, with components

        $$
        f = (f_1, f_2, \ldots, f_d),
        $$

        then we define the *Lebesgue integral* of $f$ over $U$ to be the $d$-dimensional vector whose entries are the separate Lebesgue integrals $\int_U f_j \, d\mu$, for $j=1,2,\ldots,d$.
        """
        from .operators import Operators

        return Operators.integrate(
            function=self, measurable_set=measurable_set, measure=measure
        )

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
        from ..spaces.domain import Domain
        from .measurable_function import MeasurableFunction

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
            ).rename(name)

            if len(unspecified_parameters) == 0:
                return MeasurableFunction(
                    domain=self.measurable_domain,
                    sig_alg=self.sig_alg,
                    measure=self.measure,
                    mapping=mapping.rename(name),
                    name=name,
                )

            elif len(unspecified_parameters) != 0 and len(specified_parameters) != 0:
                domain_name = f"{self.domain.name}|{{{parameter_string}}}"
                domain = Domain(mapping.index, name=domain_name)

                return ParametrizedMeasurableFunction.from_domains(
                    measurable_domain=self.measurable_domain,
                    complete_domain=domain,
                    sig_alg=self.sig_alg,
                    mapping=mapping,
                    measure=self.measure,
                    name=name,
                )

        else:
            try:
                result = super().__call__(**kwargs)
                if not isinstance(result, Real):
                    result._data.name = result.name
                return result
            except Exception as e:
                raise ValueError(
                    "Error while evaluating the parametrized measurable function on the given arguments."
                ) from e

    def __iter__(self) -> Iterator[tuple[dict, MeasurableFunction]]:
        """Pass."""
        unique_params = (
            self.data.index.to_frame()[self.parameter_names]
            .drop_duplicates()
            .iterrows()
        )
        for _, params in unique_params:
            yield params.to_dict(), self(**params.to_dict())

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a concise string representation of the function.

        Returns
        -------
        repr_str : str
            The string representation of the function.
        """
        if self.variable_names is not None:
            parameter_list = ", ".join(self.parameter_names)
            measurable_list = ", ".join(self.measurable_names)
            return (
                f"{type(self)._repr_name}(parameters=({parameter_list}), "
                f"measurable_vars=({measurable_list}), "
                f"domain={self.measurable_domain.name}, "
                f"sig_alg={self.sig_alg.name}, "
                f"measure={self.measure.name if self.measure is not None else None}, "
                f"name={self.name})"
            )
        else:
            return type(self)._repr_name + "(empty)"
