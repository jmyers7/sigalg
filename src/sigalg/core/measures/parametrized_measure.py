"""A class representing a parametrized measure."""

from __future__ import annotations

from collections.abc import Hashable, Iterator
from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ..functions.multivariate_function import MultivariateFunction

if TYPE_CHECKING:
    from ...typing.index_like import IndexLike
    from ...typing.mapping_like import MappingLike
    from ...typing.measure_domain import MeasureDomain
    from ..functions.measurable_vector import MeasurableVector
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.domain import Domain
    from .measure import Measure


class ParametrizedMeasure(MultivariateFunction):
    r"""A class representing a parametrized measure.

    The `__init__` constructor is not meant to be used directly. Instead, the user should use the `from_domains` class method.

    See the Notes section below for the mathematical details.

    Examples
    --------
    >>> from sigalg.core import (
    ...     Domain,
    ...     ParametrizedMeasure,
    ...     SigmaAlgebra,
    ... )

    Define a 1-dimensional parameter domain and a sigma-algebra on a domain.

    >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
    >>> X = Domain.from_sequence(size=3, variable_name="x")
    >>> F = SigmaAlgebra(
    ...     domain=X,
    ...     mapping={
    ...         0: 0,
    ...         1: 1,
    ...         2: 1,
    ...     },
    ...     variable_names=["u"],
    ... )

    Define the mapping of a parametrized measure.

    >>> def mapping(*, theta, u):  # noqa: D103
    ...     if theta == 0:
    ...         if u == 0:
    ...             return 1
    ...         else:
    ...             return 2
    ...     if theta == 1:
    ...         if u == 0:
    ...             return 4
    ...         else:
    ...             return 0

    Instantiate a parametrized measure and print it.

    >>> mu = ParametrizedMeasure.from_domains(
    ...     measure_domain=F,
    ...     parameter_domain=Theta,
    ...     mapping=mapping,
    ... )
    >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
    Parametrized measure 'mu':
             measure
    theta u
    0     0        1
          1        2
    1     0        4
          1        0

    Evaluate at a parameter to obtain a measure.

    >>> print(mu(theta=0))  # doctest: +NORMALIZE_WHITESPACE
    Measure 'mu(theta=0)':
       measure
    u
    0        1
    1        2

    Evaluate at an atom identifer to get an instance of `MultivariateFunction`.

    >>> print(mu(u=0))  # doctest: +NORMALIZE_WHITESPACE
    Function 'mu(u=0)':
           measure
    theta
    0            1
    1            4

    Notes
    -----
    Let $(X, \mathcal{F})$ be a measurable space and $\Theta$ a nonempty set. A *parametrized measure* is a function

    $$
    \mu : \Theta \times \mathcal{F} \to \mathbb{R}
    $$

    such that, for each fixed $\theta \in \Theta$, the partial function

    $$
    \mu(\theta, -): \mathcal{F} \to \mathbb{R}, \quad U \mapsto \mu(\theta,U),
    $$

    is a measure on the $\sigma$-algebra $\mathcal{F}$. The set $\Theta$ is called the *parameter domain* and elements $\theta\in \Theta$ are called *parameters*.
    """

    _repr_name = "ParametrizedMeasure"
    _str_name = "Parametrized measure"
    _default_name = "mu"
    _properties = MultivariateFunction._properties + [
        "_parameter_domain",
        "_parameter_names",
    ]

    # --------------------- constructors --------------------- #

    # TODO: input validation
    # TODO: bring signature and implementation into alignment with ParametrizedMeasurableFunction
    @classmethod
    def from_domains(
        cls,
        measure_domain: MeasureDomain,
        parameter_domain: IndexLike | None,
        mapping: MappingLike,
        kind: Literal["measure", "probability"] = "measure",
        output_name: str = "measure",
        name: Hashable = "mu",
    ) -> ParametrizedMeasure:
        """Construct a parametrized measure from a measure domain and parameter domain.

        Parameters
        ----------
        measure_domain : MeasureDomain
            The domain of the measure, if a `SigmaAlgebra` is provided. If an `IndexLike` object that can be coerced to a `Domain` is provided, the sigma-algebra of the measure will be the power-set sigma-algebra of the domain.
        parameter_domain : IndexLike | None
            The parameter domain for the measure.
        mapping : MappingLike
            The mapping of the parametrized measure.
        kind : Literal["measure", "probability"], default="measure"
            The kind of the parametrized measure.
        output_name : str, default="measure"
            The name of the output variable for the parametrized measure.
        name : Hashable | None, default="mu"
            The name of the parametrized measure. If `None`, a default name will be assigned.

        Returns
        -------
        param_measure : ParametrizedMeasure
            The constructed parametrized measure.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedMeasure,
        ...     SigmaAlgebra,
        ... )

        Define a 1-dimensional parameter domain and a sigma-algebra on a domain.

        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> X = Domain.from_sequence(size=3, variable_name="x")
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     },
        ...     variable_names=["u"],
        ... )

        Define the mapping of a parametrized measure.

        >>> def mapping(*, theta, u):  # noqa: D103
        ...     if theta == 0:
        ...         if u == 0:
        ...             return 1
        ...         else:
        ...             return 2
        ...     if theta == 1:
        ...         if u == 0:
        ...             return 4
        ...         else:
        ...             return 0

        Instantiate a parametrized measure and print it.

        >>> mu = ParametrizedMeasure.from_domains(
        ...     measure_domain=F,
        ...     parameter_domain=Theta,
        ...     mapping=mapping,
        ... )
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu':
                 measure
        theta u
        0     0        1
              1        2
        1     0        4
              1        0

        Evaluate at a parameter to obtain a measure.

        >>> print(mu(theta=0))  # doctest: +NORMALIZE_WHITESPACE
        Measure 'mu(theta=0)':
           measure
        u
        0        1
        1        2

        Evaluate at an atom identifer to get an instance of `MultivariateFunction`.

        >>> print(mu(u=0))  # doctest: +NORMALIZE_WHITESPACE
        Function 'mu(u=0)':
               measure
        theta
        0            1
        1            4
        """
        from ...validation.measure_domain_validator import MeasureDomainValidator
        from ..spaces.domain import Domain
        from .parametrized_probability_measure import ParametrizedProbabilityMeasure

        if parameter_domain is not None:
            parameter_domain = Domain(parameter_domain)

        v = MeasureDomainValidator(measure_domain=measure_domain)

        if parameter_domain is not None:
            domain = Domain.cartesian_product(
                factors=[parameter_domain, v.sig_alg.atom_space]
            )
        else:
            domain = None

        measure = cls(
            domain=domain,
            mapping=mapping,
            output_name=output_name,
            name=name,
        )

        measure._init_measure_attrs(
            parameter_domain=parameter_domain,
            sig_alg=v.sig_alg,
            kind=kind,
        )

        if not measure._is_measure():
            raise ValueError(
                "A measure must have non-negative values. This is not true."
            )
        if kind == "probability" and not measure._sum_to_one():
            raise ValueError(
                "For each unique set of parameters, the values of the measure must sum to 1. This is not true."
            )
        if kind == "probability":
            measure.__class__ = ParametrizedProbabilityMeasure
            output_name = "probability"

        return measure

    def _init_measure_attrs(
        self,
        parameter_domain: IndexLike | None = None,
        sig_alg: SigmaAlgebra | None = None,
        kind: Literal["measure", "probability"] | None = None,
    ) -> None:
        self._parameter_domain = parameter_domain
        self._sig_alg = sig_alg
        self._kind = kind

    # TODO: move measure checks to MappingValidator?
    def _is_measure(self) -> bool:
        return all(self.data >= 0)

    def _sum_to_one(self) -> bool:
        return all(np.abs(self.data.groupby(self.parameter_names).sum() - 1) < 1e-8)

    @classmethod
    def from_rand(
        cls,
        domain_dims: tuple[int],
        output_name: Hashable = "measure",
        variable_names: list[Hashable] | None = None,
        variable_name_prefix: str | None = None,
        distribution: Literal["uniform", "poisson"] = "uniform",
        max_value: int = 10,
        rate: float = 5.0,
        name: Hashable | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> ParametrizedMeasure:
        """Generate a random parametrized measure.

        The measure dimension will be the last dimension of the domain. See the Examples below.

        Parameters
        ----------
        domain_dims : tuple[int]
            The dimensions of the domain of the function.
        output_name : Hashable, default="measure"
            The name of the outputs of the function.
        variable_names : list[Hashable] | None, default=None
            The names of the variables. If `None`, either `variable_name_prefix` will be used to generate names or default names will be generated.
        variable_name_prefix : str | None, default=None
            The prefix for generating variable names. If `None`, either default names will be generated or `variable_names` must be provided.
        distribution : Literal["uniform", "poisson"], default="uniform"
            The distribution to use for generating random values.
        max_value : int, default=10
            The maximum value for the uniform distribution.
        rate : float, default=5.0
            The rate parameter for the Poisson distribution.
        name : Hashable | None, default=None
            The name of the function. If `None`, a default name will be used.
        random_state : int | np.random.Generator | None, default=None
            The random state for reproducibility.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import ParametrizedMeasure
        >>> rng = np.random.default_rng(42)

        Generate a random parametrized measure with values drawn from a uniform distribution on the integers in `[0, 1111)`.

        >>> mu = ParametrizedMeasure.from_rand(
        ...     domain_dims=(2, 3),
        ...     variable_name_prefix="x",
        ...     distribution="uniform",
        ...     max_value=1111,
        ...     name="mu",
        ...     random_state=rng,
        ... )
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu':
                 measure
        x_0 x_1
        0   0         99
            1        859
            2        727
        1   0        487
            1        481
            2        953

        Hold a parameter fixed to obtain an actual measure.

        >>> print(mu(x_0=0))  # doctest: +NORMALIZE_WHITESPACE
        Measure 'mu(x_0=0)':
            measure
        x_1
        0        99
        1       859
        2       727

        Generate a random parametrized measure with values drawn from a Poisson distribution.

        >>> nu = ParametrizedMeasure.from_rand(
        ...     domain_dims=(2, 3),
        ...     variable_name_prefix="x",
        ...     distribution="poisson",
        ...     rate=3.0,
        ...     name="nu",
        ...     random_state=rng,
        ... )
        >>> print(nu)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'nu':
                 measure
        x_0 x_1
        0   0          3
            1          2
            2          5
        1   0          1
            1          7
            2          1

        Hold a parameter fixed to obtain an actual measure.

        >>> print(nu(x_0=0))  # doctest: +NORMALIZE_WHITESPACE
        Measure 'nu(x_0=0)':
            measure
        x_1
        0          3
        1          2
        2          5
        """
        from ..spaces.domain import Domain

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
        if distribution not in ("uniform", "poisson"):
            raise ValueError(f"Unsupported distribution: {distribution}")
        if not isinstance(max_value, int):
            raise TypeError("`max_value` must be an integer.")
        if max_value < 0:
            raise ValueError("`max_value` must be non-negative.")
        if not isinstance(rate, (int, float)):
            raise TypeError("`rate` must be a number.")
        if rate <= 0:
            raise ValueError("`rate` must be positive.")
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

        if distribution == "uniform":
            arr = rng.integers(low=0, high=max_value, size=domain_dims)
        elif distribution == "poisson":
            arr = rng.poisson(lam=rate, size=domain_dims)

        function = MultivariateFunction.from_numpy(
            arr=arr,
            output_name=output_name,
            variable_names=variable_names,
            variable_name_prefix=variable_name_prefix,
            name=name,
        )

        last_dim = domain_dims[-1]
        domain = function.domain
        measure_domain = Domain(
            list(range(last_dim)), variable_names=[domain.variable_names[-1]]
        )

        return function.to_measure(measure_domain=measure_domain, kind="measure")

    # --------------------- properties --------------------- #

    @property
    def parameter_domain(self) -> Domain | None:
        """Get the parameter domain of the parametrized measure.

        Returns
        -------
        parameter_domain : Domain | None
            The parameter domain associated with the parametrized measure, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedMeasure,
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
        ...     variable_names=["u"],
        ... )
        >>> mu = ParametrizedMeasure.from_domains(
        ...     measure_domain=F,
        ...     parameter_domain=Theta,
        ...     mapping=lambda *, theta, u: theta + u,
        ... )
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu':
                 measure
        theta u
        0     0        0
              1        1
        1     0        1
              1        2
        >>> print(mu.parameter_domain)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'Theta':
         theta
             0
             1
        """
        return self._parameter_domain

    @property
    def parameter_names(self) -> list[Hashable] | None:
        """Get the parameter names of the parametrized measure.

        Returns
        -------
        parameter_names : list[Hashable] | None
            The parameter names associated with the parametrized measure, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedMeasure,
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
        ...     variable_names=["u"],
        ... )
        >>> mu = ParametrizedMeasure.from_domains(
        ...     measure_domain=F,
        ...     parameter_domain=Theta,
        ...     mapping=lambda *, theta, u: theta + u,
        ... )
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu':
                    measure
        theta u
        0     0        0
              1        1
        1     0        1
              1        2
        >>> print(mu.parameter_names)
        ['theta']
        """
        if self._parameter_names is None and self.sig_alg is not None:
            self._parameter_names = [
                name
                for name in self.variable_names
                if name not in self.sig_alg.variable_names
            ]

        return self._parameter_names

    @property
    def measure_domain_names(self) -> list[Hashable] | None:
        """Get the domain names of the parametrized measure.

        Returns
        -------
        domain_names : list[Hashable] | None
            The domain names associated with the parametrized measure, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedMeasure,
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
        ...     variable_names=["u"],
        ... )
        >>> mu = ParametrizedMeasure.from_domains(
        ...     measure_domain=F,
        ...     parameter_domain=Theta,
        ...     mapping=lambda *, theta, u: theta + u,
        ... )
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu':
                    measure
        theta u
        0     0        0
              1        1
        1     0        1
              1        2
        >>> print(mu.measure_domain_names)
        ['u']
        """
        return self.sig_alg.variable_names if self.sig_alg is not None else None

    @property
    def sig_alg(self) -> SigmaAlgebra | None:
        """Get the sigma-algebra of the parametrized measure.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra associated with the parametrized measure, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedMeasure,
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
        ...     variable_names=["u"],
        ... )
        >>> mu = ParametrizedMeasure.from_domains(
        ...     measure_domain=F,
        ...     parameter_domain=Theta,
        ...     mapping=lambda *, theta, u: theta + u,
        ... )
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu':
                    measure
        theta u
        0     0        0
              1        1
        1     0        1
              1        2
        >>> print(mu.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
           u
        x
        0  0
        1  1
        2  1
        """
        return self._sig_alg

    @property
    def kind(self) -> Literal["measure", "probability"]:
        """Get the kind of the parametrized measure.

        Returns
        -------
        kind : Literal["measure", "probability"]
            The kind of the parametrized measure, which can be "measure" or "probability".

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedMeasure,
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
        ...     variable_names=["u"],
        ... )
        >>> mu = ParametrizedMeasure.from_domains(
        ...     measure_domain=F,
        ...     parameter_domain=Theta,
        ...     mapping=lambda *, theta, u: theta + u,
        ... )
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu':
                    measure
        theta u
        0     0        0
              1        1
        1     0        1
              1        2
        >>> print(mu.kind)
        measure
        """
        return self._kind

    # --------------------- probability methods --------------------- #

    def __rshift__(self, vec: MeasurableVector) -> ParametrizedMeasure:
        """Push forward the parametrized measure through a measurable vector.

        Calls the method `Operators.pushforward`. See the documentation for that method for more details.

        Parameters
        ----------
        vec : MeasurableVector
            The measurable vector along which to push forward the measure.

        Returns
        -------
        pushforward : ParametrizedMeasure
            The parametrized measure pushed forward along the measurable vector.
        """
        from ..functions.operators import Operators

        return Operators.pushforward(vec=vec, measure=self)

    # --------------------- data access methods --------------------- #

    def __call__(
        self, *args, **kwargs
    ) -> Real | MultivariateFunction | ParametrizedMeasure | Measure:
        """Call the parametrized measure with the given arguments.

        There are three types of arguments that may be provided to the `__call__` method:

        1. An instance of `MeasurableSet` in the sigma-algebra (or a list of points that can be made into one) as a positional argument.
        2. Parameters (of the parametrized measure) as keyword arguments.
        3. Atom identifiers of the sigma-algebra as keyword arguments.

        The `__call__` method will either return a real number, an instance of `MultivariateFunction`, an instance of `Measure`, or an instance of `ParametrizedMeasure` depending on what types of arguments are provided.

        1. If either an instance of `MeasurableSet` or *all* atom identifiers are provided, along with *all* parameters, then the measure of the measurable set is returned. If only a partial set of parameters is provided, then an instance of `MultivariateFunction` is returned.
        2. If no `MeasurableSet` is provided and no atom identifiers are provided, than either a `Measure` or `ParametrizedMeasure` is returned depending on whether all or only a partial set of parameters are provided.

        Parameters
        ----------
        *args : tuple
            Positional arguments.
        **kwargs : dict
            Keyword arguments.

        Raises
        ------
        ValueError
            If an invalid combination of positional and keyword arguments is provided, or if the measurable set (if passed) is not in the sigma-algebra of the parametrized measure.

        Examples
        --------
        Define a parametrized measure on a sigma-algebra with 2-dimensional atom identifiers and a 2-dimensional parameter domain.

        >>> from sigalg.core import Domain, ParametrizedMeasure, SigmaAlgebra
        >>> X = Domain.from_sequence(size=5)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: ("a", "b"),
        ...         1: ("a", "b"),
        ...         2: ("c", "d"),
        ...         3: ("e", "f"),
        ...         4: ("e", "f"),
        ...     },
        ...     variable_names=["letter1", "letter2"],
        ... )
        >>> Theta = Domain([(1, 2), (3, 4)], variable_names=["theta1", "theta2"])
        >>> def mapping(*, theta1, theta2, letter1, letter2):
        ...     return (
        ...         (theta1 + theta2)
        ...         * (ord(letter1) - ord("a") + 1)
        ...         * (ord(letter2) - ord("a") + 1)
        ...     )
        >>> mu = ParametrizedMeasure.from_domains(
        ...     measure_domain=F,
        ...     parameter_domain=Theta,
        ...     mapping=mapping,
        ... )
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu':
                                       measure
        theta1 theta2 letter1 letter2
        1      2      a       b              6
                      c       d             36
                      e       f             90
        3      4      a       b             14
                      c       d             84
                      e       f            210

        Get a measurable set from the sigma-algebra.

        >>> A = F.get_set([0, 1])

        Call with a `MeasurableSet` instance as a positional argument.

        >>> print(mu(A))  # doctest: +NORMALIZE_WHITESPACE
        Function 'mu(A)':
                       measure
        theta1 theta2
        1      2             6
        3      4            14

        Call with a list of points as a positional argument.

        >>> print(mu([0, 1]))  # doctest: +NORMALIZE_WHITESPACE
        Function 'mu(set)':
                       measure
        theta1 theta2
        1      2             6
        3      4            14

        Call with a `MeasurableSet` instance as a positional argument and some parameters as keyword arguments.

        >>> print(mu(A, theta1=1))  # doctest: +NORMALIZE_WHITESPACE
        Function 'mu(A)(theta1=1)':
                measure
        theta2
        2             6

        Call with a list of points as a positional argument and some parameters as keyword arguments.

        >>> print(mu([0, 1], theta1=1))  # doctest: +NORMALIZE_WHITESPACE
        Function 'mu(set)(theta1=1)':
                measure
        theta2
        2             6

        Call with a `MeasurableSet` instance as a positional argument and all parameters as keyword arguments.

        >>> print(mu(A, theta1=1, theta2=2))
        6

        Call with a list of points as a positional argument and all parameters as keyword arguments.

        >>> print(mu([0, 1], theta1=1, theta2=2))
        6

        Call with all parameters and all atom identifiers as keyword arguments.

        >>> print(mu(theta1=1, theta2=2, letter1="a", letter2="b"))
        6

        Call with partial atom identifiers and partial parameters as keyword arguments.

        >>> print(mu(theta1=1, letter1="a"))  # doctest: +NORMALIZE_WHITESPACE
        Function 'mu(theta1=1, letter1=a)':
                        measure
        theta2 letter2
        2      b              6

        Call with some parameters as keyword arguments but no measurable set.

        >>> print(mu(theta1=1))  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu(theta1=1)':
                                measure
        theta2 letter1 letter2
        2      a       b              6
               c       d             36
               e       f             90

        Call with all parameters as keyword arguments but no measurable set.

        >>> print(mu(theta1=1, theta2=2))  # doctest: +NORMALIZE_WHITESPACE
        Measure 'mu(theta1=1, theta2=2)':
                           measure
        letter1 letter2
        a       b                6
        c       d               36
        e       f               90
        >>>

        Obtain a measure with iterative calls.

        >>> print(mu(theta1=1)(theta2=2))  # doctest: +NORMALIZE_WHITESPACE
        Measure 'mu(theta1=1)(theta2=2)':
                         measure
        letter1 letter2
        a       b                6
        c       d               36
        e       f               90
        """
        from ..spaces.measurable_set import MeasurableSet
        from .measure import Measure
        from .parametrized_probability_measure import ParametrizedProbabilityMeasure

        if self.data is None:
            raise NotImplementedError(
                "The __call__ method is not yet implemented for parametrized measures without data."
            )

        if len(args) == 1:
            measurable_set = args[0]
            if isinstance(measurable_set, list):
                measurable_set = self.sig_alg.get_set(measurable_set, name="set")
            elif not isinstance(measurable_set, MeasurableSet):
                raise TypeError(
                    "The provided measurable_set (as a positional argument) must be an instance of MeasurableSet or a list of points."
                )
        elif len(args) > 1:
            raise ValueError(
                "Only one positional argument is allowed, which should be a measurable set."
            )
        else:
            measurable_set = None

        provided_parameters = {
            name: value
            for name, value in kwargs.items()
            if name in self.parameter_names
        }
        provided_atom_ids = {
            name: value
            for name, value in kwargs.items()
            if name in self.measure_domain_names
        }

        if measurable_set and provided_atom_ids:
            raise ValueError(
                "Cannot provide both a measurable set and atom identifiers as arguments."
            )
        if not set(provided_parameters) <= set(self.parameter_names):
            unknown_parameters = set(provided_parameters) - set(self.parameter_names)
            raise ValueError(
                f"Unknown parameter names: {unknown_parameters}. "
                f"Expected parameters from {self.parameter_names}"
            )
        if not set(provided_atom_ids) <= set(self.measure_domain_names):
            unknown_atom_ids = set(provided_atom_ids) - set(self.measure_domain_names)
            raise ValueError(
                f"Unknown atom identifier names: {unknown_atom_ids}. "
                f"Expected atom identifiers from {self.measure_domain_names}"
            )

        if set(provided_parameters) == set(self.parameter_names):
            mapping = super().__call__(**provided_parameters)

            measure = Measure(
                domain=self.sig_alg,
                mapping=mapping.function,
                name=mapping.name,
                output_name=self.output_name,
                kind=self.kind,
            )

            if measurable_set:
                return measure(measurable_set)
            elif provided_atom_ids:
                return measure(**provided_atom_ids)
            else:
                return measure

        if measurable_set:
            if self.data is None:
                raise ValueError(
                    "The data attribute of the parametrized measure is None. Cannot call the measure with a measurable set as an argument without all parameters."
                )

            # TODO: this is slow. a pure-pandas version that voids the call through `indicator` and RandomVariable?
            mapping = (
                (self.data * measurable_set.indicator.atom_data)
                .groupby(self.parameter_names)
                .sum()
            )

            function = MultivariateFunction(
                domain=self.parameter_domain,
                mapping=mapping,
                output_name=self.output_name,
                name=f"{self.name}({measurable_set.name})",
            )

            if not provided_parameters:
                return function
            else:
                return function(**provided_parameters)

        elif measurable_set is None:
            partial_function = super().__call__(
                **provided_atom_ids, **provided_parameters
            )

            if set(provided_atom_ids) == set(self.measure_domain_names):
                return partial_function
            elif not provided_atom_ids:
                measure = ParametrizedMeasure(
                    domain=partial_function.domain,
                    mapping=partial_function.function,
                    name=partial_function.name,
                    output_name=self.output_name,
                    # kind=self.kind,
                )
                # HACK: the call to the ParametrizedMeasure constructor uses input validation to screen out probability measures, so we manually change the class
                if self.kind == "probability":
                    measure.__class__ = ParametrizedProbabilityMeasure

                measure._init_measure_attrs(
                    parameter_domain=None,
                    sig_alg=self.sig_alg,
                    kind=self.kind,
                )

                return measure

            else:
                return partial_function

        else:
            raise ValueError(
                "Invalid combination of positional and keyword arguments. Please read the docstring of the `__call__` method of the `ParametrizedMeasure` class for valid argument combinations."
            )

    @staticmethod
    def _to_series(data: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(data, pd.Series):
            return data
        else:
            return data.apply(tuple, axis=1)

    def __iter__(self) -> Iterator[tuple[dict, Measure]]:
        """Pass."""
        unique_params = (
            self.data.index.to_frame()[self.parameter_names]
            .drop_duplicates()
            .iterrows()
        )
        for _, params in unique_params:
            yield params.to_dict(), self(**params.to_dict())

    # --------------------- equality --------------------- #

    # TODO: docstring!
    def __eq__(
        self,
        other: ParametrizedMeasure,
        rtol=1e-5,
        atol=1e-8,
    ) -> bool:
        """Test equality of two parametrized measures."""
        from ..utils.utils import _to_df

        if not isinstance(other, ParametrizedMeasure):
            return False
        if self.sig_alg != other.sig_alg:
            return False
        if self.parameter_names is None or other.parameter_names is None:
            return TypeError(
                "Cannot compare parametrized measures when one (or both) does not have parameter names."
            )
        if set(self.parameter_names) != set(other.parameter_names):
            return False
        if len(self.domain) != len(other.domain):
            return False

        parameter_names = self.parameter_names
        self_sig_alg_var_names = [
            f"{name}_self" for name in self.sig_alg.variable_names
        ]
        other_sig_alg_var_names = [
            f"{name}_other" for name in other.sig_alg.variable_names
        ]

        self_data = (
            self.data.reorder_levels(parameter_names + self.measure_domain_names)
            .sort_index(level=parameter_names)
            .reset_index(self.measure_domain_names)
            .add_suffix("_self")
            .reset_index()
        )
        other_data = (
            other.data.reorder_levels(parameter_names + other.measure_domain_names)
            .sort_index(level=parameter_names)
            .reset_index(other.measure_domain_names)
            .add_suffix("_other")
            .reset_index()
        )

        # print(self_data)
        # print(other_data)

        if isinstance(self.sig_alg.data.index, pd.MultiIndex):
            other_sig_alg_data = other.sig_alg.data.reorder_levels(
                self.sig_alg.domain.variable_names
            )
        else:
            other_sig_alg_data = other.sig_alg.data

        self_sig_alg_sorted = (
            _to_df(self.sig_alg.data.sort_index()).add_suffix("_self").reset_index()
        )
        other_sig_alg_sorted = (
            _to_df(other_sig_alg_data.sort_index()).add_suffix("_other").reset_index()
        )

        # print(self_sig_alg_sorted)
        # print(other_sig_alg_sorted)

        parameter_df = (
            self_data[parameter_names].drop_duplicates().reset_index(drop=True)
        )

        # TODO: check merge logic — possibly change to `on`?
        combined_sig_alg_data = pd.merge(
            left=parameter_df,
            right=self_sig_alg_sorted,
            how="cross",
        )
        combined_sig_alg_data = pd.merge(
            left=combined_sig_alg_data,
            right=other_sig_alg_sorted,
            on=self.sig_alg.domain.variable_names,
        )

        # print(combined_sig_alg_data)

        data = pd.merge(
            left=combined_sig_alg_data,
            right=self_data,
            on=parameter_names + self_sig_alg_var_names,
        )
        # print(data)
        data = pd.merge(
            left=data,
            right=other_data,
            on=parameter_names + other_sig_alg_var_names,
        )
        # print(data)

        return np.allclose(
            data[self.output_name + "_self"],
            data[other.output_name + "_other"],
            rtol=rtol,
            atol=atol,
        )

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a concise string representation of the parametrized measure.

        Returns
        -------
        repr_str : str
            The string representation of the parametrized measure.
        """
        if self.parameter_names is not None and self.measure_domain_names is not None:
            parameter_list = ", ".join(self.parameter_names)
            domain_list = ", ".join(self.measure_domain_names)
            return (
                f"{type(self)._repr_name}(parameters=({parameter_list}), "
                f"domain_vars=({domain_list}), "
                f"sig_alg={self.sig_alg.name}, "
                f"output_name={self.output_name}, "
                f"name={self.name})"
            )
        else:
            return type(self)._repr_name + "(empty)"
