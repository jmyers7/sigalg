"""A class representing a parametrized measure."""

from __future__ import annotations

import inspect
from collections.abc import Hashable
from itertools import product
from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy.stats import rv_discrete

from ..functions.multivariate_function import MultivariateFunction

if TYPE_CHECKING:
    from ...typing.mapping_like import MappingLike
    from ...typing.measure_domain import MeasureDomain
    from ..functions.measurable_vector import MeasurableVector
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.domain import Domain
    from .measure import Measure


class ParametrizedMeasure(MultivariateFunction):
    r"""A class representing a parametrized measure.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    measure_domain : MeasureDomain | None, default=None
        The domain of the measure, if a `SigmaAlgebra` is provided. If an `IndexLike` object that can be coerced to a `Domain` is provided, the sigma-algebra of the measure will be the power-set sigma-algebra of the domain.
    parameter_domain : Domain | None, default=None
        The domain of the parameters for the parametrized measure.
    domain : Domain | None, default=None
        The domain of the parametrized measure.
    mapping : MappingLike | None, default=None
        The mapping of the parametrized measure.
    kind : Literal["measure", "probability"], default="measure"
        The kind of the parametrized measure.
    output_name : str, default="measure"
        The name of the output variable for the parametrized measure.
    name : Hashable | None, default=None
        The name of the parametrized measure. If `None`, a default name will be assigned.

    Examples
    --------
    >>> from math import comb
    >>> from sigalg.core import (
    ...     Domain,
    ...     ParametrizedMeasure,
    ...     SampleSpace,
    ... )
    >>> Omega = SampleSpace.from_sequence(size=3, variable_name="omega")
    >>> Theta = Domain([0.0, 0.25, 0.75, 1.0], name="Theta", variable_names=["theta"])
    >>> def mapping(*, theta, omega):
    ...     return comb(2, omega) * theta**omega * (1 - theta) ** (2 - omega)
    >>> mu = ParametrizedMeasure(
    ...     measure_domain=Omega, parameter_domain=Theta, mapping=mapping
    ... )
    >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
    Parametrized measure 'mu':
                 measure
    theta omega
    0.00  0       1.0000
          1       0.0000
          2       0.0000
    0.25  0       0.5625
          1       0.3750
          2       0.0625
    0.75  0       0.0625
          1       0.3750
          2       0.5625
    1.00  0       0.0000
          1       0.0000
          2       1.0000

    Notes
    -----
    Let $(X, \mathcal{F})$ be a measurable space and $\Theta$ a nonempty set. A *parametrized measure* is a function

    $$
    \mu : \mathcal{F} \times X \to \mathbb{R}
    $$

    such that, for each fixed $\theta \in \Theta$, the partial function

    $$
    \mu(-, \theta): \mathcal{F} \to \mathbb{R}, \quad U \mapsto \mu(U,\theta),
    $$

    is a measure on the $\sigma$-algebra $\mathcal{F}$. The set $\Theta$ is called the *parameter domain* and elements $\theta\in \Theta$ are called *parameters*.
    """

    _repr_name = "ParametrizedMeasure"
    _str_name = "Parametrized measure"
    _default_name = "mu"
    _properties = MultivariateFunction._properties + ["_parameter_names"]

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        measure_domain: MeasureDomain | None = None,
        parameter_domain: Domain | None = None,
        domain: Domain | None = None,
        mapping: MappingLike | None = None,
        kind: Literal["measure", "probability"] = "measure",
        output_name: str = "measure",
        name: Hashable | None = None,
    ) -> None:
        from ...validation.measure_domain_validator import MeasureDomainValidator
        from .parametrized_probability_measure import ParametrizedProbabilityMeasure

        v = MeasureDomainValidator(measure_domain=measure_domain)

        measure_domain, parameter_domain, domain = self._generate_components(
            v.domain, parameter_domain, domain
        )

        if kind == "probability":
            self.__class__ = ParametrizedProbabilityMeasure
            output_name = "probability"

        super().__init__(
            domain=domain,
            mapping=mapping,
            output_name=output_name,
            name=name,
        )

        self._sig_alg = v.sig_alg
        self._measure_domain = measure_domain
        self._parameter_domain = parameter_domain
        self._kind = kind

    @classmethod
    def _generate_components(cls, measure_domain, parameter_domain, domain):
        from ..spaces.domain import Domain

        parameters_given = (
            measure_domain is not None,
            parameter_domain is not None,
            domain is not None,
        )
        if parameters_given == (1, 1, 1):
            raise ValueError(
                "If a measure_domain and parameter_domain are given, domain must be None."
            )
        elif parameters_given == (1, 1, 0):
            domain = Domain.cartesian_product([parameter_domain, measure_domain])
        elif parameters_given == (1, 0, 1):
            pass
        elif parameters_given == (1, 0, 0):
            pass
        elif parameters_given == (0, 1, 1):
            raise ValueError(
                "If parameter_domain is given, the measure_domain must be given and domain must be None."
            )
        elif parameters_given == (0, 1, 0):
            pass
        elif parameters_given == (0, 0, 1):
            raise ValueError(
                "If domain is given, the measure_domain must also be given."
            )
        elif parameters_given == (0, 0, 0):
            pass

        return measure_domain, parameter_domain, domain

    @staticmethod
    def _flatten(t):
        if isinstance(t[0], tuple) and isinstance(t[1], tuple):
            return t[0] + t[1]
        if isinstance(t[0], tuple) and not isinstance(t[1], tuple):
            return t[0] + (t[1],)
        if not isinstance(t[0], tuple) and isinstance(t[1], tuple):
            return (t[0],) + t[1]
        if not isinstance(t[0], tuple) and not isinstance(t[1], tuple):
            return t

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

    @classmethod
    def from_scipy(
        cls,
        dist: rv_discrete,
        support: tuple[Hashable, list],
        parameter_domain: Domain,
        name: Hashable = "P",
    ) -> ParametrizedMeasure:
        """Initialize the parametrized probability measure from a discrete SciPy probability distribution.

        Parameters
        ----------
        dist : rv_discrete
            A discrete SciPy probability distribution.
        support : tuple[Hashable, list]
            A tuple containing the name of the support variable and a list of its possible values.
        parameter_domain : Domain
            The domain of the parameters for the parametrized probability measure.
        name : Hashable, default="P"
            The name of the parametrized probability measure.

        Raises
        ------
        TypeError
            If `dist` is not a discrete SciPy distribution, or if `parameter_domain` is not an instance of `Domain`, or if `support` is not a 2-tuple of a hashable name and a list of values.
        ValueError
            If `support` is not a 2-tuple of a hashable name and a list of values.

        Examples
        --------
        >>> from scipy.stats import binom, hypergeom
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedMeasure,
        ... )
        >>> Theta_P = Domain(
        ...     [(2, 0.25), (3, 0.75)],
        ...     name="Theta_P",
        ...     variable_names=["n", "p"],
        ... )
        >>> P = ParametrizedMeasure.from_scipy(
        ...     dist=binom,
        ...     support=("k", [0, 1, 2, 3]),
        ...     parameter_domain=Theta_P,
        ... )
        >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'P':
                    probability
        n p    k
        2 0.25 0       0.562500
               1       0.375000
               2       0.062500
               3       0.000000
        3 0.75 0       0.015625
               1       0.140625
               2       0.421875
               3       0.421875
        >>> Theta_Q = Domain(
        ...     [(5, 3, 3), (10, 5, 5)],
        ...     name="Theta_Q",
        ...     variable_names=["M", "n", "N"],
        ... )
        >>> Q = ParametrizedMeasure.from_scipy(
        ...     dist=hypergeom,
        ...     support=("k", [0, 1, 2, 3, 4, 5]),
        ...     parameter_domain=Theta_Q,
        ...     name="Q",
        ... )
        >>> print(Q)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'Q':
                  probability
        M  n N k
        5  3 3 0     0.000000
               1     0.300000
               2     0.600000
               3     0.100000
               4     0.000000
               5     0.000000
        10 5 5 0     0.003968
               1     0.099206
               2     0.396825
               3     0.396825
               4     0.099206
               5     0.003968
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from ..spaces.domain import Domain
        from ..spaces.sample_space import SampleSpace

        if not isinstance(parameter_domain, Domain):
            raise TypeError("parameter_domain must an instance of Domain")
        if not isinstance(dist, rv_discrete):
            raise TypeError("dist must be a discrete scipy distribution (rv_discrete)")
        if not isinstance(support, tuple) or len(support) != 2:
            raise ValueError("support must be a 2-tuple (name, values)")
        if not isinstance(support[0], Hashable):
            raise TypeError("support[0] must be hashable")
        if not isinstance(support[1], list):
            raise TypeError("support[1] must be a list")

        sample_space = SampleSpace(support[1], variable_names=[support[0]])
        sig_alg = SigmaAlgebra.power_set(sample_space)
        parameters = parameter_domain.variable_names

        tuples = list(product(parameter_domain.data, sig_alg.atom_space.data))
        tuples = [cls._flatten(t) for t in tuples]
        data = pd.MultiIndex.from_tuples(
            tuples,
            names=parameter_domain.data.names + sig_alg.atom_space.data.names,
        )
        domain = Domain(data)

        parameter_names = [
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
            for name in parameters
        ] + [inspect.Parameter(support[0], inspect.Parameter.KEYWORD_ONLY)]
        sig = inspect.Signature(parameter_names)

        def mapping(**kwargs):
            bound = sig.bind(**kwargs)
            return dist.pmf(**bound.arguments)

        mapping.__signature__ = sig

        return cls(
            measure_domain=sig_alg,
            domain=domain,
            mapping=mapping,
            name=name,
            kind="probability",
        )

    # --------------------- properties --------------------- #

    @property
    def sig_alg(self) -> SigmaAlgebra | None:
        """Get the sigma-algebra of the parametrized measure.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra associated with the parametrized measure, or `None` if not set.

        Examples
        --------
        >>> from math import comb
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedMeasure,
        ...     SampleSpace,
        ... )
        >>> Omega = SampleSpace.from_sequence(size=3, variable_name="omega")
        >>> Theta = Domain([0.0, 0.25, 0.75, 1.0], name="Theta", variable_names=["theta"])
        >>> def mapping(*, theta, omega):
        ...     return comb(2, omega) * theta**omega * (1 - theta) ** (2 - omega)
        >>> mu = ParametrizedMeasure(
        ...     measure_domain=Omega, parameter_domain=Theta, mapping=mapping
        ... )
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu':
                     measure
        theta omega
        0.00  0       1.0000
              1       0.0000
              2       0.0000
        0.25  0       0.5625
              1       0.3750
              2       0.0625
        0.75  0       0.0625
              1       0.3750
              2       0.5625
        1.00  0       0.0000
              1       0.0000
              2       1.0000
        >>> print(mu.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'power_set':
               atom_ID
        omega
        0            0
        1            1
        2            2
        """
        return self._sig_alg

    @property
    def measure_domain(self) -> SigmaAlgebra | None:
        """Get the measure domain of the parametrized measure, i.e., the sigma-algebra associated with the measure.

        Returns
        -------
        measure_domain : SigmaAlgebra | None
            The measure domain associated with the parametrized measure, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import Domain, ParametrizedMeasure, SigmaAlgebra
        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> X = Domain.from_sequence(size=3, variable_name="u")
        >>> def mapping(*, theta, u):
        ...     return theta + u
        >>> mu = ParametrizedMeasure(
        ...     measure_domain=X,
        ...     parameter_domain=Theta,
        ...     mapping=mapping,
        ... )
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu':
                 measure
        theta u
        0     0        0
              1        1
              2        2
        1     0        1
              1        2
              2        3
        >>> print(mu.measure_domain)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'X':
         u
         0
         1
         2
        """
        return self._measure_domain

    @property
    def parameter_domain(self) -> Domain | None:
        """Get the parameter domain of the parametrized measure.

        Returns
        -------
        parameter_domain : Domain | None
            The parameter domain associated with the parametrized measure, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import Domain, ParametrizedMeasure, SigmaAlgebra
        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> X = Domain.from_sequence(size=3, variable_name="u")
        >>> def mapping(*, theta, u):
        ...     return theta + u
        >>> mu = ParametrizedMeasure(
        ...     measure_domain=X,
        ...     parameter_domain=Theta,
        ...     mapping=mapping,
        ... )
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu':
                 measure
        theta u
        0     0        0
              1        1
              2        2
        1     0        1
              1        2
              2        3
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
        >>> from sigalg.core import Domain, ParametrizedMeasure, SigmaAlgebra
        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> X = Domain.from_sequence(size=3, variable_name="u")
        >>> def mapping(*, theta, u):
        ...     return theta + u
        >>> mu = ParametrizedMeasure(
        ...     measure_domain=X,
        ...     parameter_domain=Theta,
        ...     mapping=mapping,
        ... )
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu':
                 measure
        theta u
        0     0        0
              1        1
              2        2
        1     0        1
              1        2
              2        3
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
    def domain_names(self) -> list[Hashable] | None:
        """Get the domain names of the parametrized measure.

        Returns
        -------
        domain_names : list[Hashable] | None
            The domain names associated with the parametrized measure, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import Domain, ParametrizedMeasure, SigmaAlgebra
        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> X = Domain.from_sequence(size=3, variable_name="u")
        >>> def mapping(*, theta, u):
        ...     return theta + u
        >>> mu = ParametrizedMeasure(
        ...     measure_domain=X,
        ...     parameter_domain=Theta,
        ...     mapping=mapping,
        ... )
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu':
                 measure
        theta u
        0     0        0
              1        1
              2        2
        1     0        1
              1        2
              2        3
        >>> print(mu.domain_names)
        ['u']
        """
        return self.sig_alg.variable_names if self.sig_alg is not None else None

    @property
    def kind(self) -> Literal["measure", "probability"]:
        """Get the kind of the parametrized measure.

        Returns
        -------
        kind : Literal["measure", "probability"]
            The kind of the parametrized measure, which can be "measure" or "probability".
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
        >>> mu = ParametrizedMeasure(
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

        if self.measure_domain is None:
            raise ValueError(
                "The measure_domain of the parametrized measure is None. "
                "Cannot evaluate the measure without a measure_domain."
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
            name: value for name, value in kwargs.items() if name in self.domain_names
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
        if not set(provided_atom_ids) <= set(self.domain_names):
            unknown_atom_ids = set(provided_atom_ids) - set(self.domain_names)
            raise ValueError(
                f"Unknown atom identifier names: {unknown_atom_ids}. "
                f"Expected atom identifiers from {self.domain_names}"
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

            sig_alg_data_df = pd.DataFrame(
                self.sig_alg.data.to_list(),
                index=self.sig_alg.data.index,
                columns=self.domain_names,
            )

            combined_data = pd.concat(
                [
                    measurable_set.indicator.data.rename("ind"),
                    sig_alg_data_df,
                ],
                axis=1,
            )

            atom_indicator = (
                combined_data.drop_duplicates().set_index(self.domain_names).squeeze()
            )

            data = (
                self.data.unstack(self.domain_names, sort=False)
                .fillna(0.0)
                .dot(atom_indicator)
                .rename(self.output_name)
            )

            function = MultivariateFunction(
                domain=self.parameter_domain,
                mapping=data,
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

            if set(provided_atom_ids) == set(self.domain_names):
                return partial_function
            elif not provided_atom_ids:
                return ParametrizedMeasure(
                    measure_domain=self.sig_alg,
                    domain=partial_function.domain,
                    mapping=partial_function.function,
                    name=partial_function.name,
                    output_name=self.output_name,
                    kind=self.kind,
                )
            else:
                return partial_function

        else:
            raise ValueError(
                "Invalid combination of positional and keyword arguments. Please read the docstring of the `__call__` method of the `ParametrizedMeasure` class for valid argument combinations."
            )

    def __repr__(self) -> str:
        """Return a concise string representation of the parametrized measure.

        Returns
        -------
        repr_str : str
            The string representation of the parametrized measure.
        """
        if self.parameter_names is not None and self.domain_names is not None:
            parameter_list = ", ".join(self.parameter_names)
            domain_list = ", ".join(self.domain_names)
            return (
                f"{type(self)._repr_name}(parameter_names=({parameter_list}), "
                f"domain_var_names=({domain_list}), "
                f"domain={self.measure_domain.name}, "
                f"output_name={self.output_name}, "
                f"name={self.name})"
            )
        else:
            return type(self)._repr_name + "(empty)"
