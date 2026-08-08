"""A class representing a parametrized probability measure."""

from __future__ import annotations

import inspect
from collections.abc import Hashable
from typing import TYPE_CHECKING

import numpy as np

from .parametrized_measure import ParametrizedMeasure

if TYPE_CHECKING:
    from scipy.stats import rv_discrete

    from ...typing.index_like import IndexLike
    from ...typing.mapping_like import MappingLike
    from ...typing.measure_domain import MeasureDomain
    from ..spaces.domain import Domain


class ParametrizedProbabilityMeasure(ParametrizedMeasure):
    r"""A class representing a parametrized probability measure.

    The `__init__` constructor is not meant to be used directly. Instead, the user should use the `from_domains` class method.

    See the Notes section below for the mathematical details.

    Examples
    --------
    >>> from math import comb
    >>> from sigalg.core import (
    ...     Domain,
    ...     ParametrizedProbabilityMeasure,
    ...     SampleSpace,
    ... )

    Define a 1-dimensional parameter domain and sample space.

    >>> Theta = Domain([0.0, 0.25, 0.75, 1.0], name="Theta", variable_names=["theta"])
    >>> Omega = SampleSpace.from_sequence(size=3, variable_name="omega")

    Define a binomial probability distribution Bin(n=2,theta), parametrized by theta.

    >>> def mapping(*, theta, omega):
    ...     return comb(2, omega) * theta**omega * (1 - theta) ** (2 - omega)
    >>> P = ParametrizedProbabilityMeasure.from_domains(
    ...     measure_domain=Omega, parameter_domain=Theta, mapping=mapping
    ... )
    >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
    Parametrized probability measure 'P':
             probability
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

    Evaluate at a parameter to obtain a probability.

    >>> print(P(theta=0.25))  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'P(theta=0.25)':
           probability
    omega
    0           0.5625
    1           0.3750
    2           0.0625

    Notes
    -----
    Let $(\Omega, \mathcal{F})$ be a measurable space and $\Theta$ a nonempty set. A *parametrized probability measure* is a function

    $$
    P : \Theta \times \mathcal{F} \to \mathbb{R}
    $$

    such that, for each fixed $\theta \in \Theta$, the partial function

    $$
    P(\theta, -): \mathcal{F} \to \mathbb{R}, \quad U \mapsto P(\theta,U),
    $$

    is a probability measure on the $\sigma$-algebra $\mathcal{F}$. The set $\Theta$ is called the *parameter domain* and elements $\theta\in \Theta$ are called *parameters*.
    """

    _repr_name = "ParametrizedProbabilityMeasure"
    _str_name = "Parametrized probability measure"
    _default_name = "P"

    # --------------------- constructors --------------------- #

    @classmethod
    def from_domains(
        self,
        measure_domain: MeasureDomain,
        parameter_domain: IndexLike | None,
        mapping: MappingLike,
        output_name: str = "probability",
        name: Hashable = "P",
        **kwargs,
    ) -> ParametrizedProbabilityMeasure:
        """Construct a parametrized probability measure from a measure domain and parameter domain.

        Parameters
        ----------
        measure_domain : MeasureDomain
            The domain of the measure, if a `SigmaAlgebra` is provided. If an `IndexLike` object that can be coerced to a `Domain` is provided, the sigma-algebra of the measure will be the power-set sigma-algebra of the domain.
        parameter_domain : IndexLike | None
            The parameter domain for the measure.
        mapping : MappingLike
            The mapping of the parametrized measure.
        output_name : str, default="probability"
            The name of the output variable for the parametrized probability measure.
        name : Hashable | None, default="P"
            The name of the parametrized probability measure. If `None`, a default name will be assigned.
        **kwargs : Any
            Additional keyword arguments passed to the underlying constructor.

        Returns
        -------
        param_measure : ParametrizedProbabilityMeasure
            The constructed parametrized probability measure.

        Examples
        --------
        >>> from math import comb
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedProbabilityMeasure,
        ...     SampleSpace,
        ... )

        Define a 1-dimensional parameter domain and sample space.

        >>> Theta = Domain([0.0, 0.25, 0.75, 1.0], name="Theta", variable_names=["theta"])
        >>> Omega = SampleSpace.from_sequence(size=3, variable_name="omega")

        Define a binomial probability distribution Bin(n=2,theta), parametrized by theta.

        >>> def mapping(*, theta, omega):
        ...     return comb(2, omega) * theta**omega * (1 - theta) ** (2 - omega)
        >>> P = ParametrizedProbabilityMeasure.from_domains(
        ...     measure_domain=Omega, parameter_domain=Theta, mapping=mapping
        ... )
        >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'P':
                    probability
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

        Evaluate at a parameter to obtain a probability.

        >>> print(P(theta=0.25))  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P(theta=0.25)':
                probability
        omega
        0           0.5625
        1           0.3750
        2           0.0625
        """
        return super().from_domains(
            measure_domain=measure_domain,
            parameter_domain=parameter_domain,
            mapping=mapping,
            kind="probability",
            output_name=output_name,
            name=name,
            **kwargs,
        )

    @classmethod
    def from_rand(
        cls,
        domain_dims: tuple[int],
        output_name: Hashable = "probability",
        variable_names: list[Hashable] | None = None,
        variable_name_prefix: str | None = None,
        name: Hashable | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> ParametrizedProbabilityMeasure:
        """Generate a random parametrized probability measure.

        The measure dimension will be the last dimension of the domain. See the Examples below.

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
        name : Hashable | None, default=None
            The name of the function. If `None`, a default name will be used.
        random_state : int | np.random.Generator | None, default=None
            The random state for reproducibility.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import ParametrizedProbabilityMeasure
        >>> rng = np.random.default_rng(42)

        Generate a random parametrized probability measure.

        >>> P = ParametrizedProbabilityMeasure.from_rand(
        ...     domain_dims=(2, 3),
        ...     variable_name_prefix="x",
        ...     random_state=rng
        ... )
        >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'f':
                 probability
        x_0 x_1
        0   0       0.368430
            1       0.630960
            2       0.000610
        1   0       0.898607
            1       0.004016
            2       0.097377

        Hold a parameter fixed to obtain an actual probability measure.

        >>> print(P(x_0=0))  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'f(x_0=0)':
             probability
        x_1
        0        0.36843
        1        0.63096
        2        0.00061
        """
        from ..functions.multivariate_function import MultivariateFunction
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

        last_dim = domain_dims[-1]
        arr = rng.dirichlet(alpha=(1 / last_dim,) * last_dim, size=domain_dims[:-1])

        function = MultivariateFunction.from_numpy(
            arr=arr,
            output_name=output_name,
            variable_names=variable_names,
            variable_name_prefix=variable_name_prefix,
            name=name,
        )

        domain = function.domain
        measure_domain = Domain(
            list(range(last_dim)), variable_names=[domain.variable_names[-1]]
        )

        return function.to_measure(measure_domain=measure_domain, kind="probability")

    @classmethod
    def from_scipy(
        cls,
        dist: rv_discrete,
        support: tuple[Hashable, list],
        parameter_domain: Domain,
        name: Hashable = "P",
    ) -> ParametrizedProbabilityMeasure:
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

        Returns
        -------
        param_prob_measure : ParametrizedProbabilityMeasure
            The constructed parametrized probability measure.

        Examples
        --------
        >>> from scipy.stats import binom, hypergeom
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedProbabilityMeasure,
        ... )
        >>> Theta_P = Domain(
        ...     [(2, 0.25), (3, 0.75)],
        ...     name="Theta_P",
        ...     variable_names=["n", "p"],
        ... )
        >>> P = ParametrizedProbabilityMeasure.from_scipy(
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
        >>> Q = ParametrizedProbabilityMeasure.from_scipy(
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
        from scipy.stats import rv_discrete

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

        parameters = parameter_domain.variable_names
        parameter_names = [
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
            for name in parameters
        ] + [inspect.Parameter(support[0], inspect.Parameter.KEYWORD_ONLY)]
        sig = inspect.Signature(parameter_names)

        def mapping(**kwargs):
            bound = sig.bind(**kwargs)
            return dist.pmf(**bound.arguments)

        mapping.__signature__ = sig

        return cls.from_domains(
            measure_domain=sample_space,
            parameter_domain=parameter_domain,
            mapping=mapping,
            output_name="probability",
            name=name,
        )
