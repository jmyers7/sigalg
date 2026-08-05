"""A class representing a parametrized probability measure."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import numpy as np

from .parametrized_measure import ParametrizedMeasure

if TYPE_CHECKING:
    from ...typing.index_like import IndexLike
    from ...typing.mapping_like import MappingLike
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.domain import Domain


class ParametrizedProbabilityMeasure(ParametrizedMeasure):
    r"""A class representing a parametrized probability measure.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    measure_domain : SigmaAlgebra | IndexLike | None, default=None
        The domain of the probability measure, if a `SigmaAlgebra` is provided. If an `IndexLike` object that can be coerced to a `Domain` is provided, the sigma-algebra of the measure will be the power-set sigma-algebra of the domain.
    parameter_domain : Domain | None, default=None
        The domain of the parameters for the parametrized probability measure.
    domain : Domain | None, default=None
        The domain of the parametrized probability measure.
    mapping : MappingLike | None, default=None
        The mapping of the parametrized probability measure.
    output_name : str, default="probability"
        The name of the output variable for the parametrized probability measure.
    name : Hashable | None, default=None
        The name of the parametrized probability measure. If `None`, a default name will be assigned.
    **kwargs
        Keyword arguments to catch unexpected parameters.

    Examples
    --------
    >>> from math import comb
    >>> from sigalg.core import (
    ...     Domain,
    ...     ParametrizedProbabilityMeasure,
    ...     SampleSpace,
    ... )
    >>> Omega = SampleSpace.from_sequence(size=3, variable_name="omega")
    >>> Theta = Domain([0.0, 0.25, 0.75, 1.0], name="Theta", variable_names=["theta"])
    >>> def mapping(*, theta, omega):
    ...     return comb(2, omega) * theta**omega * (1 - theta) ** (2 - omega)
    >>> P = ParametrizedProbabilityMeasure(
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

    Notes
    -----
    Let $(\Omega, \mathcal{F})$ be a measurable space and $\Theta$ a nonempty set. A *parametrized probability measure* is a function

    $$
    P : \mathcal{F} \times \Theta \to \mathbb{R}
    $$

    such that, for each fixed $\theta \in \Theta$, the partial function

    $$
    P(-, \theta): \mathcal{F} \to \mathbb{R}, \quad U \mapsto P(U,\theta),
    $$

    is a probability measure on the $\sigma$-algebra $\mathcal{F}$. The set $\Theta$ is called the *parameter domain* and elements $\theta\in \Theta$ are called *parameters*.
    """

    _repr_name = "ParametrizedProbabilityMeasure"
    _str_name = "Parametrized probability measure"
    _default_name = "P"

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        measure_domain: SigmaAlgebra | IndexLike | None = None,
        parameter_domain: Domain | None = None,
        domain: Domain | None = None,
        mapping: MappingLike | None = None,
        output_name: str = "probability",
        name: Hashable | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            measure_domain=measure_domain,
            parameter_domain=parameter_domain,
            domain=domain,
            mapping=mapping,
            kind="probability",
            output_name=output_name,
            name=name,
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
