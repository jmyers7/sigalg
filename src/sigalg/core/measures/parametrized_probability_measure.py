"""A class representing a parametrized probability measure."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING

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
