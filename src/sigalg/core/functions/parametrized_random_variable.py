"""Marker class for a parametrized random variable."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

from .parametrized_measurable_function import ParametrizedMeasurableFunction

if TYPE_CHECKING:
    from ...typing.index_like import IndexLike
    from ...typing.mapping_like import MappingLike
    from ..measures.probability_measure import ProbabilityMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra


class ParametrizedRandomVariable(ParametrizedMeasurableFunction):
    """A class representing a parametrized random variable.

    The `__init__` constructor is not meant to be used directly. Instead, the user should use the `from_domains` class method.

    See the documentation for `ParametrizedMeasurableFunction` for more details.
    """

    _repr_name = "ParametrizedRandomVariable"
    _str_name = "Parametrized random variable"
    _default_name = "X"

    # --------------------- constructors --------------------- #

    @classmethod
    def from_domains(
        cls,
        measurable_domain: IndexLike,
        parameter_domain: IndexLike,
        sig_alg: SigmaAlgebra,
        mapping: MappingLike,
        measure: ProbabilityMeasure,
        name: Hashable = "X",
    ) -> ParametrizedRandomVariable:
        """Construct a parametrized random variable from a measurable domain and parameter domain.

        See the documentation for `ParametrizedMeasurableFunction` for usage examples.

        Parameters
        ----------
        measurable_domain : IndexLike
            The measurable domain of the random variable.
        parameter_domain : IndexLike
            The parameter domain of the random variable.
        sig_alg : SigmaAlgebra
            The sigma-algebra of the underlying measurable space.
        mapping : MappingLike
            The mapping of the parametrized random variable.
        measure : ProbabilityMeasure
            The probability measure on the underlying measurable space.
        name : Hashable, default="X"
            The name of the parametrized random variable.

        Returns
        -------
        param_rv : ParametrizedRandomVariable
            The constructed parametrized random variable.
        """
        from ..measures.probability_measure import ProbabilityMeasure

        if not isinstance(measure, ProbabilityMeasure):
            raise TypeError("The measure must be an instance of ProbabilityMeasure.")

        function = super().from_domains(
            measurable_domain=measurable_domain,
            parameter_domain=parameter_domain,
            sig_alg=sig_alg,
            measure=measure,
            mapping=mapping,
            name=name,
        )

        return function
