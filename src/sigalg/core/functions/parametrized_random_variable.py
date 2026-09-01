"""Marker class for a parametrized random variable."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .parametrized_measurable_function import ParametrizedMeasurableFunction

if TYPE_CHECKING:
    from collections.abc import Hashable

    from ...typing.mapping_like import MappingLike
    from ..measures.probability_measure import ProbabilityMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.domain import Domain


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
        measurable_domain: Domain,
        parameter_domain: Domain | None = None,
        complete_domain: Domain | None = None,
        sig_alg: SigmaAlgebra | None = None,
        measure: ProbabilityMeasure | None = None,
        mapping: MappingLike | None = None,
        parameter_names: list[Hashable] | None = None,
        complete_domain_name: Hashable | None = None,
        parameter_domain_name: Hashable | None = None,
        output_name: Hashable | None = None,
        name: Hashable | None = None,
    ) -> ParametrizedRandomVariable:
        """Construct a parametrized random variable from a measurable domain and parameter domain.

        See the documentation for `ParametrizedMeasurableFunction` for usage examples.

        Parameters
        ----------
        measurable_domain : IndexLike
            The measurable domain of the function.
        parameter_domain : IndexLike
            The parameter domain of the function.
        sig_alg : SigmaAlgebra
            The sigma-algebra of the underlying measurable space.
        measure : Probability | None, default=None
            The probability measure of the underlying probability space.
        mapping : MappingLike
            The mapping of the parametrized function.
        parameter_names : list[Hashable] | None, default=None
            The names of the parameters of the function.
        complete_domain_name : Hashable | None, default=None
            The name of the domain of the function.
        parameter_domain_name : Hashable | None, default=None
            The name of the parameter domain.
        output_name : Hashable | None, default=None
            The name of the output variable for the parametrized function. If `None`, a default will be generated.
        name : Hashable | None, default=None
            The name of the parametrized function. If `None`, a default will be generated.

        Returns
        -------
        param_rv : ParametrizedRandomVariable
            The constructed parametrized random variable.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedRandomVariable,
        ...     ProbabilityMeasure,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )

        Define a 1-dimensional parameter domain and measurable domain.

        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> Omega = SampleSpace.from_sequence(size=3)

        Define a sigma-algebra and probability measure.

        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     domain=F,
        ...     mapping={
        ...         0: 0.2,
        ...         1: 0.8,
        ...     },
        ... )

        Define the mapping of a parametrized random variable.

        >>> mapping = {
        ...     (0, 0): 1,  # (theta, omega) = (0, 0), etc ...
        ...     (0, 1): 2,
        ...     (0, 2): 2,
        ...     (1, 0): 0,
        ...     (1, 1): -3,
        ...     (1, 2): -3,
        ... }

        Instantiate a parametrized random variable and print it.

        >>> X = ParametrizedRandomVariable.from_domains(
        ...     measurable_domain=Omega,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     measure=P,
        ...     mapping=mapping,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized random variable 'X':
        theta  0  1
        omega
        0      1  0
        1      2 -3
        2      2 -3

        The printout displays the random variable as a 2-dimensional array, where the parameters run alog the horizontal axis and the "measurable" variables along the vertical.

        Evaluate the function at a parameter to obtain a random variable.

        >>> print(X(theta=0))  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X(theta=0)':
            X(theta=0)
        omega
        0           1
        1           2
        2           2

        Evaluate the function at a "measurable" variable to get an instance of `Function`.

        >>> print(X(omega=0))  # doctest: +NORMALIZE_WHITESPACE
        Function 'X(omega=0)':
                X(omega=0)
        theta
        0                1
        1                0
        """
        from ..measures.probability_measure import ProbabilityMeasure

        if not isinstance(measure, ProbabilityMeasure):
            raise TypeError("The measure must be an instance of ProbabilityMeasure.")

        function = super().from_domains(
            measurable_domain=measurable_domain,
            parameter_domain=parameter_domain,
            complete_domain=complete_domain,
            sig_alg=sig_alg,
            measure=measure,
            mapping=mapping,
            parameter_names=parameter_names,
            complete_domain_name=complete_domain_name,
            parameter_domain_name=parameter_domain_name,
            output_name=output_name,
            name=name,
        )

        return function
