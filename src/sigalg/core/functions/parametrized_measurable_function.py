"""A class representing a parametrized measure."""

from __future__ import annotations

from collections.abc import Callable
from functools import cached_property
from numbers import Real
from typing import TYPE_CHECKING, Literal

from .function import Function

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator

    import pandas as pd

    from ...typing.mapping_like import MappingLike
    from ..indices.index import Index
    from ..measures.measure import Measure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.domain import Domain
    from ..spaces.measurable_space import MeasurableSpace
    from ..spaces.measure_space import MeasureSpace
    from .measurable_function import MeasurableFunction

from .operators import OperatorsMethods


class ParametrizedMeasurableFunction(Function, OperatorsMethods):
    r"""A class representing a parametrized measurable function.

    The `__init__` constructor is not meant to be used directly. Instead, the user should use the `from_domains` class method.

    See the Notes section below for the mathematical details.

    Examples
    --------
    >>> from sigalg.core import (
    ...     Domain,
    ...     ParametrizedMeasurableFunction,
    ...     SigmaAlgebra,
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

    >>> mapping = {
    ...     (0, 0): 1,  # (theta, x) = (0, 0), etc ...
    ...     (0, 1): 2,
    ...     (0, 2): 2,
    ...     (1, 0): 0,
    ...     (1, 1): -3,
    ...     (1, 2): -3,
    ... }

    Instantiate a parametrized measurable function and print it.

    >>> f = ParametrizedMeasurableFunction.from_domains(
    ...     measurable_domain=X,
    ...     parameter_domain=Theta,
    ...     sig_alg=F,
    ...     mapping=mapping,
    ... )
    >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
    Parametrized measurable function 'f':
    theta  0  1
    x
    0      1  0
    1      2 -3
    2      2 -3

    The printout displays the function as a 2-dimensional array, where the parameters run alog the horizontal axis and the "measurable" variables along the vertical.

    Evaluate the function at a parameter to obtain a measurable function.

    >>> print(f(theta=0))  # doctest: +NORMALIZE_WHITESPACE
    Measurable function 'f(theta=0)':
       f(theta=0)
    x
    0           1
    1           2
    2           2

    Evaluate the function at a "measurable" variable to get an instance of `Function`.

    >>> print(f(x=0))  # doctest: +NORMALIZE_WHITESPACE
    Function 'f(x=0)':
           f(x=0)
    theta
    0           1
    1           0

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
    _properties = Function._properties + []

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
        parameter_names: list[Hashable] | None = None,
        complete_domain_name: Hashable | None = None,
        parameter_domain_name: Hashable | None = None,
        output_name: Hashable | None = None,
        name: Hashable | None = None,
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

        >>> mapping = {
        ...     (0, 0): 1,  # (theta, x) = (0, 0), etc ...
        ...     (0, 1): 2,
        ...     (0, 2): 2,
        ...     (1, 0): 0,
        ...     (1, 1): -3,
        ...     (1, 2): -3,
        ... }

        Instantiate a parametrized measurable function and print it.

        >>> f = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     mapping=mapping,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'f':
        theta  0  1
        x
        0      1  0
        1      2 -3
        2      2 -3

        The printout displays the function as a 2-dimensional array, where the parameters run alog the horizontal axis and the "measurable" variables along the vertical.

        Evaluate the function at a parameter to obtain a measurable function.

        >>> print(f(theta=0))  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'f(theta=0)':
            f(theta=0)
        x
        0           1
        1           2
        2           2

        Evaluate the function at a "measurable" variable to get an instance of `Function`.

        >>> print(f(x=0))  # doctest: +NORMALIZE_WHITESPACE
        Function 'f(x=0)':
                f(x=0)
        theta
        0           1
        1           0

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
        from ...validation.measurable_func_normalizer import MeasurableFuncNormalizer
        from ...validation.parametrized_domain_constructor import (
            ParametrizedDomainConstructor,
        )
        from ..measures.probability_measure import ProbabilityMeasure
        from .parametrized_random_variable import ParametrizedRandomVariable

        if measurable_domain is None and sig_alg is None:
            raise TypeError(
                "One or the other of measurable_domain or sig_alg must be given."
            )

        u = MeasurableFuncNormalizer(
            domain=measurable_domain,
            sig_alg=sig_alg,
            measure=measure,
        )

        measurable_domain = u.domain
        sig_alg = u.sig_alg
        measure = u.measure

        v = ParametrizedDomainConstructor(
            component_domain=measurable_domain,
            parameter_domain=parameter_domain,
            complete_domain=complete_domain,
            parameter_names=parameter_names,
            parameter_domain_name=parameter_domain_name,
            complete_domain_name=complete_domain_name,
        )

        complete_domain = v.complete_domain
        parameter_names = v.parameter_names
        parameter_domain_name = v.parameter_domain_name

        function = cls(
            domain=complete_domain,
            mapping=mapping,
            output_name=output_name,
            parameter_names=parameter_names,
            name=name,
        )

        function.sig_alg = sig_alg
        function.measure = measure
        function.parameter_names = parameter_names
        function.parameter_domain_name = parameter_domain_name

        if parameter_domain is not None and sig_alg not in function.lattice:
            raise ValueError(
                "For each parameter level, the function needs to be measurable with respect to the given sigma-algebra. This is not true."
            )

        if isinstance(measure, ProbabilityMeasure):
            function.__class__ = ParametrizedRandomVariable

        return function

    @classmethod
    def _from_validated(
        cls,
        *,
        data: pd.Series,
        sig_alg: SigmaAlgebra,
        measure: Measure | None,
        complete_domain_name: Hashable | None,
        parameter_domain_name: Hashable | None,
        parameter_names: list[Hashable],
        name: Hashable,
        **kwargs,
    ) -> ParametrizedMeasurableFunction:
        from ..measures.probability_measure import ProbabilityMeasure
        from .parametrized_random_variable import ParametrizedRandomVariable

        function = super()._from_validated(
            data=data,
            kind="any",
            name=name,
            domain_kind="Domain",
            domain_name=complete_domain_name,
            index_kind="Index",
            index_name=None,
        )
        function.sig_alg = sig_alg
        function.measure = measure
        function.parameter_names = parameter_names
        function.parameter_domain_name = parameter_domain_name

        if isinstance(measure, ProbabilityMeasure):
            function.__class__ = ParametrizedRandomVariable

        return function

    # --------------------- properties --------------------- #

    @cached_property
    def parameter_domain(self) -> Domain | None:
        """Get the parameter domain of the function.

        Returns
        -------
        parameter_domain : Domain | None
            The parameter domain of the function, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import Domain, ParametrizedMeasurableFunction, SigmaAlgebra
        >>> Theta = Domain.cartesian_power(
        ...     [0, 1], n=2, variable_names=["theta_0", "theta_1"], name="Theta"
        ... )
        >>> X = Domain.from_sequence(size=3, variable_name="x")
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: (0, 1),
        ...         1: (1, 2),
        ...         2: (1, 2),
        ...     },
        ... )
        >>> mapping = {
        ...     (0, 0, 0): 0,  # (theta_0, theta_1, x) = (0, 0, 0), etc...
        ...     (0, 0, 1): 1,
        ...     (0, 0, 2): 1,
        ...     (0, 1, 0): 2,
        ...     (0, 1, 1): 8,
        ...     (0, 1, 2): 8,
        ...     (1, 0, 0): 4,
        ...     (1, 0, 1): 1,
        ...     (1, 0, 2): 1,
        ...     (1, 1, 0): 5,
        ...     (1, 1, 1): 0,
        ...     (1, 1, 2): 0,
        ... }
        >>> f = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     mapping=mapping,
        ... )
        >>> print(f.parameter_domain)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'Theta':
         theta_0  theta_1
               0        0
               0        1
               1        0
               1        1
        """
        from ..spaces.domain import Domain

        if self.data is not None:
            data = (
                self.data.reset_index()[self.parameter_names]
                .drop_duplicates()
                .set_index(self.parameter_names)
                .index
            )

            return Domain._from_validated(data=data, name=self.parameter_domain_name)
        else:
            return None

    @cached_property
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
        >>> mapping = {
        ...     (0, 0): 1,  # (theta, x) = (0, 0), etc ...
        ...     (0, 1): 2,
        ...     (0, 2): 2,
        ...     (1, 0): 0,
        ...     (1, 1): -3,
        ...     (1, 2): -3,
        ... }
        >>> f = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     mapping=mapping,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'f':
        theta  0  1
        x
        0      1  0
        1      2 -3
        2      2 -3
        >>> print(f.measurable_domain)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'X':
         x
         0
         1
         2
        """
        return self.sig_alg.domain if self.sig_alg else None

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
        >>> mapping = {
        ...     (0, 0): 1,  # (theta, x) = (0, 0), etc ...
        ...     (0, 1): 2,
        ...     (0, 2): 2,
        ...     (1, 0): 0,
        ...     (1, 1): -3,
        ...     (1, 2): -3,
        ... }
        >>> f = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     mapping=mapping,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'f':
        theta  0  1
        x
        0      1  0
        1      2 -3
        2      2 -3
        >>> f.measurable_names
        ['x']
        """
        return self.measurable_domain.variable_names

    @cached_property
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
        >>> mapping = {
        ...     (0, 0): 1,  # (theta, x) = (0, 0), etc ...
        ...     (0, 1): 2,
        ...     (0, 2): 2,
        ...     (1, 0): 0,
        ...     (1, 1): -3,
        ...     (1, 2): -3,
        ... }
        >>> f = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     mapping=mapping,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'f':
        theta  0  1
        x
        0      1  0
        1      2 -3
        2      2 -3
        >>> f.measurable_space
        MeasurableSpace(domain=X, sig_alg=F)
        """
        from ..spaces.measurable_space import MeasurableSpace

        return MeasurableSpace._from_validated(sig_alg=self.sig_alg)

    @cached_property
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
        >>> mapping = {
        ...     (0, 0): 1,  # (theta, x) = (0, 0), etc ...
        ...     (0, 1): 2,
        ...     (0, 2): 2,
        ...     (1, 0): 0,
        ...     (1, 1): -3,
        ...     (1, 2): -3,
        ... }
        >>> f = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     measure=mu,
        ...     mapping=mapping,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'f':
        theta  0  1
        x
        0      1  0
        1      2 -3
        2      2 -3
        >>> f.measure_space
        MeasureSpace(domain=X, sig_alg=F, measure=mu)
        """
        from ..spaces.measure_space import MeasureSpace

        return (
            MeasureSpace._from_validated(measure=self.measure)
            if self.measure is not None
            else None
        )

    @cached_property
    def generated_sig_alg(self) -> SigmaAlgebra | None:
        r"""Get the sigma-algebra generated by the function.

        See the Notes section below for the mathematical details.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra generated by the parametrized measurable function.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     ParametrizedMeasurableFunction,
        ...     SigmaAlgebra,
        ... )

        Define a 1-dimensional domain, a 1-dimensional parameter space, and a sigma-algebra.

        >>> X = Domain.from_sequence(size=3)
        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     },
        ... )

        Define a parametrized measurable function.

        >>> mapping = {
        ...     (0, 0): 1,  # (theta, x) = (0, 0), etc ...
        ...     (0, 1): 2,
        ...     (0, 2): 2,
        ...     (1, 0): -3,
        ...     (1, 1): 4,
        ...     (1, 2): 4,
        ... }
        >>> f = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     mapping=mapping,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'f':
        theta  0  1
        x
        0      1 -3
        1      2  4
        2      2  4

        Print the sigma-algebra generated by `f` and its atom space.

        >>> print(f.generated_sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'sigma(f)':
        theta  0  1
        x
        0      1 -3
        1      2  4
        2      2  4
        >>> print(f.generated_sig_alg.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'sigma(f)':
         f_0  f_1
           1   -3
           2    4

        For a second example, consider a function parametrized by a 2-dimensional space.

        >>> Phi = Domain.cartesian_power([0, 1], n=2, variable_names=["phi_0", "phi_1"], name="Phi")
        >>> mapping = {
        ...     (0, 0, 0): 1,  # (phi_0, phi_1, x) = (0, 0, 0), etc ...
        ...     (0, 0, 1): 2,
        ...     (0, 0, 2): 2,
        ...     (0, 1, 0): -3,
        ...     (0, 1, 1): 4,
        ...     (0, 1, 2): 4,
        ...     (1, 0, 0): 0,
        ...     (1, 0, 1): 1,
        ...     (1, 0, 2): 1,
        ...     (1, 1, 0): 8,
        ...     (1, 1, 1): 0,
        ...     (1, 1, 2): 0,
        ... }
        >>> g = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Phi,
        ...     sig_alg=F,
        ...     mapping=mapping,
        ...     name="g",
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'g':
        phi_0  0     1
        phi_1  0  1  0  1
        x
        0      1 -3  0  8
        1      2  4  1  0
        2      2  4  1  0

        Print the sigma-algebra generated by `g` and its atom space.

        >>> print(g.generated_sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'sigma(g)':
        phi_0  0     1
        phi_1  0  1  0  1
        x
        0      1 -3  0  8
        1      2  4  1  0
        2      2  4  1  0

        >>> print(g.generated_sig_alg.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Domain 'sigma(g)':
         g_0_0  g_0_1  g_1_0  g_1_1
             1     -3      0      8
             2      4      1      0

        Notes
        -----
        Let $f: \Theta \times X \to \mathbb{R}$ be a parametrized measurable function relative to a $\sigma$-algebra on $X$. Then for each parameter $\theta \in \Theta$ we have the partial function

        $$
        f_\theta : X \to \mathbb{R}, \quad x \mapsto f(\theta,x).
        $$

        We define the *$\sigma$-algebra generated by $f$*, denoted $\sigma(f)$, to be the $\sigma$-algebra given by the join

        $$
        \sigma(f) = \vee_{\theta\in \Theta} \sigma(f_\theta).
        $$
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from ..spaces.domain import Domain

        if self.data is not None:
            data = self.data.unstack(level=self.parameter_names)

            implied_domain = Domain._from_validated(data=data.index, name="implied")

            if implied_domain != self.measurable_domain or data.isna().sum().sum() != 0:
                raise ValueError(
                    "For each parameter level, the function needs to be defined on the entire measurable domain. This is not true."
                )

            def _normalize_param(param):
                return (
                    str(param).replace("(", "").replace(")", "").replace(", ", "_")
                    if isinstance(param, tuple)
                    else param
                )

            variable_names = [
                f"{self.name}_{_normalize_param(param)}"
                for param in self.parameter_domain.data.to_list()
            ]

            return SigmaAlgebra._from_validated(
                data=data,
                variable_names=variable_names,
                name=f"sigma({self.name})",
                domain_kind=type(self.measurable_domain).__name__,
                domain_name=self.measurable_domain.name,
                index_kind="Index",
                index_name=self.parameter_domain_name,
            )

    # --------------------- data methods --------------------- #

    def __call__(
        self, *args, **kwargs
    ) -> Real | MeasurableFunction | Function | ParametrizedMeasurableFunction:
        """Call the parametrized function.

        The return value is determined by the following rules:

        1. If all parameters and all measurable arguments are provided (as keyword arguments), a real number is returned.

        2. If all parameters are provided (as keyword arguments) and a `Set` on which the component functions are constant is provided (as a positional argument), a real number is returned.

        3. If all parameters are provided (as keyword arguments) but no measurable arguments are provided, a measurable function is returned.

        4. If a partial set of parameters is provided (as keyword arguments) and no other arguments are provided, a parametrized measurable function is provided.

        5. In all other cases, a `Function` is returned.

        Parameters
        ----------
        *args : positional arguments
            Positional arguments for the function.
        **kwargs : keyword arguments
            Keyword arguments for the function.

        Examples
        --------
        >>> from sigalg.core import Domain, ParametrizedMeasurableFunction, Set, SigmaAlgebra

        Define a 2-dimensional parameter space, a 1-dimensional domain, and a sigma-algebra with 2-dimensional atom identifiers.

        >>> Theta = Domain.cartesian_power(
        ...     [0, 1], n=2, variable_names=["theta_0", "theta_1"], name="Theta"
        ... )
        >>> X = Domain.from_sequence(size=4, variable_name="x")
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: (0, 1),
        ...         1: (1, 2),
        ...         2: (1, 2),
        ...         3: (2, 3),
        ...     },
        ... )

        Define a parametrized measurable function.

        >>> mapping = {
        ...     (0, 0, 0): 0,  # (theta_0, theta_1, x) = (0, 0, 0), etc...
        ...     (0, 0, 1): 1,
        ...     (0, 0, 2): 1,
        ...     (0, 0, 3): 1,
        ...     (0, 1, 0): 2,
        ...     (0, 1, 1): 8,
        ...     (0, 1, 2): 8,
        ...     (0, 1, 3): 8,
        ...     (1, 0, 0): 4,
        ...     (1, 0, 1): 1,
        ...     (1, 0, 2): 1,
        ...     (1, 0, 3): 1,
        ...     (1, 1, 0): 5,
        ...     (1, 1, 1): 0,
        ...     (1, 1, 2): 0,
        ...     (1, 1, 3): 0,
        ... }
        >>> f = ParametrizedMeasurableFunction.from_domains(
        ...     measurable_domain=X,
        ...     parameter_domain=Theta,
        ...     sig_alg=F,
        ...     mapping=mapping,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'f':
        theta_0  0     1
        theta_1  0  1  0  1
        x
        0        0  2  4  5
        1        1  8  1  0
        2        1  8  1  0
        3        1  8  1  0

        Rules 1 and 2: Calling with complete sets of parameters and either a `Set` instance, a list of points, or a complete set of measurable arguments.

        >>> U = Set([1, 2, 3], domain=X, name="U")
        >>> f(U, theta_0=0, theta_1=0)
        1
        >>> f([1, 2, 3], theta_0=0, theta_1=0)
        1
        >>> f(x=2, theta_0=0, theta_1=1)
        8

        Rule 3: Calling with a complete set of parameters, and no other arguments.

        >>> print(f(theta_0=0, theta_1=1))  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'f(theta_0=0, theta_1=1)':
           f(theta_0=0, theta_1=1)
        x
        0                        2
        1                        8
        2                        8
        3                        8

        Rule 4: Calling with a partial set of parameters, and no other arguments.

        >>> print(f(theta_0=0))  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'f(theta_0=0)':
        theta_1  0  1
        x
        0        0  2
        1        1  8
        2        1  8
        3        1  8

        Rule 5: All other types of calls yield instances of `Function`.

        >>> print(f(U))  # doctest: +NORMALIZE_WHITESPACE
        Function 'f(U)':
                         f(U)
        theta_0 theta_1
        0       0           1
                1           8
        1       0           1
                1           0
        >>> print(f([1, 2, 3]))  # doctest: +NORMALIZE_WHITESPACE
        Function 'f(set)':
                       f(set)
        theta_0 theta_1
        0       0           1
                1           8
        1       0           1
                1           0
        >>> print(f(U, theta_0=1))  # doctest: +NORMALIZE_WHITESPACE
        Function 'f(U, theta_0=1)':
                 f(U, theta_0=1)
        theta_1
        0                      1
        1                      0
        >>> print(f([1, 2, 3], theta_0=1))  # doctest: +NORMALIZE_WHITESPACE
        Function 'f(set, theta_0=1)':
                 f(set, theta_0=1)
        theta_1
        0                        1
        1                        0
        >>> print(f(x=1, theta_0=1))  # doctest: +NORMALIZE_WHITESPACE
        Function 'f(theta_0=1, x=1)':
                 f(theta_0=1, x=1)
        theta_1
        0                        1
        1                        0

        One last demonstration: We obtain a measurable with iterative calls.

        >>> print(f(theta_0=0)(theta_1=1))  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'f(theta_0=0)(theta_1=1)':
           f(theta_0=0)(theta_1=1)
        x
        0                        2
        1                        8
        2                        8
        3                        8
        """
        from ..spaces.set import Set
        from .measurable_function import MeasurableFunction

        if self.data is None:
            raise NotImplementedError(
                "The __call__ method is not yet implemented for functions without data."
            )

        measurable_set = None
        if len(args) != 0:
            if len(args) != 1:
                raise ValueError("Only one positional argument may be passed.")

            measurable_set = args[0]

            if not isinstance(measurable_set, Set):
                if isinstance(measurable_set, list):
                    measurable_set = Set(
                        measurable_set, domain=self.sig_alg.domain, name="set"
                    )
                else:
                    raise TypeError(
                        "The only allowed type of positional argument is an instance of Set."
                    )

            join = measurable_set.generated_sig_alg | self.generated_sig_alg
            atom_id = measurable_set.atom_id(join)

            if not atom_id:
                raise ValueError(
                    "Cannot call an instance of ParametrizedMeasurableFunction on a set on which the component functions are not constant."
                )

            ordered_atom_id = tuple(atom_id[name] for name in join.variable_names)
            atom_data = self.lattice.get_atom_data(join).loc[ordered_atom_id]

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
        all_parameters_specified = len(unspecified_parameters) == 0
        no_parameters_specified = len(specified_parameters) == 0

        if not no_specified_measurables and measurable_set:
            raise ValueError(
                "Cannot provide both a set (as a positional argument) and measurable variables (as keyword arguments)."
            )

        if no_specified_measurables:
            if no_parameters_specified:
                name = f"{self.name}({measurable_set.name})"

                return Function._from_validated(
                    data=atom_data.rename(name),
                    kind="any",
                    domain_kind="Domain",
                    domain_name=self.parameter_domain.name,
                    index_kind="Index",
                    index_name=None,
                    name=name,
                )

            else:
                parameter_string = ", ".join(
                    f"{name}={value}" for name, value in specified_parameters.items()
                )
                name = f"{self.name}({parameter_string})"
                data = self.data.xs(
                    key=tuple(specified_parameters.values()),
                    level=tuple(specified_parameters.keys()),
                ).rename(name)

                if all_parameters_specified:
                    if measurable_set:
                        ordered_parameters = tuple(
                            specified_parameters[name] for name in atom_data.index.names
                        )
                        return atom_data.loc[ordered_parameters].astype(Real)

                    else:
                        return MeasurableFunction._from_validated(
                            data=data.rename(name),
                            name=name,
                            sig_alg=self.sig_alg,
                            measure=self.measure,
                            index_kind="Index",
                            index_name=None,
                        )

                else:
                    parameter_domain_name = (
                        f"{self.parameter_domain_name}|{{{parameter_string}}}"
                    )

                    if measurable_set:
                        name = f"{self.name}({measurable_set.name}, {parameter_string})"
                        data = atom_data.xs(
                            key=tuple(specified_parameters.values()),
                            level=tuple(specified_parameters.keys()),
                        )

                        return Function._from_validated(
                            data=data.rename(name),
                            kind="any",
                            domain_kind="Domain",
                            domain_name=parameter_domain_name,
                            index_kind="Index",
                            index_name=None,
                            name=name,
                        )

                    else:
                        domain_name = f"{self.domain.name}|{{{parameter_string}}}"

                        return ParametrizedMeasurableFunction._from_validated(
                            data=data,
                            name=name,
                            sig_alg=self.sig_alg,
                            measure=self.measure,
                            complete_domain_name=domain_name,
                            parameter_domain_name=parameter_domain_name,
                            parameter_names=unspecified_parameters,
                        )

        else:
            try:
                result = super().__call__(**kwargs)
                if not isinstance(result, Real):
                    result.data.name = result.name
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

    def __str__(self) -> str:
        """Return a detailed string representation of the function.

        Returns
        -------
        repr_str : str
            The string representation of the function.
        """
        import pandas as pd

        if isinstance(self.data, pd.Series):
            return f"{type(self)._str_name} '{self.name}':\n{self.data.unstack(level=self.parameter_names)}"
        elif isinstance(self.data, Callable):
            return self.__repr__()
        else:
            return f"{type(self)._str_name} '{self.name}': empty"

    # --------------------- arithmetic operations --------------------- #

    def _apply_binary_operation(
        self,
        other: Function | Real,
        operation: Callable,
        op_symbol: str,
        reverse: bool = False,
        domain_name: Hashable | None = None,
        index: Index | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
    ) -> Function:
        """Pass."""
        from .measurable_function import MeasurableFunction

        if isinstance(other, Real):
            sig_alg = self.sig_alg
            measure = self.measure
            complete_domain_name = None
            parameter_domain_name = None
            parameter_names = None

        elif isinstance(other, MeasurableFunction):
            if self.sig_alg <= other.sig_alg:
                sig_alg = other.sig_alg

            elif self.sig_alg > other.sig_alg:
                sig_alg = self.sig_alg

            else:
                raise ValueError(
                    f"Cannot {op_symbol} measurable functions on incompatible measurable spaces."
                )

            measure = MeasurableFunction._get_max_measure([self, other])
            complete_domain_name = self.domain.name
            parameter_domain_name = self.parameter_domain_name
            parameter_names = self.parameter_names
            other = ParametrizedMeasurableFunction._from_validated(
                data=other.data,
                sig_alg=sig_alg,
                measure=measure,
                complete_domain_name=complete_domain_name,
                parameter_domain_name=parameter_domain_name,
                parameter_names=parameter_names,
                name=other.name,
            )

        elif isinstance(other, ParametrizedMeasurableFunction):
            raise NotImplementedError(
                "Arithmetic between two ParametrizedMeasurableFunction instances is not implemented yet."
            )

        # TODO: implement arithmetic!
        else:
            raise NotImplementedError(
                f"Arithmetic not implemented between ParametrizedMeasurableFunction and {type(other).__name__}."
            )

        return Function._apply_binary_operation(
            self=self,
            other=other,
            operation=operation,
            op_symbol=op_symbol,
            reverse=reverse,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            sig_alg=sig_alg,
            measure=measure,
            complete_domain_name=complete_domain_name,
            parameter_domain_name=parameter_domain_name,
            parameter_names=parameter_names,
        )
