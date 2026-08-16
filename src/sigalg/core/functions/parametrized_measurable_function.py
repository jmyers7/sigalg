"""A class representing a parametrized measure."""

from __future__ import annotations

from collections.abc import Callable
from functools import cached_property
from numbers import Real
from typing import TYPE_CHECKING

from .function import Function

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterator

    import pandas as pd

    from ...typing.mapping_like import MappingLike
    from ..measures.measure import Measure
    from ..measures.parametrized_measure import ParametrizedMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.domain import Domain
    from ..spaces.measurable_space import MeasurableSpace
    from ..spaces.measure_space import MeasureSpace
    from ..spaces.set import Set
    from .measurable_function import MeasurableFunction


class ParametrizedMeasurableFunction(Function):
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

        function = cls(
            domain=complete_domain,
            mapping=mapping,
            output_name=output_name,
            name=name,
        )

        function.sig_alg = sig_alg
        function.measure = measure
        function.parameter_names = v.parameter_names
        function.parameter_domain_name = v.parameter_domain_name

        if (
            parameter_domain is not None
            and not sig_alg.is_power_set
            and sig_alg not in function.lattice
        ):
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
        name: Hashable,
        sig_alg: SigmaAlgebra,
        measure: Measure | None,
        domain_name: Hashable | None,
        parameter_domain_name: Hashable | None,
        parameter_names: list[Hashable],
    ) -> ParametrizedMeasurableFunction:
        from ..measures.probability_measure import ProbabilityMeasure
        from .parametrized_random_variable import ParametrizedRandomVariable

        function = super()._from_validated(
            data=data,
            kind="any",
            name=name,
            domain_kind="Domain",
            domain_name=domain_name,
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
            MeasureSpace._from_validated(measure=self.measure) if self.measure else None
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

    # --------------------- measurable-related methods --------------------- #

    # TODO: stale docstring
    def atom_data(
        self, sig_alg: SigmaAlgebra | None = None
    ) -> pd.Series | pd.DataFrame | None:
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

        Define a 1-dimensional parameter space, a 1-dimensional domain, and a sigma-algebra.

        >>> Theta = Domain.from_sequence(size=2, variable_name="theta", name="Theta")
        >>> X = Domain.from_sequence(size=4, variable_name="x")
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ... )

        Define a parametrized measurable function.

        >>> mapping = {
        ...     (0, 0): 1,  # (theta, x) = (0, 0), etc ...
        ...     (0, 1): 2,
        ...     (0, 2): 2,
        ...     (0, 3): 2,
        ...     (1, 0): 0,
        ...     (1, 1): -3,
        ...     (1, 2): -3,
        ...     (1, 3): -3,
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
        3      2 -3

        By leaving the parameter to `atom_data` as its default `None`, it computes the unique values of the parametrized measurable function on each of the atoms of the underlying sigma-algebra (accessed through the `sig_alg` attribute).

        >>> print(f.atom_data())  # doctest: +NORMALIZE_WHITESPACE
        theta  0  1
        u
        0      1  0
        1      2 -3
        2      2 -3

        Note that the function is also measurable with respect to the following finer sigma-algebra.

        >>> G = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ...     variable_names=["v"],
        ...     name="G",
        ... )
        >>> f in G
        True

        We may thus pass `G` into the `atom_data` method to get the unique values of the function on each of the atoms of `G`.

        >>> print(f.atom_data(G))  # doctest: +NORMALIZE_WHITESPACE
        theta  0  1
        v
        0      1  0
        1      2 -3
        2      2 -3

        """
        if self.data is not None:
            if sig_alg is None:
                sig_alg = self.sig_alg
            self.lattice.add(sig_alg)
            return self.lattice.get_atom_data(sig_alg)
        else:
            return None

    # --------------------- measure-related methods --------------------- #

    def integrate(
        self,
        measurable_set: Set | None = None,
        measure: Measure | ParametrizedMeasure | None = None,
    ) -> Function:
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
        integral : Real | pd.Series | Function
            Returns the following:

            * If `measure` is a `Measure`, returns a `Function` representing the integral of the function with respect to the measure over the specified set for each parameter value.

            * If `measure` is a `ParametrizedMeasure`, returns a `Function` representing the integral of the function with respect to the measure over the specified set for each parameter value.

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
        self, *args, **kwargs
    ) -> Real | MeasurableFunction | Function | ParametrizedMeasurableFunction:
        """Call the parametrized function.

        The return value is determined by the following rules:

        1. If all parameters and all measurable arguments are provided (as keyword arguments), a real number is returned.

        2. If all parameters are provided (as keyword arguments) and an atom of the underlying sigma-algebra is provided (as a positional argument), a real number is returned.

        3. If all parameters are provided (as keyword arguments) but no measurable arguments are provided, a measurable function is returned.

        4. If a partial set of parameters is provided (as keyword arguments) and no measurable arguments are provided, a parametrized measurable function is provided.

        5. In all other cases, a function is returned.

        Parameters
        ----------
        *args : positional arguments
            Positional arguments for the function.
        **kwargs : keyword arguments
            Keyword arguments for the function.

        Examples
        --------
        >>> from sigalg.core import Domain, ParametrizedMeasurableFunction, SigmaAlgebra

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

        Call the function with all parameters and all "measurable" arguments.

        >>> f(x=0, theta_0=1, theta_1=0)
        4

        Call the function with all parameters and an atom of the sigma-algebra (as a positional argument).

        >>> A = F.get_set([1, 2])
        >>> f(A, theta_0=0, theta_1=1)
        8

        Call the function with all parameters, but no other arguments. An instance of `MeasurableFunction` is returned.

        >>> measurable_function = f(theta_0=1, theta_1=1)
        >>> print(measurable_function)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'f(theta_0=1, theta_1=1)':
           f(theta_0=1, theta_1=1)
        x
        0                        5
        1                        0
        2                        0
        3                        0

        Call the function with a partial set of parameters, but no other arguments. An instance of `ParametrizedMeasurableFunction` is returned.

        >>> param_measurable_function = f(theta_0=1)
        >>> print(param_measurable_function)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measurable function 'f(theta_0=1)':
        theta_1  0  1
        x
        0        4  5
        1        1  0
        2        1  0
        3        1  0

        Call the function with a partial set of parameters and a "measurable" argument. An instance of `Function` is returned.

        >>> function = f(theta_0=1, x=0)
        >>> print(function)  # doctest: +NORMALIZE_WHITESPACE
        Function 'f(theta_0=1, x=0)':
                 f(theta_0=1, x=0)
        theta_1
        0                        4
        1                        5
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
            if not isinstance(args[0], Set):
                raise TypeError(
                    "The only allowed type of positional argument is an instance of MeasurableSet."
                )
            measurable_set = args[0]

            if measurable_set not in self.sig_alg:
                raise ValueError(
                    "Cannot call an instance of ParametrizedMeasurableVector on a set which is not an atom in the underlying sigma-algebra."
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
        all_parameters_specified = len(unspecified_parameters) == 0

        if no_specified_measurables:
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
                    atom_data = self.atom_data()
                    atom_id = measurable_set.atom_id(self.sig_alg)
                    ordered_atom_id = tuple(
                        atom_id[name] for name in atom_data.index.names
                    )
                    result = atom_data.loc[ordered_atom_id]
                    ordered_parameters = tuple(
                        specified_parameters[name] for name in result.index.names
                    )

                    return result.loc[ordered_parameters].astype(Real)

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
                domain_name = f"{self.domain.name}|{{{parameter_string}}}"
                parameter_domain_name = (
                    f"{self.parameter_domain_name}|{{{parameter_string}}}"
                )

                return ParametrizedMeasurableFunction._from_validated(
                    data=data,
                    name=name,
                    sig_alg=self.sig_alg,
                    measure=self.measure,
                    domain_name=domain_name,
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
