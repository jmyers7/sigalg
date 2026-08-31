"""A class representing a parametrized measure."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from functools import cached_property
from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ..functions.function import Function

if TYPE_CHECKING:
    from ...typing.mapping_like import MappingLike
    from ...typing.measure_domain import MeasureDomain
    from ..functions.measurable_vector import MeasurableVector
    from ..sigma_algebras.lattice import Lattice
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.domain import Domain
    from ..spaces.set import Set
    from .measure import Measure


class ParametrizedMeasure(Function):
    r"""A class representing a parametrized measure.

    The `__init__` constructor is not meant to be used directly. Instead, the user should use the `from_domains` class method.

    See the Notes section below for the mathematical details.

    Examples
    --------
    >>> from sigalg.core import Domain, ParametrizedMeasure, SigmaAlgebra

    Define a 2-dimensional parameter domain, a 1-dimensional domain, and a sigma-algebra with 2-dimensional atom identifiers.

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

    Define a parametrized measure and print it.

    >>> mapping = {
    ...     (0, 0, 0, 1): 0,  # (theta_0, theta_1, F_0, F_1) = (0, 0, 0, 1), etc...
    ...     (0, 0, 1, 2): 1,
    ...     (0, 1, 0, 1): 2,
    ...     (0, 1, 1, 2): 8,
    ...     (1, 0, 0, 1): 4,
    ...     (1, 0, 1, 2): 1,
    ...     (1, 1, 0, 1): 5,
    ...     (1, 1, 1, 2): 0,
    ... }
    >>> mu = ParametrizedMeasure.from_domains(
    ...     measure_domain=F,
    ...     parameter_domain=Theta,
    ...     mapping=mapping,
    ... )
    >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
    Parametrized measure 'mu':
    theta_0    0     1
    theta_1    0  1  0  1
    F_0 F_1
    0   1      0  2  4  5
    1   2      1  8  1  0

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

    # --------------------- constructors --------------------- #

    @classmethod
    def from_domains(
        cls,
        measure_domain: MeasureDomain,
        parameter_domain: Domain | None = None,
        complete_domain: Domain | None = None,
        mapping: MappingLike | None = None,
        kind: Literal["param_measure", "param_probability"] = "param_measure",
        parameter_names: list[Hashable] | None = None,
        complete_domain_name: Hashable | None = None,
        parameter_domain_name: Hashable | None = None,
        output_name: Hashable | None = None,
        name: Hashable | None = None,
    ) -> ParametrizedMeasure:
        """Construct a parametrized measure from a measure domain and parameter domain.

        Parameters
        ----------
        measure_domain : MeasureDomain
            The domain of the measure, if a `SigmaAlgebra` is provided. If an `IndexLike` object that can be coerced to a `Domain` is provided, the sigma-algebra of the measure will be the power-set sigma-algebra of the domain.
        parameter_domain : Domain | None, default=None
            The parameter domain for the measure.
        complete_domain : Domain | None, default=None
            If `measure_domain` and `parameter_domain` are specified, the domain of the parametrized measure will be the Cartesian product of the two. Alternativaly, the user may specify the domain of the measure directly through this parameter.
        mapping : MappingLike | None, default=None
            The mapping of the parametrized measure.
        kind : Literal["param_measure", "param_probability"], default="param_measure"
            The kind of the parametrized measure.
        parameter_names : list[Hashable] | None, default=None
            The names of the parameters of the measure.
        complete_domain_name : Hashable | None, default=None
            The name of the domain of the measure.
        parameter_domain_name : Hashable | None, default=None
            The name of the parameter domain.
        output_name : Hashable | None, default=None
            The name of the output variable for the parametrized measure. If `None`, a default will be generated.
        name : Hashable | None, default="mu"
            The name of the parametrized measure. If `None`, a default name will be assigned.

        Returns
        -------
        param_measure : ParametrizedMeasure
            The constructed parametrized measure.

        Examples
        --------
        >>> from sigalg.core import Domain, ParametrizedMeasure, SigmaAlgebra

        Define a 2-dimensional parameter domain, a 1-dimensional domain, and a sigma-algebra with 2-dimensional atom identifiers.

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

        Define a parametrized measure and print it.

        >>> mapping = {
        ...     (0, 0, 0, 1): 0,  # (theta_0, theta_1, F_0, F_1) = (0, 0, 0, 1), etc...
        ...     (0, 0, 1, 2): 1,
        ...     (0, 1, 0, 1): 2,
        ...     (0, 1, 1, 2): 8,
        ...     (1, 0, 0, 1): 4,
        ...     (1, 0, 1, 2): 1,
        ...     (1, 1, 0, 1): 5,
        ...     (1, 1, 1, 2): 0,
        ... }
        >>> mu = ParametrizedMeasure.from_domains(
        ...     measure_domain=F,
        ...     parameter_domain=Theta,
        ...     mapping=mapping,
        ... )
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu':
        theta_0    0     1
        theta_1    0  1  0  1
        F_0 F_1
        0   1      0  2  4  5
        1   2      1  8  1  0
        """
        from ...validation.measure_domain_normalizer import MeasureDomainNormalizer
        from ...validation.parametrized_domain_constructor import (
            ParametrizedDomainConstructor,
        )
        from .parametrized_probability_measure import ParametrizedProbabilityMeasure

        if cls is ParametrizedProbabilityMeasure:
            kind = "param_probability"

        u = MeasureDomainNormalizer(measure_domain=measure_domain)

        measure_domain = u.domain

        v = ParametrizedDomainConstructor(
            component_domain=measure_domain,
            parameter_domain=parameter_domain,
            complete_domain=complete_domain,
            parameter_names=parameter_names,
            parameter_domain_name=parameter_domain_name,
            complete_domain_name=complete_domain_name,
        )

        complete_domain = v.complete_domain
        parameter_names = v.parameter_names
        parameter_domain_name = v.parameter_domain_name

        measure = cls(
            domain=complete_domain,
            mapping=mapping,
            kind=kind,
            output_name=output_name,
            parameter_names=parameter_names,
            name=name,
        )

        measure.sig_alg = u.sig_alg
        measure.parameter_names = parameter_names
        measure.parameter_domain_name = parameter_domain_name

        if kind == "param_probability":
            measure.__class__ = ParametrizedProbabilityMeasure
            output_name = "probability"

        return measure

    @classmethod
    def _from_validated(
        cls,
        *,
        data: pd.Series,
        sig_alg: SigmaAlgebra,
        kind: Literal["param_measure", "param_probability"],
        complete_domain_name: Hashable,
        parameter_domain_name: Hashable | None,
        parameter_names: list[Hashable],
        name: Hashable,
    ) -> ParametrizedMeasure:
        from ..measures.parametrized_probability_measure import (
            ParametrizedProbabilityMeasure,
        )

        function = super()._from_validated(
            data=data,
            kind=kind,
            name=name,
            domain_kind="Domain",
            domain_name=complete_domain_name,
            index_kind="Index",
            index_name=None,
        )
        function.sig_alg = sig_alg
        function.parameter_names = parameter_names
        function.parameter_domain_name = parameter_domain_name

        if kind == "param_probability":
            function.__class__ = ParametrizedProbabilityMeasure

        return function

    @classmethod
    def from_rand(
        cls,
        measure_domain: MeasureDomain,
        parameter_domain: Domain | None = None,
        complete_domain: Domain | None = None,
        num_null_atoms: int = 0,
        kind: Literal["param_measure", "param_probability"] = "param_measure",
        distribution: Literal["uniform", "poisson", "dirichlet"] = "uniform",
        max_value: int = 10,
        rate: float = 5.0,
        parameter_names: list[Hashable] | None = None,
        complete_domain_name: Hashable | None = None,
        parameter_domain_name: Hashable | None = None,
        output_name: Hashable | None = None,
        name: Hashable | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> ParametrizedMeasure:
        """Generate a random parametrized measure.

        Parameters
        ----------
        measure_domain : MeasureDomain
            The domain of the measure, if a `SigmaAlgebra` is provided. If an `IndexLike` object that can be coerced to a `Domain` is provided, the sigma-algebra of the measure will be the power-set sigma-algebra of the domain.
        parameter_domain : Domain | None, default=None
            The parameter domain for the measure.
        complete_domain : Domain | None, default=None
            If `measure_domain` and `parameter_domain` are specified, the domain of the parametrized measure will be the Cartesian product of the two. Alternativaly, the user may specify the domain of the measure directly through this parameter.
        num_null_atoms : int, default=0
            The number of atoms in the sigma-algebra that should be assigned a measure of 0.
        kind : Literal["param_measure", "param_probability"], default="param_measure"
            The kind of measure to generate. If `'param_probability'`, generates a probability measure using a Dirichlet distribution.
        distdistribution : Literal["uniform", "poisson", "dirichlet"], default="uniform"
            The type of distribution from which to sample the values of the measure.
        max_value : int, default=10
            The maximum value for uniform integer sampling when `distribution='uniform'`. Integers are sampled from the interval `[1, max_value)`.
        rate : float, default=5.0
            The rate parameter for Poisson sampling when `distribution='poisson'`.
        parameter_names : list[Hashable] | None, default=None
            The names of the parameters of the measure.
        complete_domain_name : Hashable | None, default=None
            The name of the domain of the measure.
        parameter_domain_name : Hashable | None, default=None
            The name of the parameter domain.
        output_name : Hashable | None, default=None
            The name of the output variable for the parametrized measure. If `None`, a default will be generated.
        name : Hashable | None, default=None
            The name of the function. If `None`, a default name will be used.
        random_state : int | np.random.Generator | None, default=None
            The random state for reproducibility.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import Domain, ParametrizedMeasure, SigmaAlgebra
        >>> rng = np.random.default_rng(42)

        Define a 2-dimensional parameter domain, a 1-dimensional domain, and a sigma-algebra with four atoms.

        >>> Theta = Domain.cartesian_power(
        ...     [0, 1], n=2, variable_names=["theta_0", "theta_1"], name="Theta"
        ... )
        >>> X = Domain.from_sequence(size=5, variable_name="x")
        >>> F = SigmaAlgebra(domain=X, mapping=dict(zip(X, [0, 1, 1, 2, 3])))

        Generate a random parametrized measure with values drawn from a uniform distribution on the integers in `[0, 10)` and with one null atom.

        >>> mu = ParametrizedMeasure.from_rand(
        ...     measure_domain=F,
        ...     parameter_domain=Theta,
        ...     random_state=rng,
        ...     num_null_atoms=1,
        ... )
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu':
        theta_0  0     1
        theta_1  0  1  0  1
        F
        0        4  1  6  9
        1        2  0  5  0
        2        1  7  1  4
        3        0  8  0  7

        Generate a random parametrized measure with values drawn from a Poisson distribution with `rate=5.0`.

        >>> nu = ParametrizedMeasure.from_rand(
        ...     measure_domain=F,
        ...     parameter_domain=Theta,
        ...     distribution="poisson",
        ...     random_state=rng,
        ...     name="nu",
        ... )
        >>> print(nu)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'nu':
        theta_0  0     1
        theta_1  0  1  0  1
        F
        0        3  6  3  5
        1        3  7  5  8
        2        5  2  3  5
        3        6  9  8  7

        Generate a parametrized probability measure with values drawn from a Dirichlet distribution.

        >>> P = ParametrizedMeasure.from_rand(
        ...     measure_domain=F,
        ...     parameter_domain=Theta,
        ...     distribution="dirichlet",
        ...     random_state=rng,
        ... )
        >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized probability measure 'P':
        theta_0         0                       1
        theta_1         0             1         0         1
        F
        0        0.029892  6.987893e-02  0.137316  0.007920
        1        0.628674  8.958606e-02  0.000001  0.029446
        2        0.000276  8.405348e-01  0.735544  0.776833
        3        0.341159  2.074313e-07  0.127139  0.185800
        """
        from ...validation.measure_domain_normalizer import MeasureDomainNormalizer
        from ...validation.parametrized_domain_constructor import (
            ParametrizedDomainConstructor,
        )
        from .parametrized_probability_measure import ParametrizedProbabilityMeasure

        if distribution not in ["uniform", "poisson", "dirichlet"]:
            raise ValueError(
                "distribution must be either 'uniform', 'poisson', or 'dirichlet'."
            )
        if not isinstance(max_value, int) or max_value < 2:
            raise ValueError("max_value must be an integer >= 2.")
        if not isinstance(rate, Real) or rate <= 0:
            raise ValueError("rate must be a positive number.")
        if output_name is not None and not isinstance(output_name, Hashable):
            raise TypeError("If given, output_name must be hashable.")
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("If given, name must be hashable.")
        if random_state is not None and not isinstance(
            random_state, (int, np.random.Generator)
        ):
            raise TypeError(
                "random_state must be an integer, np.random.Generator, or None."
            )

        if (
            cls is ParametrizedProbabilityMeasure
            or kind == "param_probability"
            or distribution == "dirichlet"
        ):
            kind = "param_probability"
            distribution = "dirichlet"
            name = name if name else "P"
        else:
            kind = "param_measure"
            name = name if name else "mu"

        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        u = MeasureDomainNormalizer(measure_domain=measure_domain)

        measure_domain = u.domain
        sig_alg = u.sig_alg

        v = ParametrizedDomainConstructor(
            component_domain=measure_domain,
            parameter_domain=parameter_domain,
            complete_domain=complete_domain,
            parameter_names=parameter_names,
            parameter_domain_name=parameter_domain_name,
            complete_domain_name=complete_domain_name,
        )

        complete_domain = v.complete_domain
        parameter_names = v.parameter_names
        parameter_domain_name = v.parameter_domain_name
        complete_domain_name = v.complete_domain_name

        if not isinstance(num_null_atoms, int) or num_null_atoms > len(measure_domain):
            raise ValueError(
                "num_null_atoms must be an integer no larger than the number of atoms in the sigma-algebra."
            )

        if distribution == "uniform":
            arr = rng.integers(
                low=1,
                high=max_value,
                size=(len(measure_domain) - num_null_atoms, len(parameter_domain)),
            )

        elif distribution == "poisson":
            arr = rng.poisson(
                lam=rate,
                size=(len(measure_domain) - num_null_atoms, len(parameter_domain)),
            )

        else:
            arr = rng.dirichlet(
                alpha=(1 / (len(measure_domain) - num_null_atoms),)
                * (len(measure_domain) - num_null_atoms),
                size=len(parameter_domain),
            ).T

        arr = np.vstack(
            (arr, np.zeros(shape=(num_null_atoms, len(parameter_domain)), dtype=int))
        )
        idx = np.argsort(
            rng.random((len(measure_domain), len(parameter_domain))), axis=0
        )
        arr = np.take_along_axis(arr, idx, axis=0).ravel(order="F")

        if output_name is None:
            output_name = name

        data = pd.Series(arr, index=complete_domain.data, name=output_name)

        return cls._from_validated(
            data=data,
            sig_alg=sig_alg,
            kind=kind,
            complete_domain_name=complete_domain_name,
            parameter_domain_name=parameter_domain_name,
            parameter_names=parameter_names,
            name=name,
        )

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
        >>> from sigalg.core import Domain, ParametrizedMeasure, SigmaAlgebra

        Define a parametrized measure.

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
        ...     (0, 0, 0, 1): 0,  # (theta_0, theta_1, F_0, F_1) = (0, 0, 0, 1), etc...
        ...     (0, 0, 1, 2): 1,
        ...     (0, 1, 0, 1): 2,
        ...     (0, 1, 1, 2): 8,
        ...     (1, 0, 0, 1): 4,
        ...     (1, 0, 1, 2): 1,
        ...     (1, 1, 0, 1): 5,
        ...     (1, 1, 1, 2): 0,
        ... }
        >>> mu = ParametrizedMeasure.from_domains(
        ...     measure_domain=F,
        ...     parameter_domain=Theta,
        ...     mapping=mapping,
        ... )
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu':
        theta_0    0     1
        theta_1    0  1  0  1
        F_0 F_1
        0   1      0  2  4  5
        1   2      1  8  1  0

        Print the parameter domain of the measure.

        >>> print(mu.parameter_domain)  # doctest: +NORMALIZE_WHITESPACE
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

    @property
    def measure_domain_names(self) -> list[Hashable] | None:
        """Get the domain names of the parametrized measure.

        Returns
        -------
        domain_names : list[Hashable] | None
            The domain names associated with the parametrized measure, or `None` if not set.

        Examples
        --------
        >>> from sigalg.core import Domain, ParametrizedMeasure, SigmaAlgebra

        Define a parametrized measure.

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
        ...     (0, 0, 0, 1): 0,  # (theta_0, theta_1, F_0, F_1) = (0, 0, 0, 1), etc...
        ...     (0, 0, 1, 2): 1,
        ...     (0, 1, 0, 1): 2,
        ...     (0, 1, 1, 2): 8,
        ...     (1, 0, 0, 1): 4,
        ...     (1, 0, 1, 2): 1,
        ...     (1, 1, 0, 1): 5,
        ...     (1, 1, 1, 2): 0,
        ... }
        >>> mu = ParametrizedMeasure.from_domains(
        ...     measure_domain=F,
        ...     parameter_domain=Theta,
        ...     mapping=mapping,
        ... )
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu':
        theta_0  0     1
        theta_1  0  1  0  1
        F_0 F_1
        0   1    0  2  4  5
        1   2    1  8  1  0

        Print the variable names of the measure domain.

        >>> mu.measure_domain_names
        ['F_0', 'F_1']
        """
        return self.sig_alg.variable_names if self.sig_alg is not None else None

    @cached_property
    def lattice(self) -> Lattice | None:
        """Return the downward lattice of all sigma-algebras contained in the domain sigma-algebra of the parametrized measure.

        Returns
        -------
        down_lattice : Lattice | None
            The downward lattice of all sigma-algebras contained in the domain sigma-algebra of the measure.

        Examples
        --------
        >>> from sigalg.core import Domain, ParametrizedMeasure, SigmaAlgebra

        Define a parametrized measure.

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
        ...     (0, 0, 0, 1): 0,  # (theta_0, theta_1, F_0, F_1) = (0, 0, 0, 1), etc...
        ...     (0, 0, 1, 2): 1,
        ...     (0, 1, 0, 1): 2,
        ...     (0, 1, 1, 2): 8,
        ...     (1, 0, 0, 1): 4,
        ...     (1, 0, 1, 2): 1,
        ...     (1, 1, 0, 1): 5,
        ...     (1, 1, 1, 2): 0,
        ... }
        >>> mu = ParametrizedMeasure.from_domains(
        ...     measure_domain=F,
        ...     parameter_domain=Theta,
        ...     mapping=mapping,
        ... )
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu':
        theta_0  0     1
        theta_1  0  1  0  1
        F_0 F_1
        0   1    0  2  4  5
        1   2    1  8  1  0

        The lattice of the measure is initialized with the domain sigma-algebra `F`.

        >>> mu.lattice
        Lattice(base=F, type=downward, num_sig_algs=1)

        Define a sub-sigma-algebra of `F` and check that it is in the lattice.

        >>> G = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 0,
        ...     },
        ...     name="G",
        ... )
        >>> G in mu.lattice
        True

        Check that the lattice now includes `G`.

        >>> mu.lattice
        Lattice(base=F, type=downward, num_sig_algs=2)

        Since `G` is a sub-sigma-algebra of `F`, we may restrict the measure `mu` to `G`.

        >>> print(mu | G)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu|G':
        theta_0  0      1
        theta_1  0   1  0  1
        G
        0        1  10  5  5

        Define another sigma-algebra, and check if it is a sub-sigma-algebra of `F`.

        >>> H = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ...     name="H",
        ... )
        >>> H in mu.lattice
        False

        Check that `H` was not added to the lattice.

        >>> mu.lattice
        Lattice(base=F, type=downward, num_sig_algs=2)
        """
        if self.sig_alg is not None:
            return self.sig_alg.down_lattice
        else:
            return None

    # --------------------- methods --------------------- #

    def restrict_to(
        self,
        obj: SigmaAlgebra | Set | list[Hashable],
        normalize: bool = False,
        subset_name: Hashable | None = "A",
        name: Hashable | None = None,
    ) -> Measure:
        """Pass."""
        from .._utils.utils import to_df

        sig_alg = obj

        if sig_alg is self.sig_alg:
            return self

        if sig_alg not in self.lattice:
            raise TypeError(
                "If given obj is a sigma-algebra, it must be a sub-sigma-algebra of the sigma-algebra of the measure."
            )

        atom_data = to_df(self.lattice.get_atom_data(sig_alg), "_alg")
        unstacked_self_data = self.data.unstack(level=self.parameter_names)

        if name is None:
            name = f"{self.name}|{sig_alg.name}"

        data = (
            pd.concat([atom_data, unstacked_self_data], axis=1)
            .groupby(list(atom_data.columns))
            .sum()
        )
        data.index.names = sig_alg.variable_names
        data.columns = unstacked_self_data.columns
        data = data.stack(level=data.columns.names)
        data = data.reorder_levels(
            unstacked_self_data.columns.names + sig_alg.variable_names
        ).sort_index()

        return type(self)._from_validated(
            data=data,
            sig_alg=sig_alg,
            kind=self.kind,
            complete_domain_name=f"{self.parameter_domain.name} x {sig_alg.name}",
            parameter_domain_name=self.parameter_domain.name,
            parameter_names=self.parameter_names,
            name=name,
        )

    # --------------------- dunder operators --------------------- #

    def __or__(
        self,
        obj: SigmaAlgebra | Set | list[Hashable],
    ) -> ParametrizedMeasure:
        """Restrict the measure to a sub-sigma-algebra.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The sub-sigma-algebra to which to restrict the measure.

        Returns
        -------
        measure : Measure
            A new measure restricted to the new sigma-algebra.
        """
        return self.restrict_to(obj)

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
    ) -> Real | Function | ParametrizedMeasure | Measure:
        """Call the parametrized measure.

        The return value is determined by the following rules:

        1. If all parameters are provided and a complete set of atom identifiers are provided (as keyword arguments), a real number is returned. This number is the measure of the atom under the given parameters.

        2. If all parameters are provided (as keyword arguments) and a measurable `Set` or list of points is provided (as a positional argument), a real number is returned. This number is the measure of the set under the given parameters.

        3. If all parameters are provided (as keyword arguments) and no other arguments are provided, a `Measure` is return.

        4. If a partial set of parameters is provided (as keyword arguments) and no other arguments are provided, a parametrized measure is provided.

        5. In all other cases, a `Function` is returned.

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
        >>> import numpy as np
        >>> from sigalg.core import Domain, ParametrizedMeasure, Set, SigmaAlgebra
        >>> rng = np.random.default_rng(42)

        Define a 2-dimensional parameter domain, a 1-dimensional domain, and a sigma-algebra. Then, define a parametrized measure.

        >>> Theta = Domain.cartesian_power(
        ...     [0, 1], n=2, variable_names=["theta_0", "theta_1"], name="Theta"
        ... )
        >>> X = Domain.from_sequence(size=5, variable_name="x")
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: (0, 1),
        ...         1: (0, 1),
        ...         2: (1, 2),
        ...         3: (1, 2),
        ...         4: (2, 3),
        ...     },
        ... )
        >>> mu = ParametrizedMeasure.from_rand(
        ...     measure_domain=F,
        ...     parameter_domain=Theta,
        ...     random_state=rng,
        ...     num_null_atoms=1,
        ... )
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu':
        theta_0  0     1
        theta_1  0  1  0  1
        F_0 F_1
        0   1    1  8  1  0
        1   2    4  0  0  4
        2   3    0  7  6  7

        Rules 1 and 2: Calling with complete sets of parameters and either a `Set` instance, a list of points, or a complete set of atom identifiers.

        >>> U = Set([2, 3, 4], X, name="U")
        >>> mu(U, theta_0=0, theta_1=1)
        7
        >>> mu([2, 3, 4], theta_0=0, theta_1=1)
        7
        >>> mu(F_0=0, F_1=1, theta_0=0, theta_1=1)
        8

        Rule 3: Calling with a complete set of parameters, and no other arguments.

        >>> print(mu(theta_0=0, theta_1=1))  # doctest: +NORMALIZE_WHITESPACE
        Measure 'mu(theta_0=0, theta_1=1)':
                    mu(theta_0=0, theta_1=1)
        F_0 F_1
        0   1                           8
        1   2                           0
        2   3                           7

        Rule 4: Calling with a partial set of parameters, and no other arguments.

        >>> print(mu(theta_0=0))  # doctest: +NORMALIZE_WHITESPACE
        Parametrized measure 'mu(theta_0=0)':
        theta_1  0  1
        F_0 F_1
        0   1    1  8
        1   2    4  0
        2   3    0  7

        Rule 5: All other types of calls yield instances of `Function`.

        >>> print(mu(U))  # doctest: +NORMALIZE_WHITESPACE
        Function 'mu(U)':
                         mu(U)
        theta_0 theta_1
        0       0            4
                1            7
        1       0            6
                1           11
        >>> print(mu([2, 3, 4]))  # doctest: +NORMALIZE_WHITESPACE
        Function 'mu(set)':
                         mu(set)
        theta_0 theta_1
        0       0              4
                1              7
        1       0              6
                1             11
        >>> print(mu(U, theta_0=0))  # doctest: +NORMALIZE_WHITESPACE
        Function 'mu(U, theta_0=0)':
                mu(U, theta_0=0)
        theta_1
        0                       4
        1                       7
        >>> print(mu([2, 3, 4], theta_0=0))  # doctest: +NORMALIZE_WHITESPACE
        Function 'mu(set, theta_0=0)':
                 mu(set, theta_0=0)
        theta_1
        0                         4
        1                         7
        >>> print(mu(F_0=0, theta_0=0))  # doctest: +NORMALIZE_WHITESPACE
        Function 'mu(theta_0=0, F_0=0)':
                     mu(theta_0=0, F_0=0)
        theta_1 F_1
        0       1                       1
        1       1                       8

        One last demonstration: We obtain a measure with iterative calls.

        >>> print(mu(theta_0=0)(theta_1=1))  # doctest: +NORMALIZE_WHITESPACE
        Measure 'mu(theta_0=0)(theta_1=1)':
                 mu(theta_0=0)(theta_1=1)
        F_0 F_1
        0   1                           8
        1   2                           0
        2   3                           7
        """
        from ..spaces.set import Set
        from .measure import Measure
        from .parametrized_probability_measure import ParametrizedProbabilityMeasure

        if self.data is None:
            raise NotImplementedError(
                "The __call__ method is not yet implemented for parametrized measures without data."
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

            if self.sig_alg not in measurable_set.lattice:
                raise ValueError(
                    "Cannot call an instance of ParametrizedMeasure on a set which is not in the underlying sigma-algebra."
                )

            atom_data = measurable_set.lattice.get_atom_data(self.sig_alg)

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
        atom_id_kwargs = {
            name: param
            for name, param in kwargs.items()
            if name in self.measure_domain_names
        }

        no_specified_atom_ids = len(atom_id_kwargs) == 0
        all_parameters_specified = len(unspecified_parameters) == 0
        no_parameters_specified = len(specified_parameters) == 0

        if not no_specified_atom_ids and measurable_set:
            raise ValueError(
                "Cannot provide both a set (as a positional argument) and atom IDs (as keyword arguments)."
            )

        if no_specified_atom_ids:
            if no_parameters_specified:
                name = f"{self.name}({measurable_set.name})"
                data = (
                    (self.data * atom_data)
                    .unstack(level=self.measure_domain_names)
                    .sum(axis=1)
                ).rename(name)

                return Function._from_validated(
                    data=data,
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
                        return (atom_data * data).sum().astype(Real)

                    else:
                        kind = (
                            "probability"
                            if type(self) is ParametrizedProbabilityMeasure
                            else "measure"
                        )
                        return Measure._from_validated(
                            data=data,
                            kind=kind,
                            sig_alg=self.sig_alg,
                            name=name,
                        )

                else:
                    parameter_domain_name = (
                        f"{self.parameter_domain_name}|{{{parameter_string}}}"
                    )

                    if measurable_set:
                        name = f"{self.name}({measurable_set.name}, {parameter_string})"
                        data = (
                            (data * atom_data)
                            .unstack(level=self.measure_domain_names)
                            .sum(axis=1)
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
                        kind = (
                            "param_probability"
                            if type(self) is ParametrizedProbabilityMeasure
                            else "param_measure"
                        )

                        return ParametrizedMeasure._from_validated(
                            data=data,
                            sig_alg=self.sig_alg,
                            kind=kind,
                            complete_domain_name=domain_name,
                            parameter_domain_name=parameter_domain_name,
                            parameter_names=unspecified_parameters,
                            name=name,
                        )

        else:
            try:
                result = super().__call__(**kwargs)

                if hasattr(result, "data"):
                    if isinstance(result.data, pd.Series) and result.data.empty:
                        return 0.0
                    else:
                        result.data.name = result.name

                return result

            except Exception as e:
                raise ValueError(
                    "Error while evaluating the parametrized measure on the given arguments."
                ) from e

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
    def __eq__(self, other: ParametrizedMeasure) -> bool:
        """Test equality of two parametrized measures."""
        from .._utils.utils import add_subscript, to_df

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

        self_data = self.data.reorder_levels(
            parameter_names + self.measure_domain_names
        ).sort_index(level=parameter_names)
        other_data = other.data.reorder_levels(
            parameter_names + other.measure_domain_names
        ).sort_index(level=parameter_names)

        self_sig_alg_variable_names_subscripted = add_subscript(
            self.sig_alg.variable_names, "ID"
        )

        self_data = self_data.unstack(level=parameter_names)
        self_data.columns = self_data.columns.to_flat_index()
        self_data.index.names = self_sig_alg_variable_names_subscripted

        other_data = other_data.unstack(level=parameter_names)
        other_data.columns = other_data.columns.to_flat_index()

        sig_alg_data = to_df(other.lattice.get_atom_data(self.sig_alg))
        sig_alg_data.columns = self_sig_alg_variable_names_subscripted

        self_data_with_other_sig_alg_ids = pd.merge(
            left=sig_alg_data,
            right=self_data,
            left_on=self_sig_alg_variable_names_subscripted,
            right_index=True,
        ).drop(columns=self_sig_alg_variable_names_subscripted)

        return bool(
            (self_data_with_other_sig_alg_ids.reindex(other_data.index) == other_data)
            .all()
            .all()
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
                f"name={self.name})"
            )
        else:
            return type(self)._repr_name + "(empty)"

    def __str__(self) -> str:
        """Return a detailed string representation of the measure.

        Returns
        -------
        repr_str : str
            The string representation of the measure.
        """
        if isinstance(self.data, pd.Series):
            return f"{type(self)._str_name} '{self.name}':\n{self.data.unstack(level=self.parameter_names)}"
        elif isinstance(self.data, Callable):
            return self.__repr__()
        else:
            return f"{type(self)._str_name} '{self.name}': empty"
