"""A class representing a measure on a sigma-algebra."""

from __future__ import annotations

from collections.abc import Callable, Hashable
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
    from ..indices.index import Index
    from ..sigma_algebras.lattice import Lattice
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.set import Set


class Measure(Function):
    r"""A class representing a measure on a sigma-algebra.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    domain : MeasureDomain, default=None
        The domain of the measure. Either a `SigmaAlgebra` or an `IndexLike` object that can be coerced into a `Domain`; in the latter case the domain will be set to the power set of the domain.
    mapping : MappingLike | None, default=None
        A mapping from the domain to the measure values.
    kind : Literal["measure", "probability"], default="measure"
        The kind of measure. If `measure`, the measure can only take non-negative values; if `probability`, the measure can only take non-negative values that sum to 1 and it will be promoted to an instance of the subclass `ProbabilityMeasure` and the `output_name` will be set to `probability`.
    output_name : str, default="measure"
        The name of the output variable of the measure.
    name : Hashable, default="mu"
        The name of the measure

    Examples
    --------
    >>> from sigalg.core import Domain, Measure, SigmaAlgebra

    Define a measure on a sigma-algebra with two atoms.

    >>> X = Domain.from_sequence(size=3)
    >>> F = SigmaAlgebra(
    ...    domain=X,
    ...    mapping={
    ...        0: 0,
    ...        1: 0,
    ...        2: 1,
    ...    },
    ... )
    >>> mu = Measure(domain=F, mapping={0: 1, 1: 2})
    >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
    Measure 'mu':
             mu
    F
    0   1
    1   2

    Define a measure directly on a domain, which will use the power-set sigma-algebra by default.

    >>> nu = Measure(
    ...     domain=X,
    ...     mapping={
    ...         0: 3,
    ...         1: 1,
    ...         2: 4,
    ...     },
    ...     name="nu",
    ... )
    >>> print(nu)  # doctest: +NORMALIZE_WHITESPACE
    Measure 'nu':
       nu
    x
    0   3
    1   1
    2   4
    >>> print(nu.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'R':
       R
    x
    0  0
    1  1
    2  2

    Define the same measure directly on a `list` of points.

    >>> nu = Measure(
    ...     domain=[0, 1, 2],
    ...     mapping={
    ...         0: 3,
    ...         1: 1,
    ...         2: 4,
    ...     },
    ...     name="nu",
    ... )
    >>> print(nu)  # doctest: +NORMALIZE_WHITESPACE
    Measure 'nu':
       nu
    x
    0   3
    1   1
    2   4
    >>> print(nu.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'R':
       R
    x
    0  0
    1  1
    2  2

    Measures may also be defined from callables.

    >>> xi = Measure(domain=F, mapping=lambda u: u+1, name="xi")
    >>> print(xi)  # doctest: +NORMALIZE_WHITESPACE
    Measure 'xi':
       xi
    F
    0   1
    1   2

    Define a probability measure using the `Measure` constructor with the parameter `kind` set to `probability`.

    >>> P = Measure(
    ...     domain=X,
    ...     mapping={
    ...         0: 0.5,
    ...         1: 0.2,
    ...         2: 0.3,
    ...     },
    ...     kind="probability",
    ...     name="P",
    ... )
    >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'P':
         P
    x
    0  0.5
    1  0.2
    2  0.3
    >>> type(P).__name__
    'ProbabilityMeasure'
    >>> print(P.domain)  # doctest: +NORMALIZE_WHITESPACE
    Domain 'X':
     x
     0
     1
     2

    Notes
    -----
    Let $(X, \mathcal{F})$ be a measurable space consisting of a $\sigma$-algebra $\mathcal{F}$ on a set $X$. A *measure* $\mu$ is a countably additive function $\mu: \mathcal{F} \to [0,\infty)$. Here, *countable additivity* means that

    $$
    \mu \left( \bigcup_{k=1}^\infty A_k \right) = \sum_{k=1}^\infty \mu(A_k)
    $$

    for all collections $\{A_k\}_{k=1}^\infty$ of pairwise disjoint measurable sets. If $X$ is finite (as it always is, in SigAlg), then $\mu$ needs only to be finitely additive in order to be countably additive.
    """

    _default_name = "mu"
    _str_name = "Measure"
    _repr_name = "Measure"
    _properties = []

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        domain: MeasureDomain | None = None,
        mapping: MappingLike | None = None,
        kind: Literal["measure", "probability"] = "measure",
        domain_kind: Literal["Domain", "SampleSpace"] = "Domain",
        output_name: Hashable | None = None,
        name: Hashable | None = None,
    ) -> None:
        from ...validation.measure_domain_normalizer import MeasureDomainNormalizer
        from .probability_measure import ProbabilityMeasure

        if mapping is not None and domain is None:
            raise TypeError("If mapping is given, then the domain must be given too.")

        v = MeasureDomainNormalizer(measure_domain=domain, kind=kind)

        domain = v.domain
        self.sig_alg = v.sig_alg

        super().__init__(
            domain=domain,
            mapping=mapping,
            kind=kind,
            domain_kind=domain_kind,
            output_name=output_name,
            name=name,
        )

        if kind == "probability":
            self.__class__ = ProbabilityMeasure

    @classmethod
    def _from_validated(
        cls,
        *,
        data: pd.Series,
        kind: Literal["measure", "probability"],
        sig_alg: SigmaAlgebra,
        name: Hashable,
    ):
        from ..measures.probability_measure import ProbabilityMeasure

        measure = super()._from_validated(
            data=data,
            kind=kind,
            name=name,
            domain_kind=sig_alg.domain_kind,
            domain_name=sig_alg.name,
            index_kind="Index",
            index_name=None,
        )
        measure.sig_alg = sig_alg

        if kind == "probability":
            measure.__class__ = ProbabilityMeasure

        return measure

    @classmethod
    def counting(
        cls,
        domain: MeasureDomain,
        output_name: Hashable | None = None,
        name: Hashable = "C",
    ) -> Measure:
        r"""Create a counting measure on a sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        domain : MeasureDomain
            The domain of the measure. Either a `SigmaAlgebra` or an `IndexLike` object that can be coerced into a `Domain`; in the latter case the domain of the measure will be set to the power-set of the `Domain` instance.
        name : Hashable, default="C"
            A name for the measure.

        Examples
        --------
        >>> from sigalg.core import Domain, Measure, SigmaAlgebra
        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 0,
        ...         3: 0,
        ...     }
        ... )
        >>> print(F)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
           F
        x
        0  0
        1  1
        2  0
        3  0
        >>> C = Measure.counting(domain=F)
        >>> print(C)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'C':
           C
        F
        0  3
        1  1
        >>> D = Measure.counting(domain=X, name="D")
        >>> print(D)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'D':
           D
        x
        0  1
        1  1
        2  1
        3  1

        Notes
        -----
        Let $(X,\mathcal{F})$ be a finite measurable space. The *counting measure* on $\mathcal{F}$ is the unique measure $C$ for which

        $$
        C(A) = |A|
        $$

        for all atoms $A$ of $\mathcal{F}$. Here, $|A|$ is the cardinality of $A$.
        """
        from ...validation.measure_domain_normalizer import MeasureDomainNormalizer

        if output_name is None:
            output_name = name

        v = MeasureDomainNormalizer(measure_domain=domain)

        mapping = v.sig_alg.atom_id_to_cardinality
        data = pd.Series(mapping, index=v.domain.data, name=output_name)

        return cls._from_validated(
            data=data,
            kind="measure",
            sig_alg=v.sig_alg,
            name=name,
        )

    @classmethod
    def from_rand(
        cls,
        domain: MeasureDomain,
        num_null_atoms: int = 0,
        kind: Literal["measure", "probability"] = "measure",
        distribution: Literal["uniform", "poisson", "dirichlet"] = "uniform",
        max_value: int = 10,
        rate: float = 5.0,
        output_name: Hashable | None = None,
        name: Hashable | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> Measure:
        """Generate a random measure.

        Parameters
        ----------
        domain : MeasureDomain
            The domain of the measure. Either a `SigmaAlgebra` or an `IndexLike` object that can be coerced into a `Domain`; in the latter case the domain will be set to the power-set of the `Domain` instance.
        num_null_atoms : int, default=0
            The number of atoms in the sigma-algebra that should be assigned a measure of 0.
        kind : Literal["probability", "measure"], default="measure"
            The kind of measure to generate. If `"probability"`, generates a probability measure using a Dirichlet distribution. If `"measure"`, generates a general measure with integer values. If the method is called on the `ProbabilityMeasure` class, this parameter is ignored and a probability measure is always generated.
        distribution : Literal["uniform", "poisson"], default="uniform"
            The distribution to use when `kind="measure"`. If `"uniform"`, samples integers uniformly from `[min_value, max_value]`. If `"poisson"`, samples from a Poisson distribution with parameter `rate`.
        min_value : int, default=1
            The minimum value for uniform integer sampling (only used when `kind="measure"` and `distribution="uniform"`).
        max_value : int, default=10
            The maximum value for uniform integer sampling (only used when `kind="measure"` and `distribution="uniform"`).
        rate : float, default=5.0
            The rate parameter for Poisson sampling (only used when `kind="measure"` and `distribution="poisson"`).
        output_name: Hashable | None = None
        name : Hashable | None, default=None
            The name of the measure. If `None`, a default will be generated.
        random_state : int | np.random.Generator | None, default=None
            An optional random seed.

        Raises
        ------
        TypeError
            If `num_null_atoms` is not an integer. If `random_state` is not an integer, `np.random.Generator`, or `None`. If `name` is not hashable. If `min_value` or `max_value` are not integers.
        ValueError
            If `num_null_atoms` is negative or greater than or equal to the number of atoms in the sigma-algebra (if given) or the size of the sample space (if given). If `kind` is not "probability" or "measure". If `distribution` is not "uniform" or "poisson". If `min_value > max_value`. If `rate` is not positive.

        Returns
        -------
        random_measure : Measure
            A randomly generated measure. If `kind="probability"`, returns a `ProbabilityMeasure`; otherwise returns a `Measure`.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import Domain, Measure, SigmaAlgebra
        >>> rng = np.random.default_rng(42)

        Define a 1-dimensional domain and a sigma-algebra with four atoms.

        >>> X = Domain.from_sequence(size=5, variable_name="x")
        >>> F = SigmaAlgebra(domain=X, mapping=dict(zip(X, [0, 1, 1, 2, 3])))

        Generate a random measure with values drawn from a uniform distribution on the integers in `[0, 10)` and with one null atom.

        >>> mu = Measure.from_rand(
        ...     domain=F,
        ...     num_null_atoms=1,
        ...     random_state=rng,
        ... )
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'mu':
           mu
        F
        0   0
        1   6
        2   1
        3   7

        Generate a random measure with values drawn from a Poisson distribution with `rate=5.0`.

        >>> nu = Measure.from_rand(
        ...     domain=F,
        ...     distribution="poisson",
        ...     random_state=rng,
        ...     name="nu",
        ... )
        >>> print(nu)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'nu':
           nu
        F
        0   2
        1   7
        2   7
        3   5

        Generate a random probability measure with values drawn from a Dirichlet distribution.

        >>> P = Measure.from_rand(
        ...     domain=F,
        ...     num_null_atoms=1,
        ...     distribution="dirichlet",
        ...     random_state=rng,
        ... )
        >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                  P
        F
        0  0.060728
        1  0.757515
        2  0.181758
        3  0.000000
        """
        from ...validation.measure_domain_normalizer import MeasureDomainNormalizer
        from .probability_measure import ProbabilityMeasure

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
            cls is ProbabilityMeasure
            or kind == "probability"
            or distribution == "dirichlet"
        ):
            kind = "probability"
            distribution = "dirichlet"
            name = name if name else "P"
        else:
            kind = "measure"
            name = name if name else "mu"

        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        v = MeasureDomainNormalizer(measure_domain=domain)

        domain = v.domain
        sig_alg = v.sig_alg

        if not isinstance(num_null_atoms, int) or num_null_atoms > len(domain):
            raise ValueError(
                "num_null_atoms must be an integer no larger than the number of atoms in the sigma-algebra."
            )

        if distribution == "uniform":
            arr = rng.integers(
                low=1,
                high=max_value,
                size=len(domain) - num_null_atoms,
            )

        elif distribution == "poisson":
            arr = rng.poisson(
                lam=rate,
                size=len(domain) - num_null_atoms,
            )

        else:
            arr = rng.dirichlet(
                alpha=(1 / (len(domain) - num_null_atoms),)
                * (len(domain) - num_null_atoms)
            ).T

        arr = np.concat([arr, np.zeros(num_null_atoms, dtype=int)])
        rng.shuffle(arr)

        if name is None:
            name = cls._default_name
        if output_name is None:
            output_name = name

        data = pd.Series(arr, index=domain.data, name=output_name)

        return cls._from_validated(
            data=data,
            kind=kind,
            sig_alg=sig_alg,
            name=name,
        )

    # --------------------- properties --------------------- #

    @cached_property
    def non_null_atoms(self) -> list[Set] | None:
        """Get the non-null atoms of the sigma-algebra of the measure.

        Examples
        --------
        >>> from sigalg.core import Domain, Measure, SigmaAlgebra
        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 0,
        ...     },
        ... )
        >>> for A in mu.non_null_atoms:
        ...     print(f"Atom id of non-null atom: {A.name}")
        Atom id of non-null atom: 0
        Atom id of non-null atom: 1
        """
        return self.sig_alg.non_null_atoms(measure=self)

    @cached_property
    def lattice(self) -> Lattice | None:
        """Pass."""
        if self.sig_alg is not None:
            return self.sig_alg.down_lattice
        else:
            return None

    # --------------------- methods --------------------- #

    def equal_almost_everywhere(
        self,
        first: Function,
        second: Function,
        tol: float = 1e-8,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        r"""Determine whether two measurable vectors are equal almost everywhere.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        first : MeasurableVector
            The first measurable vector.
        second : MeasurableVector
            The second measurable vector.
        tol : float, default=1e-8
            The tolerance below which a measure is considered to be zero for the purposes of this comparison.
        rtol : float, default=1e-5
            The relative tolerance for `np.isclose` when comparing the measurable vectors.
        atol : float, default=1e-8
            The absolute tolerance for `np.isclose` when comparing the measurable vectors.

        Returns
        -------
        equal_ae : bool
            `True` if the measurable vectors are equal almost everywhere; `False` otherwise.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Measure,
        ...     RandomVariable,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 1.2,
        ...         1: 2.3,
        ...         2: 0.0,
        ...     },
        ... )
        >>> X = RandomVariable.with_uniform(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 1.0,
        ...         1: 2.0,
        ...         2: 3.0,
        ...     },
        ... )
        >>> Y = RandomVariable.with_uniform(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 1.0,
        ...         1: 2.0,
        ...         2: 4.0,
        ...     },
        ...     name="Y",
        ... )
        >>> Z = RandomVariable.with_uniform(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 1.0,
        ...         1: 3.0,
        ...         2: 3.0,
        ...     },
        ...     name="Z",
        ... )
        >>> print(mu.equal_almost_everywhere(X, Y))
        True
        >>> print(mu.equal_almost_everywhere(X, Z))
        False
        >>> U = RandomVector.with_uniform(
        ...     domain=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 2),
        ...     },
        ...     name="U",
        ... )
        >>> V = RandomVector.with_uniform(
        ...     domain=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (-1, 4),
        ...     },
        ...     name="V",
        ... )
        >>> W = RandomVector.with_uniform(
        ...     domain=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (-1, 1),
        ...         2: (3, 2),
        ...     },
        ...     name="W",
        ... )
        >>> print(mu.equal_almost_everywhere(U, V))
        True
        >>> print(mu.equal_almost_everywhere(U, W))
        False

        Notes
        -----
        Two measurable vectors $f,g:X \to \mathbb{R}^d$ defined on a measure space $(X, \mathcal{F}, \mu)$ are *equal almost everywhere* if

        $$
        \mu \left( \{x \in X : f(x) \neq g(x)\} \right) = 0.
        $$
        """
        from .._utils.utils import to_df

        if first not in self.sig_alg or second not in self.sig_alg:
            raise ValueError(
                "Both functions must be measurable with respect to the sigma-algebra of the measure."
            )
        if first.dimension != second.dimension:
            raise ValueError("The measurable vectors must have the same dimension.")
        if not first.is_measurable(self.sig_alg) or not second.is_measurable(
            self.sig_alg
        ):
            raise ValueError(
                "The measurable vectors must be measurable with respect to the sigma-algebra of the measure."
            )

        sig_alg_df = to_df(self.sig_alg.data)

        first_df = (
            pd.concat([sig_alg_df, first.data], axis=1)
            .drop_duplicates()
            .set_index(list(sig_alg_df.columns))
        )
        second_df = (
            pd.concat([sig_alg_df, second.data], axis=1)
            .drop_duplicates()
            .set_index(list(sig_alg_df.columns))
        )
        first_arr = first_df.to_numpy()
        second_arr = second_df.to_numpy()
        prob_arr = self.data.to_numpy()

        if first.dimension == 1:
            are_different = (
                ~np.isclose(first_arr, second_arr, rtol=rtol, atol=atol)
            ).squeeze()
        else:
            are_different = ~np.all(
                np.isclose(first_arr, second_arr, rtol=rtol, atol=atol), axis=1
            )

        measure_different = np.sum(are_different.astype(float) * prob_arr)

        return bool(measure_different < tol)

    def restrict_to(
        self,
        obj: SigmaAlgebra | Set | list[Hashable],
        normalize: bool = False,
        subset_name: Hashable | None = "A",
        name: Hashable | None = None,
    ) -> Measure:
        """Restrict the measure to a sub-sigma-algebra and return a new measure.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The sub-sigma-algebra to which to restrict the measure.

        Returns
        -------
        measure : Measure
            A new measure restricted to a sub-sigma-algebra.
        name : Hashable | None, default=None
            The name of the restriction. If `None`, a default will be generated.

        Examples
        --------
        >>> from sigalg.core import Domain, Measure, Set, SigmaAlgebra

        Define a sigma-algebra, a sub-sigma-algebra, and a measure on the larger sigma-algebra.

        >>> X = Domain.from_sequence(size=5)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...         4: 2,
        ...     },
        ... )
        >>> G = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 0,
        ...         3: 1,
        ...         4: 1,
        ...     },
        ...     name="G",
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 1,
        ...         1: 3,
        ...         2: 4,
        ...     },
        ... )

        Restrict the measure using the `restrict_to` method.

        >>> mu_G = mu.restrict_to(G)
        >>> print(mu_G)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'mu|G':
             mu|G
        G
        0       4
        1       4

        Restrict the measure using the `|` operator.

        >>> mu_G = mu | G
        >>> print(mu_G)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'mu|G':
             mu|G
        G
        0       4
        1       4

        We may also restrict the measure to a subset in its sigma-algebra.

        >>> U = Set([2, 3, 4], domain=X, name="U")
        >>> print(mu | U)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'mu|U':
           mu|U
        F
        1     3
        2     4

        We may use the `restrict_to` method, passing `normalize=True`, to create a probability measure.

        >>> P = mu.restrict_to(U, normalize=True, name="P")
        >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                  P
        F
        1  0.428571
        2  0.571429
        """
        import pandas as pd

        from .._utils import to_df
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from ..spaces.set import Set

        if isinstance(obj, SigmaAlgebra):
            sig_alg = obj

            # TODO: add fast path if pandas_all_equal
            if sig_alg is self.sig_alg:
                return self

            if sig_alg not in self.lattice:
                raise TypeError(
                    "If given obj is a sigma-algebra, it must be a sub-sigma-algebra of the sigma-algebra of the measure."
                )

            atom_data = to_df(self.lattice.get_atom_data(sig_alg), "_alg")

            if name is None:
                name = f"{self.name}|{sig_alg.name}"

            data = (
                pd.concat([atom_data, self.data], axis=1)
                .groupby(list(atom_data.columns))[self.name]
                .sum()
                .rename(name)
            )
            data.index.names = sig_alg.variable_names

            return type(self)._from_validated(
                data=data,
                kind=self.kind,
                sig_alg=sig_alg,
                name=name,
            )

        elif isinstance(obj, Set | list):
            subset = obj

            if not isinstance(subset, Set):
                subset = Set(subset, domain=self.sig_alg.domain, name=subset_name)
            if self.sig_alg not in subset.lattice:
                raise ValueError(
                    "If given, subset must be in the sigma-algebra of the measure."
                )

            if name is None:
                name = f"{self.name}|{subset.name}"

            atom_data = subset.lattice.get_atom_data(self.sig_alg)
            data = self.data[atom_data != 0].rename(name)

            if normalize:
                if self(subset) < 1e-10:
                    raise ValueError(
                        "Cannot normalize the restrict measure on a subset of measure 0."
                    )
                data /= self(subset)

            return Measure._from_validated(
                data=data,
                kind="measure" if not normalize else "probability",
                sig_alg=self.sig_alg | subset,
                name=name,
            )

    def is_absolutely_continuous(
        self, base_measure: Measure, tol: float = 1e-8
    ) -> bool:
        """Pass."""
        if self.sig_alg != base_measure.sig_alg:
            raise ValueError(
                "Only measures with the same sigma-algebra may be compared for absolute continuity."
            )

        self_data = (self | base_measure.sig_alg).data.reindex(base_measure.data.index)
        base_data = base_measure.data

        return bool(((base_data > tol) | (self_data < tol)).all())

    def __contains__(self, sig_alg: SigmaAlgebra) -> bool:
        """Pass."""
        if self.sig_alg is not None:
            return sig_alg in self.lattice

    def get_random_set(
        self,
        num_atoms: int,
        is_null: bool = False,
        name: Hashable = "A",
        random_state: int | np.random.Generator | None = None,
    ) -> Set:
        """Get a random (possibly null) measurable set from the sigma-algebra of the measure.

        Parameters
        ----------
        num_atoms : int
            The number of atoms to include in the random set.
        is_null : bool, default=False
            If `True`, the random set will be a null set (i.e., it will have measure zero).
        name : Hashable, default="A"
            The name of the random set.
        random_state : int | np.random.Generator | None, default=None
            An optional random seed.

        Raises
        ------
        TypeError
            If `num_atoms` is not an integer, or if `random_state` is not an integer, `np.random.Generator`, or `None`.
        ValueError
            If `num_atoms` is not a positive integer.

        Returns
        -------
        random_set : MeasurableSet
            A random measurable set from the sigma-algebra of the measure.
        """
        if not isinstance(num_atoms, int):
            raise TypeError("num_atoms must be an integer.")
        if num_atoms <= 0:
            raise ValueError("num_atoms must be a positive integer.")
        if random_state is not None and not isinstance(
            random_state, (int, np.random.Generator)
        ):
            raise TypeError(
                "random_state must be an integer, np.random.Generator, or None."
            )

        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        if not is_null:
            return self.sig_alg.get_random_set(
                num_atoms=num_atoms, random_state=random_state
            )
        else:
            null_IDs = list(self.data[self.data == 0].index)
            atom_IDs = rng.choice(
                null_IDs,
                size=min(num_atoms, len(null_IDs)),
                replace=False,
            )
            points = [
                point for id in atom_IDs for point in self.sig_alg.atom_id_to_points[id]
            ]

            return self.sig_alg.get_set(points, name=name)

    # --------------------- dunder operators --------------------- #

    def __or__(
        self,
        obj: SigmaAlgebra | Set | list[Hashable],
    ) -> Measure:
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

    def __rshift__(self, vec: MeasurableVector) -> Measure:
        """Pushforward the measure through a measurable vector.

        Calls the method `Operators.pushforward`. See the documentation of that method for more information.

        Parameters
        ----------
        vec : MeasurableVector
            The measurable vector through which to push the measure forward.

        Returns
        -------
        measure : Measure
            The pushforward measure.
        """
        from ..functions.operators import Operators

        return Operators.pushforward(vec=vec, measure=self)

    # TODO: add docstring
    def __lshift__(self, other: Measure):
        """Pass."""
        return self.is_absolutely_continuous(other)

    # --------------------- data access methods --------------------- #

    def __call__(self, *args, **kwargs) -> Real | Function:
        """Get the measure.

        The return value is determined by the following rules:

        1. If a complete set of atom identifiers (as keyword arguments), a real number is returned. This number is the measure of the atom.

        2. If a measurable `Set` is provided (as a positional argument), a real number is returned. This number is the measure of the set.

        3. If a list of points is provided (as a positional argument), the method first checks if a measurable `Set` can be made. If so, a real number is returned. This number is the measure of the set.

        Parameters
        ----------
        *args : tuple
            Positional arguments.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        measure : Real | Function
            The measure of the set or a `Function` instance from a partial evaluation.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     Measure,
        ...     SigmaAlgebra,
        ... )

        Define a measure on a measurable space.

        >>> X = Domain.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (0, 2),
        ...         3: (2, 4),
        ...         4: (2, 4),
        ...         5: (2, 4),
        ...     },
        ...     variable_names=["u", "v"],
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         (1, 2): 2,
        ...         (0, 2): 4,
        ...         (2, 4): 6,
        ...     },
        ... )

        Extract a measurable set and get its measure.

        >>> U = F.get_set([0, 1, 2], name="U")
        >>> mu(U)
        6

        We may also pass in a list of points

        >>> mu([0, 1, 2])
        6

        The `Measure` subclasses `Function`, so we may also call the measure on keyword arguments corresponding to atom identifiers.

        >>> mu(u=0, v=2)
        4

        Partial calls are also possible.

        >>> print(mu(v=2))  # doctest: +NORMALIZE_WHITESPACE
        Function 'mu(v=2)':
           mu(v=2)
        u
        1        2
        0        4
        """
        from .._utils.utils import to_df
        from ..spaces.set import Set

        measurable_set = None

        if len(args) == 1 and len(kwargs) == 0:
            if isinstance(args[0], Set):
                measurable_set = args[0]
            elif isinstance(args[0], list):
                measurable_set = self.sig_alg.get_set(args[0])
            else:
                raise TypeError(
                    "If a positional argument is passed into a measure, it must be an instance of MeasurableSet or a list of points corresponding to a measurable set in the sigma-algebra."
                )

        elif len(args) == 0 and len(kwargs) > 0:
            if not set(kwargs.keys()) <= set(self.variable_names):
                raise ValueError(
                    "The keyword arguments passed to a measure must be atom identifiers of the sigma-algebra."
                )

        else:
            raise ValueError(
                "A measure may only be called with a MeasurableSet (or list of points corresponding to a measurable set) as a positional argument, or atom identifiers as keyword arguments."
            )

        if measurable_set is not None:
            sig_alg_data = to_df(self.sig_alg.data)

            indicator_atom_data = (
                pd.concat([sig_alg_data, measurable_set.indicator_data], axis=1)
                .drop_duplicates()
                .set_index(list(sig_alg_data.columns))
                .squeeze(axis=1)
            )
            return (self.data * indicator_atom_data).sum().astype(Real)

        else:
            result = super().__call__(**kwargs)

            if hasattr(result, "data"):
                if isinstance(result.data, pd.Series) and result.data.empty:
                    return 0.0
                else:
                    result.data.name = result.name

            return result

    def to_function(self) -> Function:
        """Promote to a `Function` instance."""
        return Function._from_validated(
            data=self.data,
            kind="any",
            domain_kind=type(self.domain).__name__,
            domain_name=self.domain.name,
            index_kind="Index",
            index_name=None,
            name=self.name,
        )

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a concise string representation of the measure.

        Returns
        -------
        repr_str : str
            A string representation of the measure.
        """
        if self.data is None:
            return type(self)._repr_name + "(empty)"
        else:
            return (
                type(self)._repr_name + f"(domain={self.sig_alg.domain.name}, "
                f"sig_alg={self.sig_alg.name}, "
                f"name={self.name})"
            )

    # --------------------- equality --------------------- #

    def __eq__(self, other: Measure) -> bool:
        """Check equality with another measure.

        Two measures are considered equal if they have the same sigma-algebras and identical values for each atom.

        Parameters
        ----------
        other : Measure
            The other measure to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the two measures are equal, `False` otherwise.
        """
        # HACK: this branch catches the error in one of the tests for the factorization of the joint distribution in the tests
        if not isinstance(other, Measure):
            if isinstance(other, Function):
                self_domain = self.domain.data
                other_domain = other.domain.data

                intersection = self_domain.intersection(other_domain)
                if not intersection.equals(self_domain):
                    return False

                complement_domain = other_domain.difference(self_domain)

                return np.allclose(self.data, other.data[intersection]) and np.allclose(
                    other.data[complement_domain], 0.0
                )

            raise TypeError("Can only compare with another Measure instance.")

        if self.sig_alg != other.sig_alg:
            return False

        atom_data = self.lattice.get_atom_data(other.sig_alg)
        self_data = self.data.copy()
        other_data = other.data

        if other.sig_alg.dimension > 1:
            self_data.index = pd.MultiIndex.from_frame(
                atom_data.reindex(self_data.index), names=other.variable_names
            )

        else:
            self_data.index = pd.Index(
                atom_data.reindex(self_data.index), name=other.variable_names[0]
            )

        return np.array_equal(other_data, self_data.reindex(other_data.index))

    # --------------------- comparison methods --------------------- #

    def is_restriction_of(self, other: Measure) -> bool:
        """Check whether this measure is the restriction of the other measure to a sub-sigma-algebra.

        Returns
        -------
        is_le : bool
            `True` if this measure is the restriction of the other measure or is equal to it, `False` otherwise.
        """
        if self is other:
            return True

        return bool((self.sig_alg <= other.sig_alg) and (self == other | self.sig_alg))

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
        **kwargs,
    ) -> Function:
        """Apply a binary operation to this measure."""
        return Function._apply_binary_operation(
            self=self.to_function(),
            other=other,
            operation=operation,
            op_symbol=op_symbol,
            reverse=reverse,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
        )
