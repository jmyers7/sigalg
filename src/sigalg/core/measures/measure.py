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
    domain : MeasureDomain | None, default=None
        The domain of the measure. Either a `SigmaAlgebra` or an `IndexLike` object that can be coerced into a `Domain`. In the latter case, the domain will be set to the power set of the domain.
    mapping : MappingLike | None, default=None
        A mapping from the domain to the measure values.
    kind : Literal["measure", "probability"], default="measure"
        The kind of measure.
    output_name : Hashable | None, default=None
        The name of the output variable of the measure. If `None`, a default will be generated.
    name : Hashable | None, default=None
        The name of the measure. If `None`, a default will be generated.

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
            The domain of the measure. Either a `SigmaAlgebra` or an `IndexLike` object that can be coerced into a `Domain`. In the latter case, the domain of the measure will be set to the power set of the domain.
        output_name : Hashable | None, default=None
            The output name of the measure. If `None`, a default will be generated.
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
            The domain of the measure. Either a `SigmaAlgebra` or an `IndexLike` object that can be coerced into a `Domain`; in the latter case, the domain will be set to the power set of the domain.
        num_null_atoms : int, default=0
            The number of atoms in the sigma-algebra that should be assigned a measure of 0.
        kind : Literal["probability", "measure"], default="measure"
            The kind of measure to generate. If `'probability'`, generates a probability measure using a Dirichlet distribution.
        distribution : Literal["uniform", "poisson", "dirichlet"], default="uniform"
            The type of distribution from which to sample the values of the measure.
        max_value : int, default=10
            The maximum value for uniform integer sampling when `distribution='uniform'`. Integers are sampled from the interval `[1, max_value)`.
        rate : float, default=5.0
            The rate parameter for Poisson sampling when `distribution='poisson'`.
        output_name : Hashable | None = None
            Output name of measure. If `None`, a default will be generated.
        name : Hashable | None, default=None
            The name of the measure. If `None`, a default will be generated.
        random_state : int | np.random.Generator | None, default=None
            An optional random seed.

        Returns
        -------
        random_measure : Measure
            A randomly generated measure.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import Domain, Measure, SigmaAlgebra
        >>> rng = np.random.default_rng(42)

        Define a 1-dimensional domain and a sigma-algebra with four atoms.

        >>> X = Domain.from_sequence(size=5, variable_name="x")
        >>> F = SigmaAlgebra(domain=X, mapping=dict(zip(X, [0, 1, 1, 2, 3])))

        Generate a random measure with values drawn from a uniform distribution on the integers in `[1, 10)` and with one null atom.

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
        0  0.164599
        1  0.535368
        2  0.300033
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
            arr = rng.dirichlet(alpha=(1,) * (len(domain) - num_null_atoms)).T

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

        Returns
        -------
        non_null_atoms : list[Set] | None
            A list of the atoms of the sigma-algebra that have non-zero measure.

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
        """Return the downward lattice of all sigma-algebras contained in the domain sigma-algebra of the measure.

        Returns
        -------
        down_lattice : Lattice | None
            The downward lattice of all sigma-algebras contained in the domain sigma-algebra of the measure.

        Examples
        --------
        >>> from sigalg.core import Domain, Measure, SigmaAlgebra

        Define a measure on a measurable space.

        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 9,
        ...         1: 8,
        ...         2: 7,
        ...     },
        ... )

        The lattice of the measure is initialized with the domain sigma-algebra `F`.

        >>> mu.lattice
        Lattice(base=F, type=downward, num_sig_algs=1)

        Define a sub-sigma-algebra of `F` and check that it is in the lattice.

        >>> G = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 1,
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
        Measure 'mu|G':
           mu|G
        G
        0     9
        1    15

        Define another sigma-algebra, and check if it is a sub-sigma-algebra of `F`.

        >>> H = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
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

    # --------------------- measure methods --------------------- #

    def equal_almost_everywhere(
        self,
        first: Function,
        second: Function,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        r"""Determine whether two functions are equal almost everywhere.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        first : Functin
            The first function.
        second : Function
            The second function.
        rtol : float, default=1e-5
            The relative tolerance for `np.isclose` when comparing the functions.
        atol : float, default=1e-8
            The absolute tolerance for `np.isclose` when comparing the functions.

        Returns
        -------
        equal_ae : bool
            `True` if the measurable vectors are equal almost everywhere; `False` otherwise.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     Measure,
        ...     MeasurableFunction,
        ...     SigmaAlgebra,
        ... )

        Define a measure space and three measurable functions.

        >>> X = Domain.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,  # null atom
        ...         1: 1,
        ...         2: 1,
        ...     }
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 0,
        ...         1: 2,
        ...     },
        ... )
        >>> f = MeasurableFunction(
        ...     domain=X,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 2,
        ...     },
        ... )
        >>> g = MeasurableFunction(
        ...     domain=X,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0,
        ...         1: 2,
        ...         2: 2,
        ...     },
        ...     name="g",
        ... )
        >>> h = MeasurableFunction(
        ...     domain=X,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 1,
        ...         1: 3,
        ...         2: 3,
        ...     },
        ...     name="h",
        ... )

        Notice that `f` and `g` are equal, except on the atom with identifier `0`. However, this is a null atom.

        >>> mu.equal_almost_everywhere(f, g)
        True

        The functions `f` and `h` differ on a non-null atom.

        >>> mu.equal_almost_everywhere(f, h)
        False

        Notes
        -----
        Two functions $f,g:X \to \mathbb{R}^d$ defined on a measure space $(X, \mathcal{F}, \mu)$ are *equal almost everywhere* if

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

        return bool(measure_different < atol)

    def restrict_to(
        self,
        obj: SigmaAlgebra | Set | list[Hashable],
        normalize: bool = False,
        subset_name: Hashable | None = "A",
        name: Hashable | None = None,
    ) -> Measure:
        """Restrict the measure to either a sub-sigma-algebra or measurable set.

        Parameters
        ----------
        obj : SigmaAlgebra | Set | list[Hashable]
            The sub-sigma-algebra or `Set` instance to which to restrict the measure.
        normalize : bool, default=False
            If the measure is restricted to a `Set` instance, whether to normalize the measure values to create a probability measure.
        subset_name : Hashable | None, default="A"
            The name of the subset. Only used if `obj` is a `list[Hashable]`.
        name : Hashable | None, default=None
            The name of the restriction. If `None`, a default will be generated.

        Returns
        -------
        measure : Measure
            The restricted measure.

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

    def is_absolutely_continuous(self, other: Measure, tol: float = 1e-8) -> bool:
        r"""Determine whether the current measure is absolutely continuous with respect to another.

        See the Notes section below for the mathematical details,

        Parameters
        ----------
        other : Measure
            The other measure.
        tol : float, default=1e-8
            A tolerance below which a quantity will be deemed `0`.

        Returns
        -------
        is_absolutely_continuous : bool
            A Boolean flagging whether the current measure is absolutely continuous with respect to the `other` measure.

        Examples
        --------
        >>> from sigalg.core import Domain, Measure, SigmaAlgebra

        Define three measures on a measurable space.

        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 2,
        ...     },
        ... )
        >>> nu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...     },
        ...     name="nu",
        ... )
        >>> xi = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 0,
        ...     },
        ...     name="xi",
        ... )

        Notice that the only `nu`-null atom is also `mu`-null, and so `mu` is absolutely continuous with respect to `nu`.

        >>> mu.is_absolutely_continuous(nu)
        True

        Note that there is a `xi`-null atom that is not `mu`-null, and so `mu` is not absolutely continuous with respect to `xi`.

        >>> mu.is_absolutely_continuous(xi)
        False

        We can check these conditions using the `<<` operators, as well.

        >>> mu << nu
        True
        >>> mu << xi
        False

        Notes
        -----
        Let $\mu$ and $\nu$ be two measures on a measurable space $(X,\mathcal{F})$. We shall say that $\mu$ is *absolutely continuous* with respect to $\nu$, and write $\mu \ll \nu$, provided that $\mu(U)=0$ whenever $\nu(U)=0$, for all $U\in \mathcal{F}$.
        """
        if self.sig_alg != other.sig_alg:
            raise ValueError(
                "Only measures with the same sigma-algebra may be compared for absolute continuity."
            )

        self_data = (self | other.sig_alg).data.reindex(other.data.index)
        base_data = other.data

        return bool(((base_data > tol) | (self_data < tol)).all())

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
            If `True`, the random set will be a null set.
        name : Hashable, default="A"
            The name of the random set.
        random_state : int | np.random.Generator | None, default=None
            An optional random seed.

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
        """Restrict the measure to either a sub-sigma-algebra or measurable set.

        Internally calls the `restrict_to` method. See the docstring there for more details.
        """
        return self.restrict_to(obj)

    def __rshift__(self, vec: MeasurableVector) -> Measure:
        """Push the measure forward through a measurable vector.

        Internally calls the `Operators.pushforward` method. See the docstring there for more details.
        """
        from ..functions.operators import Operators

        return Operators.pushforward(vec=vec, measure=self)

    def __lshift__(self, other: Measure):
        """Determine whether the current measure is absolutely continuous with respect to another.

        Internally calls the `is_absolutely_continuous` method. See the docstring there for more details.
        """
        return self.is_absolutely_continuous(other)

    def __contains__(self, sig_alg: SigmaAlgebra) -> bool:
        """Check whether a sigma-algebra is contained in the domain sigma-algebra of this measure."""
        if self.sig_alg is not None:
            return sig_alg in self.lattice

    # --------------------- data access methods --------------------- #

    def __call__(self, *args, **kwargs) -> Real | Function:
        """Call the measure.

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
                    "If a positional argument is passed into a measure, it must be an instance of Set or a list of points corresponding to a measurable set in the sigma-algebra."
                )

        elif len(args) == 0 and len(kwargs) > 0:
            if not set(kwargs.keys()) <= set(self.variable_names):
                raise ValueError(
                    "The keyword arguments passed to a measure must be atom identifiers of the sigma-algebra."
                )

        else:
            raise ValueError(
                "A measure may only be called with a Set (or list of points corresponding to a measurable set) as a positional argument, or atom identifiers as keyword arguments."
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

    # --------------------- conversion methods --------------------- #

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

    # --------------------- equality and comparison methods --------------------- #

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
        from .._utils.utils import to_df

        if self.sig_alg != other.sig_alg:
            return False

        atom_data = to_df(self.lattice.get_atom_data(other.sig_alg))
        atom_data.columns = other.sig_alg.variable_names

        data = (
            pd.concat([self.data, atom_data], axis=1)
            .set_index(other.sig_alg.variable_names)
            .squeeze(axis=1)
        )

        return bool(data.equals(other.data))

    def is_close(self, other: Measure, rtol: float = 1e-5, atol=1e-8) -> bool:
        """Check if two measures have approximately the same values.

        Parameters
        ----------
        other : Measure
            The other measure to compare with.
        rtol : float, default=1e-5
            Relative tolerance for comparing values.
        atol : float, default=1e-8
            Absolute tolerance for comparing values.

        Returns
        -------
        is_close : bool
            `True` if the two measures have approximately the same values, `False` otherwise.
        """
        from .._utils.utils import to_df

        if self.sig_alg != other.sig_alg:
            return False

        atom_data = to_df(self.lattice.get_atom_data(other.sig_alg))
        atom_data.columns = other.sig_alg.variable_names

        data = (
            pd.concat([self.data, atom_data], axis=1)
            .set_index(other.sig_alg.variable_names)
            .squeeze(axis=1)
        )

        return np.allclose(data, other.data, rtol=rtol, atol=atol)

    def is_restriction_of(self, other: Measure, rtol: float = 1e-5, atol=1e-8) -> bool:
        """Check whether this measure is the restriction of the other measure to a sub-sigma-algebra.

        Parameters
        ----------
        other : Measure
            The other measure to compare to.
        rtol : float, default=1e-5
            Relative tolerance for comparing values.
        atol : float, default=1e-8
            Absolute tolerance for comparing values.

        Examples
        --------
        >>> from sigalg.core import Domain, Measure, SigmaAlgebra

        Define a measurable space and a sub-sigma-algebra.

        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ... )
        >>> G = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 1,
        ...     },
        ...     name="G",
        ... )

        Define two measures.

        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 3,
        ...     },
        ... )
        >>> nu = Measure(
        ...     domain=G,
        ...     mapping={
        ...         0: 1,
        ...         1: 5,
        ...     },
        ...     name="nu",
        ... )

        Note that `nu` is indeed the restriction of `mu` to the sub-sigma-algebra `G`.

        >>> nu.is_restriction_of(mu)
        True

        Returns
        -------
        is_le : bool
            `True` if this measure is the restriction of the other measure or is equal to it, `False` otherwise.
        """
        if self is other:
            return True

        return bool(
            (self.sig_alg <= other.sig_alg)
            and self.is_close(other | self.sig_alg, rtol=rtol, atol=atol)
        )

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
        """Apply a binary operation to this measure.

        See the method `Function._apply_binary_operation` for more details on the parameters and behavior.
        """
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
