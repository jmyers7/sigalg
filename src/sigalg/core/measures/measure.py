"""A class representing a measure on a sigma-algebra."""

from __future__ import annotations

from collections.abc import Hashable
from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy.stats import dirichlet

from ..functions.multivariate_function import MultivariateFunction

if TYPE_CHECKING:
    from ...typing.mapping_like import MappingLike
    from ...typing.measure_domain import MeasureDomain
    from ..functions.measurable_vector import MeasurableVector
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.measurable_set import MeasurableSet


class Measure(MultivariateFunction):
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
    Define a measure on a sigma-algebra with two atoms.

    >>> from sigalg.core import Domain, Measure, SigmaAlgebra
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
             measure
    atom_ID
    0              1
    1              2

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
             measure
    point
    0              3
    1              1
    2              4
    >>> print(nu.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'R':
              point
    point
    0             0
    1             1
    2             2

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
             measure
    point
    0              3
    1              1
    2              4
    >>> print(nu.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'R':
              point
    point
    0             0
    1             1
    2             2

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
            probability
    point
    0               0.5
    1               0.2
    2               0.3
    >>> print(type(P))
    <class 'sigalg.core.measures.probability_measure.ProbabilityMeasure'>
    >>> print(P.domain)  # doctest: +NORMALIZE_WHITESPACE
    Domain 'X':
     point
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
    _properties = MultivariateFunction._properties + ["_sig_alg", "_non_null_atoms"]

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        domain: MeasureDomain | None = None,
        mapping: MappingLike | None = None,
        kind: Literal["measure", "probability"] = "measure",
        output_name: str = "measure",
        name: Hashable = "mu",
    ) -> None:
        from ...validation.measure_domain_validator import MeasureDomainValidator
        from .probability_measure import ProbabilityMeasure

        v = MeasureDomainValidator(measure_domain=domain, kind=kind)
        output_name = "probability" if kind == "probability" else output_name

        super().__init__(
            domain=v.domain,
            mapping=mapping,
            output_name=output_name,
            name=name,
            kind=kind,
        )

        self._kind = kind
        self._sig_alg = v.sig_alg

        if kind == "probability":
            self.__class__ = ProbabilityMeasure

    @classmethod
    def counting(cls, domain: MeasureDomain, name: Hashable = "C") -> Measure:
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
        >>> F = SigmaAlgebra.from_rand(num_atoms=2, domain=X, random_state=42)
        >>> print(F)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                atom_ID
        point
        0             0
        1             1
        2             0
        3             0
        >>> C = Measure.counting(domain=F)
        >>> print(C)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'C':
                measure
        atom_ID
        0              3
        1              1
        >>> D = Measure.counting(domain=X, name="D")
        >>> print(D)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'D':
                measure
        point
        0             1
        1             1
        2             1
        3             1

        Notes
        -----
        Let $(X,\mathcal{F})$ be a finite measurable space. The *counting measure* on $\mathcal{F}$ is the unique measure $C$ for which

        $$
        C(A) = |A|
        $$

        for all atoms $A$ of $\mathcal{F}$. Here, $|A|$ is the cardinality of $A$.
        """
        from ...validation.measure_domain_validator import MeasureDomainValidator

        v = MeasureDomainValidator(measure_domain=domain)

        mapping = v.sig_alg.atom_id_to_cardinality

        return cls(domain=v.sig_alg, mapping=mapping, name=name)

    @classmethod
    def from_rand(
        cls,
        domain: MeasureDomain,
        num_null_atoms: int = 0,
        kind: Literal["probability", "measure"] = "measure",
        distribution: Literal["uniform", "poisson"] = "uniform",
        min_value: int = 1,
        max_value: int = 10,
        rate: float = 5.0,
        name: Hashable | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> Measure:
        """Generate a random measure.

        This method generates either a random probability measure (using a Dirichlet distribution) or a random general measure (using uniform or Poisson-distributed integers).

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
        Generate a random probability measure.

        >>> import numpy as np
        >>> from sigalg.core import Measure, ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> rng = np.random.default_rng(seed=42)
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> P = ProbabilityMeasure.from_rand(domain=F, random_state=rng)
        >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                    probability
        atom_ID
        0           0.492826
        1           0.507174

        Generate a random measure with uniform integers.

        >>> mu = Measure.from_rand(domain=F, random_state=rng)
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'mu':
                 measure
        atom_ID
        0              1
        1              9

        Generate a random measure with Poisson integers.

        >>> nu = Measure.from_rand(domain=Omega, random_state=rng, distribution="poisson", name="nu")
        >>> print(nu)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'nu':
             measure
        sample
        0          7
        1          9
        2          5

        Generate a sparse measure with null atoms.

        >>> xi = Measure.from_rand(domain=Omega, random_state=rng, num_null_atoms=1, name="xi")
        >>> print(xi)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'xi':
             measure
        sample
        0          7
        1          0
        2          8
        """
        from ...validation.measure_domain_validator import MeasureDomainValidator

        if not isinstance(num_null_atoms, int):
            raise TypeError("num_null_atoms must be an integer.")
        if num_null_atoms < 0:
            raise ValueError("num_null_atoms must be non-negative.")
        if random_state is not None and not isinstance(
            random_state, (int, np.random.Generator)
        ):
            raise TypeError(
                "random_state must be an integer, np.random.Generator, or None."
            )
        if not isinstance(name, Hashable):
            raise TypeError("name must be hashable.")
        if kind not in ["probability", "measure"]:
            raise ValueError('kind must be either "probability" or "measure".')
        if distribution not in ["uniform", "poisson"]:
            raise ValueError('distribution must be either "uniform" or "poisson".')
        if not isinstance(min_value, int) or not isinstance(max_value, int):
            raise TypeError("min_value and max_value must be integers.")
        if min_value > max_value:
            raise ValueError("min_value must be less than or equal to max_value.")
        if rate <= 0:
            raise ValueError("rate must be positive.")

        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        is_prob_measure = cls.__name__ == "ProbabilityMeasure"
        kind = "probability" if is_prob_measure else kind

        if name is None:
            name = "P" if kind == "probability" else "mu"

        v = MeasureDomainValidator(measure_domain=domain, kind=kind)

        if num_null_atoms >= len(v.domain):
            raise ValueError(
                "num_null_atoms must be less than either the number of atoms of sig_alg (if given) or the size of the domain (if given)."
            )

        if kind == "probability":
            values_arr = (
                dirichlet.rvs(
                    alpha=[
                        1,
                    ]
                    * (len(v.domain) - num_null_atoms),
                    random_state=rng,
                )
                .squeeze()
                .tolist()
            )
            values_arr = (
                [values_arr] if not isinstance(values_arr, list) else values_arr
            )
            values_arr = values_arr + [0.0] * num_null_atoms
        else:
            if distribution == "uniform":
                values_arr = rng.integers(
                    low=min_value,
                    high=max_value + 1,
                    size=len(v.domain) - num_null_atoms,
                ).tolist()
            else:
                values_arr = rng.poisson(
                    lam=rate,
                    size=len(v.domain) - num_null_atoms,
                ).tolist()
            values_arr = values_arr + [0] * num_null_atoms

        rng.shuffle(values_arr)
        mapping = dict(zip(v.domain, values_arr))

        return cls(domain=v.sig_alg, mapping=mapping, name=name, kind=kind)

    # --------------------- properties --------------------- #

    @property
    def sig_alg(self) -> SigmaAlgebra | None:
        """Get the sigma-algebra on which the measure is defined.

        The `sig_alg` property is settable. The new sigma-algebra must be a sub-sigma-algebra of the current sigma-algebra. The measure will be restricted to the new sigma-algebra.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra on which the measure is defined.

        Examples
        --------
        >>> from sigalg.core import Domain, Measure, SigmaAlgebra
        >>> X = Domain.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=X,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...     },
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 3,
        ...     },
        ... )
        >>> print(mu.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
               atom_ID
        point
        0            0
        1            1
        2            2
        3            2
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
        >>> mu.sig_alg = G
        >>> print(mu.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
               atom_ID
        point
        0            0
        1            1
        2            1
        3            1
        >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'mu|G':
                 measure
        atom_ID
        0              1
        1              5
        """
        return self._sig_alg

    @sig_alg.setter
    def sig_alg(self, sig_alg: SigmaAlgebra) -> None:
        """Set the sigma-algebra on which the measure is defined.

        The new sigma-algebra must be a sub-sigma-algebra of the current sigma-algebra. The measure will be restricted to the new sigma-algebra.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The new sigma-algebra on which the measure is defined.

        Raises
        ------
        TypeError
            If `sig_alg` is not a `SigmaAlgebra` instance.
        ValueError
            If `sig_alg` is not a sub-sigma-algebra of the current sigma-algebra, or if the measure has no data.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from ..utils.utils import _to_df

        if not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be a SigmaAlgebra instance.")
        if not sig_alg <= self._sig_alg:
            raise ValueError(
                "sig_alg must be a sub-sigma-algebra of the current sigma-algebra."
            )
        if self.data is None:
            raise ValueError("Cannot set sig_alg when the measure has no data.")

        super = self._sig_alg
        sub = sig_alg

        super_data = _to_df(super.data)
        sub_data = _to_df(sub.data, "_sub")

        mapping = (
            pd.concat(
                [super_data, sub_data],
                axis=1,
            )
            .drop_duplicates(list(super_data.columns))
            .set_index(list(super_data.columns))
        )

        # TODO: check merge logic — possibly change to `on`?
        mapping = pd.merge(
            left=mapping, right=self.data, left_index=True, right_index=True
        )
        mapping = mapping.groupby(by=list(sub_data.columns), sort=False)[
            self.output_name
        ].sum()

        mapping.index = sub.atom_space.data

        if sub != super:
            name = f"{self.name}|{sub.name}"
        else:
            name = self.name

        new = type(self)(domain=sub, mapping=mapping, name=name)
        self.__dict__.update(new.__dict__)

    @property
    def non_null_atoms(self) -> list[MeasurableSet] | None:
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

    @property
    def kind(self) -> Literal["measure", "probability"]:
        """Get the kind of the measure.

        Returns
        -------
        kind : Literal["measure", "probability"]
            The kind of the measure, which can be "measure" or "probability".
        """
        return self._kind

    # --------------------- methods --------------------- #

    def equal_almost_everywhere(
        self,
        first: MeasurableVector,
        second: MeasurableVector,
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

        Raises
        ------
        TypeError
            If `first` or `second` are not `MeasurableVector` instances.
        ValueError
            If `first` or `second` are not measurable with respect to the sigma-algebra of this measure, or if they have different dimensions.

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
        from ..functions.measurable_vector import MeasurableVector
        from ..utils.utils import _to_df

        if not isinstance(first, MeasurableVector) or not isinstance(
            second, MeasurableVector
        ):
            raise TypeError("first and second must be MeasurableVector instances.")
        if first.dimension != second.dimension:
            raise ValueError("The measurable vectors must have the same dimension.")
        if not first.is_measurable(self.sig_alg) or not second.is_measurable(
            self.sig_alg
        ):
            raise ValueError(
                "The measurable vectors must be measurable with respect to the sigma-algebra of the measure."
            )

        sig_alg_df = _to_df(self.sig_alg.data)

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

        return measure_different < tol

    def restrict_to(self, sig_alg: SigmaAlgebra, in_place: bool = False) -> Measure:
        """Restrict the measure to a sub-sigma-algebra.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The sub-sigma-algebra to which to restrict the measure.
        in_place : bool, default=False
            Whether to modify the current instance in place.

        Returns
        -------
        measure : Measure
            The current measure restricted to the new sigma-algebra if `in_place` is `True`, otherwise a new instance of `Measure`.

        Examples
        --------
        Define a sigma-algebra, a sub-sigma-algebra, and a measure on the larger sigma-algebra.

        >>> from sigalg.core import Domain, Measure, SigmaAlgebra
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

        >>> mu_G = mu.restrict_to(sig_alg=G)
        >>> print(mu_G)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'mu|G':
                 measure
        atom_ID
        0              4
        1              4

        Restrict the measure using the `|` operator.

        >>> mu_G = mu | G
        >>> print(mu_G)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'mu|G':
                 measure
        atom_ID
        0              4
        1              4
        """
        if in_place:
            if self.sig_alg != sig_alg:
                self.sig_alg = sig_alg
            return self
        else:
            measure = Measure(
                domain=self.sig_alg,
                mapping=self.data,
                name=self.name,
                kind=self.kind,
                output_name=self.output_name,
            )
            measure.sig_alg = sig_alg
            return measure

    def get_random_set(
        self,
        num_atoms: int,
        is_null: bool = False,
        name: Hashable = "A",
        random_state: int | np.random.Generator | None = None,
    ) -> MeasurableSet:
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

    def __or__(self, sig_alg: SigmaAlgebra) -> Measure:
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
        return self.restrict_to(sig_alg=sig_alg)

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
        return bool(((other.data > 1e-8) | (self.data < 1e-8)).all())

    # --------------------- data access methods --------------------- #

    # TODO: Check that the `point` argument matches the variables names of the underlying domain of the sigma-algebra
    def __call__(self, *args, **kwargs):
        """Get the measure of an event.

        One may pass arguments in one of the following ways:

        * A `MeasurableSet` instance as a (single) positional argument or a keyword argument named `measurable_set`.
        * A list of points as a (single) positional argument or a keyword argument named `measurable_set`. The list of points must correspond to a measurable event in the sigma-algebra of the measure.
        * A single point as a keyword argument named `point`. The point must correspond to a measurable (singleton) set in the sigma-algebra of the probability measure.
        * An atom ID of the sigma-algebra as a keyword argument.

        This method calls the `__call__` method of the parent class `MultivariateFunction` and hence allows partially applied calls. See the docstring of the parent class for details.

        Parameters
        ----------
        *args : tuple
            Positional arguments.
        **kwargs : dict
            Keyword arguments.

        Raises
        ------
        ValueError
            If the set is not measurable with respect to the sigma-algebra of the measure.

        Returns
        -------
        measure : Real
            The measure of the set.

        Examples
        --------
        >>> from sigalg.core import (
        ...     Domain,
        ...     Measure,
        ...     SigmaAlgebra,
        ... )
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
        ...     variable_names=["F_0", "F_1"],
        ... )
        >>> mu = Measure(
        ...     domain=F,
        ...     mapping={
        ...         (1, 2): 2,
        ...         (0, 2): 4,
        ...         (2, 4): 6,
        ...     },
        ... )
        >>> # Call on `MeasurableSet` instances as positional or keyword arguments
        >>> A = F.get_set([0, 1, 2])
        >>> print(mu(A))
        6
        >>> print(mu(measurable_set=A))
        6
        >>> # Call on a list as a positional or keyword argument
        >>> print(mu([0, 1, 2]))
        6
        >>> print(mu(measurable_set=[0, 1, 2]))
        6
        >>> # Call on a sample point as a keyword argument
        >>> print(mu(point=2))
        4
        >>> print(mu(F_0=0, F_1=2))
        4
        >>> # Evaluate the measure of a set using curried calls
        >>> print(mu(F_0=0)(F_1=2))
        4
        >>> print(mu(F_1=2)(F_0=0))
        4
        """
        from ..spaces.measurable_set import MeasurableSet

        measurable_set = None
        if len(args) == 1 and len(kwargs) == 0:
            if isinstance(args[0], MeasurableSet):
                measurable_set = args[0]
            if isinstance(args[0], list):
                measurable_set = self.sig_alg.get_set(args[0])
            if isinstance(args[0], Hashable):
                measurable_set = self.sig_alg.get_set([args[0]])
        elif "measurable_set" in kwargs and len(kwargs) == 1 and len(args) == 0:
            if isinstance(kwargs["measurable_set"], MeasurableSet):
                measurable_set = kwargs["measurable_set"]
            if isinstance(kwargs["measurable_set"], list):
                measurable_set = self.sig_alg.get_set(kwargs["measurable_set"])
        elif "point" in kwargs and len(kwargs) == 1 and len(args) == 0:
            measurable_set = self.sig_alg.get_set([kwargs["point"]])

        if measurable_set is not None and isinstance(measurable_set, MeasurableSet):
            if not measurable_set.sig_alg <= self.sig_alg:
                raise ValueError("Measurable set is not in the domain of the measure.")

            ones = pd.Series(
                [1] * len(measurable_set), index=measurable_set.data, name="indicator"
            )
            # TODO: check merge logic — possibly change to `on`?
            df = pd.merge(
                left=self.sig_alg.data,
                right=ones,
                how="left",
                left_index=True,
                right_index=True,
            ).fillna(0)

            if isinstance(self.sig_alg.data, pd.Series):
                index_name = self.sig_alg.data.name
            else:
                index_name = self.sig_alg.data.columns.to_list()

            atom_indicator = df.drop_duplicates().set_index(index_name).squeeze()

            return self.data[atom_indicator.astype(bool)].sum().astype(Real)

        try:
            return super().__call__(*args, **kwargs)
        except (TypeError, ValueError) as e:
            raise ValueError(
                "Error while evaluating a measure. Perhaps the callable function was not constructed properly due to an invalid parameter name."
            ) from e

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
            if isinstance(other, MultivariateFunction):
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

        if isinstance(self.sig_alg.data.index, pd.MultiIndex):
            other_sig_alg_data = other.sig_alg.data.reorder_levels(
                self.sig_alg.domain.variable_names
            )
        else:
            other_sig_alg_data = other.sig_alg.data

        self_sig_alg_sorted = self.sig_alg.data.sort_index()
        other_sig_alg_sorted = other_sig_alg_data.sort_index()

        self_sig_alg_var_names = [
            f"{name}_self" for name in self.sig_alg.variable_names
        ]
        other_sig_alg_var_names = [
            f"{name}_other" for name in other.sig_alg.variable_names
        ]

        # TODO: check merge logic — possibly change to `on`?
        self_merged = pd.merge(
            left=self_sig_alg_sorted.to_frame().add_suffix("_self")
            if isinstance(self_sig_alg_sorted, pd.Series)
            else self_sig_alg_sorted.add_suffix("_self"),
            right=self.data.rename("self"),
            left_on=self_sig_alg_var_names,
            right_index=True,
        )

        other_merged = pd.merge(
            left=other_sig_alg_sorted.to_frame().add_suffix("_other")
            if isinstance(other_sig_alg_sorted, pd.Series)
            else other_sig_alg_sorted.add_suffix("_other"),
            right=other.data.rename("other"),
            left_on=other_sig_alg_var_names,
            right_index=True,
        )

        return self_merged["self"].equals(other_merged["other"])

    # --------------------- comparison methods --------------------- #

    def __le__(self, other: Measure) -> bool:
        """Check whether this measure is the restriction of the other measure to a sub-sigma-algebra.

        Returns
        -------
        is_le : bool
            `True` if this measure is the restriction of the other measure or is equal to it, `False` otherwise.
        """
        return bool(
            (self.sig_alg < other.sig_alg) and (self == other | self.sig_alg)
        ) or (self == other)
