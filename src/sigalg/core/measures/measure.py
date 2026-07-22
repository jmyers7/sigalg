"""A class representing a measure on a sigma-algebra."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ..functions.multivariate_function import MultivariateFunction
from ..functions.operators import OperatorsMethods

if TYPE_CHECKING:
    from ...validation.mapping_validator import MappingLike
    from ..spaces.domain import Domain
    from ..spaces.sample_space import SampleSpace
    from ..functions.random_vector import RandomVector
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra


class Measure(MultivariateFunction, OperatorsMethods):
    r"""A class representing a measure on a sigma-algebra.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    sig_alg : SigmaAlgebra | None, default=None
        The sigma-algebra on which the measure is defined.
    sample_space : SampleSpace | IndexLike | None, default=None
        The sample space on which the measure is defined.
    domain : Domain | None, default=None
        The domain of the measure. This parameter is not intended to be set by the user.
    mapping : MappingLike | Callable | None, default=None
        A mapping from the domain to the measure values.
    kind : Literal["any", "probabilities"], default="any"
        The kind of measure. If "any", the measure can take any non-negative values. If "probabilities", the measure will be promoted to an instance of `ProbabilityMeasure` and the `output_name` will be set to "probability".
    output_name : str, default="measure"
        The name of the output variable of the measure.
    name : Hashable | None, default=None
        A name for the measure. If `None`, the default name `mu` is used.

    Raises
    ------
    TypeError
        If `sig_alg` is not a `SigmaAlgebra` instance.
    ValueError
        If both `sig_alg` and `sample_space` are provided.

    Examples
    --------
    Define a measure on a sigma-algebra with two atoms.

    >>> from sigalg.core import Measure, SampleSpace, SigmaAlgebra
    >>> Omega = SampleSpace.from_sequence(size=3)
    >>> F = SigmaAlgebra(
    ...    sample_space=Omega,
    ...    mapping={
    ...        0: 0,
    ...        1: 0,
    ...        2: 1,
    ...    },
    ... )
    >>> mu = Measure(sig_alg=F, mapping={0: 1, 1: 2})
    >>> print(mu)  # doctest: +NORMALIZE_WHITESPACE
    Measure 'mu':
             measure
    atom_ID
    0              1
    1              2

    Define a measure directly on a sample space, which will use the power-set sigma-algebra by default.

    >>> nu = Measure(
    ...     sample_space=Omega,
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
    sample
    0              3
    1              1
    2              4

    Define a probability measure using the `Measure` constructor with the parameter `kind` set to `probabilities`.

    >>> P = Measure(
    ...     sample_space=Omega,
    ...     mapping={
    ...         0: 0.5,
    ...         1: 0.2,
    ...         2: 0.3,
    ...     },
    ...     kind="probabilities",
    ...     name="P",
    ... )
    >>> print(P)  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'P':
            probability
    sample
    0               0.5
    1               0.2
    2               0.3
    >>> print(type(P))
    <class 'sigalg.core.measures.probability_measure.ProbabilityMeasure'>

    Notes
    -----
    Let $(\Omega, \mathcal{F})$ be a measurable space consisting of a $\sigma$-algebra $\mathcal{F}$ on a set $\Omega$. A *measure* $\mu$ is a countably additive function $\mu: \mathcal{F} \to [0,\infty)$. Here, *countable additivity* means that

    $$
    \mu \left( \bigcup_{k=1}^\infty A_k \right) = \sum_{k=1}^\infty \mu(A_k)
    $$

    for all collections $\{A_k\}_{k=1}^\infty$ of pairwise disjoint measurable sets. If $\Omega$ is finite (as it always is, in SigAlg), then $\mu$ needs only to be finitely additive in order to be countably additive.
    """

    _default_name = "mu"
    _repr_name = "Measure"
    _properties = MultivariateFunction._properties + ["_sig_alg"]

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        sig_alg: SigmaAlgebra | None = None,
        sample_space: SampleSpace | None = None,
        domain: Domain | None = None,
        mapping: MappingLike | Callable | None = None,
        kind: Literal["any", "probabilities"] = "any",
        output_name: str = "measure",
        name: Hashable | None = None,
        **kwargs,
    ) -> None:
        from ..spaces.sample_space import SampleSpace
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .probability_measure import ProbabilityMeasure

        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be a SigmaAlgebra instance, if given.")
        if sample_space is not None and not isinstance(sample_space, SampleSpace):
            sample_space = SampleSpace(sample_space)
        if (sig_alg is not None) and (sample_space is not None):
            raise ValueError("Cannot provide both sig_alg and sample_space.")

        if name is None:
            name = self._default_name

        if domain is None:
            domain = sig_alg.atom_space if sig_alg is not None else sample_space

        output_name = "probability" if kind == "probabilities" else output_name

        super().__init__(
            domain=domain,
            mapping=mapping,
            output_name=output_name,
            name=name,
            kind=kind,
        )

        if sig_alg is not None:
            self._sig_alg = sig_alg
        elif sample_space is not None:
            self._sig_alg = SigmaAlgebra.power_set(sample_space)

        if kind == "probabilities":
            self.__class__ = ProbabilityMeasure

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
        >>> from sigalg.core import Measure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...     },
        ... )
        >>> mu = Measure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 3,
        ...     },
        ... )
        >>> print(mu.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
               atom_ID
        sample
        0            0
        1            1
        2            2
        3            2
        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
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
        sample
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
        """Set the sigma-algebra on which the probability measure is defined.

        The new sigma-algebra must be a sub-sigma-algebra of the current sigma-algebra. The probability measure will be restricted to the new sigma-algebra.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The new sigma-algebra on which the probability measure is defined.

        Raises
        ------
        TypeError
            If `sig_alg` is not a `SigmaAlgebra` instance.
        ValueError
            If `sig_alg` is not a sub-sigma-algebra of the current sigma-algebra, or if the probability measure has no data.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be a SigmaAlgebra instance.")
        if not sig_alg <= self._sig_alg:
            raise ValueError(
                "sig_alg must be a sub-sigma-algebra of the current sigma-algebra."
            )
        if self.data is None:
            raise ValueError(
                "Cannot set sig_alg when the probability measure has no data."
            )

        super = self._sig_alg
        sub = sig_alg

        mapping = pd.concat(
            [super.data.rename("super_ID"), sub.data.rename("sub_ID")],
            axis=1,
        ).drop_duplicates("super_ID")

        if super.dimension > 1:
            mapping = mapping.set_index(
                pd.MultiIndex.from_tuples(
                    list(mapping["super_ID"]), names=super.variable_names
                )
            ).drop(columns=["super_ID"])
        else:
            mapping = mapping.set_index("super_ID")

        mapping = pd.merge(mapping, self.data, left_index=True, right_index=True)
        mapping = mapping.groupby(by="sub_ID", sort=False)[self.output_name].sum()
        mapping.index = sub.atom_space.data

        if sub != super:
            name = f"{self.name}|{sub.name}"
        else:
            name = self.name

        new = type(self)(sig_alg=sub, mapping=mapping, name=name)
        self.__dict__.update(new.__dict__)

    @property
    def sample_space(self) -> SampleSpace:
        """Get the sample space of the probability measure.

        The `sample_space` property is settable. The new sample space must contain the same number of sample points. If the probability measure does not have a sigma-algebra, the sample space cannot be set.

        Returns
        -------
        sample_space : SampleSpace
            The sample space on which the probability measure is defined.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 2,
        ...         3: 2,
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.2,
        ...         1: 0.3,
        ...         2: 0.5,
        ...     },
        ... )
        >>> print(P.sample_space)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega':
         sample
              0
              1
              2
              3
        >>> S = SampleSpace(["a", "b", "c", "d"], name="S")
        >>> P.sample_space = S
        >>> print(P.sample_space)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'S':
         sample
              a
              b
              c
              d
        """
        return self._sig_alg._sample_space if self.sig_alg is not None else None

    @sample_space.setter
    def sample_space(self, sample_space: SampleSpace) -> None:
        """Set the sample space of the probability measure.

        The new sample space must contain the same number of sample points.

        Parameters
        ----------
        sample_space : SampleSpace
            The new sample space on which the probability measure is defined.

        Raises
        ------
        ValueError
            If the probability measure does not have a sigma-algebra.
        """
        self.sig_alg.sample_space = sample_space

    # --------------------- methods --------------------- #

    def equal_almost_everywhere(
        self,
        first: RandomVector,
        second: RandomVector,
        tol: float = 1e-8,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        r"""Determine whether two random vectors are equal almost everywhere.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        first : RandomVector
            The first random vector.
        second : RandomVector
            The second random vector.
        tol : float, default=1e-8
            The tolerance below which a measure is considered to be zero for the purposes of this comparison.
        rtol : float, default=1e-5
            The relative tolerance for `np.isclose` when comparing the random vectors.
        atol : float, default=1e-8
            The absolute tolerance for `np.isclose` when comparing the random vectors.

        Raises
        ------
        TypeError
            If `first` or `second` are not `RandomVector` instances.
        ValueError
            If `first` or `second` are from a different sample space than this measure's sample space, or if they have different dimensions.

        Returns
        -------
        equal_ae : bool
            `True` if the random vectors are equal almost everywhere; `False` otherwise.

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
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 1.2,
        ...         1: 2.3,
        ...         2: 0.0,
        ...     },
        ... )
        >>> X = RandomVariable(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 1.0,
        ...         1: 2.0,
        ...         2: 3.0,
        ...     },
        ... )
        >>> Y = RandomVariable(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 1.0,
        ...         1: 2.0,
        ...         2: 4.0,
        ...     },
        ...     name="Y",
        ... )
        >>> Z = RandomVariable(
        ...     sample_space=Omega,
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
        >>> U = RandomVector(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 2),
        ...     },
        ...     name="U",
        ... )
        >>> V = RandomVector(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (-1, 4),
        ...     },
        ...     name="V",
        ... )
        >>> W = RandomVector(
        ...     sample_space=Omega,
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
        Two random vectors $X,Y:\Omega \to \mathbb{R}^d$ defined on a measure space $(\Omega, \mathcal{F}, \mu)$ are *equal almost everywhere* if

        $$
        \mu \left( \{\omega \in \Omega : X(\omega) \neq Y(\omega)\} \right) = 0.
        $$
        """
        from ..functions.random_variable import RandomVector

        if not isinstance(first, RandomVector) or not isinstance(second, RandomVector):
            raise TypeError("first and second must be RandomVector instances.")
        if first.dimension != second.dimension:
            raise ValueError("The random vectors must have the same dimension.")
        if (
            first.sample_space != self.sig_alg.sample_space
            or second.sample_space != self.sig_alg.sample_space
        ):
            raise ValueError("Random vectors must be from this measure's sample space.")

        first_df = (
            pd.concat([self.sig_alg.data, first.data], axis=1)
            .drop_duplicates()
            .set_index("atom_ID")
        )
        second_df = (
            pd.concat([self.sig_alg.data, second.data], axis=1)
            .drop_duplicates()
            .set_index("atom_ID")
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

        >>> from sigalg.core import Measure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=5)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...         4: 2,
        ...     },
        ... )
        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
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
        ...     sig_alg=F,
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
            prob_measure = type(self)(
                sig_alg=self.sig_alg, mapping=self.data, name=self.name
            )
            if self.sig_alg != sig_alg:
                prob_measure.sig_alg = sig_alg
            return prob_measure

    def __or__(self, sig_alg: SigmaAlgebra) -> Measure:
        """Restrict the probability measure to a sub-sigma-algebra.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The sub-sigma-algebra to which to restrict the probability measure.

        Returns
        -------
        prob_measure : ProbabilityMeasure
            A new probability measure restricted to the new sigma-algebra.
        """
        return self.restrict_to(sig_alg=sig_alg)

    def __rshift__(self, rv: RandomVector) -> Measure:
        """Pass."""
        from ..functions.operators import Operators

        return Operators.pushforward(rv=rv, measure=self)

    # --------------------- data access methods --------------------- #

    def __call__(self, *args, **kwargs):
        """Get the probability of an event.

        One may pass arguments in one of the following ways:

        * An `Event` instance as a (single) positional argument or a keyword argument named `event`.
        * A list of sample points as a (single) positional argument or a keyword argument named `event`. The list of sample points must correspond to a measurable event in the sigma-algebra of the probability measure.
        * A single sample point as a keyword argument named `sample_point`. The sample point must correspond to a measurable (singleton) event in the sigma-algebra of the probability measure.
        * An atom ID of the sigma-algebra as a keyword argument.

        This method calls the parent `__call__` method of the parent class `MultivariateFunction` and hence allows curried calls. See the docstring of the parent class for details.

        Parameters
        ----------
        *args : tuple
            Positional arguments.
        **kwargs : dict
            Keyword arguments.

        Raises
        ------
        ValueError
            If the event is not measurable with respect to the sigma-algebra of the probability measure.

        Returns
        -------
        probability : Real
            The probability of the event.

        Examples
        --------
        >>> from sigalg.core import (
        ...     ProbabilityMeasure,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
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
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         (1, 2): 0.2,
        ...         (0, 2): 0.2,
        ...         (2, 4): 0.6,
        ...     },
        ... )
        >>> # Call on `Event` instances as positional or keyword arguments
        >>> A = F.get_event([0, 1, 2])
        >>> print(P(A))
        0.4
        >>> print(P(event=A))
        0.4
        >>> # Call on a list as a positional or keyword argument
        >>> print(P([0, 1, 2]))
        0.4
        >>> print(P(event=[0, 1, 2]))
        0.4
        >>> # Call on a sample point as a keyword argument
        >>> print(P(sample_point=2))
        0.2
        >>> print(P(F_0=0, F_1=2))
        0.2
        >>> # Evaluate the probability of an event using curried calls
        >>> print(P(F_0=0)(F_1=2))
        0.2
        >>> print(P(F_1=2)(F_0=0))
        0.2
        """
        from ..spaces.event import Event

        event = None
        if len(args) == 1 and len(kwargs) == 0:
            if isinstance(args[0], Event):
                event = args[0]
            if isinstance(args[0], list):
                event = self.sig_alg.get_event(args[0])
            if isinstance(args[0], Hashable):
                event = self.sig_alg.get_event([args[0]])
        elif "event" in kwargs and len(kwargs) == 1 and len(args) == 0:
            if isinstance(kwargs["event"], Event):
                event = kwargs["event"]
            if isinstance(kwargs["event"], list):
                event = self.sig_alg.get_event(kwargs["event"])
        elif "sample_point" in kwargs and len(kwargs) == 1 and len(args) == 0:
            event = self.sig_alg.get_event([kwargs["sample_point"]])

        if event is not None and isinstance(event, Event):
            if not event.sig_alg <= self.sig_alg:
                raise ValueError(
                    "Event is not in the domain of the probability measure."
                )
            df = pd.concat([event.indicator.data, self.sig_alg.data], axis=1)
            if isinstance(self.sig_alg.data, pd.Series):
                index_name = self.sig_alg.data.name
            else:
                index_name = self.sig_alg.data.columns.to_list()
            atom_indicator = df.drop_duplicates().set_index(index_name).squeeze()
            return self.data[atom_indicator.astype(bool)].sum()

        try:
            return super().__call__(*args, **kwargs)
        except (TypeError, ValueError) as e:
            raise ValueError(
                "Error while evaluating a probability measure. Perhaps the callable function was not constructed properly due to an invalid parameter name."
            ) from e

    # --------------------- equality --------------------- #

    def __eq__(self, other: Measure) -> bool:
        """Check equality with another measure.

        Two measures are considered equal if they have the same sigma-algebras and identical values for each atom. They may have different names and still be considered equal.

        Parameters
        ----------
        other : Measure
            The other measure to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the two measures are equal, `False` otherwise.
        """
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

        self_atom_mapping = {
            atom_id: frozenset(sample_ids)
            for atom_id, sample_ids in self.sig_alg.atom_id_to_sample_ids.items()
        }
        other_atom_mapping = {
            atom_id: frozenset(sample_ids)
            for atom_id, sample_ids in other.sig_alg.atom_id_to_sample_ids.items()
        }

        s1 = self.data.rename(index=self_atom_mapping).sort_index()
        s2 = other.data.rename(index=other_atom_mapping).sort_index()
        return s1.index.equals(s2.index) and (s1 - s2).abs().lt(1e-8).all()
