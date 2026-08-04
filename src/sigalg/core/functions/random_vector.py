"""A class representing a random vector."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import pandas as pd

from .measurable_vector import MeasurableVector

if TYPE_CHECKING:
    from ...typing.index_like import IndexLike
    from ...typing.mapping_like import MappingLike
    from ..measures.probability_measure import ProbabilityMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra


class RandomVector(MeasurableVector):
    r"""A class representing a random vector.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    domain : IndexLike | None, default=None
        The sample space of the underlying probability space.
    sig_alg : SigmaAlgebra | None, default=None
        The sigma algebra of the underlying probability space.
    measure : ProbabilityMeasure | None, default=None
        The probability measure of the underlying probability space. This is a required argument. The default `None` is only provided to maintain consistency with the
        parent class `MeasurableVector`, which does not require a probability measure.
    mapping : MappingLike | None, default=None
        The mapping defining the random vector.
    index : IndexLike | Index | None, default=None
        The index of the random vector.
    name : Hashable, default="X"
        The name of the random vector.

    Examples
    --------
    >>> from sigalg.core import (
    ...     MeasurableSpace,
    ...     ProbabilityMeasure,
    ...     ProbabilitySpace,
    ...     RandomVector,
    ...     SampleSpace,
    ...     SigmaAlgebra,
    ... )

    Generate a 2-dimensional random vector on a pre-existing sample space from a dictionary mapping using the `with_uniform` class method. The power-set sigma-algebra and uniform probability measure are automatically generated.

    >>> Omega = SampleSpace.from_sequence(size=3)
    >>> X = RandomVector.with_uniform(
    ...     domain=Omega,
    ...     mapping={
    ...         0: (1, 1),
    ...         1: (1, 1),
    ...         2: (2, 2),
    ...     },
    ... )
    >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
    Random vector 'X':
    index   0  1
    sample
    0       1  1
    1       1  1
    2       2  2
    >>> print(X.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'power_set':
            atom_ID
    sample
    0             0
    1             1
    2             2
    >>> print(X.measure)  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'U':
            probability
    sample
    0          0.333333
    1          0.333333
    2          0.333333

    Generate a random vector on a pre-existing probability space.

    >>> F = SigmaAlgebra(
    ...     domain=Omega,
    ...     mapping={
    ...         0: 0,
    ...         1: 0,
    ...         2: 1,
    ...     }
    ... )
    >>> P = ProbabilityMeasure(
    ...     domain=F,
    ...     mapping={
    ...         0: 0.5,
    ...         1: 0.5,
    ...     },
    ... )
    >>> prob_space = ProbabilitySpace(Omega, F, P)
    >>> Z = RandomVector(
    ...     *prob_space,
    ...     mapping={
    ...         0: (1, 1),
    ...         1: (1, 1),
    ...         2: (2, 2),
    ...     },
    ...     name="Z",
    ... )
    >>> print(Z.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'F':
            atom_ID
    sample
    0             0
    1             0
    2             1
    >>> print(Z.measure)  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'P':
             probability
    atom_ID
    0                0.5
    1                0.5

    Attempt to define a random vector that is not F-measurable

    >>> try:
    ...     W = RandomVector(
    ...         *prob_space,
    ...         mapping={
    ...             0: (1, 2),
    ...             1: (3, 4),
    ...             2: (5, 6),
    ...         },
    ...         name="W",
    ...     )
    ... except ValueError as e:
    ...     print(e)
    Function W is not measurable.

    Generate a 2-dimensional random vector from a function on a sample space and a custom index.

    >>> S = SampleSpace([(0, 1), (1, 2)], variable_names=["x", "y"], name="S")
    >>> def mapping(*, x, y):  # noqa: D103
    ...     return (x + y, x)
    >>> V = RandomVector.with_uniform(
    ...     domain=S,
    ...     mapping=mapping,
    ...     index=[1, 2],
    ...     name="V",
    ... )
    >>> print(V)  # doctest: +NORMALIZE_WHITESPACE
    Random vector 'V':
    index  1  2
    x y
    0 1    1  0
    1 2    3  1

    Notes
    -----
    Given a probability space $(\Omega,\mathcal{F},P)$, a *random vector* is an $\mathcal{F}$-measurable function $X: \Omega \to \mathbb{R}^d$, where $d$ is the *dimension* of the vector and $\mathbb{R}^d$ is equipped with its Borel $\sigma$-algebra. If $\Omega$ is finite (as it always is, in SigAlg), so that $\mathcal{F}$ is determined by its atoms, then $X$ is $\mathcal{F}$-measurable if and only if $X$ is constant on the atoms of $\mathcal{F}$.
    """

    _repr_name = "RandomVector"
    _str_name = "Random vector"
    _default_name = "X"

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        domain: IndexLike | None = None,
        sig_alg: SigmaAlgebra | None = None,
        measure: ProbabilityMeasure | None = None,
        mapping: MappingLike | None = None,
        index: IndexLike | None = None,
        name: Hashable = "X",
    ) -> None:
        from ..measures.probability_measure import ProbabilityMeasure
        from ..spaces.sample_space import SampleSpace
        from .random_variable import RandomVariable

        if (domain is not None or sig_alg is not None) and not isinstance(
            measure, ProbabilityMeasure
        ):
            raise ValueError("measure must be a probability measure.")

        if domain is not None and not isinstance(domain, SampleSpace):
            domain = SampleSpace(domain)

        super().__init__(
            domain=domain,
            sig_alg=sig_alg,
            measure=measure,
            mapping=mapping,
            index=index,
            name=name,
        )

        if self.dimension == 1 and not isinstance(self, RandomVariable):
            self._data = (
                self._data.squeeze(axis=1)
                if isinstance(self._data, pd.DataFrame)
                else self._data
            )
            self._data.name = self._name
            self._index = None
            self.__class__ = RandomVariable

    @classmethod
    def with_uniform(
        cls,
        domain: IndexLike | None = None,
        sig_alg: SigmaAlgebra | None = None,
        mapping: MappingLike | None = None,
        index: IndexLike | None = None,
        name: Hashable = "X",
    ) -> RandomVector:
        """Construct a random vector on a given measurable space with a uniform probability measure.

        Parameters
        ----------
        domain : IndexLike | None, default=None
            The sample space of the underlying probability space.
        sig_alg : SigmaAlgebra | None, default=None
            The sigma algebra of the underlying probability space.
        mapping : MappingLike | None, default=None
            The mapping defining the random vector.
        index : IndexLike | None, default=None
            The index of the random vector.
        name : Hashable, default="X"
            The name of the random vector.

        Returns
        -------
        random_vector : RandomVector
            A random vector on the given measurable space with a uniform probability measure.
        """
        from ..measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if sig_alg is None and domain is not None:
            sig_alg = SigmaAlgebra.power_set(domain)
        if domain is None and sig_alg is not None:
            domain = sig_alg.domain

        measure = ProbabilityMeasure.uniform(sig_alg)

        return cls(
            domain=domain,
            sig_alg=sig_alg,
            measure=measure,
            mapping=mapping,
            index=index,
            name=name,
        )
