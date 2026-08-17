"""A class representing a random vector."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .measurable_vector import MeasurableVector

if TYPE_CHECKING:
    from collections.abc import Hashable

    from ...typing.index_like import IndexLike
    from ...typing.mapping_like import MappingLike
    from ..measures.measure import Measure
    from ..measures.probability_measure import ProbabilityMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.domain import Domain
    from ..spaces.measure_space import MeasureSpace


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
    i  0  1
    s
    0  1  1
    1  1  1
    2  2  2
    >>> print(X.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'R':
       R
    s
    0  0
    1  1
    2  2
    >>> print(X.measure)  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'U':
               U
    s
    0   0.333333
    1   0.333333
    2   0.333333

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
       F
    s
    0  0
    1  0
    2  1
    >>> print(Z.measure)  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'P':
             P
    F
    0      0.5
    1      0.5

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
        domain_kind: Literal["Domain", "SampleSpace"] = "SampleSpace",
        domain_name: Hashable | None = None,
        output_name: Hashable | None = None,
        index: IndexLike | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
    ) -> None:
        from ..measures.probability_measure import ProbabilityMeasure

        if not isinstance(measure, ProbabilityMeasure):
            raise TypeError(
                "Please pass an instance of ProbabilityMeasure into the constructor for RandomVector/RandomVariable."
            )

        super().__init__(
            domain=domain,
            sig_alg=sig_alg,
            measure=measure,
            mapping=mapping,
            domain_kind=domain_kind,
            output_name=output_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
        )

    @classmethod
    def with_uniform(
        cls,
        domain: IndexLike | None = None,
        sig_alg: SigmaAlgebra | None = None,
        mapping: MappingLike | None = None,
        domain_kind: Literal["Domain", "SampleSpace"] = "SampleSpace",
        domain_name: Hashable | None = None,
        output_name: Hashable | None = None,
        index: IndexLike | None = None,
        index_kind: Literal["Index", "Time"] = "Index",
        index_name: Hashable | None = None,
        name: Hashable | None = None,
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
        from .random_variable import RandomVariable

        if name is None:
            name = RandomVector._default_name

        result = MeasurableVector(
            domain=domain,
            sig_alg=sig_alg,
            mapping=mapping,
            domain_kind=domain_kind,
            domain_name=domain_name,
            output_name=output_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
        )

        result.measure = ProbabilityMeasure.uniform(result.sig_alg)

        if result.dimension == 1:
            result.__class__ = RandomVariable
        else:
            result.__class__ = RandomVector

        return result

    # --------------------- properties --------------------- #

    @property
    def sample_space(self) -> Domain:
        """Pass."""
        return self.domain

    @property
    def prob_measure(self) -> Measure:
        """Pass."""
        return self.measure

    @property
    def prob_space(self) -> MeasureSpace:
        """Pass."""
        return self.measure_space
