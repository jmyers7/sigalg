"""A class representing a random vector."""

from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING, Literal

from .measurable_vector import MeasurableVector

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from ...typing.index_like import IndexLike
    from ...typing.mapping_like import MappingLike
    from ..indices.index import Index
    from ..measures.measure import Measure
    from ..measures.probability_measure import ProbabilityMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.domain import Domain
    from ..spaces.measure_space import MeasureSpace
    from .function import Function


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
                "Instances of RandomVector may only be created with probability measures. If you want a default uniform measure, use the 'with_uniform' class method."
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

    # --------------------- conversion methods --------------------- #

    def to_measurable_vec(self) -> MeasurableVector:
        """Promote to a `MeasurableVector` instance."""
        return MeasurableVector._from_validated(
            data=self.data,
            sig_alg=self.sig_alg,
            measure=None,
            index_kind=type(self.index).__name__ if self.index is not None else "Index",
            index_name=self.index.name if self.index is not None else None,
            name=self.name,
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
    ) -> Function:
        """Apply a binary operation to this random vector.

        Parameters
        ----------
        self : MeasurableVector
            The left operand (or right if reverse=True).
        other : MeasurableVector | Real
            The right operand (or left if reverse=True).
        operation : Callable
            The pandas operation to apply (e.g., `lambda a, b: a + b`).
        op_symbol : str
            Symbol representing the operation (e.g., '+', '-', '*').
        reverse : bool, default=False
            Whether this is a reverse operation (e.g., __radd__ vs __add__).

        Returns
        -------
        result : MeasurableVector
            A new measurable vector representing the result of the operation.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import (
        ...     Domain,
        ...     MeasurableFunction,
        ...     MeasurableVector,
        ...     Measure,
        ...     SigmaAlgebra,
        ... )
        >>> rng = np.random.default_rng(42)

        Define two functions on a measurable space with 2-dimensional outputs and print their sum.

        >>> X = Domain([(1, 2), (3, 4), (5, 6), (7, 8)], variable_names=["u", "v"])
        >>> F = SigmaAlgebra(domain=X, mapping=dict(zip(X, [0, 1, 1, 2])))
        >>> f = MeasurableVector.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     dim=2,
        ...     random_state=rng,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'f':
        i    0  1
        u v
        1 2  0  7
        3 4  6  4
        5 6  6  4
        7 8  4  8
        >>> g = MeasurableVector.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     dim=2,
        ...     name="g",
        ...     random_state=rng,
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector 'g':
        i    0  1
        u v
        1 2  0  6
        3 4  2  0
        5 6  2  0
        7 8  5  9
        >>> print(f + g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable vector '(f + g)':
        i    0   1
        u v
        1 2  0  13
        3 4  8   4
        5 6  8   4
        7 8  9  17

        Since both functions have the same sigma-algebra, the sigma-algebra passes through to the sum.

        >>> (f + g).measurable_space
        MeasurableSpace(domain=X, sig_alg=F)

        The same is true for differences of functions with 1-dimensional outputs, for example.

        >>> f = MeasurableFunction.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     random_state=rng,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'f':
                f
        u v
        1 2  7
        3 4  7
        5 6  7
        7 8  7
        >>> g = MeasurableFunction.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     name="g",
        ...     random_state=rng,
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'g':
                g
        u v
        1 2  7
        3 4  5
        5 6  5
        7 8  1
        >>> print(f - g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function '(f - g)':
                (f - g)
        u v
        1 2        0
        3 4        2
        5 6        2
        7 8        6
        >>> (f - g).measurable_space
        MeasurableSpace(domain=X, sig_alg=F)

        Arithmetic operations between two measurable functions does not strictly require that they are both defined on the same sigma-algebra. If one sigma-algebra is a sub-sigma-algebra of another, then the result of an arithmetic operation will be defined on the larger sigma-algebra.

        >>> G = SigmaAlgebra(domain=X, mapping=dict(zip(X, [0, 1, 1, 1])), name="G")
        >>> G <= F
        True
        >>> f = MeasurableFunction.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     random_state=rng,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'f':
                f
        u v
        1 2  8
        3 4  4
        5 6  4
        7 8  5
        >>> g = MeasurableFunction.from_rand(
        ...     domain=X,
        ...     sig_alg=G,
        ...     name="g",
        ...     random_state=rng,
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'g':
                g
        u v
        1 2  3
        3 4  1
        5 6  1
        7 8  1
        >>> print(f * g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function '(f * g)':
                (f * g)
        u v
        1 2       24
        3 4        4
        5 6        4
        7 8        5
        >>> (f * g).measurable_space
        MeasurableSpace(domain=X, sig_alg=F)

        If two measurable functions carry the same measure, this measure will pass through to the result of an arithmetic operation between them.

        >>> mu = Measure(domain=F, mapping=dict(zip(F.atom_space, [4, 2, 7])))
        >>> f = MeasurableFunction.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     measure=mu,
        ...     random_state=rng,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'f':
                f
        u v
        1 2  9
        3 4  7
        5 6  7
        7 8  6
        >>> g = MeasurableFunction.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     measure=mu,
        ...     name="g",
        ...     random_state=rng,
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'g':
                g
        u v
        1 2  4
        3 4  8
        5 6  8
        7 8  5
        >>> print(f / g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function '(f / g)':
                (f / g)
        u v
        1 2    2.250
        3 4    0.875
        5 6    0.875
        7 8    1.200
        >>> (f / g).measure_space
        MeasureSpace(domain=X, sig_alg=F, measure=mu)

        Again, the arithmetic operations do not strictly require that measurable functions carry the same measure, as long as one is defined on a sub-sigma-algebra of another and is the restriction of the measure on the larger sigma-algebra. Then the result of an arithmetic operation will carry the larger sigma-algebra and its measure.

        >>> nu = Measure(domain=G, mapping=dict(zip(G.atom_space, [4, 9])), name="nu")
        >>> nu <= mu  # This checks if nu is the restrictio of mu to G
        True

        >>> f = MeasurableFunction.from_rand(
        ...     domain=X,
        ...     sig_alg=F,
        ...     measure=mu,
        ...     random_state=rng,
        ... )
        >>> print(f)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'f':
                f
        u v
        1 2  4
        3 4  4
        5 6  4
        7 8  2
        >>> g = MeasurableFunction.from_rand(
        ...     domain=X,
        ...     sig_alg=G,
        ...     measure=nu,
        ...     name="g",
        ...     random_state=rng,
        ... )
        >>> print(g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function 'g':
                g
        u v
        1 2  0
        3 4  5
        5 6  5
        7 8  5
        >>> print(f**g)  # doctest: +NORMALIZE_WHITESPACE
        Measurable function '(f ** g)':
                (f ** g)
        u v
        1 2       1.0
        3 4    1024.0
        5 6    1024.0
        7 8      32.0
        >>> (f**g).measure_space
        MeasureSpace(domain=X, sig_alg=F, measure=mu)
        """
        from ..measures.probability_measure import ProbabilityMeasure
        from .function import Function

        if isinstance(other, Real):
            super_sig_alg = self.sig_alg
            measure = self.measure
            self_promoted = self

        elif isinstance(other, RandomVector):
            if self.sig_alg <= other.sig_alg:
                super_sig_alg = other.sig_alg
            elif self.sig_alg > other.sig_alg:
                super_sig_alg = self.sig_alg
            else:
                raise ValueError(
                    f"Cannot {op_symbol} measurable functions on incompatible measurable spaces."
                )

            measure = self._get_max_measure([self, other])
            self_promoted = self

        elif isinstance(other, MeasurableVector):
            if self.sig_alg <= other.sig_alg:
                super_sig_alg = other.sig_alg
            elif self.sig_alg > other.sig_alg:
                super_sig_alg = self.sig_alg
            else:
                raise ValueError(
                    f"Cannot {op_symbol} measurable functions on incompatible measurable spaces."
                )

            measure = self._get_max_measure([self, other])

            if isinstance(measure, ProbabilityMeasure):
                self_promoted = self
            else:
                self_promoted = self.to_measurable_vec()
                measure = None

        elif isinstance(other, Function):
            if self.sig_alg in other.lattice:
                super_sig_alg = self.sig_alg
                measure = self.measure
                self_promoted = self

            else:
                self_promoted = self.to_function()
                super_sig_alg = None
                measure = None

        else:
            raise NotImplementedError(
                f"Arithmetic not implemented between RandomVector and {type(other).__name__}."
            )

        return Function._apply_binary_operation(
            self=self_promoted,
            other=other,
            operation=operation,
            op_symbol=op_symbol,
            reverse=reverse,
            domain_name=domain_name,
            index=index,
            index_kind=index_kind,
            index_name=index_name,
            name=name,
            sig_alg=super_sig_alg,
            measure=measure,
        )
