"""A class representing a random vector."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .measurable_vector import MeasurableVector

if TYPE_CHECKING:
    from collections.abc import Hashable

    import numpy as np

    from ...typing.index_like import IndexLike
    from ...typing.mapping_like import MappingLike
    from ..measures.measure import Measure
    from ..measures.probability_measure import ProbabilityMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.domain import Domain
    from ..spaces.measure_space import MeasureSpace
    from ..spaces.set import Set


class RandomVector(MeasurableVector):
    r"""A class representing a random vector.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    domain : IndexLike | None, default=None
        The domain of the function.
    sig_alg : SigmaAlgebra | None, default=None
        The sigma-algebra of the underlying probability space.
    measure : ProbabilityMeasure | None, default=None
        The probability measure of the underlying probability space. This is a required argument. The default `None` is only provided to maintain consistency with the parent class `MeasurableVector`, which does not require a probability measure.
    mapping : MappingLike | None, default=None
        The mapping defining the random vector.
    domain_kind : Literal["Domain", "SampleSpace"], default="Domain"
        The type of the domain.
    domain_name : Hashable | None, default=None
        The name of the domain.
    output_name : Hashable | None, default=None
        The name of the outputs of the function. If `None`, a default will be generated.
    index : IndexLike | None, default=None
        The index for the outputs of the function. Only used if the outputs are multi-dimensional.
    index_kind : Literal["Index", "Time"], default="Index"
        The kind of index. Only used if the outputs are multi-dimensional.
    index_name : Hashable | None, default=None
        The name of the index. Only used if the outputs are multi-dimensional.
    name : Hashable | None, default=None
        The name of the function. If `None`, a default name will be generated.

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
    i      0  1
    omega
    0      1  1
    1      1  1
    2      2  2
    >>> print(X.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'R':
          R
    omega
    0     0
    1     1
    2     2
    >>> print(X.measure)  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'U':
                  U
    omega
    0      0.333333
    1      0.333333
    2      0.333333

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
    omega
    0     0
    1     0
    2     1
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
            domain_name=domain_name,
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
            The domain of the function.
        sig_alg : SigmaAlgebra | None, default=None
            The sigma-algebra of the underlying probability space.
        mapping : MappingLike | None, default=None
            The mapping defining the random vector.
        domain_kind : Literal["Domain", "SampleSpace"], default="Domain"
            The type of the domain.
        domain_name : Hashable | None, default=None
            The name of the domain.
        output_name : Hashable | None, default=None
            The name of the outputs of the function. If `None`, a default will be generated.
        index : IndexLike | None, default=None
            The index for the outputs of the function. Only used if the outputs are multi-dimensional.
        index_kind : Literal["Index", "Time"], default="Index"
            The kind of index. Only used if the outputs are multi-dimensional.
        index_name : Hashable | None, default=None
            The name of the index. Only used if the outputs are multi-dimensional.
        name : Hashable | None, default=None
            The name of the function. If `None`, a default name will be generated.

        Returns
        -------
        random_vector : RandomVector
            A random vector on the given measurable space with a uniform probability measure.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace, SigmaAlgebra

        Define a `2`-dimensional random vector on a measure space with the uniform probability measure.

        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     },
        ... )
        >>> X = RandomVector.with_uniform(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: (0, 1),
        ...         1: (2, 1),
        ...         2: (2, 1),
        ...     },
        ... )
        >>> print(X.measure_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, F, U)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
         omega
             0
             1
             2
        <BLANKLINE>
        * Sigma algebra 'F':
               F
        omega
        0      0
        1      1
        2      1
        <BLANKLINE>
        * Probability measure 'U':
             U
        F
        0  0.5
        1  0.5
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
    def sample_space(self) -> Domain | None:
        """Get the domain of the underlying probability space of the random vector.

        This is an alias for the `domain` property of the parent class `Function`.
        """
        return self.domain

    @property
    def prob_measure(self) -> Measure | None:
        """Get the probability measure of the underlying probability space of the random vector.

        This is an alias for the `measure` property of the parent class `MeasurableVector`.
        """
        return self.measure

    @property
    def prob_space(self) -> MeasureSpace | None:
        """Get the underlying probability space of the random vector.

        This is an alias for the `measure_space` property of the parent class `MeasurableVector`.
        """
        return self.measure_space

    # --------------------- function methods --------------------- #

    def restrict_to(
        self,
        subset: Set | list[Hashable],
        subset_name: Hashable | None = "A",
    ) -> RandomVector:
        """Restrict the measurable vector to a measurable subset.

        Parameters
        ----------
        subset : Set | list[Hashable]
            The set to restrict the measurable vector to.
        subset_name : Hashable, default="A"
            The name to use for the subset. Ignored if `subset` is an instance of `Set`.

        Returns
        -------
        restriction : RandomVector
            The restriction of the function.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace, Set, SigmaAlgebra

        Define a probability space and 2-dimensional random vector.

        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     domain=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     domain=F,
        ...     mapping={
        ...         0: 0.2,
        ...         1: 0.5,
        ...         2: 0.3,
        ...     },
        ... )
        >>> X = RandomVector(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...         3: (5, 6),
        ...     },
        ... )

        Restrict the random vector to a set using the `restrict_to` method.

        >>> A = Set([1, 2, 3], domain=Omega)
        >>> X_A = X.restrict_to(A)
        >>> print(X_A)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X|A':
        i      0  1
        omega
        1      3  4
        2      3  4
        3      5  6

        Print its measure space consisting of the restricted sigma-algebra and (normalized) restricted measure.

        >>> print(X_A.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (A, F|A, P|A)
        ===============================
        <BLANKLINE>
        * Sample space 'A':
         omega
             1
             2
             3
        <BLANKLINE>
        * Sigma algebra 'F|A':
               F|A
        omega
        1        1
        2        1
        3        2
        <BLANKLINE>
        * Probability measure 'P|A':
             P|A
        F
        1  0.625
        2  0.375

        Compute the same restriction using the overloaded `|` operator.

        >>> print(X | A)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X|A':
        i      0  1
        omega
        1      3  4
        2      3  4
        3      5  6
        """
        return super().restrict_to(
            subset=subset,
            subset_name=subset_name,
            normalize=True,
        )

    # --------------------- probability methods --------------------- #

    def sample(
        self,
        size: int = 1,
        name: Hashable | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> MeasureSpace:
        """Generate random samples from the range space of this random vector.

        Parameters
        ----------
        size : int, default=1
            Number of samples to generate. Must be positive.
        name : Hashable | None, default=None
            A name for the random sample. If `None`, a default will be generatd.
        random_state : int | np.random.Generator | None, default=None
            Random seed or generator for reproducibility.

        Returns
        -------
        sample : MeasureSpace
            An instance of `MeasureSpace` whose domain consists of the random samples and whose measure is a counting measure giving the number of each sample produced.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import ProbabilityMeasure, RandomVariable, RandomVector, SampleSpace, SigmaAlgebra
        >>> rng = np.random.default_rng(42)

        Generate a random probability space and 2-dimensional random vector.

        >>> Omega = SampleSpace.from_sequence(size=10)
        >>> F = SigmaAlgebra.from_rand(
        ...     domain=Omega,
        ...     num_atoms=4,
        ...     random_state=rng,
        ... )
        >>> P = ProbabilityMeasure.from_rand(domain=F, random_state=rng)
        >>> X = RandomVector.from_rand(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     max_value=10,
        ...     dim=2,
        ...     random_state=rng,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        i      0  1
        omega
        0      2  6
        1      2  6
        2      1  7
        3      2  6
        4      7  3
        5      2  6
        6      0  9
        7      2  6
        8      0  9
        9      2  6

        Print the underlying measure space of the random vector.

        >>> print(X.measure_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, F, P)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
            omega
                0
                1
                2
                3
                4
                5
                6
                7
                8
                9
        <BLANKLINE>
        * Sigma algebra 'F':
                F
        omega
        0     1
        1     1
        2     3
        3     1
        4     2
        5     1
        6     0
        7     1
        8     0
        9     1
        <BLANKLINE>
        * Probability measure 'P':
                    P
        F
        1  0.049134
        3  0.207580
        2  0.082504
        0  0.660782

        Sample from the range of the random vector.

        >>> X_sample = X.sample(size=1_000, random_state=rng)
        >>> print(X_sample.measure)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'C':
                    C
        X_0 X_1
        0   9    663
        1   7    210
        7   3     81
        2   6     46

        Sample from a 1-dimensional random variable.

        >>> Y = RandomVariable.from_rand(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
        ...     max_value=10,
        ...     random_state=rng,
        ...     name="Y",
        ... )
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'Y':
                Y
        omega
        0      4
        1      4
        2      5
        3      4
        4      7
        5      4
        6      6
        7      4
        8      6
        9      4
        >>> Y_sample = Y.sample(size=1_000, random_state=rng)
        >>> print(Y_sample.measure)  # doctest: +NORMALIZE_WHITESPACE
        Measure 'C':
                C
        Y
        6  650
        5  219
        7   92
        4   39
        """
        from ..measures.probability_measure import ProbabilityMeasure
        from .operators import Operators

        if not isinstance(self.measure, ProbabilityMeasure):
            raise TypeError("Cannot sample from a non-random-vector.")

        if self.data is not None:
            return Operators.pushforward(vec=self, measure=self.measure).sample(
                size=size, random_state=random_state
            )
        else:
            raise ValueError("Cannot sample from an empty measurable vector instance.")

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
