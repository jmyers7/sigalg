"""A class representing a random vector."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from itertools import combinations
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .operators import OperatorsMethods

if TYPE_CHECKING:
    from ...processes.base.stochastic_process import StochasticProcess
    from ...validation.index_validator import IndexLike
    from ...validation.mapping_validator import MappingLike
    from ..functions.random_variable import RandomVariable
    from ..indices.index import Index
    from ..measures.probability_measure import ProbabilityMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.measurable_set import MeasurableSet
    from ..spaces.probability_space import ProbabilitySpace
    from ..spaces.sample_space import SampleSpace


class RandomVector(OperatorsMethods):
    r"""A class representing a random vector.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    sample_space : SampleSpace | IndexLike | None, default=None
        The sample space of the underlying probability space.
    sig_alg : SigmaAlgebra | None, default=None
        The sigma algebra of the underlying probability space.
    prob_measure : ProbabilityMeasure | None, default=None
        The probability measure of the underlying probability space.
    mapping : MappingLike | Callable | None, default=None
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

    Generate a 2-dimensional random vector on a pre-existing sample space from a dictionary mapping. The power-set sigma-algebra and uniform probability measure are automatically generated

    >>> Omega = SampleSpace.from_sequence(size=3)
    >>> X = RandomVector(
    ...     sample_space=Omega,
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
    >>> print(X.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'U':
            probability
    sample
    0          0.333333
    1          0.333333
    2          0.333333

    Generate a random vector on a pre-existing event space. A uniform probability measure is automatically generated

    >>> F = SigmaAlgebra(
    ...     sample_space=Omega,
    ...     mapping={
    ...         0: 0,
    ...         1: 0,
    ...         2: 1,
    ...     },
    ... )
    >>> measurable_space = MeasurableSpace(Omega, F)
    >>> Y = RandomVector(
    ...     *measurable_space,
    ...     mapping={
    ...         0: (1, 1),
    ...         1: (1, 1),
    ...         2: (2, 2),
    ...     },
    ...     name="Y",
    ... )
    >>> print(Y.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'F':
            atom_ID
    sample
    0             0
    1             0
    2             1
    >>> print(Y.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'U':
             probability
    atom_ID
    0                0.5
    1                0.5


    Generate a random vector on a pre-existing probability space

    >>> P = ProbabilityMeasure(
    ...     sig_alg=F,
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
    >>> print(Z.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'P':
             probability
    atom_ID
    0                0.5
    1                0.5

    Attempt to define a random vector that is not F-measurable

    >>> W = RandomVector(
    ...     *prob_space,
    ...     mapping={
    ...         0: (1, 2),  # <- Not constant on atom A_0 = {0, 1}
    ...         1: (3, 4),  # <- Not constant on atom A_0 = {0, 1}
    ...         2: (5, 6),
    ...     },
    ...     name="W",
    ... )  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: Random vector W is not measurable.

    Generate a 2-dimensional random vector from a function on a sample space and a custom index.

    >>> S = SampleSpace([(0, 1), (1, 2)], variable_names=["x", "y"], name="S")
    >>> def mapping(*, x, y):  # noqa: D103
    ...     return (x + y, x)
    >>> V = RandomVector(
    ...     sample_space=S,
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
    Given a probability space $(\Omega,\mathcal{F},P)$, a *random vector* is an $\mathcal{F}$-measurable function $X: \Omega \to \mathbb{R}^d$, where $d$ is the *dimension* of the vector and $\mathbb{R}^d$ is equipped with its Borel $\sigma$-algebra. The image $X(\omega)\in \mathbb{R}^d$ of a sample point $\omega \in \Omega$ is called a *feature vector*.

    If $\Omega$ is finite (as it always is, in SigAlg), so that $\mathcal{F}$ is determined by its atoms, then $X$ is $\mathcal{F}$-measurable if and only if $X$ is constant on the atoms of $\mathcal{F}$.
    """

    _properties = [
        "_atom_data",
        "_dimension",
        "_components",
        "_component_names",
        "_generated_sig_alg",
        "_range",
        "_is_identity",
    ]
    _repr_name = "Random vector"

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        sample_space: SampleSpace | IndexLike | None = None,
        sig_alg: SigmaAlgebra | None = None,
        prob_measure: ProbabilityMeasure | None = None,
        mapping: MappingLike | Callable | None = None,
        index: Index | IndexLike | None = None,
        name: Hashable = "X",
    ) -> None:
        from ...validation.mapping_validator import MappingValidator
        from ..indices.index import Index
        from ..spaces.probability_space import ProbabilitySpace
        from ..spaces.sample_space import SampleSpace

        if sample_space is not None and not isinstance(sample_space, SampleSpace):
            sample_space = SampleSpace(sample_space)
        if index is not None and not isinstance(index, Index):
            index = Index(indices=index)

        v = MappingValidator(
            mapping=mapping,
            domain=sample_space,
            output_name=name,
            index=index,
            multi_dim=True,
            name=name,
        )
        self._data = v.data
        self._index = v.index
        self._name = v.name
        sample_space = v.domain

        if self._data is not None:
            if not isinstance(sample_space, SampleSpace):
                sample_space = SampleSpace.from_domain(domain=sample_space)
            if sample_space.name == "D":
                sample_space.name = "Omega"
            if sample_space.dimension == 1 and sample_space.variable_names == ["point"]:
                sample_space.variable_names = ["sample"]
            if sample_space.dimension > 1 and sample_space.variable_names == [
                f"point_{i}" for i in range(sample_space.dimension)
            ]:
                sample_space.variable_names = [
                    f"sample_{i}" for i in range(sample_space.dimension)
                ]
            self._data.index = sample_space.data

        self._prob_space = ProbabilitySpace(
            sample_space=sample_space,
            sig_alg=sig_alg,
            prob_measure=prob_measure,
        )

        if self.sig_alg is not None and not self.sig_alg.is_power_set:
            combined_data = pd.concat(
                [self._data, self.sig_alg.data], axis=1
            ).drop_duplicates()
            if len(combined_data) != self.sig_alg.num_atoms:
                raise ValueError(f"Random vector {self._name} is not measurable.")

        self._initialize_property_caches()

    def _initialize_property_caches(self, exceptions: set | None = None) -> None:
        if exceptions is None:
            exceptions = set()
        for property in set(self._properties) - exceptions:
            setattr(self, property, None)

    @classmethod
    def from_constant(
        cls,
        sample_space: SampleSpace | IndexLike,
        sig_alg: SigmaAlgebra | None = None,
        prob_measure: ProbabilityMeasure | None = None,
        constant: Hashable | None = None,
        index: Index | IndexLike | None = None,
        name: Hashable = "X",
    ) -> RandomVector:
        """Create a `RandomVector` that maps every sample point in the domain to the same constant output vector.

        Parameters
        ----------
        sample_space: SampleSpace | IndexLike
            The sample space of the underlying probability space.
        sig_alg: SigmaAlgebra | None, default=None
            The sigma-algebra of the underlying probability space. If `None`, the power set sigma-algebra is used.
        prob_measure: ProbabilityMeasure | None, default=None
            The probability measure of the underlying probability space. If `None`, the uniform probability measure is used.
        constant : Hashable | None, default=None
            The constant output vector that every sample point in the domain maps to.
        index : IndexLike | Index | None, default=None
            The index of the random vector.
        name : Hashable, default="X"
            The name of the random vector.

        Raises
        ------
        TypeError
            If `constant` is not a `Hashable`.
        ValueError
            If `constant` is a tuple and its length does not match the length of `index`.

        Returns
        -------
        rv : RandomVector
            A random vector mapping every sample point in the domain to the same constant output vector.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> X = RandomVector.from_constant(sample_space=Omega, constant=(1, 2), index=[1, 2])
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        index    1  2
        sample
        0        1  2
        1        1  2
        2        1  2
        >>> Y = RandomVector.from_constant(sample_space=Omega, constant=2, name="Y")
        >>> print(Y) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
                Y
        sample
        0       2
        1       2
        2       2
        """
        from ..indices.index import Index
        from ..spaces.sample_space import SampleSpace

        if sample_space is not None and not isinstance(sample_space, SampleSpace):
            sample_space = SampleSpace(sample_space)
        if index is not None and not isinstance(index, Index):
            index = Index(indices=index)

        if not isinstance(constant, Hashable):
            raise TypeError("constant must be a Hashable.")
        if (
            index is not None
            and isinstance(constant, tuple)
            and len(constant) != len(index)
        ):
            raise ValueError(
                "Length of constant tuple must match the length of the index."
            )

        if constant is not None:
            if isinstance(constant, tuple):
                mapping = dict.fromkeys(sample_space.data, constant)
            elif index is not None:
                mapping = dict.fromkeys(sample_space.data, (constant,) * len(index))
            else:
                mapping = dict.fromkeys(sample_space.data, constant)
        else:
            mapping = None

        return cls(
            sample_space=sample_space,
            sig_alg=sig_alg,
            prob_measure=prob_measure,
            mapping=mapping,
            index=index,
            name=name,
        )

    @classmethod
    def from_identity(
        cls,
        sample_space: SampleSpace | IndexLike,
        sig_alg: SigmaAlgebra | None = None,
        prob_measure: ProbabilityMeasure | None = None,
        index: Index | IndexLike | None = None,
        name: Hashable = "X",
    ) -> RandomVector:
        """Create a random vector that maps every sample point in the domain to itself.

        For this construction method, the sigma-algebra must be the power set.

        Parameters
        ----------
        sample_space: SampleSpace | IndexLike
            The sample space of the underlying probability space.
        sig_alg: SigmaAlgebra | None, default=None
            The sigma-algebra of the underlying probability space. The sigma-algebra must be the power-set. This parameter is here only for consistency with other constructors.
        prob_measure: ProbabilityMeasure | None, default=None
            The probability measure of the underlying probability space. If `None`, the uniform probability measure will be created.
        index : Index | IndexLike | None, default=None
            The index of the random vector.
        name : Hashable, default="X"
            The name of the random vector.

        Raises
        ------
        ValueError
            If the sigma-algebra is not the power set (if given), or if the length of the index (if given) does not match the dimension of the sample space.

        Returns
        -------
        rv : RandomVector
            A random vector mapping every sample point in the domain to itself.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace.cartesian_power(
        ...     [0, 1], n=2, name="Omega", variable_names=["x", "y"]
        ... )
        >>> X = RandomVector.from_identity(sample_space=Omega)
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        index  0  1
        x y
        0 0    0  0
          1    0  1
        1 0    1  0
          1    1  1
        >>> print(X.range)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, power_set, U)
        =======================================
        * Sample space 'Omega':
         x  y
         0  0
         0  1
         1  0
         1  1
        <BLANKLINE>
        * Sigma algebra 'power_set':
            atom_ID
        x y
        0 0  (0, 0)
          1  (0, 1)
        1 0  (1, 0)
          1  (1, 1)
        <BLANKLINE>
        * Probability measure 'U':
             probability
        x y
        0 0         0.25
          1         0.25
        1 0         0.25
          1         0.25
        >>> S = SampleSpace(indices=["a", "b"], name="S")
        >>> Y = RandomVector.from_identity(sample_space=S, name="Y")
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
                Y
        sample
        a       a
        b       b
        >>> print(Y.range)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (S, power_set, U)
        ===================================
        <BLANKLINE>
        * Sample space 'S':
        sample
             a
             b
        <BLANKLINE>
        * Sigma algebra 'power_set':
               atom_ID
        sample
        a            a
        b            b
        <BLANKLINE>
        * Probability measure 'U':
                probability
        sample
        a               0.5
        b               0.5
        """
        from ..indices.index import Index
        from ..spaces.sample_space import SampleSpace

        if sig_alg is not None and not sig_alg.is_power_set:
            raise ValueError(
                "The sigma-algebra must be the power set for an identity random vector."
            )
        if index is not None and len(index) != sample_space.dimension:
            raise ValueError(
                "The length of the index must match the dimension of the sample space."
            )

        if sample_space is not None and not isinstance(sample_space, SampleSpace):
            sample_space = SampleSpace(sample_space)
        if index is not None and not isinstance(index, Index):
            index = Index(indices=index)

        mapping = sample_space.data.to_frame()
        if mapping.shape[1] == 1:
            mapping = mapping.squeeze(axis=1)
            mapping.name = name
        else:
            mapping = sample_space.data.to_frame()
            if index is None:
                index = Index.from_sequence(size=mapping.shape[1])
            mapping.columns = index.data

        rv = cls(
            sample_space=sample_space,
            sig_alg=sig_alg,
            prob_measure=prob_measure,
            mapping=mapping,
            name=name,
        )

        rv._range = rv.prob_space
        rv._is_identity = True

        return rv

    @classmethod
    def from_randint(
        cls,
        sample_space: SampleSpace | IndexLike,
        sig_alg: SigmaAlgebra | None = None,
        prob_measure: ProbabilityMeasure | None = None,
        low: int = 0,
        high: int = 2,
        dim: int | None = None,
        index: Index | IndexLike | None = None,
        random_state: int | np.random.Generator | None = None,
        name: Hashable = "X",
    ) -> RandomVariable | RandomVector:
        """Generate a random vector with integer outputs uniformly sampled from the range [low, high).

        Parameters
        ----------
        sample_space: SampleSpace | IndexLike
            The sample space of the underlying probability space.
        sig_alg: SigmaAlgebra | None, default=None
            The sigma-algebra of the underlying probability space. If `None`, the power set sigma-algebra is used.
        prob_measure: ProbabilityMeasure | None, default=None
            The probability measure of the underlying probability space. If `None`, the uniform probability measure is used.
        low : int, default=0
            The lower bound (inclusive) of the random integers.
        high : int, default=2
            The upper bound (exclusive) of the random integers.
        dim : int | None, default=None
            The dimension of the random vector. Either `dim` or `index` may be provided to set the dimension of the random vector. If neither is provided, `dim` will default to `1`.
        index : Index | IndexLike | None, default=None
            The index of the random vector. Either `dim` or `index` may be provided to set the dimension of the random vector. If neither is provided, `dim` will default to `1`.
        random_state : int | np.random.Generator | None, default=None
            An optional seed for the random number generator, or a `np.random.Generator` instance to use directly. If an integer is provided, a new generator is created with that seed. If a generator is provided, it is used directly. If `None`, the random number generator is not seeded.
        name : Hashable, default="X"
            The name of the random vector.

        Raises
        ------
        TypeError
            If `low` or `high` are not integers, if `sig_alg` is not an instance of `SigmaAlgebra` or `None`, or if `random_state` is not an integer, `np.random.Generator`, or `None`.
        ValueError
            If `dim` is not positive, if both `dim` and `index` are provided, or if `low` is greater than or equal to `high`.

        Returns
        -------
        rv : RandomVariable | RandomVector
            A random vector (or random variable, if the dimension is 1) with integer outputs uniformly sampled from the range [low, high).

        Examples
        --------
        Create a 2-dimensional random vector with integer outputs uniformly sampled from the range [0, 5).

        >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> X = RandomVector.from_randint(sample_space=Omega, low=0, high=5, dim=2, random_state=42)
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        index   0  1
        sample
        0       0  3
        1       3  2
        2       2  4

        Create a random variable by not specifying the dimension or index. The default dimension is 1.

        >>> Y = RandomVector.from_randint(sample_space=Omega, low=1, high=10, random_state=42, name="Y")
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'Y':
                Y
        sample
        0       1
        1       7
        2       6

        Create a 2-dimensional random vector with outputs uniformly sampled from the range [0, 5) with a custom sigma-algebra and probability measure.

        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 1,
        ...         1: 1,
        ...         2: 0,
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.2,
        ...         1: 0.8,
        ...     },
        ... )
        >>> Z = RandomVector.from_randint(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     prob_measure=P,
        ...     low=0,
        ...     high=5,
        ...     index=[1, 2],
        ...     random_state=42,
        ...     name="Z",
        ... )
        >>> print(Z)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Z':
        index   1  2
        sample
        0       0  3
        1       0  3
        2       3  2

        """
        from ..indices.index import Index
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from ..spaces.sample_space import SampleSpace

        if not isinstance(sample_space, SampleSpace):
            sample_space = SampleSpace(indices=sample_space)
        if index is not None and not isinstance(index, Index):
            index = Index(indices=index)
        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be an instance of SigmaAlgebra, if given.")
        if dim is not None and not isinstance(dim, int):
            raise TypeError("dim must be a positive integer, if given.")
        if dim is not None and dim <= 0:
            raise ValueError("dim must be positive, if given.")
        if random_state is not None and not isinstance(
            random_state, (int, np.random.Generator)
        ):
            raise TypeError(
                "random_state must be an integer, np.random.Generator, or None."
            )
        if dim is not None and index is not None:
            raise ValueError("Both dim and index cannot be provided.")

        if not isinstance(low, int) or not isinstance(high, int):
            raise TypeError("low and high must be integers.")
        if low >= high:
            raise ValueError("low must be less than high.")

        if dim is None and index is None:
            dim = 1
        if index is not None:
            dim = len(index)

        if sig_alg is None:
            sig_alg = SigmaAlgebra.power_set(sample_space)

        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        mapping = rng.integers(low, high, size=(sig_alg.num_atoms, dim))
        mapping = pd.DataFrame(mapping, index=sig_alg.atom_space.data)
        mapping = pd.merge(
            left=sig_alg.data, right=mapping, left_on="atom_ID", right_index=True
        ).drop(columns="atom_ID")

        rv = cls(
            sample_space=sample_space,
            sig_alg=sig_alg,
            prob_measure=prob_measure,
            mapping=mapping,
            index=index,
            name=name,
        )

        if rv.dimension == 1:
            return rv.to_random_variable()
        else:
            return rv

    @classmethod
    def from_randnorm(
        cls,
        sample_space: SampleSpace | IndexLike,
        sig_alg: SigmaAlgebra | None = None,
        prob_measure: ProbabilityMeasure | None = None,
        loc: float = 0.0,
        scale: float = 1.0,
        dim: int | None = None,
        index: Index | IndexLike | None = None,
        random_state: int | np.random.Generator | None = None,
        name: Hashable = "X",
    ) -> RandomVector:
        """Generate a random vector with outputs sampled from a normal distribution with specified mean and standard deviation.

        Parameters
        ----------
        sample_space: SampleSpace | IndexLike
            The sample space of the underlying probability space.
        sig_alg: SigmaAlgebra | None, default=None
            The sigma-algebra of the underlying probability space. If `None`, the power set sigma-algebra is used.
        prob_measure: ProbabilityMeasure | None, default=None
            The probability measure of the underlying probability space. If `None`, the uniform probability measure is used.
        loc : float, default=0.0
            The mean of the normal distribution.
        scale : float, default=1.0
            The standard deviation of the normal distribution.
        dim : int | None, default=None
            The dimension of the random vector. Either `dim` or `index` may be provided to set the dimension of the random vector. If neither is provided, `dim` will default to `1`.
        index : Index | IndexLike | None, default=None
            The index of the random vector. Either `dim` or `index` may be provided to set the dimension of the random vector. If neither is provided, `dim` will default to `1`.
        random_state : int | np.random.Generator | None, default=None
            An optional seed for the random number generator, or a `np.random.Generator` instance to use directly. If an integer is provided, a new generator is created with that seed. If a generator is provided, it is used directly. If `None`, the random number generator is not seeded.
        name : Hashable, default="X"
            The name of the random vector.

        Returns
        -------
        self : RandomVector
            A random vector with outputs sampled from a normal distribution with specified mean and standard deviation.

        Examples
        --------
        Create a 2-dimensional random vector with floating-point outputs sampled from a standard normal distribution.

        >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> X = RandomVector.from_randnorm(sample_space=Omega, dim=2, random_state=42)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        index          0         1
        sample
        0       0.304717 -1.039984
        1       0.750451  0.940565
        2      -1.951035 -1.302180

        Create a random variable by not specifying the dimension or index. The default dimension is 1.

        >>> Y = RandomVector.from_randnorm(sample_space=Omega, random_state=42, name="Y")
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'Y':
                       Y
        sample
        0       0.304717
        1      -1.039984
        2       0.750451

        Create a 2-dimensional random vector with outputs uniformly sampled from the range [0, 5) with a custom sigma-algebra and probability measure.

        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 1,
        ...         1: 1,
        ...         2: 0,
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.2,
        ...         1: 0.8,
        ...     },
        ... )
        >>> Z = RandomVector.from_randnorm(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     prob_measure=P,
        ...     index=[1, 2],
        ...     random_state=42,
        ...     name="Z",
        ... )
        >>> print(Z)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Z':
        index          1         2
        sample
        0       0.304717 -1.039984
        1       0.304717 -1.039984
        2       0.750451  0.940565
        """
        from ..indices.index import Index
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from ..spaces.sample_space import SampleSpace

        if not isinstance(sample_space, SampleSpace):
            sample_space = SampleSpace(indices=sample_space)
        if index is not None and not isinstance(index, Index):
            index = Index(indices=index)
        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be an instance of SigmaAlgebra, if given.")
        if dim is not None and not isinstance(dim, int):
            raise TypeError("dim must be a positive integer, if given.")
        if dim is not None and dim <= 0:
            raise ValueError("dim must be positive, if given.")
        if random_state is not None and not isinstance(
            random_state, (int, np.random.Generator)
        ):
            raise TypeError(
                "random_state must be an integer, np.random.Generator, or None."
            )
        if dim is not None and index is not None:
            raise ValueError("Both dim and index cannot be provided.")

        if not isinstance(loc, Real) or not isinstance(scale, Real):
            raise TypeError("loc and scale must be real numbers.")
        if scale <= 0:
            raise ValueError("scale must be positive.")

        if dim is None and index is None:
            dim = 1
        if index is not None:
            dim = len(index)

        if sig_alg is None:
            sig_alg = SigmaAlgebra.power_set(sample_space)

        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        mapping = rng.normal(loc, scale, size=(sig_alg.num_atoms, dim))
        mapping = pd.DataFrame(mapping, index=sig_alg.atom_space.data)
        mapping = pd.merge(
            left=sig_alg.data, right=mapping, left_on="atom_ID", right_index=True
        ).drop(columns="atom_ID")

        rv = cls(
            sample_space=sample_space,
            sig_alg=sig_alg,
            prob_measure=prob_measure,
            mapping=mapping,
            index=index,
            name=name,
        )

        if rv.dimension == 1:
            return rv.to_random_variable()
        else:
            return rv

    @classmethod
    def concatenate(
        cls,
        factors: list[RandomVariable | RandomVector | Real],
        index: Index | IndexLike | None = None,
        name: Hashable | None = None,
    ) -> RandomVector:
        """Concatenate a list of random vectors or scalars into a single random vector.

        Parameters
        ----------
        factors : list[RandomVariable | RandomVector | Real]
            A list of random vectors or scalars to combine.
        index : Index | IndexLike | None, default=None
            The index of the resulting random vector. If `None`, the index will be generated by concatenating the indices of the input random vectors, provided that they are disjoint; otherwise, a new default index will be generated.
        name : Hashable | None, default=None
            The name of the resulting random vector. If `None`, the name will be generated by concatenating the names of the input random vectors.

        Raises
        ------
        TypeError
            If `factors` is not a list, if any element of `factors is not a `RandomVariable`, `RandomVector`, or scalar, or if `name` is not a `Hashable` or `None`.
        ValueError
            If there is not at least one `RandomVector` instance in `factors`, or if the random vectors in `factors` are not defined on the same probability space.

        Returns
        -------
        concatenation : RandomVector
            A new random vector created by combining the input random vectors.

        Examples
        --------
        Generate a random probability space.

        >>> from sigalg.core import (
        ...     Index,
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra.from_rand(
        ...     sample_space=Omega,
        ...     num_atoms=3,
        ...     random_state=42,
        ... )
        >>> P = ProbabilityMeasure.from_rand(sig_alg=F, random_state=42)
        >>> I = Index([0, 1, 2])

        Generate two random vector with disjoint indices.

        >>> X = RandomVector.from_randint(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     prob_measure=P,
        ...     index=I,
        ...     random_state=42,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        index   0  1  2
        sample
        0       0  1  1
        1       0  0  1
        2       0  1  1
        3       0  1  0
        >>> J = Index([3, 4], name="J")
        >>> Y = RandomVector.from_randint(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     prob_measure=P,
        ...     index=J,
        ...     random_state=42,
        ...     name="Y",
        ... )
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
        index   3  4
        sample
        0       0  1
        1       1  0
        2       0  1
        3       0  1

        Concatenate the two random vectors.

        >>> XY = RandomVector.concatenate([X, Y])
        >>> print(XY)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'XY':
        index   0  1  2  3  4
        sample
        0       0  1  1  0  1
        1       0  0  1  1  0
        2       0  1  1  0  1
        3       0  1  0  0  1

        Generate a random variable.

        >>> Z = RandomVariable.from_randint(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     prob_measure=P,
        ...     random_state=42,
        ...     name="Z",
        ... )
        >>> print(Z)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'Z':
                Z
        sample
        0       0
        1       1
        2       0
        3       1

        Concatenate random variables and vectors, along with scalars using the `|` operator.

        >>> ZX2Y = Z | X | 2 | Y
        >>> print(ZX2Y)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'ZX2Y':
        index   Z  0  1  2  2  3  4
        sample
        0       0  0  1  1  2  0  1
        1       1  0  0  1  2  1  0
        2       0  0  1  1  2  0  1
        3       1  0  1  0  2  0  1

        From a concatenation with a custom index and name.

        >>> W = RandomVector.concatenate([0, Z, X], index=[0, 1, 2, 3, 4], name="W")
        >>> print(W)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'W':
        index   0  1  2  3  4
        sample
        0       0  0  0  1  1
        1       0  1  0  0  1
        2       0  0  0  1  1
        3       0  1  0  1  0

        """
        from ..indices.index import Index
        from .random_variable import RandomVariable

        if not isinstance(factors, list):
            raise TypeError(
                "factors must be a list of instances of RandomVector and scalars."
            )
        actual_rvs = [rv for rv in factors if isinstance(rv, RandomVector)]
        if not actual_rvs:
            raise ValueError("There must be at least one random vector in `factors`.")
        prob_space = actual_rvs[0].prob_space
        if any(rv.prob_space != prob_space for rv in actual_rvs):
            raise ValueError(
                "All RandomVector instances must be defined on the same probability space."
            )
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("If given, name must be a Hashable.")

        try:
            factors = [
                RandomVariable.from_constant(*prob_space, constant=rv, name=str(rv))
                if not isinstance(rv, RandomVector)
                else rv
                for rv in factors
            ]
        except TypeError as e:
            raise TypeError(
                "Cannot form constant random variables from the factors."
            ) from e

        factors = [rv.to_random_vector() for rv in factors]
        indices = [rv.index for rv in factors]

        ignore_index = any(
            len(idx1 & idx2) >= 1 for idx1, idx2 in combinations(indices, 2)
        )

        if name is None:
            name = "".join(rv.name for rv in factors)

        combined_data = pd.concat(
            [rv.data for rv in factors], axis=1, ignore_index=ignore_index
        )
        if index is not None:
            if not isinstance(index, Index):
                index = Index(indices=index)
            combined_data.columns = index.data

        return cls(
            *factors[0].prob_space, mapping=combined_data, index=index, name=name
        )

    def __or__(self, other: RandomVector | MeasurableSet) -> RandomVector:
        """Concatenate the current instance with a second random vector, random variable, or scalar into a single random vector, or restrict the random vector to an event.

        Calls `RandomVector.concatenate` if `other` is a `RandomVector` or scalar, or `RandomVector.restrict_to` if `other` is an `MeasurableSet`. See the documentation for those methods for more details.
        """
        from ..spaces.measurable_set import MeasurableSet

        if isinstance(other, MeasurableSet):
            return self.restrict_to(event=other)
        else:
            return RandomVector.concatenate([self, other])

    @classmethod
    def cartesian_product(
        cls,
        factors: list[RandomVector],
        index: Index | IndexLike | None = None,
        name: Hashable | None = None,
    ) -> RandomVector:
        """Form the Cartesian product of a list of random vectors.

        Parameters
        ----------
        factors : list[RandomVector]
            The factors of the Cartesian product.
        index : Index | IndexLike | None, default=None
            The index of the Cartesian product. If `None`, a default index will be generated.
        name : Hashable | None, default=None
            The name of the Cartesian product. If `None`, a default will be generated from the names of the random vectors in `factors`.

        Raises
        ------
        TypeError
            If `factors` is not a list of random vectors.

        Returns
        -------
        product : RandomVector
            The Cartesian product of the random vectors.

        Examples
        --------
        Define two 1-dimensional sample spaces and a 2-dimensional one. Then, define two 2-dimensional random vectors `X` and `Y` and a random variable `Z`, and form their Cartesian product.

        >>> from sigalg.core import (
        ...     RandomVariable,
        ...     RandomVector,
        ...     SampleSpace,
        ... )
        >>> Omega1 = SampleSpace.from_sequence(size=4, name="Omega1", variable_name="x")
        >>> Omega2 = SampleSpace([(1, "a"), (2, "b")], name="Omega2", variable_names=["y", "z"])
        >>> Omega3 = SampleSpace.from_sequence(
        ...     size=2,
        ...     initial_index=4,
        ...     name="Omega3",
        ...     variable_name="w",
        ... )
        >>> X = RandomVector(
        ...     sample_space=Omega1,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...         3: (5, 6),
        ...     },
        ... )
        >>> Y = RandomVector(
        ...     sample_space=Omega2,
        ...     mapping={
        ...         (1, "a"): (7, 8),
        ...         (2, "b"): (9, 10),
        ...     },
        ...     name="Y",
        ... )
        >>> Z = RandomVariable(
        ...     sample_space=Omega3,
        ...     mapping={
        ...         4: 1,
        ...         5: 3,
        ...     },
        ...     name="Z",
        ... )
        >>> product = RandomVector.cartesian_product(factors=[X, Y, Z])
        >>> print(product)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X x Y x Z':
        index    0  1  2   3  4
        x y z w
        0 1 a 4  1  2  7   8  1
              5  1  2  7   8  3
          2 b 4  1  2  9  10  1
              5  1  2  9  10  3
        1 1 a 4  3  4  7   8  1
              5  3  4  7   8  3
          2 b 4  3  4  9  10  1
              5  3  4  9  10  3
        2 1 a 4  3  4  7   8  1
              5  3  4  7   8  3
          2 b 4  3  4  9  10  1
              5  3  4  9  10  3
        3 1 a 4  5  6  7   8  1
              5  5  6  7   8  3
          2 b 4  5  6  9  10  1
              5  5  6  9  10  3

        Compute the Cartesian product of `X` and `Y` using the `@` operator.

        >>> product = X @ Y
        >>> print(product)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X x Y':
        index  0  1  2   3
        x y z
        0 1 a  1  2  7   8
          2 b  1  2  9  10
        1 1 a  3  4  7   8
          2 b  3  4  9  10
        2 1 a  3  4  7   8
          2 b  3  4  9  10
        3 1 a  5  6  7   8
          2 b  5  6  9  10
        """
        from ..indices.index import Index
        from ..spaces.sample_space import SampleSpace

        if index is not None and not isinstance(index, Index):
            index = Index(index)

        if not isinstance(factors, list) or not all(
            isinstance(rv, RandomVector) for rv in factors
        ):
            raise TypeError("factors must be a list of RandomVectors.")

        mapping = factors[0].data
        sample_space = SampleSpace.cartesian_product(
            [rv.sample_space for rv in factors]
        )

        for rv in factors[1:]:
            mapping = pd.merge(
                left=mapping,
                right=rv.data,
                how="cross",
            )
        mapping.index = sample_space.data

        if index is None:
            index = Index(indices=list(range(mapping.shape[1])))
        mapping.columns = index.data

        if name is None:
            name = " x ".join([rv.name for rv in factors])

        if all(rv.is_identity for rv in factors):
            return cls.from_identity(
                sample_space=sample_space,
                name=name,
                index=index,
            )

        else:
            return cls(
                sample_space=sample_space,
                mapping=mapping,
                index=index,
                name=name,
            )

    def __matmul__(self, other: RandomVector) -> RandomVector:
        """Form the Cartesian product of a pair of random vectors.

        Calls the `RandomVector.cartesian_product` method. See the documentation of that method for details.
        """
        return RandomVector.cartesian_product([self, other])

    @classmethod
    def cartesian_power(
        cls,
        rv: RandomVector,
        power: int,
        index: Index | IndexLike | None = None,
    ) -> RandomVector:
        """Form the Cartesian power of a random vector.

        Parameters
        ----------
        rv : RandomVector
            The base of the Cartesian power.
        power : int
            The power of the Cartesian power.
        index : Index | IndexLike | None, default=None
            The index of the Cartesian power. If `None`, a default index will be generated.

        Raises
        ------
        TypeError
            If `rv` is not a `RandomVector` or if `power` is not an integer.
        ValueError
            If `power` is not positive.

        Examples
        --------
        Define a 2-dimensional random vector `X`.

        >>> from sigalg.core import (
        ...     ProbabilityMeasure,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace.from_sequence(size=4, variable_name="x")
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ...     variable_names=["u"],
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.2,
        ...         1: 0.4,
        ...         2: 0.4,
        ...     },
        ... )
        >>> X = RandomVector(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     prob_measure=P,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...         3: (5, 6),
        ...     },
        ... )

        Compute the second Cartesian power of the random vector `X` and print its probability space.

        >>> X_2 = RandomVector.cartesian_power(X, 2)
        >>> print(X_2)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X ^ 2':
        index    0  1  2  3
        x_0 x_1
        0   0    1  2  1  2
            1    1  2  3  4
            2    1  2  3  4
            3    1  2  5  6
        1   0    3  4  1  2
            1    3  4  3  4
            2    3  4  3  4
            3    3  4  5  6
        2   0    3  4  1  2
            1    3  4  3  4
            2    3  4  3  4
            3    3  4  5  6
        3   0    5  6  1  2
            1    5  6  3  4
            2    5  6  3  4
            3    5  6  5  6
        >>> print(X_2.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega ^ 2, F ^ 2, P ^ 2)
        ===========================================
        <BLANKLINE>
        * Sample space 'Omega ^ 2':
         x_0  x_1
           0    0
           0    1
           0    2
           0    3
           1    0
           1    1
           1    2
           1    3
           2    0
           2    1
           2    2
           2    3
           3    0
           3    1
           3    2
           3    3
        <BLANKLINE>
        * Sigma algebra 'F ^ 2':
                atom_ID
        x_0 x_1
        0   0    (0, 0)
            1    (0, 1)
            2    (0, 1)
            3    (0, 2)
        1   0    (1, 0)
            1    (1, 1)
            2    (1, 1)
            3    (1, 2)
        2   0    (1, 0)
            1    (1, 1)
            2    (1, 1)
            3    (1, 2)
        3   0    (2, 0)
            1    (2, 1)
            2    (2, 1)
            3    (2, 2)
        <BLANKLINE>
        * Probability measure 'P ^ 2':
                probability
        u_0 u_1
        0   0           0.04
            1           0.08
            2           0.08
        1   0           0.08
            1           0.16
            2           0.16
        2   0           0.08
            1           0.16
            2           0.16

        Compute the third Cartesian power using the `^` operator.

        >>> X_3 = X ^ 3
        >>> print(X_3)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X ^ 3':
        index        0  1  2  3  4  5
        x_0 x_1 x_2
        0   0   0    1  2  1  2  1  2
                1    1  2  1  2  3  4
                2    1  2  1  2  3  4
                3    1  2  1  2  5  6
            1   0    1  2  3  4  1  2
        ...         .. .. .. .. .. ..
        3   2   3    5  6  3  4  5  6
            3   0    5  6  5  6  1  2
                1    5  6  5  6  3  4
                2    5  6  5  6  3  4
                3    5  6  5  6  5  6
        <BLANKLINE>
        [64 rows x 6 columns]
        """
        from ..indices.index import Index

        if not isinstance(rv, RandomVector):
            raise TypeError("rv must be a RandomVector.")
        if not isinstance(power, int):
            raise TypeError("power must be an integer.")
        if power <= 0:
            raise ValueError("power must be a positive integer.")
        if index is not None and not isinstance(index, Index):
            index = Index(index)

        variable_names = list(rv.data.index.names)
        reset_data = []
        product_variable_names = []

        for k in range(power):
            reset_data.append(rv.data.reset_index().add_suffix(f"_{k}"))
            product_variable_names += [f"{name}_{k}" for name in variable_names]

        power_data = reset_data[0]

        for data in reset_data[1:]:
            power_data = pd.merge(
                left=power_data,
                right=data,
                how="cross",
            )
        power_data.set_index(product_variable_names, inplace=True)

        if index is None:
            index = Index(indices=list(range(power_data.shape[1])))
        power_data.columns = index.data

        sample_space = rv.sample_space ^ power
        prob_measure = rv.prob_measure ^ power
        name = f"{rv.name} ^ {power}"

        if rv.is_identity:
            return RandomVector.from_identity(
                sample_space=sample_space,
                prob_measure=prob_measure,
                name=name,
                index=index,
            )

        else:
            return RandomVector(
                sample_space=sample_space,
                prob_measure=prob_measure,
                mapping=power_data,
                index=index,
                name=name,
            )

    def __xor__(self, power: int) -> RandomVector:
        """Form the Cartesian power of this instance of `RandomVector`.

        Calls the `RandomVector.cartesian_power` method. See the documentation of that method for details.
        """
        return RandomVector.cartesian_power(rv=self, power=power)

    @classmethod
    def indicator_of(
        cls,
        event: MeasurableSet,
        dim: int = 1,
        index: Index | IndexLike | None = None,
        name: Hashable | None = None,
    ) -> RandomVector:
        r"""Create the indicator random vector of a given event of a given dimension.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        event : MeasurableSet
            The event for which the indicator random vector is to be created.
        dim : int, default=1
            The dimension of the indicator random vector.
        index : Index | IndexLike | None, default=None
            The index of the indicator random vector. If `None`, a default index will be generated.
        name : Hashable | None, default=None
            The name of the indicator random vector. If `None`, a default name will be generated.

        Raises
        ------
        TypeError
            If `event` is not an instance of `MeasurableSet`, if `dim` is not an integer, or if `name` is not hashable (if given).
        ValueError
            If `dim` is not a positive integer.

        Returns
        -------
        indicator_rv : RandomVector
            The indicator random variable of the given event.

        Examples
        --------
        Get an event from a sample space.

        >>> from sigalg.core import RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_event([0, 1])

        Create an indicator random variable with default name.

        >>> I_A = RandomVector.indicator_of(event=A)
        >>> print(I_A)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'I_A':
                I_A
        sample
        0         1
        1         1
        2         0

        Create a 2-dimensional indicator random vector with custom name and index.

        >>> ind = RandomVector.indicator_of(event=A, dim=2, index=[1, 2], name="ind")
        >>> print(ind)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'ind':
        index   1  2
        sample
        0       1  1
        1       1  1
        2       0  0

        Notes
        -----
        Let $(\Omega, \mathcal{F}, P)$ be a probability space. Given an event $A\in \mathcal{F}$ and a dimension $d$, the *indicator random vector* is the random vector $I_A: \Omega \to \mathbb{R}^d$ such that

        $$
        I_A(\omega) = \begin{cases}
        (1, 1, \ldots, 1) & : \omega \in A,\\
        (0, 0, \ldots, 0) & : \omega \notin A.
        \end{cases}
        $$
        """
        from ..indices.index import Index
        from ..spaces.measurable_set import MeasurableSet

        if index is not None and not isinstance(index, Index):
            index = Index(index)

        if not isinstance(event, MeasurableSet):
            raise TypeError("event must be an MeasurableSet.")
        if not isinstance(dim, int):
            raise TypeError("dim must be an integer.")
        if dim <= 0:
            raise ValueError("dim must be a positive integer.")
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("If given, name must be hashable.")

        if dim == 1:
            return event.indicator
        data = pd.concat([event.indicator.data] * dim, axis=1)

        if name is None:
            name = event.indicator.name
        if index is None:
            index = Index(indices=list(range(dim)))

        data.columns = index.data

        return cls(
            sample_space=event.sample_space,
            sig_alg=event.indicator.sig_alg,
            mapping=data,
            index=index,
            name=name,
        )

    # --------------------- properties --------------------- #

    @property
    def data(self) -> pd.Series | pd.DataFrame | None:
        """Get the underlying data of the random vector.

        Returns
        -------
        data : pd.Series | pd.DataFrame | None
            A `pd.Series` (if the random vector is 1-dimensional) or `pd.DataFrame` (if the random vector is 2-dimensional or higher), or `None`.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     },
        ... )
        >>> X = RandomVector(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...     },
        ... )
        >>> print(X.data)  # doctest: +NORMALIZE_WHITESPACE
        index   0  1
        sample
        0       1  2
        1       3  4
        2       3  4
        >>> Y = RandomVector(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 2,
        ...     },
        ...     name="Y",
        ... )
        >>> print(Y.data)  # doctest: +NORMALIZE_WHITESPACE
        sample
        0    1
        1    2
        2    2
        Name: Y, dtype: int64
        """
        return self._data

    @property
    def atom_data(self) -> pd.Series | pd.DataFrame | None:
        """Get the underlying data of the random vector, grouped by atom identifiers.

        Returns
        -------
        atom_data : pd.Series | pd.DataFrame | None
            A `pd.Series` (if the random vector is 1-dimensional) or `pd.DataFrame` (if the random vector is 2-dimensional or higher), or `None`.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     },
        ... )
        >>> X = RandomVector(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...     },
        ... )
        >>> print(X.atom_data)  # doctest: +NORMALIZE_WHITESPACE
        index    0  1
        atom_ID
        0        1  2
        1        3  4
        >>> Y = RandomVector(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 2,
        ...     },
        ...     name="Y",
        ... )
        >>> print(Y.atom_data)  # doctest: +NORMALIZE_WHITESPACE
        atom_ID
        0    1
        1    2
        Name: Y, dtype: int64
        """
        if self._atom_data is None and self.data is not None:
            self._atom_data = (
                pd.concat([self.data, self.sig_alg.data], axis=1)
                .drop_duplicates()
                .set_index("atom_ID")
            ).squeeze(axis=1)

            if self.index is not None:
                self._atom_data.columns = self.index.data

        return self._atom_data

    @property
    def dimension(self) -> int | None:
        """Get the dimension of the random vector.

        Returns
        -------
        dimension : int | None
            The dimension of the random vector, or `None`.
        """
        if self._dimension is None and self.data is not None:
            if isinstance(self.data, pd.Series):
                self._dimension = 1
            else:
                self._dimension = self.data.shape[1]

        return self._dimension

    @property
    def components(self) -> list[RandomVariable] | None:
        r"""Get the component random variables of the random vector.

        See the Notes section below for the mathematical details.

        Raises
        ------
        ValueError
            If `self` has an empty `data` attribute.

        Returns
        -------
        components : list[RandomVariable] | None
            A list of the component random variables of the random vector.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> X = RandomVector.from_randint(
        ...     sample_space=Omega,
        ...     low=0,
        ...     high=3,
        ...     dim=2,
        ...     random_state=42,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        index   0  1
        sample
        0       0  2
        1       1  1
        2       1  2
        >>> for component in X.components:
        ...     print(component)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_0':
                X_0
        sample
        0         0
        1         1
        2         1
        Random variable 'X_1':
                X_1
        sample
        0         2
        1         1
        2         2
        >>> Y = RandomVector.from_randint(
        ...     sample_space=Omega,
        ...     low=0,
        ...     high=3,
        ...     dim=1,
        ...     random_state=42,
        ...     name="Y",
        ... )
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'Y':
                Y
        sample
        0       0
        1       2
        2       1
        >>> for component in Y.components:
        ...     print(component)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'Y':
                Y
        sample
        0       0
        1       2
        2       1

        Notes
        -----
        If $X: \Omega \to \mathbb{R}^d$ is a random vector, then for each $\omega \in \Omega$ we may write

        $$
        X(\omega) = (X_1(\omega),X_2(\omega),\ldots, X_d(\omega))
        $$

        where $X_j: \Omega \to \mathbb{R}$ is the *$j$-th component random variable* of $X$.
        """
        from .random_variable import RandomVariable

        if self._components is None and self.data is not None:
            if self.dimension == 1:
                if isinstance(self, RandomVariable):
                    self._components = [self]
                else:
                    self._components = [self.to_random_variable()]
            else:
                self._components = [
                    self.get_component_rv(idx).with_name(name)
                    for idx, name in zip(self.index, self.component_names)
                ]
        return self._components

    @property
    def component_names(self) -> list[Hashable] | None:
        """Get the names of the component random variables of the random vector.

        Returns
        -------
        component_names : list[Hashable] | None
            A list of the names of the component random variables of the random vector, or `None`.
        """
        if self._component_names is None and self.data is not None:
            if self.index is not None:
                self._component_names = [
                    f"{self.name}_{idx}".replace(".", "_") for idx in self.index
                ]
            else:
                self._component_names = [self.name]

        return self._component_names

    @property
    def name(self) -> Hashable:
        """Get the name of the random vector.

        Returns
        -------
        name : Hashable
            The name of the random vector.
        """
        return self._name

    @name.setter
    def name(self, name: Hashable) -> None:
        """Set the name of the random vector.

        Parameters
        ----------
        name : Hashable
            The new name for the random vector.

        Raises
        ------
        TypeError
            If `name` is not hashable.
        """
        if not isinstance(name, Hashable):
            raise TypeError("name must be a Hashable.")

        self._name = name
        self._components = None
        self._component_names = None
        self._generated_sig_alg = None
        self._range = None
        if isinstance(self._data, pd.Series):
            self._data.name = name

    def with_name(self, name: Hashable) -> RandomVector:
        """Set the name of the random vector and return self for chaining.

        Parameters
        ----------
        name : Hashable
            The new name for the random vector.

        Returns
        -------
        self : RandomVector
            Returns self to allow method chaining.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> X = RandomVector(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: (0, 2),
        ...         1: (1, 1),
        ...         2: (1, 2),
        ...     },
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        index   0  1
        sample
        0       0  2
        1       1  1
        2       1  2
        >>> Y = X.with_name("Y")
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
        index   0  1
        sample
        0       0  2
        1       1  1
        2       1  2
        """
        self.name = name
        return self

    @property
    def index(self) -> Index | None:
        """Get the index of the random vector.

        Returns
        -------
        index : Index | None
            The index of the random vector, or `None` if the random vector is 1-dimensional or has not been set.

        Examples
        --------
        >>> from sigalg.core import Index, RandomVector, SampleSpace
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> X = RandomVector(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (5, 6),
        ...     },
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        index   0  1
        sample
        0       1  2
        1       3  4
        2       5  6
        >>> print(X.index)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I':
         index
             0
             1
        >>> J = Index(["a", "b"], variable_names=["letter"], name="J")
        >>> X.index = J
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        letter  a  b
        sample
        0       1  2
        1       3  4
        2       5  6
        >>> print(X.index)  # doctest: +NORMALIZE_WHITESPACE
        Index 'J':
         letter
              a
              b
        >>> Y = RandomVector(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 1,
        ...         1: 2,
        ...         2: 3,
        ...     },
        ...     name="Y",
        ... )
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
                Y
        sample
        0       1
        1       2
        2       3
        >>> print(Y.index)
        None
        """
        return self._index

    @index.setter
    def index(self, index: Index | IndexLike) -> None:
        """Set the index of the random vector.

        Parameters
        ----------
        index : Index | IndexLike
            The new index for the random vector.

        Raises
        ------
        TypeError
            If `index` is not an instance of `Index`.
        ValueError
            If the random vector has a non-empty `data` attribute and the length of `index` does not match the dimension of the random vector.
        """
        from ..indices.index import Index

        if not isinstance(index, Index):
            index = Index(index)

        if self.data is not None:
            if len(index) != self.dimension:
                raise ValueError(
                    "index size must match the dimension of the random vector."
                )
            self.data.columns = index.data
            self.atom_data.columns = index.data

        self._components = None
        self._component_names = None
        self._generated_sig_alg = None
        self._range = None
        self._index = index

    @property
    def generated_sig_alg(self) -> SigmaAlgebra | None:
        r"""Get the sigma-algebra generated by a random vector.

        See the Notes section below for the mathematical details.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra induced by the random vector.

        Examples
        --------
        >>> from sigalg.core import (
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
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
        >>> X = RandomVector(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...     },
        ... )
        >>> sig_X = X.generated_sig_alg
        >>> print(sig_X)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'sigma(X)':
               atom_ID
        sample
        0       (1, 2)
        1       (3, 4)
        2       (3, 4)
        3       (3, 4)
        >>> print(sig_X <= F)
        True

        Notes
        -----
        A random vector $X: \Omega \to \mathbb{R}^d$ on a probability space $(\Omega, \mathcal{F},P)$ generates a $\sigma$-algebra denoted $\sigma(X)$. On a finite sample space $\Omega$, this $\sigma$-algebra is determined by its atoms, which are the nonempty preimages

        $$
        \{X=x\} \stackrel{\text{def}}{=} \{ \omega \in \Omega : X(\omega) = x\},
        $$

        for $x\in \mathbb{R}^d$. The atom identifiers may thus be taken as the vectors $x\in \mathbb{R}^d$ in the range of $X$.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if self._generated_sig_alg is None and self.data is not None:
            self._generated_sig_alg = SigmaAlgebra.from_random_vector(self)
        return self._generated_sig_alg

    @property
    def prob_space(self) -> ProbabilitySpace | None:
        """Get the probability space on which the random vector is defined.

        Returns
        -------
        prob_space : ProbabilitySpace | None
            The probability space on which the random vector is defined.

        Examples
        --------
        >>> from sigalg.core import (
        ...     ProbabilityMeasure,
        ...     ProbabilitySpace,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.2,
        ...         1: 0.8,
        ...     },
        ... )
        >>> prob_space = ProbabilitySpace(Omega, F, P)
        >>> X = RandomVector(
        ...     *prob_space,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...     },
        ... )
        >>> print(X.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, F, P)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
         sample
              0
              1
              2
        <BLANKLINE>
        * Sigma algebra 'F':
                atom_ID
        sample
        0             0
        1             1
        2             1
        <BLANKLINE>
        * Probability measure 'P':
                 probability
        atom_ID
        0                0.2
        1                0.8
        """
        return self._prob_space

    @property
    def sample_space(self) -> SampleSpace | None:
        """Get the domain of the random vector.

        The `domain` property is settable. If the random vector is not defined on an empty probability space, the new domain must have the same number of sample points as the existing domain and the sample spaces of the sigma-algebra and probability measure are updated to the new sample space. If in addition the random vector is not empty (i.e., if it has outputs), then the outputs of the random vector are remapped to the new domain according to the order of sample points in the new domain. If the random vector is defined on an empty probability space (and therefore also has no outputs), then the domain may be set freely, the sigma-algebra is updated to the power-set sigma-algebra on the new domain, and the probability measure is updated to the uniform measure on the new domain.

        Returns
        -------
        domain : SampleSpace | None
            The domain of the random vector.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.25,
        ...         1: 0.75,
        ...     },
        ... )
        >>> X = RandomVector(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     prob_measure=P,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...     },
        ... )
        >>> print(X.sample_space)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega':
         sample
              0
              1
              2
              3
        >>> print(X.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, F, P)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
         sample
              0
              1
              2
              3
        <BLANKLINE>
        * Sigma algebra 'F':
                atom_ID
        sample
        0             0
        1             0
        2             1
        3             1
        <BLANKLINE>
        * Probability measure 'P':
                 probability
        atom_ID
        0               0.25
        1               0.75
        >>> S = SampleSpace(["a", "b", "c", "d"], name="S")
        >>> X.sample_space = S
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        index   0  1
        sample
        a       1  2
        b       1  2
        c       3  4
        d       3  4
        >>> print(X.sample_space)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'S':
        sample
            a
            b
            c
            d
        >>> print(X.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (S, F, P)
        ===========================
        <BLANKLINE>
        * Sample space 'S':
        sample
            a
            b
            c
            d
        <BLANKLINE>
        * Sigma algebra 'F':
                atom_ID
        sample
        a             0
        b             0
        c             1
        d             1
        <BLANKLINE>
        * Probability measure 'P':
                 probability
        atom_ID
        0               0.25
        1               0.75
        >>> empty_RV = RandomVector(name="empty_RV")
        >>> empty_RV.sample_space = S
        >>> print(empty_RV.sample_space)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'S':
        sample
            a
            b
            c
            d
        >>> print(empty_RV.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (S, power_set, U)
        ===================================
        <BLANKLINE>
        * Sample space 'S':
        sample
            a
            b
            c
            d
        <BLANKLINE>
        * Sigma algebra 'power_set':
            atom_ID
        sample
        a            a
        b            b
        c            c
        d            d
        <BLANKLINE>
        * Probability measure 'U':
                probability
        sample
        a              0.25
        b              0.25
        c              0.25
        d              0.25
        """
        return self.prob_space.sample_space

    @sample_space.setter
    def sample_space(self, sample_space: SampleSpace | IndexLike) -> None:
        """Set the domain of the random vector.

        If the random vector is not defined on an empty probability space, the new domain must have the same number of sample points as the existing domain, and then the sample spaces of the sigma-algebra and probability measure are updated to the new sample space. If in addition the random vector is not empty (i.e., if it has outputs), then the outputs of the random vector are remapped to the new domain according to the order of sample points in the new domain. If the random vector is defined on an empty probability space (and therefore also has no outputs), then the domain may be set freely, the sigma-algebra is updated to the power-set sigma-algebra on the new domain, and the probability measure is updated to the uniform measure on the new domain.

        Parameters
        ----------
        sample_space : SampleSpace | IndexLike
            The new sample space for the random vector.
        """
        from ..spaces.sample_space import SampleSpace

        if not isinstance(sample_space, SampleSpace):
            sample_space = SampleSpace(sample_space)

        self.prob_space.sample_space = sample_space

        if self.data is not None:
            self.data.index = self.prob_space.sample_space.data

        new = RandomVector(
            *self.prob_space,
            mapping=self.data if self.data is not None else None,
            index=self.index,
            name=self.name,
        )

        self.__dict__.update(new.__dict__)

    @property
    def sig_alg(self) -> SigmaAlgebra | None:
        """Get the sigma-algebra on the underlying probability space.

        The `sig_alg` property is settable. If the random vector is not defined on an empty probability space, the new sigma-algebra must be a sub-sigma-algebra of the existing sigma-algebra and the probability measure is updated to be the restriction of the existing probability measure to the new sigma-algebra. If in addition the random vector is not empty (i.e., if it has outputs), then the random vector must be measurable with respect to the new sigma-algebra. If the random vector is defined on an empty probability space (and therefore also has no outputs), then the sigma-algebra may be set freely and the domain is set to the sample space of the sigma-algebra and the probability measure is the uniform measure on the sample space.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra on the domain of the random vector.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace, SigmaAlgebra
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
        ...         0: 0.05,
        ...         1: 0.75,
        ...         2: 0.2,
        ...     },
        ... )
        >>> X = RandomVector(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     prob_measure=P,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...     },
        ... )
        >>> print(X.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                atom_ID
        sample
        0             0
        1             1
        2             2
        3             2
        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...     },
        ...     name="G",
        ... )
        >>> X.sig_alg = G
        >>> print(X.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
                atom_ID
        sample
        0             0
        1             0
        2             1
        3             1
        >>> print(X.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, G, P|G)
        =================================
        <BLANKLINE>
        * Sample space 'Omega':
         sample
              0
              1
              2
              3
        <BLANKLINE>
        * Sigma algebra 'G':
                atom_ID
        sample
        0             0
        1             0
        2             1
        3             1
        <BLANKLINE>
        * Probability measure 'P|G':
                 probability
        atom_ID
        0                0.8
        1                0.2
        >>> empty_RV = RandomVector(name="empty_RV")
        >>> empty_RV.sig_alg = G
        >>> print(empty_RV.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
                atom_ID
        sample
        0             0
        1             0
        2             1
        3             1
        >>> print(empty_RV.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, G, U)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
         sample
              0
              1
              2
              3
        <BLANKLINE>
        * Sigma algebra 'G':
                atom_ID
        sample
        0             0
        1             0
        2             1
        3             1
        <BLANKLINE>
        * Probability measure 'U':
                 probability
        atom_ID
        0                0.5
        1                0.5
        """
        return self.prob_space.sig_alg

    @sig_alg.setter
    def sig_alg(self, sig_alg: SigmaAlgebra) -> None:
        """Set the sigma-algebra on the underlying probability space.

        If the random vector is not defined on an empty probability space, the new sigma-algebra must be a sub-sigma-algebra of the existing sigma-algebra and the probability measure is updated to be the restriction of the existing probability measure to the new sigma-algebra. If in addition the random vector is not empty (i.e., if it has outputs), then the random vector must be measurable with respect to the new sigma-algebra. If the random vector is defined on an empty probability space (and therefore also has no outputs), then the sigma-algebra may be set freely and the domain is set to the sample space of the sigma-algebra and the probability measure is the uniform measure on the sample space.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The new sigma-algebra for the random vector.

        Raises
        ------
        TypeError
            If `sig_alg` is not an instance of `SigmaAlgebra`.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be an instance of SigmaAlgebra.")

        self.prob_space.sig_alg = sig_alg

        new = RandomVector(
            *self.prob_space,
            mapping=self.data if self.data is not None else None,
            index=self.index,
            name=self.name,
        )

        self.__dict__.update(new.__dict__)

    @property
    def prob_measure(self) -> ProbabilityMeasure | None:
        """Get the probability measure on the underlying probability space.

        The `prob_measure` property is settable. If the random vector is not defined on an empty probability space, the new probability measure must be a probability measure on a sub-sigma-algebra of the existing sigma-algebra. If in addition the random vector is not empty (i.e., if it has outputs), then the random vector must be measurable with respect to the sub-sigma-algebra. If the random vector is defined on an empty probability space (and therefore also has no outputs), then the probability measure may be set freely and the domain is set to the sample space of the probability measure's sigma-algebra and the sigma-algebra is set to the sigma-algebra of the probability measure.

        Returns
        -------
        prob_measure : ProbabilityMeasure | None
            The probability measure on the domain of the random vector.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace, SigmaAlgebra
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
        ...         0: 0.05,
        ...         1: 0.75,
        ...         2: 0.2,
        ...     },
        ... )
        >>> X = RandomVector(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     prob_measure=P,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...     },
        ... )
        >>> print(X.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                 probability
        atom_ID
        0               0.05
        1               0.75
        2               0.20
        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...     },
        ...     name="G",
        ... )
        >>> Q = ProbabilityMeasure(
        ...     sig_alg=G,
        ...     mapping={
        ...         0: 0.1,
        ...         1: 0.9,
        ...     },
        ...     name="Q",
        ... )
        >>> X.prob_measure = Q
        >>> print(X.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'Q':
                 probability
        atom_ID
        0                0.1
        1                0.9
        >>> print(X.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, G, Q)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
         sample
              0
              1
              2
              3
        <BLANKLINE>
        * Sigma algebra 'G':
                atom_ID
        sample
        0             0
        1             0
        2             1
        3             1
        <BLANKLINE>
        * Probability measure 'Q':
                 probability
        atom_ID
        0                0.1
        1                0.9
        >>> empty_RV = RandomVector(name="empty_RV")
        >>> empty_RV.prob_measure = Q
        >>> print(empty_RV.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'Q':
                 probability
        atom_ID
        0                0.1
        1                0.9
        >>> print(empty_RV.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, G, Q)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
         sample
              0
              1
              2
              3
        <BLANKLINE>
        * Sigma algebra 'G':
                atom_ID
        sample
        0             0
        1             0
        2             1
        3             1
        <BLANKLINE>
        * Probability measure 'Q':
                 probability
        atom_ID
        0                0.1
        1                0.9
        """
        return self.prob_space.prob_measure

    @prob_measure.setter
    def prob_measure(self, prob_measure: ProbabilityMeasure) -> None:
        """Set the probability measure on the underlying probability space.

        If the random vector is not defined on an empty probability space, the new probability measure must be a probability measure on a sub-sigma-algebra of the existing sigma-algebra. If in addition the random vector is not empty (i.e., if it has outputs), then the random vector must be measurable with respect to the sub-sigma-algebra. If the random vector is defined on an empty probability space (and therefore also has no outputs), then the probability measure may be set freely and the domain is set to the sample space of the probability measure's sigma-algebra and the sigma-algebra is set to the sigma-algebra of the probability measure.

        Parameters
        ----------
        prob_measure : ProbabilityMeasure
            The new probability measure for the random vector.

        Raises
        ------
        TypeError
            If `prob_measure` is not an instance of `ProbabilityMeasure`.
        """
        from ..measures.probability_measure import ProbabilityMeasure

        if not isinstance(prob_measure, ProbabilityMeasure):
            raise TypeError("prob_measure must be an instance of ProbabilityMeasure.")

        self.prob_space.prob_measure = prob_measure

        new = RandomVector(
            *self.prob_space,
            mapping=self.data if self.data is not None else None,
            index=self.index,
            name=self.name,
        )

        self.__dict__.update(new.__dict__)

    def with_probability_measure(
        self,
        prob_measure: ProbabilityMeasure | None = None,
        probabilities: MappingLike | None = None,
        name: Hashable | None = None,
    ) -> RandomVector:
        """Set the probability measure on the domain of the random vector and return self for chaining.

        This method is equivalent to setting the `prob_measure` attribute with an instance of `ProbabilityMeasure`. The method also accepts a dictionary of probabilities as a parameter, allowing the user to bypass constructing an instance of `ProbabilityMeasure`.

        The method takes either the `probabilities` parameter or the `prob_measure` parameter, but not both. If neither parameter is provided, the method defaults to setting the probability measure to the uniform measure.

        Parameters
        ----------
        prob_measure : ProbabilityMeasure | None, default=None
            The probability measure to set on the domain of the random vector.
        probabilities : Mapping[Hashable, Real] | None, default=None
            A mapping from sample points in the domain to their corresponding probabilities.

        Raises
        ------
        ValueError
            If both `probabilities` and `prob_measure` are provided.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.05,
        ...         1: 0.95,
        ...     },
        ... )
        >>> X = RandomVector(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     prob_measure=P,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...     },
        ... )
        >>> print(X.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                 probability
        atom_ID
        0               0.05
        1               0.95
        >>> Q = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.3,
        ...         1: 0.7,
        ...     },
        ...     name="Q",
        ... )
        >>> _ = X.with_probability_measure(prob_measure=Q)
        >>> print(X.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'Q':
                 probability
        atom_ID
        0                0.3
        1                0.7
        >>> _ = X.with_probability_measure(
        ...     probabilities={
        ...         0: 0.6,
        ...         1: 0.4,
        ...     },
        ...     name="R",
        ... )
        >>> print(X.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'R':
                 probability
        atom_ID
        0                0.6
        1                0.4
        """
        from ..measures.probability_measure import ProbabilityMeasure

        if (probabilities is None) == (prob_measure is None):
            raise ValueError(
                "Must specify either probabilities or prob_measure, but not both."
            )

        if probabilities is not None:
            prob_measure = ProbabilityMeasure(
                sig_alg=self.sig_alg,
                mapping=probabilities,
                name=name,
            )

        self.prob_measure = prob_measure
        return self

    @property
    def is_identity(self) -> bool:
        """Check if the random vector is the identity mapping on its domain.

        Returns
        -------
        is_identity : bool
            `True` if the random vector is the identity mapping, `False` otherwise.
        """
        return self._is_identity

    @property
    def range(self) -> ProbabilitySpace | None:
        r"""Return the range of a random vector as a probability space with the pushforward measure.

        See the Notes section below for the mathematical details.

        Examples
        --------
        >>> from sigalg.core import (
        ...     ProbabilityMeasure,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.2,
        ...         1: 0.7,
        ...         2: 0.1,
        ...     },
        ... )
        >>> X = RandomVector(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     prob_measure=P,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...     },
        ... )
        >>> print(X.range)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (X_range, power_set, P_X)
        ===========================================
        <BLANKLINE>
        * Sample space 'X_range':
         X_0  X_1
           1    2
           3    4
        <BLANKLINE>
        * Sigma algebra 'power_set':
                atom_ID
        X_0 X_1
        1   2    (1, 2)
        3   4    (3, 4)
        <BLANKLINE>
        * Probability measure 'P_X':
                probability
        X_0 X_1
        1   2            0.2
        3   4            0.8

        Notes
        -----
        Let $X: \Omega \to \mathbb{R}^d$ be a random vector on a probability space $(\Omega, \mathcal{F},P)$. The range

        $$
        X(\Omega) = \{ X(\omega) \in \mathbb{R}^d : \omega \in \Omega \}
        $$

        of the random vector is a probability space when equipped with the *pushforward measure* $P_X$ given by

        $$
        P_X(A) = P \left( \{\omega \in \Omega \mid X(\omega) \in A \} \right),
        $$

        for all events $A \subset X(\Omega)$. In SigAlg, the $\sigma$-algebra on $X(\Omega)$ defaults to the power set.
        """
        from ..spaces.probability_space import ProbabilitySpace
        from .operators import Operators

        if self._range is None and self.data is not None:
            if self.is_identity:
                self._range = self.prob_space
            else:
                pushforward = Operators.pushforward(self, self.prob_measure)
                self._range = ProbabilitySpace(prob_measure=pushforward)

        return self._range

    # --------------------- probability methods --------------------- #

    def sample(
        self,
        size: int = 1,
        random_state: int | np.random.Generator | None = None,
    ) -> pd.Series | pd.DataFrame:
        """Generate random samples from the range probability space of this random vector.

        Parameters
        ----------
        size : int, default=1
            Number of samples to generate.
        random_state : int | np.random.Generator | None, default=None
            An optional seed for the random number generator, or a `np.random.Generator` instance to use directly. If an integer is provided, a new generator is created with that seed. If a generator is provided, it is used directly. If `None`, the random number generator is not seeded.

        Returns
        -------
        sample : pd.Series | pd.DataFrame
            If the random vector is 1-dimensional, then a `pd.Series` is returned containing the random sample. Otherwise, if the random vector is multi-dimensional, a `pd.DataFrame` is returned whose rows contain the random sample and has columns indexed by the index of the random vector.

        Examples
        --------
        Generate a random probability space and sample from a 2-dimensional random vector.

        >>> import numpy as np
        >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace, SigmaAlgebra
        >>> rng = np.random.default_rng(42)
        >>> Omega = SampleSpace.from_sequence(size=10)
        >>> F = SigmaAlgebra.from_rand(
        ...     sample_space=Omega,
        ...     num_atoms=4,
        ...     random_state=rng,
        ... )
        >>> P = ProbabilityMeasure.from_rand(sig_alg=F, random_state=rng)
        >>> X = RandomVector.from_randint(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     prob_measure=P,
        ...     high=10,
        ...     dim=2,
        ...     random_state=rng,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        index   0  1
        sample
        0       2  6
        1       2  6
        2       1  7
        3       2  6
        4       7  3
        5       2  6
        6       0  9
        7       2  6
        8       0  9
        9       2  6
        >>> print(X.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, F, P)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
         sample
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
                atom_ID
        sample
        0             1
        1             1
        2             3
        3             1
        4             2
        5             1
        6             0
        7             1
        8             0
        9             1
        <BLANKLINE>
        * Probability measure 'P':
                 probability
        atom_ID
        1           0.049134
        3           0.207580
        2           0.082504
        0           0.660782
        >>> X_sample = X.sample(size=10, random_state=rng)
        >>> print(X_sample)  # doctest: +NORMALIZE_WHITESPACE
           X_0  X_1
        0    0    9
        1    0    9
        2    1    7
        3    0    9
        4    2    6
        5    1    7
        6    0    9
        7    0    9
        8    0    9
        9    7    3

        Sample from a 1-dimensional random variable.

        >>> Y = RandomVector.from_randint(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     prob_measure=P,
        ...     high=10,
        ...     random_state=rng,
        ...     name="Y",
        ... )
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'Y':
                Y
        sample
        0       9
        1       9
        2       3
        3       9
        4       0
        5       9
        6       4
        7       9
        8       4
        9       9
        >>> Y_sample = Y.sample(size=10, random_state=rng)
        >>> print(Y_sample)  # doctest: +NORMALIZE_WHITESPACE
        0    3
        1    3
        2    4
        3    3
        4    4
        5    4
        6    4
        7    4
        8    0
        9    4
        Name: Y, dtype: int64
        """
        if self.data is not None:
            return self.range.prob_measure.sample(size=size, random_state=random_state)
        else:
            raise ValueError("Cannot sample from an empty RandomVector instance.")

    def is_measurable(self, sig_alg: SigmaAlgebra | None = None) -> bool:
        r"""Check if the random vector is measurable with respect to a given sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        sig_alg : SigmaAlgebra | None, default=None
            The sigma-algebra to check measurability against. If `None`, checks measurability with respect to the sigma-algebra on the underlying probability space.

        Returns
        -------
        is_measurable : bool
            `True` if the random vector is measurable with respect to the given sigma-algebra, `False` otherwise.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ... )
        >>> X = RandomVector(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...         3: (3, 4),
        ...     },
        ... )
        >>> Y = RandomVector(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (5, 6),
        ...         3: (7, 8),
        ...     },
        ...     name="Y",
        ... )
        >>> print(X.is_measurable(F))
        True
        >>> print(Y.is_measurable(F))
        False

        Notes
        -----
        Let $(\Omega, \mathcal{F})$ be a measurable space and $X: \Omega \to \mathbb{R}^d$ a function. In the case that $\Omega$ is finite (as in SigAlg), the $\sigma$-algebra is determined by its atoms, and the function $X$ is said to be *$\mathcal{F}$-measurable* if $X$ is constant on the atoms of $\mathcal{F}$.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be a SigmaAlgebra or None.")
        if sig_alg is not None and sig_alg.sample_space != self.sample_space:
            raise ValueError(
                "The sample space of sig_alg must match the sample space of the random vector."
            )

        if sig_alg is None:
            sig_alg = self.sig_alg
        if sig_alg.is_power_set:
            return True

        df = pd.concat([self.data, sig_alg.data], axis=1).drop_duplicates()
        return len(df) == sig_alg.num_atoms

    def to_random_variable(self) -> RandomVariable:
        """Convert a 1-dimensional random vector to an instance of `RandomVariable`.

        Raises
        ------
        ValueError
            If the random vector has dimension > 1.

        Returns
        -------
        self : RandomVariable
            The converted instance from a `RandomVector` to a `RandomVariable`.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace.from_sequence(size=2)
        >>> X = RandomVector(sample_space=Omega, mapping={0: 1, 1: 2})
        >>> X_var = X.to_random_variable()
        >>> X_var # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X':
                X
        sample
        0       1
        1       2
        """
        from .random_variable import RandomVariable

        if self.dimension != 1:
            raise ValueError(
                "Can only convert a 1-dimensional RandomVector to RandomVariable."
            )

        if isinstance(self.data, pd.DataFrame):
            self._data = self.data.squeeze(axis=1)
            self._data.name = self.name
        self.__class__ = RandomVariable
        self._index = None

        return self

    def to_random_vector(self) -> RandomVector:
        """Return self.

        This method is implemented for consistency with the same method on the subclass `RandomVariable`.
        """
        return self

    def get_inverse_image(
        self, value: Hashable | tuple[Hashable] | pd.Series
    ) -> MeasurableSet:
        """Get the inverse image of a value under the random vector.

        Parameters
        ----------
        value : Hashable | tuple[Hashable] | pd.Series
            The value to find the inverse image of. If the random vector is 1-dimensional, `value` should be a Hashable. If the random vector is multi-dimensional, `value` should be a tuple of hashables or a `pd.Series` with an index matching the variable names of the random vector.

        Raises
        ------
        ValueError
            If `value` is not in the range of the random vector.

        Returns
        -------
        event : MeasurableSet
            The event in the sigma-algebra corresponding to the inverse image of `value` under the random vector.

        Examples
        --------
        Generate a 2-dimensional random vector on a random probability space.

        >>> import numpy as np
        >>> import pandas as pd
        >>> from sigalg.core import (
        ...     ProbabilityMeasure,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> rng = np.random.default_rng(42)
        >>> Omega = SampleSpace.from_sequence(size=10)
        >>> F = SigmaAlgebra.from_rand(
        ...     sample_space=Omega,
        ...     num_atoms=3,
        ...     random_state=rng,
        ... )
        >>> P = ProbabilityMeasure.from_rand(sig_alg=F, random_state=rng)
        >>> X = RandomVector.from_randint(
        ...     sample_space=Omega, sig_alg=F, prob_measure=P, dim=2, random_state=rng
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        index   0  1
        sample
        0       1  0
        1       1  0
        2       1  1
        3       1  0
        4       1  1
        5       0  0
        6       0  0
        7       1  0
        8       1  0
        9       1  0

        Get an inverse image using the `get_inverse_image` method.

        >>> inv_1 = X.get_inverse_image((1, 0))
        >>> print(inv_1)  # doctest: +NORMALIZE_WHITESPACE
        Measurable set '{X = (1, 0)}':
         sample
              0
              1
              3
              7
              8
              9

        Get an inverse image using the overloaded operator `==`.

        >>> inv_2 = X == (1, 1)
        >>> print(inv_2)  # doctest: +NORMALIZE_WHITESPACE
        Measurable set '{X = (1, 1)}':
         sample
              2
              4

        Get an inverse image using the overloaded operator `==` and a `pd.Series`.

        >>> s = pd.Series([0, 0], index=X.index)
        >>> inv_3 = X == s
        >>> print(inv_3)  # doctest: +NORMALIZE_WHITESPACE
        Measurable set '{X = (0, 0)}':
         sample
              5
              6
        """
        if not isinstance(value, (Hashable, tuple, pd.Series)):
            raise TypeError(
                "value must be a Hashable, tuple, or pd.Series corresponding to the output of the random vector."
            )

        if self.data is None:
            raise ValueError(
                "Cannot get inverse image of a random vector without outputs."
            )

        if isinstance(value, pd.Series):
            if not isinstance(self.data, pd.DataFrame):
                raise ValueError(
                    "The random vector is 1-dimensional, but the provided value is a pd.Series."
                )
            if not value.index.equals(self.index.data):
                raise ValueError(
                    "The index of the provided value does not match the index of the random vector."
                )
            value = tuple(value)
        if isinstance(value, tuple) and len(value) != self.dimension:
            raise ValueError(
                "The dimension of the provided value does not match the dimension of the random vector."
            )

        mask = (
            (self.data == value).all(axis=1)
            if isinstance(value, tuple)
            else self.data == value
        )
        name = f"{{{self.name} = {value}}}"

        return self.sig_alg.get_event(list(self.data.index[mask]), name=name)

    # --------------------- data methods --------------------- #

    def __call__(self, key: Hashable | MeasurableSet) -> Hashable | pd.Series:
        """Evaluate a random vector on a sample point or an atom in the sigma-algebra.

        Parameters
        ----------
        key : Hashable | MeasurableSet
            A sample point in the domain or an atom in the sigma-algebra of the random vector.

        Raises
        ------
        ValueError
            If the random vector has no outputs, or if `key` is not in the domain or the sigma-algebra of the random vector, or if `key` is an event that is not an atom in the sigma-algebra.
        TypeError
            If `key` is not a Hashable (i.e., a sample point) or an MeasurableSet (i.e., an atom in the sigma-algebra).

        Returns
        -------
        output : Hashable | FeatureVector
            If `key` is a sample point, returns the output of the random vector at that sample point. If `key` is an atom in the sigma-algebra, returns the output of the random vector on the atom.

        Examples
        --------
        Define a 2-dimensional random vector.

        >>> from sigalg.core import (
        ...     ProbabilityMeasure,
        ...     RandomVariable,
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: "a",
        ...         1: "b",
        ...         2: "b",
        ...         3: "c",
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         "a": 0.2,
        ...         "b": 0.5,
        ...         "c": 0.3,
        ...     },
        ... )
        >>> X = RandomVector(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     prob_measure=P,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...         3: (5, 6),
        ...     },
        ... )

        Call the random vector on a sample point.

        >>> print(X(0))  # doctest: +NORMALIZE_WHITESPACE
        index
        0    1
        1    2
        Name: 0, dtype: int64

        Call the random vector on an atom of the underlying sigma-algebra.

        >>> A = F.get_event([1, 2])
        >>> print(X(A))  # doctest: +NORMALIZE_WHITESPACE
        index
        0    3
        1    4
        Name: A, dtype: int64

        Define a 1-dimensional random variable.

        >>> Y = RandomVariable(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     prob_measure=P,
        ...     mapping={
        ...         0: 1,
        ...         1: 3,
        ...         2: 3,
        ...         3: 5,
        ...     },
        ...     name="Y",
        ... )

        Call the random variable on a sample point.

        >>> print(Y(0))
        1

        Call the random variable on the atom.

        >>> print(Y(A))
        3
        """
        from ..spaces.measurable_set import MeasurableSet

        if self.data is None:
            raise ValueError("Cannot evaluate a random vector without outputs.")

        if not isinstance(key, (Hashable, MeasurableSet)):
            raise TypeError(
                "key must be a Hashable (i.e., a sample point) or MeasurableSet (i.e., an atom in the sigma-algebra)."
            )

        if isinstance(key, MeasurableSet):
            if key not in self.sig_alg:
                raise ValueError(
                    "The provided event is not in the sigma-algebra of the random vector."
                )
            if not key.is_atom:
                raise ValueError(
                    "The provided event is not an atom in the sigma-algebra of the random vector."
                )
            sample_point = key[0]
            output_name = key.name
        else:
            if key not in self.sample_space:
                raise ValueError(
                    "The provided sample point is not in the domain of the random vector."
                )
            sample_point = key
            output_name = key

        result = self.data.loc[sample_point]

        if isinstance(result, pd.Series):
            return result.rename(output_name)
        else:
            return result

    def __iter__(self) -> Iterator[RandomVariable]:
        """Iterate over the components of the random vector.

        Returns
        -------
        iterator : Iterator[RandomVariable]
            An iterator over the components of the random vector.
        """
        return iter(self.components)

    def restrict_to(
        self,
        event: MeasurableSet | list,
        event_name: Hashable | None = "A",
    ) -> RandomVector:
        r"""Restrict the random vector to an event.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        event : MeasurableSet | list
            The event to restrict the random vector to.
        event_name : Hashable | None, default="A"
            The name to use for the event in the name of the resulting restricted random vector. This parameter is only used if `event` is a list of sample points, and is otherwise ignored if `event` is an `MeasurableSet` instance.

        Raises
        ------
        TypeError
            If `event` is not an `MeasurableSet` or a list of sample points.
        ValueError
            If `event` is not in the sigma-algebra of the random vector.

        Returns
        -------
        restricted_rv : RandomVector
            A new `RandomVector` representing the restriction of the original random vector to the given event.

        Examples
        --------
        Define a 2-dimensional random vector on a probability space.

        >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 2,
        ...     },
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.2,
        ...         1: 0.5,
        ...         2: 0.3,
        ...     },
        ... )
        >>> X = RandomVector(
        ...     sample_space=Omega,
        ...     sig_alg=F,
        ...     prob_measure=P,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (3, 4),
        ...         3: (5, 6),
        ...     },
        ... )

        Restrict the random variable to an event using the `restrict_to` method.

        >>> A = F.get_event([1, 2, 3])
        >>> X_A = X.restrict_to(A)
        >>> print(X_A)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X|A':
        index   0  1
        sample
        1       3  4
        2       3  4
        3       5  6
        >>> print(X_A.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (A, F_A, P_A)
        ===============================
        <BLANKLINE>
        * Sample space 'A':
         sample
              1
              2
              3
        <BLANKLINE>
        * Sigma algebra 'F_A':
                atom_ID
        sample
        1             1
        2             1
        3             2
        <BLANKLINE>
        * Probability measure 'P_A':
                 probability
        atom_ID
        1              0.625
        2              0.375

        Compute the same restriction using the overloaded `|` operator.

        >>> X_A = X | A
        >>> print(X_A)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X|A':
        index   0  1
        sample
        1       3  4
        2       3  4
        3       5  6

        Restrict the random vector using a `list` with a custom name.

        >>> X_B = X.restrict_to([1, 2], event_name="B")
        >>> print(X_B)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X|B':
        index   0  1
        sample
        1       3  4
        2       3  4
        >>> print(X_B.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (B, F_B, P_B)
        ===============================
        <BLANKLINE>
        * Sample space 'B':
         sample
              1
              2
        <BLANKLINE>
        * Sigma algebra 'F_B':
                atom_ID
        sample
        1             1
        2             1
        <BLANKLINE>
        * Probability measure 'P_B':
                 probability
        atom_ID
        1                1.0

        Notes
        -----
        Let $X: \Omega \to \mathbb{R}^d$ be a random vector on a probability space $(\Omega, \mathcal{F}, P)$. If $A\in \mathcal{F}$ is an event, then we may restrict the random vector to obtain the function $X|_A : A \to \mathbb{R}^d$ on $A$. If $A$ is an event of nonzero probability, then $A$ carries the conditional probability distribution $P_A$, defined so that $P_A(B) = P(B) / P(A)$, for $B\subset A$.
        """
        from ..spaces.measurable_set import MeasurableSet
        from ..spaces.probability_space import ProbabilitySpace

        if not isinstance(event, (MeasurableSet, list)):
            raise TypeError(
                "event must be an MeasurableSet or a list of sample points."
            )

        if isinstance(event, list):
            try:
                event = self.sig_alg.get_event(event, name=event_name)
            except ValueError as e:
                raise ValueError(
                    "MeasurableSet must be in the sigma-algebra of the random vector."
                ) from e
        elif isinstance(event, MeasurableSet) and event not in self.sig_alg:
            raise ValueError(
                "MeasurableSet must be in the sigma-algebra of the random vector."
            )

        event_prob_space = ProbabilitySpace.from_event(
            event=event, prob_measure=self.prob_measure
        )
        event_data = self.data.loc[event.data]
        event_data.index = event.data
        name = f"{self.name}|{event.name}"
        event_data.name = name
        result = RandomVector(*event_prob_space, mapping=event_data, name=name)

        if result.dimension == 1:
            result = result.to_random_variable()
            result.data.name = name

        return result

    def __getitem__(self, *args) -> RandomVariable:
        """Get a sub-vector of the random vector by selecting a collection of component random variables, or a single component random variable if only one index is provided.

        Calls `get_sub_vector` with the provided indices. See the documentation of that method for details.

        Parameters
        ----------
        *args : Hashable | tuple[Hashable]
            The indices of the component random variables to select for the sub-vector.

        Returns
        -------
        sub_vector : RandomVector
            A new `RandomVector` containing only the specified component random variables.
        """
        indices = list(*args) if isinstance(args[0], tuple) else list(args)
        return self.get_sub_vector(indices=indices)

    def get_sub_vector(self, indices: list[Hashable]) -> RandomVector:
        r"""Get a sub-vector of the random vector by selecting a collection of component random variables.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        indices : list[Hashable]
            List of feature indices to select for the sub-vector.

        Returns
        -------
        sub_vector : RandomVector
            A new `RandomVector` containing only the specified feature indices.

        Raises
        ------
        ValueError
            If any index is not found or if the random vector is 1-dimensional.

        Examples
        --------
        Define a 3-dimensional random vector.

        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace.from_sequence(size=2)
        >>> X = RandomVector(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: (1, 2, 3),
        ...         1: (4, 5, 6),
        ...     },
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        index   0  1  2
        sample
        0       1  2  3
        1       4  5  6

        Get a sub-vector by using the `get_sub_vector` method.

        >>> X_sub = X.get_sub_vector([1, 2])
        >>> print(X_sub)  # doctest: +NORMALIZE_WHITESPACE
        Random vector '(X_1, X_2)':
        index   1  2
        sample
        0       2  3
        1       5  6

        Get a sub-vector by using subscript notation.

        >>> X_sub = X[0, 1]
        >>> print(X_sub)  # doctest: +NORMALIZE_WHITESPACE
        Random vector '(X_0, X_1)':
        index   0  1
        sample
        0       1  2
        1       4  5

        Notes
        -----
        Given a random vector $X: \Omega \to \mathbb{R}^d$ on a probability space $(\Omega, \mathcal{F}, P)$, for each $\omega \in \Omega$ we may write

        $$
        X(\omega) = (X_1(\omega), X_2(\omega), \ldots, X_d(\omega)),
        $$

        where $X_j: \Omega \to \mathbb{R}$ are the component random variables of $X$. We may create a *sub-vector* by choosing a collection of the component random variables to get a random vector of smaller dimension. For example, we may select the first and last random variables to create the $2$-dimensional random vector

        $$
        \omega \mapsto (X_1 (\omega), X_d(\omega)).
        $$
        """
        from .random_variable import RandomVariable

        if self.dimension == 1:
            raise ValueError("Cannot get sub-vector of a 1-dimensional RandomVector.")
        invalid_features = [
            invalid_feature
            for invalid_feature in indices
            if invalid_feature not in self.index
        ]
        if invalid_features:
            raise ValueError(f"Feature indices {invalid_features} not found.")

        sub_data = self.data[indices]

        if len(indices) == 1:
            name = f"{self.name}_{indices[0]}".replace(".", "_")
            sub_vec = RandomVariable(*self.prob_space, mapping=sub_data, name=name)
        else:
            name = (
                "("
                + ", ".join([f"{self.name}_{idx}".replace(".", "_") for idx in indices])
                + ")"
            )

        sub_vec = RandomVector(*self.prob_space, mapping=sub_data, name=name)

        if sub_vec.dimension != 1:
            sub_vec._component_names = [
                f"{self.name}_{idx}".replace(".", "_") for idx in indices
            ]
            return sub_vec
        else:
            return sub_vec.to_random_variable()

    def get_component_rv(self, index: Hashable) -> RandomVariable:
        r"""Get a component random variable of the random vector.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        index : Hashable
            The feature index for which to get the component random variable.

        Returns
        -------
        component_rv : RandomVariable
            A new `RandomVariable` representing the component random variable.

        Examples
        --------
        Define a 3-dimensional random vector.

        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace.from_sequence(size=2)
        >>> X = RandomVector(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: (1, 2, 3),
        ...         1: (4, 5, 6),
        ...     },
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        index   0  1  2
        sample
        0       1  2  3
        1       4  5  6

        Get a component random variable using the `get_component_rv` method.

        >>> X_1 = X.get_component_rv(1)
        >>> print(X_1)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_1':
                X_1
        sample
        0         2
        1         5

        Get a component random variable using subscript notation.

        >>> X_0 = X[0]
        >>> print(X_0)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_0':
                X_0
        sample
        0         1
        1         4

        Notes
        -----
        Given a random vector $X: \Omega \to \mathbb{R}^d$ on a probability space $(\Omega, \mathcal{F}, P)$, for each $\omega \in \Omega$ we may write

        $$
        X(\omega) = (X_1(\omega), X_2(\omega), \ldots, X_d(\omega)),
        $$

        where $X_j: \Omega \to \mathbb{R}$ are the *component random variables* of $X$.
        """
        return self.get_sub_vector([index]).to_random_variable()

    def item(self) -> Hashable | pd.Series:
        """Get the output value of a constant random vector.

        Returns
        -------
        output : Hashable | FeatureVector
            The single output value of the random vector. If the dimension of the random vector is > 1, then the return value is an instance of `FeatureVector`; otherwise, the return value is a `Hashable`.

        Raises
        ------
        ValueError
            If the random vector is not constant.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace.from_sequence(size=2)
        >>> X = RandomVector(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...     },
        ... )
        >>> print(X.item())  # doctest: +NORMALIZE_WHITESPACE
        index
        0    1
        1    2
        dtype: int64
        >>> Y = RandomVector(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 1,
        ...         1: 1,
        ...     },
        ...     name="Y",
        ... )
        >>> print(Y.item())
        1
        """
        if self.data is None:
            raise ValueError("Cannot retrieve the item of an empty random vector.")

        if len(self.data.drop_duplicates()) != 1:
            raise ValueError("Can only retrive the item of a constant random vector.")

        item = self(self.sample_space[0])

        if isinstance(item, pd.Series):
            item.name = None

        return item

    def round(self, decimals: int = 0) -> RandomVector:
        """Round the feature vectors of the random vector to a specified number of decimal places.

        Parameters
        ----------
        decimals : int, default=0
            The number of decimal places to round to. Must be a non-negative integer.

        Raises
        ------
        ValueError
            If `decimals` is not a non-negative integer, or if the random vector's data is not set.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace.from_sequence(size=2)
        >>> mapping = dict(zip(Omega, [(0, np.pi), (np.pi / 2, 3 * np.pi / 2)]))
        >>> X = RandomVector(sample_space=Omega, mapping=mapping)
        >>> print(np.sin(X).round())  # doctest: +NORMALIZE_WHITESPACE
        Random vector 'sin(X)':
        index     0    1
        sample
        0       0.0  0.0
        1       1.0 -1.0
        """
        if not isinstance(decimals, int) or decimals < 0:
            raise ValueError("decimals must be a non-negative integer.")
        if self._data is None:
            raise ValueError("Data must be set to round the random vector.")

        self._data = self.data.round(decimals=decimals)

        return self

    # --------------------- equality --------------------- #

    def __eq__(
        self,
        other: RandomVector | Hashable | tuple[Hashable] | pd.Series,
        rtol=1e-5,
        atol=1e-8,
    ) -> bool:
        r"""Check equality with another random vector or compute an inverse image of a value under the random vector.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        other : RandomVector | Hashable | tuple[Hashable] | pd.Series
            Another random vector to compare with, or a value for which to compute the inverse image.
        rtol : float, default=1e-5
            The relative tolerance for comparing two random vectors. This is used only when `other` is a `RandomVector`.
        atol : float, default=1e-8
            The absolute tolerance for comparing two random vectors. This is used only when `other` is a `RandomVector`.

        Returns
        -------
        output : bool | MeasurableSet
            If `other` is a `RandomVector`, returns `True` if the two random vectors are equal, and `False` otherwise. If `other` is a value, returns the event corresponding to the inverse image of that value under the random vector.

        Notes
        -----
        Two random vector $X,Y: \Omega \to \mathbb{R}^d$ on the same probability space $(\Omega, \mathcal{F}, P)$ are equal if $X(\omega) = Y(\omega)$ for all $\omega \in \Omega$.
        """
        if not isinstance(other, RandomVector):
            try:
                return self.get_inverse_image(other)
            except TypeError as e:
                raise TypeError(
                    "If comparing a RandomVector to a non-RandomVector, the other object must be a Hashable, tuple[Hashable], or pd.Series corresponding to a possible output of the random vector."
                ) from e

        if self.sample_space != other.sample_space:
            raise ValueError(
                "Cannot test equality of two random vectors defined on different sample spaces."
            )
        if self.index != other.index:
            raise ValueError(
                "Cannot test equality of two random vectors with different feature indices."
            )
        return np.allclose(
            self.data.to_numpy(), other.data.to_numpy(), rtol=rtol, atol=atol
        )

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a string representation of the random vector.

        Returns
        -------
        repr_str : str
            The string representation of the random vector.
        """
        if self.data is None:
            return f"{type(self)._repr_name} '{self.name}': empty"
        else:
            if isinstance(self.data, pd.Series):
                data = self.data.to_frame()
                data.columns = [self.name]
            else:
                data = self.data

            return f"{type(self)._repr_name} '{self.name}':\n{data}"

    # --------------------- arithmetic operations --------------------- #

    @staticmethod
    def _type(x):
        from ...processes.base.stochastic_process import StochasticProcess

        if isinstance(x, Real):
            return "Number"
        elif isinstance(x, StochasticProcess):
            return "StochasticProcess"
        else:
            return type(x).__name__

    def _apply_operation(
        self,
        other: RandomVector | Real,
        operation: Callable,
        op_symbol: str,
        reverse: bool = False,
    ) -> RandomVector:
        """Apply a binary operation to this random vector.

        Parameters
        ----------
        self : RandomVector
            The left operand (or right if reverse=True).
        other : RandomVector | Real
            The right operand (or left if reverse=True).
        operation : Callable
            The pandas operation to apply (e.g., `lambda a, b: a + b`).
        op_symbol : str
            Symbol representing the operation (e.g., '+', '-', '*').
        reverse : bool, default=False
            Whether this is a reverse operation (e.g., __radd__ vs __add__).

        Returns
        -------
        result : RandomVector
            A new random vector representing the result of the operation.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector` or a scalar.
        ValueError
            If operating on two `RandomVector` instances with different domains or dimensions.
        """
        from ...processes.base.stochastic_process import StochasticProcess
        from .random_variable import RandomVariable

        types = {self._type(self), self._type(other)}

        if types == {"RandomVariable"} or types == {"RadonNikodym", "RandomVariable"}:
            if self.prob_space.is_subspace(other.prob_space):
                super_space = other.prob_space
            elif other.prob_space.is_subspace(self.prob_space):
                super_space = self.prob_space
            else:
                raise ValueError(
                    f"Cannot {op_symbol} RandomVariables on incompatible probability spaces."
                )

            if reverse:
                new_name = f"({other.name}{op_symbol}{self.name})"
                new_values = operation(other.data, self.data).rename(new_name)
            else:
                new_name = f"({self.name}{op_symbol}{other.name})"
                new_values = operation(self.data, other.data).rename(new_name)

            return RandomVariable(*super_space, mapping=new_values, name=new_name)

        elif types == {"StochasticProcess"}:
            if self.prob_space.is_subspace(other.prob_space):
                super_space = other.prob_space
            elif other.prob_space.is_subspace(self.prob_space):
                super_space = self.prob_space
            else:
                raise ValueError(
                    f"Cannot {op_symbol} StochasticProcesses on incompatible probability spaces."
                )
            if len(self) != len(other):
                raise ValueError(
                    "The length of the StochasticProcesses must be the same."
                )
            if self.time != other.time:
                raise ValueError(
                    "The time indices of the StochasticProcesses must be the same"
                )

            if reverse:
                new_name = (
                    f"({other.name}{op_symbol}{self.name})"
                    if self.name is not None and other.name is not None
                    else None
                )
                new_values = operation(other.data, self.data).rename(new_name)
            else:
                new_name = (
                    f"({self.name}{op_symbol}{other.name})"
                    if self.name is not None and other.name is not None
                    else None
                )
                new_values = operation(self.data, other.data).rename(new_name)

            result = StochasticProcess(
                *super_space,
                name=new_name,
                time=self.time,
                is_discrete_state=self.is_discrete_state,
            ).from_pandas(data=new_values)

            return result

        elif types == {"RandomVector"}:
            if self.prob_space.is_subspace(other.prob_space):
                super_space = other.prob_space
            elif other.prob_space.is_subspace(self.prob_space):
                super_space = self.prob_space
            else:
                raise ValueError(
                    f"Cannot {op_symbol} RandomVectors on incompatible probability spaces."
                )
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")

            self_data = self.data.copy()
            other_data = other.data.copy()
            if self.dimension > 1:
                self_data.columns = pd.RangeIndex(self.dimension)
                other_data.columns = pd.RangeIndex(other.dimension)

            if reverse:
                new_name = f"({other.name}{op_symbol}{self.name})"
                new_values = operation(other_data, self_data)
            else:
                new_name = f"({self.name}{op_symbol}{other.name})"
                new_values = operation(self_data, other_data)

            return RandomVector(*super_space, mapping=new_values, name=new_name)

        elif types == {"Number", "RandomVariable"}:
            if reverse:
                new_name = f"({other}{op_symbol}{self.name})"
                new_values = operation(other, self.data).rename(new_name)
            else:
                new_name = f"({self.name}{op_symbol}{other})"
                new_values = operation(self.data, other).rename(new_name)

            return RandomVariable(*self.prob_space, mapping=new_values, name=new_name)

        elif types == {"Number", "StochasticProcess"}:
            if reverse:
                new_name = (
                    f"({other}{op_symbol}{self.name})"
                    if self.name is not None
                    else None
                )
                new_values = operation(other, self.data)
            else:
                new_name = (
                    f"({self.name}{op_symbol}{other})"
                    if self.name is not None
                    else None
                )
                new_values = operation(self.data, other)

            result = StochasticProcess(
                *self.prob_space,
                name=new_name,
                mapping=new_values,
                index=self.time,
            )

            return result

        elif types == {"Number", "RandomVector"}:
            if reverse:
                new_name = f"({other}{op_symbol}{self.name})"
                new_values = operation(other, self.data)
            else:
                new_name = f"({self.name}{op_symbol}{other})"
                new_values = operation(self.data, other)

            return RandomVector(*self.prob_space, mapping=new_values, name=new_name)

        elif types == {"RandomVariable", "RandomVector"}:
            raise TypeError(f"Unsupported types for arithmetic operations: {types}")

        elif types == {"RandomVariable", "StochasticProcess"}:
            if self.prob_space.is_subspace(other.prob_space):
                super_space = other.prob_space
            elif other.prob_space.is_subspace(self.prob_space):
                super_space = self.prob_space
            else:
                raise ValueError(
                    f"Cannot {op_symbol} a RandomVariable with a StochasticProcess on different probability spaces."
                )

            if self._type(self) == "RandomVariable":
                if reverse:
                    new_name = f"({other.name}{op_symbol}{self.name})"
                    new_values = operation(
                        other.data, self.data.values.reshape(-1, 1)
                    ).rename(new_name)
                else:
                    new_name = f"({self.name}{op_symbol}{other.name})"
                    new_values = operation(
                        self.data.values.reshape(-1, 1), other.data
                    ).rename(new_name)

                result = StochasticProcess(
                    *self.prob_space, name=new_name, time=other.time
                ).from_pandas(data=new_values)

            else:
                if reverse:
                    new_name = f"({other.name}{op_symbol}{self.name})"
                    new_values = operation(
                        other.data.values.reshape(-1, 1), self.data
                    ).rename(new_name)
                else:
                    new_name = f"({self.name}{op_symbol}{other.name})"
                    new_values = operation(
                        self.data, other.data.values.reshape(-1, 1)
                    ).rename(new_name)

                result = StochasticProcess(
                    *self.prob_space, name=new_name, time=other.time
                ).from_pandas(data=new_values)

            return result

        elif types == {"RandomVector", "StochasticProcess"}:
            raise TypeError(f"Unsupported types for arithmetic operations: {types}")

        else:
            raise TypeError(f"Unsupported types for arithmetic operations: {types}")

    def __add__(self, other: RandomVector | Real) -> RandomVector:
        """Add another random vector or a scalar to this random vector."""
        return self._apply_operation(other, lambda a, b: a + b, "+")

    def __radd__(self, other: RandomVector | Real) -> RandomVector:
        """Add another random vector or a scalar to this random vector (right-hand side)."""
        return self._apply_operation(other, lambda a, b: a + b, "+", reverse=True)

    def __sub__(self, other: RandomVector | Real) -> RandomVector:
        """Subtract another random vector or a scalar from this random vector."""
        return self._apply_operation(other, lambda a, b: a - b, "-")

    def __rsub__(self, other: RandomVector | Real) -> RandomVector:
        """Subtract this random vector from another random vector or a scalar (right-hand side)."""
        return self._apply_operation(other, lambda a, b: a - b, "-", reverse=True)

    def __mul__(self, other: RandomVector | Real) -> RandomVector:
        """Multiply this random vector by another random vector or a scalar."""
        return self._apply_operation(other, lambda a, b: a * b, "*")

    def __rmul__(self, other: RandomVector | Real) -> RandomVector:
        """Multiply another random vector or a scalar by this random vector (right-hand side)."""
        return self._apply_operation(other, lambda a, b: a * b, "*", reverse=True)

    def __truediv__(self, other: RandomVector | Real) -> RandomVector:
        """Divide this random vector by another random vector or a scalar."""
        return self._apply_operation(other, lambda a, b: a / b, "/")

    def __rtruediv__(self, other: RandomVector | Real) -> RandomVector:
        """Divide another random vector or a scalar by this random vector (right-hand side)."""
        return self._apply_operation(other, lambda a, b: a / b, "/", reverse=True)

    def __pow__(self, other: RandomVector | Real) -> RandomVector:
        """Exponentiate this random vector by another random vector or a scalar."""
        return self._apply_operation(other, lambda a, b: a**b, "**")

    def __rpow__(self, other: RandomVector | Real) -> RandomVector:
        """Exponentiate another random vector or a scalar by this random vector (right-hand side)."""
        return self._apply_operation(other, lambda a, b: a**b, "**", reverse=True)

    def __array_ufunc__(
        self, ufunc, method, *inputs, **kwargs
    ) -> RandomVector | StochasticProcess | RandomVariable:
        """Override NumPy ufuncs to operate on RandomVector instances.

        Parameters
        ----------
        ufunc : numpy.ufunc
            The ufunc object that was called.
        method : str
            A string indicating which ufunc method was called (e.g., '__call__', 'reduce', etc.).
        inputs : tuple
            A tuple of the input arguments to the ufunc.
        kwargs : dict
            A dictionary of keyword arguments passed to the ufunc.

        Returns
        -------
        result : RandomVector | StochasticProcess | RandomVariable
            A new instance of `RandomVector`, `StochasticProcess`, or `RandomVariable`
            containing the result of applying the ufunc to the inputs.
        """
        from ...processes.base.stochastic_process import StochasticProcess
        from .random_variable import RandomVariable

        if method != "__call__":
            return NotImplemented

        new_inputs = [
            input.data if isinstance(input, RandomVector) else input for input in inputs
        ]
        result_data = getattr(ufunc, method)(*new_inputs, **kwargs)
        new_name = f"{ufunc.__name__}({self.name})" if self.name is not None else None

        if isinstance(self, StochasticProcess):
            return StochasticProcess(
                *self.prob_space, name=new_name, time=self.time
            ).from_pandas(data=result_data)

        elif isinstance(self, RandomVariable):
            result_data.name = None
            return RandomVariable(*self.prob_space, mapping=result_data, name=new_name)

        else:
            return RandomVector(*self.prob_space, mapping=result_data, name=new_name)

    # --------------------- comparison methods --------------------- #

    def __bool__(self) -> bool:
        """Prevent ambiguous boolean conversion of a random vector.

        Raises
        ------
        ValueError
            Always raised to prevent ambiguous boolean evaluation.
            Use explicit methods like .all() or .any() instead.
        """
        raise ValueError(
            "The truth value of a RandomVector is ambiguous. "
            "Use .all() or .any() methods, or check specific conditions explicitly."
        )

    def all(self) -> bool:
        """Check if all values in the random vector are True.

        This method is typically used after a comparison operation to verify
        that the comparison holds for all sample points and all components.

        Returns
        -------
        all_true : bool
            True if all values across all samples and features are True.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace.from_sequence(size=2)
        >>> X = RandomVector(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: (1, 1),
        ...         1: (1, 1),
        ...     },
        ... )
        >>> print(X.all())
        True
        >>> Y = RandomVector(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: (1, 0),
        ...         1: (0, 1),
        ...     },
        ...     name="Y",
        ... )
        >>> print(Y.all())
        False
        """
        return bool(self.data.all().all() if self.dimension > 1 else self.data.all())

    def any(self) -> bool:
        """Check if any value in the random vector is True.

        This method is typically used after a comparison operation to verify
        that the comparison holds for at least one sample point or component.

        Returns
        -------
        any_true : bool
            True if any value across all samples and features is True.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace.from_sequence(size=2)
        >>> X = RandomVector(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: (0, 1),
        ...         1: (1, 0),
        ...     },
        ... )
        >>> print(X.any())
        True
        >>> Y = RandomVector(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: (0, 0),
        ...         1: (0, 0),
        ...     },
        ...     name="Y",
        ... )
        >>> print(Y.any())
        False
        """
        return bool(self.data.any().any() if self.dimension > 1 else self.data.any())

    def _apply_comparison(
        self,
        other: RandomVector | Real,
        op: Callable,
        op_symbol: str,
    ) -> RandomVector:
        """Apply a comparison operation to this random vector.

        Parameters
        ----------
        other : RandomVector | Real
            The random vector or scalar to compare with.
        op : Callable
            The numpy comparison to apply (e.g., ``operator.lt``).
        op_symbol : str
            Symbol representing the comparison (e.g., '<', '<=', '>', '>=').

        Returns
        -------
        result : RandomVector
            A new random vector of booleans representing the comparison result.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector` or scalar.
        ValueError
            If the random vectors do not have the same domain or dimension.
        """
        from ...processes.base.stochastic_process import StochasticProcess
        from .random_variable import RandomVariable

        if not isinstance(other, RandomVector) and isinstance(other, Real):
            other = RandomVector.from_constant(
                *self.prob_space, index=self.index, name=other, constant=other
            )

        elif not isinstance(other, RandomVector):
            raise TypeError("other must be a RandomVector")

        if self.sample_space != other.sample_space:
            raise ValueError("Random vectors must have the same sample_space")
        if self.dimension != other.dimension:
            raise ValueError("Random vectors must have the same dimension")

        comparison_arr = op(self.data.to_numpy(), other.data.to_numpy())
        name = (
            f"({self.name} {op_symbol} {other.name})"
            if self.name and other.name
            else None
        )

        if isinstance(self, StochasticProcess):
            return StochasticProcess(
                *self.prob_space, name=name, time=self.time
            ).from_numpy(array=comparison_arr)

        elif isinstance(self, RandomVariable):
            result = RandomVariable(
                *self.prob_space, name=name, mapping=comparison_arr.flatten()
            )
            result.data.name = name
            return result

        else:
            return RandomVector(*self.prob_space, name=name, mapping=comparison_arr)

    def __lt__(self, other: RandomVector | Real) -> RandomVector:
        r"""Check if this random vector is less than another random vector or scalar.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        other : RandomVector | Real
            The random vector or scalar to compare with.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector`.
        ValueError
            If the random vectors do not have the same domain or dimension.

        Returns
        -------
        is_lt: RandomVector
            A new `RandomVector` of booleans indicating where this random vector is less than the other random vector or scalar.

        Notes
        -----
        Let $X,Y: \Omega \to \mathbb{R}^d$ be two random vectors defined on a probability space $(\Omega, \mathcal{F},P)$, with component random variables

        $$
        X = (X_1, X_2,\ldots,X_d) \quad \text{and} \quad Y = (Y_1, Y_2, \ldots,Y_d).
        $$

        We define a third random variable $Z: \Omega \to \mathbb{R}^d$ with components

        $$
        Z = (Z_1, Z_2, \ldots, Z_d)
        $$

        such that $Z_j(\omega) = 1$ if $X_j(\omega) < Y_j(\omega)$, and $Z_j(\omega)=0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $Y$ is `other`.

        If $c$ is a scalar, then we define $Z$ by setting $Z_j(\omega) = 1$ if $X_j(\omega) < c$, and $Z_j(\omega) = 0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $c$ is `other`.
        """
        import operator

        return self._apply_comparison(other, operator.lt, "<")

    def __le__(self, other: RandomVector | Real) -> RandomVector:
        r"""Check if this random vector is less than or equal to another random vector or scalar.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        other : RandomVector | Real
            The random vector or scalar to compare with.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector`.
        ValueError
            If the random vectors do not have the same domain or dimension.

        Returns
        -------
        is_le: RandomVector
            A new `RandomVector` of booleans indicating where this random vector is less than or equal to the other random vector or scalar.

        Notes
        -----
        Let $X,Y: \Omega \to \mathbb{R}^d$ be two random vectors defined on a probability space $(\Omega, \mathcal{F},P)$, with component random variables

        $$
        X = (X_1, X_2,\ldots,X_d) \quad \text{and} \quad Y = (Y_1, Y_2, \ldots,Y_d).
        $$

        We define a third random variable $Z: \Omega \to \mathbb{R}^d$ with components

        $$
        Z = (Z_1, Z_2, \ldots, Z_d)
        $$

        such that $Z_j(\omega) = 1$ if $X_j(\omega) \leq Y_j(\omega)$, and $Z_j(\omega)=0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $Y$ is `other`.

        If $c$ is a scalar, then we define $Z$ by setting $Z_j(\omega) = 1$ if $X_j(\omega) \leq c$, and $Z_j(\omega) = 0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $c$ is `other`.
        """
        import operator

        return self._apply_comparison(other, operator.le, "<=")

    def __gt__(self, other: RandomVector | Real) -> RandomVector:
        r"""Check if this random vector is greater than another random vector or scalar.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        other : RandomVector | Real
            The random vector or scalar to compare with.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector`.
        ValueError
            If the random vectors do not have the same domain or dimension.

        Returns
        -------
        is_gt: RandomVector
            A new `RandomVector` of booleans indicating where this random vector is greater than the other random vector or scalar.

        Notes
        -----
        Let $X,Y: \Omega \to \mathbb{R}^d$ be two random vectors defined on a probability space $(\Omega, \mathcal{F},P)$, with component random variables

        $$
        X = (X_1, X_2,\ldots,X_d) \quad \text{and} \quad Y = (Y_1, Y_2, \ldots,Y_d).
        $$

        We define a third random variable $Z: \Omega \to \mathbb{R}^d$ with components

        $$
        Z = (Z_1, Z_2, \ldots, Z_d)
        $$

        such that $Z_j(\omega) = 1$ if $X_j(\omega) > Y_j(\omega)$, and $Z_j(\omega)=0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $Y$ is `other`.

        If $c$ is a scalar, then we define $Z$ by setting $Z_j(\omega) = 1$ if $X_j(\omega) > c$, and $Z_j(\omega) = 0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $c$ is `other`.
        """
        import operator

        return self._apply_comparison(other, operator.gt, ">")

    def __ge__(self, other: RandomVector | Real) -> RandomVector:
        r"""Check if this random vector is greater than or equal another random vector or scalar.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        other : RandomVector | Real
            The random vector or scalar to compare with.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector`.
        ValueError
            If the random vectors do not have the same domain or dimension.

        Returns
        -------
        is_ge: RandomVector
            A new `RandomVector` of booleans indicating where this random vector is greater than or equal the other random vector or scalar.

        Notes
        -----
        Let $X,Y: \Omega \to \mathbb{R}^d$ be two random vectors defined on a probability space $(\Omega, \mathcal{F},P)$, with component random variables

        $$
        X = (X_1, X_2,\ldots,X_d) \quad \text{and} \quad Y = (Y_1, Y_2, \ldots,Y_d).
        $$

        We define a third random variable $Z: \Omega \to \mathbb{R}^d$ with components

        $$
        Z = (Z_1, Z_2, \ldots, Z_d)
        $$

        such that $Z_j(\omega) = 1$ if $X_j(\omega) \geq Y_j(\omega)$, and $Z_j(\omega)=0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $Y$ is `other`.

        If $c$ is a scalar, then we define $Z$ by setting $Z_j(\omega) = 1$ if $X_j(\omega) \geq c$, and $Z_j(\omega) = 0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $c$ is `other`.
        """
        import operator

        return self._apply_comparison(other, operator.ge, ">=")
