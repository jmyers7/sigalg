"""A class representing a random vector."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .measurable_vector import MeasurableVector

if TYPE_CHECKING:
    from ...validation.index_validator import IndexLike
    from ...validation.mapping_validator import MappingLike
    from ..indices.index import Index
    from ..measures.probability_measure import ProbabilityMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from ..spaces.domain import Domain
    from ..spaces.probability_space import ProbabilitySpace
    from ..spaces.sample_space import SampleSpace


class RandomVector(MeasurableVector):
    r"""A class representing a random vector.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    domain : Domain | IndexLike | None, default=None
        The sample space of the underlying probability space.
    sig_alg : SigmaAlgebra | None, default=None
        The sigma algebra of the underlying probability space.
    measure : ProbabilityMeasure | None, default=None
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

    Generate a random vector on a pre-existing event space. A uniform probability measure is automatically generated

    >>> F = SigmaAlgebra(
    ...     domain=Omega,
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
    >>> print(Y.measure)  # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'U':
             probability
    atom_ID
    0                0.5
    1                0.5


    Generate a random vector on a pre-existing probability space

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
    >>> V = RandomVector(
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
    Given a probability space $(\Omega,\mathcal{F},P)$, a *random vector* is an $\mathcal{F}$-measurable function $X: \Omega \to \mathbb{R}^d$, where $d$ is the *dimension* of the vector and $\mathbb{R}^d$ is equipped with its Borel $\sigma$-algebra. The image $X(\omega)\in \mathbb{R}^d$ of a sample point $\omega \in \Omega$ is called a *feature vector*.

    If $\Omega$ is finite (as it always is, in SigAlg), so that $\mathcal{F}$ is determined by its atoms, then $X$ is $\mathcal{F}$-measurable if and only if $X$ is constant on the atoms of $\mathcal{F}$.
    """

    _repr_name = "Random vector"
    _default_name = "X"

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        domain: SampleSpace | IndexLike | None = None,
        sig_alg: SigmaAlgebra | None = None,
        measure: ProbabilityMeasure | None = None,
        mapping: MappingLike | Callable | None = None,
        index: Index | IndexLike | None = None,
        name: Hashable = "X",
    ) -> None:
        from ..measures.probability_measure import ProbabilityMeasure

        if measure is not None and not isinstance(measure, ProbabilityMeasure):
            raise ValueError("If given, measure must be a probability measure.")

        super().__init__(
            domain=domain,
            sig_alg=sig_alg,
            measure=measure,
            mapping=mapping,
            index=index,
            name=name,
        )

    # --------------------- properties --------------------- #

    @property
    def sample_space(self) -> Domain | None:
        """Get the sample space of the underlying probability space.

        This property is an alias for the `domain` property of instances of `MeasurableVector`.

        Returns
        -------
        sample_space : Domain | None
            The sample space in the underlying probability space.
        """
        return self.domain

    @property
    def prob_space(self) -> ProbabilitySpace | None:
        """Get the underlying probability space.

        This property is an alias for the `measure_space` property of instances of `MeasurableVector`.

        Returns
        -------
        prob_space : ProbabilitySpace | None
            The underlying probability space.
        """
        return self.measure_space

    @property
    def prob_measure(self) -> ProbabilityMeasure | None:
        """Get the measure of the underlying probability space.

        This property is an alias for the `measure` property of instances of `MeasurableVector`.

        Returns
        -------
        prob_measure : ProbabilityMeasure | None
            The underlying probability measure.
        """
        return self.measure

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
            An optional seed for the random number generator.

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
        ...     domain=Omega,
        ...     num_atoms=4,
        ...     random_state=rng,
        ... )
        >>> P = ProbabilityMeasure.from_rand(domain=F, random_state=rng)
        >>> X = RandomVector.from_randint(
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
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
        >>> print(X.measure_space)  # doctest: +NORMALIZE_WHITESPACE
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
        ...     domain=Omega,
        ...     sig_alg=F,
        ...     measure=P,
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
            return self.range.measure.sample(size=size, random_state=random_state)
        else:
            raise ValueError("Cannot sample from an empty RandomVector instance.")
