import pandas as pd  # noqa: D100

from ..featurized_spaces.featurized_probability_space import (
    FeaturizedProbabilitySpace,
)
from ..probability_measures.probability_measure import ProbabilityMeasure
from .random_vector import RandomVector


def pushforward(
    rv: RandomVector,
    probability_measure: ProbabilityMeasure | None = None,
) -> FeaturizedProbabilitySpace:
    """Create a featurized probability space from the range of a random vector and the pushforward of a probability measure along the random vector.

    Given a random vector `X: Omega -> S` and a probability measure `P`
    on `Omega`, constructs the featurized probability space `(range(X), F, P_X, X_range)`, where `range(X)` is the range of `X`, `F` is the power-set sigma-algebra on `range(X)`, `P_X` is the pushforward measure of `P` under `X`, and `X_range` is the feature embedding mapping each index in `range(X)` to a feature vector in the range of `X`.

    Parameters
    ----------
    rv : RandomVector
        Random vector.
    probability_measure : ProbabilityMeasure | None, default=None
        Probability measure `P` defining the probabilities on the sample space. If `None`, the uniform probability measure on the domain is used.

    Raises
    ------
    TypeError
        If `rv` is not a `RandomVector`, or if `probability_measure` is not a `ProbabilityMeasure` (if given).
    ValueError
        If `rv` is not defined on the sample space of `probability_measure` (if given).

    Returns
    -------
    fps : FeaturizedProbabilitySpace
        The resulting featurized probability space `(range(X), F, P_X, X_range)`.

    Examples
    --------
    >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace, pushforward
    >>> domain = SampleSpace.generate_default(size=3)
    >>> X = RandomVector(
    ...     outputs={"omega0": (1, 2), "omega1": (3, 4), "omega2": (3, 4)},
    ...     domain=domain,
    ...     name="X",
    ... )
    >>> print(X) # doctest: +NORMALIZE_WHITESPACE
    Random vector 'X':
    feature  X0  X1
    sample
    omega0    1   2
    omega1    3   4
    omega2    3   4
    >>> prob_measure = ProbabilityMeasure(
    ...     probabilities={"omega0": 0.2, "omega1": 0.5, "omega2": 0.3},
    ...     name="P",
    ...     sample_space=domain,
    ... )
    >>> print(pushforward(probability_measure=prob_measure, rv=X)) # doctest: +NORMALIZE_WHITESPACE
    Featurized probability space (range(X), power_set, P_X, X_range)
    ================================================================
    <BLANKLINE>
    * Sample space 'range(X)':
    ['x0', 'x1']
    <BLANKLINE>
    * Sigma algebra 'power_set':
            atom ID
    output
    x0            0
    x1            1
    <BLANKLINE>
    * Probability measure 'P_X':
            probability
    output
    x0              0.2
    x1              0.8
    <BLANKLINE>
    * Random vector 'X_range':
    feature  X0  X1
    output
    x0        1   2
    x1        3   4
    >>> Y = RandomVector(
    ...     outputs={"omega0": 1, "omega1": 2, "omega2": 2}, domain=domain, name="Y"
    ... )
    >>> print(Y) # doctest: +NORMALIZE_WHITESPACE
    Random vector 'Y':
        Y
    sample
    omega0  1
    omega1  2
    omega2  2
    >>> print(pushforward(probability_measure=prob_measure, rv=Y)) # doctest: +NORMALIZE_WHITESPACE
    Featurized probability space (range(Y), power_set, P_Y, Y_range)
    ================================================================
    <BLANKLINE>
    * Sample space 'range(Y)':
    ['y0', 'y1']
    <BLANKLINE>
    * Sigma algebra 'power_set':
            atom ID
    output
    y0            0
    y1            1
    <BLANKLINE>
    * Probability measure 'P_Y':
            probability
    output
    y0              0.2
    y1              0.8
    <BLANKLINE>
    * Random vector 'Y_range':
            Y_range
    output
    y0            1
    y1            2
    """
    from ..probability_measures.probability_measure import ProbabilityMeasure
    from ..random_objects.random_vector import RandomVector

    if not isinstance(rv, RandomVector):
        raise TypeError("rv must be a RandomVector instance.")
    if probability_measure is not None and not isinstance(
        probability_measure, ProbabilityMeasure
    ):
        raise TypeError("probability_measure must be a ProbabilityMeasure instance.")
    if (
        probability_measure is not None
        and rv.domain != probability_measure.sample_space
    ):
        raise ValueError(
            "rv must be defined on the sample space of probability_measure."
        )

    if probability_measure is None:
        probability_measure = ProbabilityMeasure.uniform(sample_space=rv.domain)

    if rv.dimension == 1:
        rv_cols = [rv.data.name]
    else:
        rv_cols = rv.data.columns.tolist()
    pushforward_probs = (
        pd.concat([rv.data, probability_measure.data], axis=1).groupby(rv_cols).sum()
    )
    pushforward_probs.index = rv.range.data.index
    measure_name = f"P_{rv.name}" if rv.name is not None else None
    pushforward_measure = ProbabilityMeasure.from_pandas(
        data=pushforward_probs.iloc[:, -1], name=measure_name
    )

    return FeaturizedProbabilitySpace(
        sample_space=rv.range.domain,
        feature_embedding=rv.range,
        probability_measure=pushforward_measure,
    )
