import pandas as pd  # noqa: D100

from ..probability_measures.probability_measure import ProbabilityMeasure
from .random_vector import RandomVector


def pushforward(
    rv: RandomVector,
    probability_measure: ProbabilityMeasure | None = None,
) -> ProbabilityMeasure:
    """Push forward a probability measure on the domain of a random vector to a probability measure on its range.

    Given a random vector `X: Omega -> S` and a probability measure `P`
    on `Omega`, constructs the probability measure `P_X` on the range `X.range`.

    Parameters
    ----------
    rv : RandomVector
        Random vector.
    probability_measure : ProbabilityMeasure | None, default=None
        Probability measure `P` defining the probabilities on the domain sample space. If `None`, the uniform probability measure on the domain is used.

    Raises
    ------
    TypeError
        If `rv` is not a `RandomVector`, or if `probability_measure` is not a `ProbabilityMeasure` (if given).
    ValueError
        If `rv` is not defined on the sample space of `probability_measure` (if given).

    Returns
    -------
    pushforward_measure : ProbabilityMeasure
        The resulting probability measure `P_X`.

    Examples
    --------
    >>> import pandas as pd
    >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace, pushforward
    >>> domain = SampleSpace.generate_sequence(size=3)
    >>> X = RandomVector(domain=domain).from_dict(
    ...     {"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (3, 4)},
    ... )
    >>> print(X) # doctest: +NORMALIZE_WHITESPACE
    Random vector 'X':
    feature  X_0  X_1
    sample
    omega_0    1   2
    omega_1    3   4
    omega_2    3   4
    >>> prob_measure = ProbabilityMeasure(sample_space=domain).from_dict(
    ...     {"omega_0": 0.2, "omega_1": 0.5, "omega_2": 0.3},
    ... )
    >>> P_X = pushforward(probability_measure=prob_measure, rv=X)
    >>> X_range = X.range
    >>> print(pd.concat([X_range.data, P_X.data], axis=1)) # doctest: +NORMALIZE_WHITESPACE
            X_0  X_1  probability
    output
    x_0       1   2          0.2
    x_1       3   4          0.8
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
    pushforward_measure = ProbabilityMeasure(name=measure_name).from_pandas(
        pushforward_probs.iloc[:, -1]
    )

    return pushforward_measure
