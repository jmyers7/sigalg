from numbers import Real  # noqa: D100

import pandas as pd

from ..probability_measures.probability_measure import ProbabilityMeasure
from ..sigma_algebras.sigma_algebra import SigmaAlgebra
from .random_vector import RandomVector


def expectation(
    rv: RandomVector,
    sigma_algebra: SigmaAlgebra | None = None,
    probability_measure: ProbabilityMeasure | None = None,
) -> RandomVector | pd.Series | Real:
    """Compute the expectation of a `RandomVector` with respect to a `ProbabilityMeasure`, optionally conditioned on a `SigmaAlgebra`.

    If the sigma algebra is given and contains an atom of probability 0, the expected value is defined to be 0 on this atom.

    Parameters
    ----------
    rv : RandomVector
        The random vector for which to compute the expectation.
    sigma_algebra : SigmaAlgebra | None, default=None
        The sigma algebra to condition on. If `None`, computes the unconditional expectation.
    probability_measure : ProbabilityMeasure | None, default=None
        The probability used to compute the expectation. If `None`, the probability measure carried by the random vector is used (accessed through its `probability_measure` attribute).

    Raises
    ------
    TypeError
        If `rv` is not a RandomVector, or if `sigma_algebra` is not a `SigmaAlgebra` or `None`, or if `probability_measure` is not a ProbabilityMeausre or `None`.

    Returns
    -------
    exp : RandomVector | pd.Series | Real
        If `sigma_algebra` is `None` and `rv` is a `RandomVector` of dimension >1, returns a `pd.Series` representing the unconditional expectation of `rv`; otherwise, if `rv` is of dimension 1, returns a `Real`. If `sigma_algebra` is provided, returns a RandomVector representing the conditional expectation of `rv` given `sigma_algebra`.

    Examples
    --------
    >>> from sigalg.core import expectation, RandomVector, SampleSpace, SigmaAlgebra
    >>> domain = SampleSpace().from_sequence(size=3, prefix="omega")
    >>> outputs = {"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (5, 6)}
    >>> probabilities = {"omega_0": 0.2, "omega_1": 0.5, "omega_2": 0.3}
    >>> X = RandomVector(domain).from_dict(outputs).with_probability_measure(probabilities)
    >>> # Compute unconditional expectation
    >>> expectation(X) # doctest: +NORMALIZE_WHITESPACE
    feature
    X_0    3.2
    X_1    4.2
    dtype: float64
    >>> # Compute conditional expectation given a sigma algebra
    >>> F = SigmaAlgebra(domain).from_dict({"omega_0": 0, "omega_1": 0, "omega_2": 1})
    >>> expectation(X, F) # doctest: +NORMALIZE_WHITESPACE
    Random vector 'E(X|F)':
    feature  E(X|F)_0  E(X|F)_1
    sample
    omega_0  2.428571  3.428571
    omega_1  2.428571  3.428571
    omega_2  5.000000  6.000000
    """
    from ..probability_measures.probability_measure import ProbabilityMeasure
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from .random_vector import RandomVector

    if not isinstance(rv, RandomVector):
        raise TypeError("rv must be a RandomVector.")
    if sigma_algebra is not None and (
        not isinstance(sigma_algebra, SigmaAlgebra)
        or sigma_algebra.sample_space != rv.domain
    ):
        raise TypeError(
            "sigma_algebra must be a SigmaAlgebra or None, and its sample space must match the domain of the random vector."
        )
    if probability_measure is not None and (
        not isinstance(probability_measure, ProbabilityMeasure)
        or probability_measure.sample_space != rv.domain
    ):
        raise TypeError(
            "probability_measure must be a ProbabilityMeasure or None, and its sample space must match the domain of the random vector."
        )

    if probability_measure is None:
        probability_measure = rv.probability_measure

    if sigma_algebra is None:
        probabilities = probability_measure.data
        return rv.data.mul(probabilities, axis=0).sum()
    else:
        df = pd.concat([rv.data, sigma_algebra.data, probability_measure.data], axis=1)

        df["normalized_prob"] = df.groupby("atom ID")["probability"].transform(
            lambda x: x / x.sum()
        )

        vector_cols = (
            rv.data.columns if isinstance(rv.data, pd.DataFrame) else [rv.data.name]
        )
        expected_df = df.groupby("atom ID", group_keys=False).apply(
            _compute_expectation_of_group, vector_cols=vector_cols, include_groups=False
        )

        outputs = {idx: tuple(row) for idx, row in expected_df.iterrows()}

        name = (
            f"E({rv.name}|{sigma_algebra.name})"
            if rv.name is not None and sigma_algebra.name is not None
            else None
        )
        result = RandomVector(domain=rv.domain, name=name).from_dict(outputs)
        result.data.fillna(0, inplace=True)
        return result


def _compute_expectation_of_group(group, vector_cols):
    weights = group["normalized_prob"].values[:, None]
    expected = (group[vector_cols].values * weights).sum(axis=0)
    return pd.DataFrame([expected] * len(group), index=group.index, columns=vector_cols)


# def variance(rv: RandomVector):


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
        Probability measure `P` defining the probabilities on the domain sample space. If `None`, the probability measure carried by the random vector is used (accessed through its `probability_measure` attribute).

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
        probability_measure = rv.probability_measure

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
