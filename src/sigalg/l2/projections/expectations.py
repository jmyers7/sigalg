from typing import TYPE_CHECKING  # noqa: D100

import pandas as pd

if TYPE_CHECKING:
    from ...core.random_objects.random_vector import RandomVector
    from ...core.sigma_algebras.sigma_algebra import SigmaAlgebra


def expectation(
    rv: RandomVector,
    sigma_algebra: SigmaAlgebra | None = None,
) -> RandomVector | pd.Series:
    """Compute the expectation of a RandomVector, optionally conditioned on a SigmaAlgebra.

    Parameters
    ----------
    rv : RandomVector
        The random vector for which to compute the expectation.
    sigma_algebra : SigmaAlgebra | None, default=None
        The sigma algebra to condition on. If `None`, computes the unconditional expectation.

    Raises
    ------
    TypeError
        If `rv` is not a RandomVector or if `sigma_algebra` is not a SigmaAlgebra or None.
    ValueError
        If `rv` does not have a probability measure or if the sample space of `sigma_algebra` does not match the domain of `rv`.

    Returns
    -------
    exp : RandomVector | pd.Series
        If `sigma_algebra` is `None`, returns a pd.Series representing the unconditional expectation of `rv`. If `sigma_algebra` is provided, returns a RandomVector representing the conditional expectation of `rv` given `sigma_algebra`.

    Examples
    --------
    >>> from sigalg.core import RandomVector, SampleSpace, SigmaAlgebra
    >>> from sigalg.l2 import expectation
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
    from ...core.random_objects.random_vector import RandomVector
    from ...core.sigma_algebras.sigma_algebra import SigmaAlgebra

    if not isinstance(rv, RandomVector):
        raise TypeError("rv must be a RandomVector.")
    if sigma_algebra is not None and not isinstance(sigma_algebra, SigmaAlgebra):
        raise TypeError("sigma_algebra must be a SigmaAlgebra or None.")
    if rv.probability_measure is None:
        raise ValueError("rv must have a probability_measure to compute expectation.")

    if sigma_algebra is None:
        probabilities = rv.probability_measure.data
        return rv.data.mul(probabilities, axis=0).sum()
    else:
        if sigma_algebra.sample_space != rv.domain:
            raise ValueError(
                "SigmaAlgebra sample_space must match RandomVector domain."
            )

        df = pd.concat(
            [rv.data, sigma_algebra.data, rv.probability_measure.data], axis=1
        )

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
            if rv.name and sigma_algebra.name
            else None
        )
        return RandomVector(rv.domain, name=name).from_dict(outputs)


def _compute_expectation_of_group(group, vector_cols):
    weights = group["normalized_prob"].values[:, None]
    expected = (group[vector_cols].values * weights).sum(axis=0)
    return pd.DataFrame([expected] * len(group), index=group.index, columns=vector_cols)
