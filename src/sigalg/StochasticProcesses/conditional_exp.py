import pandas as pd
from collections.abc import Callable


def conditional_exp(omega, RV, sigma_algebra, name="expectation"):
    """
    Compute conditional expectation E(random variable | sigma-algebra).

    Parameters
    ----------
    omega : pandas.DataFrame
        Sample space DataFrame with a 'p' column for probabilities
    RV : callable
        Function that takes omega and returns a Series.
    sigma_algebra : list
        Column names generating the conditioning sigma-algebra
    name : str, default="expectation"
        Name for the resulting expectation column

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by sigma-algebra values with conditional
        expectations

    Raises
    ------
    ValueError
        If setup_sample_space() has not been called
    
    Examples
    --------
    >>> # P(X_1 = 0) = 0.5, P(X_1 = 1) = 0.5
    >>> init_prob = np.array([0.5, 0.5])
    >>> # P(X_{n+1} = 0 | X_n = 0) = 0.7, P(X_{n+1} = 1 | X_n = 0) = 0.3
    >>> # P(X_{n+1} = 0 | X_n = 1) = 0.2, P(X_{n+1} = 1 | X_n = 1) = 0.8
    >>> transition_matrix = np.array([[0.7, 0.3], [0.2, 0.8]])
    >>> # Instantiate Markov chain modeling coin flips with momentum
    >>> mc = FirstOrderMarkovChain(
    ...     transition_matrix,
    ...     init_prob,
    ... )
    >>> mc.generate_sample_space(trajectory_length=4)
    >>> print(mc.omega)
        X1  X2  X3  X4       p
    0    0   0   0   0  0.1715
    1    0   0   0   1  0.0735
    2    0   0   1   0  0.0210
    3    0   0   1   1  0.0840
    4    0   1   0   0  0.0210
    5    0   1   0   1  0.0090
    6    0   1   1   0  0.0240
    7    0   1   1   1  0.0960
    8    1   0   0   0  0.0490
    9    1   0   0   1  0.0210
    10   1   0   1   0  0.0060
    11   1   0   1   1  0.0240
    12   1   1   0   0  0.0560
    13   1   1   0   1  0.0240
    14   1   1   1   0  0.0640
    15   1   1   1   1  0.2560
    >>> # Define the random variable S3 = X1 + X2 + X3
    >>> def S3(omega):
    ...     return omega["X1"] + omega["X2"] + omega["X3"]
    >>> # Compute the conditional expectation of S3 given X1 and X2
    >>> cond_exp_S3_given_X1X2 = conditional_exp(
    ...     omega=mc.omega, RV=S3, sigma_algebra=["X1", "X2"], name="E(S3 | X1, X2)"
    ... )
    >>> # Print the conditional expectation
    >>> print(cond_exp_S3_given_X1X2)
       X1  X2  E(S3 | X1, X2)
    0   0   0             0.3
    1   0   1             1.8
    2   1   0             1.3
    3   1   1             2.8
    """
    if not isinstance(omega, pd.DataFrame) or "p" not in omega.columns:
        raise ValueError(
            "omega must be a pandas DataFrame with a 'p' column for probabilities."
        )

    if not isinstance(RV, Callable):
        raise ValueError("RV must be a callable that takes omega and returns a Series.")

    p_cond_col = f"p_{'_'.join(sigma_algebra)}"
    omega[p_cond_col] = omega.groupby(sigma_algebra)["p"].transform(
        lambda x: x / x.sum()
    )

    RV_values = RV(omega)

    result = omega.groupby(sigma_algebra).apply(
        lambda g: (RV_values.loc[g.index] * g[p_cond_col]).sum(),
        include_groups=False,
    )

    omega.drop(columns=[p_cond_col], inplace=True)

    return result.to_frame(name=name).reset_index()
