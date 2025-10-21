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
