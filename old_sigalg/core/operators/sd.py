# from ..rvs.random_variable import RandomVariable
# import numpy as np


# def sd(of: RandomVariable) -> float:
#     """
#     Computes the standard deviation of a random variable.

#     Parameters
#     ----------
#     of : RandomVariable
#         The random variable for which to compute the standard deviation.

#     Returns
#     -------
#     float
#         The standard deviation of the random variable.
    
#     Raises
#     ------
#     ValueError
#         If sample_space is not an instance of ProbabilitySpace.
    
#     Examples
#     --------
#     >>> import sigalg as sa
#     >>> import pandas as pd
#     >>> def prob_measure(row: pd.Series) -> float:
#         numerical_row = row.apply(lambda x: 0 if x == "T" else 1)
#         num_heads = numerical_row.sum()
#         num_tails = len(row) - num_heads
#         return 0.75**num_heads * 0.25**num_tails
#     >>> state_space = ["T", "H"]
#     >>> time = sa.DiscreteTime.from_list([0, 1, 2])
#     >>> probability_space = sa.ProbabilitySpace.create_sequence_space(
#     ...     state_space, time
#     ... ).add_prob_measure(prob_measure)
#     >>> def X_function(row: pd.Series) -> float:
#     ...     numerical_row = row.apply(lambda x: 0 if x == "T" else 1)
#     ...     return numerical_row.sum()
#     >>> X = sa.RandomVariable(sample_space=probability_space, rv_function=X_function)
#     >>> print(
#     ...     pd.concat(
#     ...         [
#     ...             probability_space,
#     ...             probability_space.probabilities,
#     ...             X.values.rename("X")
#     ...         ],
#     ...         axis=1,
#     ...     )
#     ... )
#               0  1  2  probability  X
#     sequence                         
#     omega1    T  T  T     0.015625  0
#     omega2    T  T  H     0.046875  1
#     omega3    T  H  T     0.046875  1
#     omega4    T  H  H     0.140625  2
#     omega5    H  T  T     0.046875  1
#     omega6    H  T  H     0.140625  2
#     omega7    H  H  T     0.140625  2
#     omega8    H  H  H     0.421875  3
#     >>> print("Standard Deviation of X:", sa.sd(of=X))
#     Standard Deviation of X: 0.75
#     """
#     rv = of
#     return np.sqrt(rv.variance())
