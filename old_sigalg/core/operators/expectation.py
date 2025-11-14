# from ..sigma_algebras import SigmaAlgebra
# from ..spaces.probability_space import ProbabilitySpace
# from ..rvs import RandomVariable
# import pandas as pd


# def expectation(
#     of: RandomVariable, given: SigmaAlgebra | None = None
# ) -> float | RandomVariable:
#     """
#     Computes the expectation of a random variable given an optional sigma-algebra.

#     Parameters
#     -------
#     of : RandomVariable
#         The random variable for which to compute the expectation.
#     given : SigmaAlgebra | None
#         If provided, computes the conditional expectation of the random variable
#         given the provided sigma algebra. Otherwise, computes the unconditional expectation.

#     Returns
#     -------
#     float | RandomVariable
#         The expectation value as a float if no sigma algebra is provided,
#         or a RandomVariable representing the conditional expectation if a sigma-algebra is provided.

#     Raises
#     ------
#     ValueError
#         If the sample space of the random variable is not a ProbabilitySpace,
#         or if the sample space of the given sigma algebra does not match
#         the sample space of the random variable.
    
#     Examples
#     --------
#     >>> import sigalg as sa
#     >>> import pandas as pd
#     >>> def prob_measure(row: pd.Series) -> float:
#     ...     numerical_row = row.apply(lambda x: 0 if x == "T" else 1)
#     ...     num_heads = numerical_row.sum()
#     ...     num_tails = len(row) - num_heads
#     ...     return 0.75**num_heads * 0.25**num_tails
#     >>> state_space = ["T", "H"]
#     >>> time = sa.DiscreteTime.from_list([0, 1, 2])
#     >>> probability_space = sa.ProbabilitySpace.create_sequence_space(
#     ...     state_space, time
#     ... ).add_prob_measure(prob_measure)
#     >>> def X_function(row: pd.Series) -> float:
#     ...     numerical_row = row.apply(lambda x: 0 if x == "T" else 1)
#     ...     return numerical_row.sum()
#     >>> X = sa.RandomVariable(sample_space=probability_space, rv_function=X_function)
#     >>> atom_labels = pd.Series(
#     ...     index=probability_space.index,
#     ...     data=[0, 0, 1, 1, 2, 2, 3, 3],
#     ...     name="atom_id",
#     ... )
#     >>> F = sa.SigmaAlgebra(probability_space, atom_labels=atom_labels)
#     >>> X.values.name = "X"
#     >>> cond_exp = sa.expectation(of=X, given=F).values
#     >>> cond_exp.name = "E[X|F]"
#     >>> print(
#     ...     pd.concat(
#     ...         [
#     ...             probability_space,
#     ...             probability_space.probabilities,
#     ...             F.atom_labels,
#     ...             X.values,
#     ...             cond_exp,
#     ...         ],
#     ...         axis=1,
#     ...     )
#     ... )
#               0  1  2  probability  atom_id  X  E[X|F]
#     sequence                                          
#     omega1    T  T  T     0.015625        0  0    0.75
#     omega2    T  T  H     0.046875        0  1    0.75
#     omega3    T  H  T     0.046875        1  1    1.75
#     omega4    T  H  H     0.140625        1  2    1.75
#     omega5    H  T  T     0.046875        2  1    1.75
#     omega6    H  T  H     0.140625        2  2    1.75
#     omega7    H  H  T     0.140625        3  2    2.75
#     omega8    H  H  H     0.421875        3  3    2.75
#     """
#     rv = of
#     if not isinstance(rv.sample_space, ProbabilitySpace):
#         raise ValueError(
#             "sample_space must be an instance of ProbabilitySpace to compute expectation."
#         )
#     probabilities = rv.sample_space.probabilities

#     if given is None:
#         return (rv.values * probabilities).sum()

#     if not given.sample_space.index.equals(rv.sample_space.index):
#         raise ValueError(
#             "The sample space of the given sigma algebra must match the sample space of the random variable."
#         )

#     grouped = pd.DataFrame(
#         {"rv_values": rv.values, "probabilities": probabilities}
#     ).groupby(given.atom_ids)

#     conditional_expectation_values = grouped.apply(
#         lambda df: (
#             (df["rv_values"] * df["probabilities"]).sum() / df["probabilities"].sum()
#             if df["probabilities"].sum() > 0
#             else 0.0
#         )
#     )
#     return RandomVariable(
#         sample_space=given.sample_space,
#         rv_function=lambda row: conditional_expectation_values.loc[
#             given.atom_ids.loc[row.name]
#         ],
#     )
