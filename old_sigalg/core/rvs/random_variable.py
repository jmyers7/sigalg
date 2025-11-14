# from ..spaces import SampleSpace
# from ..sigma_algebras import SigmaAlgebra
# from typing import Callable
# import pandas as pd


# class RandomVariable:
#     """
#     Base class for random variables.

#     A random variable is a function on a sample space. Here, the sample space is modeled
#     by an instance of `SampleSpace` called `sample_space`. The random variable is defined by a callable function (called `rv_function`) that takes a sample point (row of the sample space) as input and returns the value of the random variable at that sample point. The values of the random variable on the sample space are stored in an instance of `pd.Series` called `values`.

#     A random variable is also supposed to be measurable with respect to a sigma-algebra on the sample space. This means that the random variable is constant on the atoms of the sigma-algebra, or equivalently that the sigma-algebra induced by the random variable is a sub-sigma-algebra of the given sigma-algebra. We check for measurability using the `is_measurable()` method.

#     If `X` is an instance of `RandomVariable`, then it has the attribute `X.sigma_algebra` which is the sigma-algebra induced by `X`. But it also has the attribute `X.sample_space.sigma_algebra`, which is the sigma-algebra on the underlying sample space. So, be aware that there are two different sigma-algebras involved here.

#     Attributes
#     ----------
#     sample_space : SampleSpace
#         The sample space over which the random variable is defined.
#     rv_function : Callable[[pd.Series], float]
#         A callable function that takes a sample point (row of sample space) as input
#         and returns the value of the random variable at that sample point.
#     values: pd.Series
#         The values of the random variable on the sample space.
#     sigma_algebra : SigmaAlgebra
#         The sigma algebra on the sample space with respect to which the random variable is measurable. Computed as a @property.

#     Examples
#     --------
#     >>> import sigalg as sa
#     >>> import pandas as pd
#     >>> def X_function(row: pd.Series) -> float:
#     ...     numerical_row = row.apply(lambda x: 0 if x == "T" else 1)
#     ...     return numerical_row.sum()
#     >>> state_space = ["T", "H"]
#     >>> time = sa.DiscreteTime.from_list([0, 1, 2])
#     >>> sample_space = sa.SampleSpace.create_sequence_space(state_space, time)
#     >>> X = sa.RandomVariable(sample_space=sample_space, rv_function=X_function)
#     >>> print(pd.concat([sample_space, X.values.rename("X")], axis=1))
#               0  1  2  X
#     sequence
#     omega1    T  T  T  0
#     omega2    T  T  H  1
#     omega3    T  H  T  1
#     omega4    T  H  H  2
#     omega5    H  T  T  1
#     omega6    H  T  H  2
#     omega7    H  H  T  2
#     omega8    H  H  H  3
#     """

#     def __init__(
#         self,
#         sample_space: SampleSpace,
#         rv_function: Callable[[pd.Series], float],
#     ) -> None:
#         """
#         Initializes the RandomVariable with the given sample space and
#         random variable function.

#         Parameters
#         ----------
#         sample_space : SampleSpace
#             The sample space over which the random variable is defined.
#         rv_function : Callable[[pd.Series], float]
#             A callable function that takes a sample point (row of sample space) as input
#             and returns the value of the random variable at that sample point.

#         Raises
#         ------
#         ValueError
#             If sample_space is not an instance of SampleSpace or if rv_function
#             is not callable.

#         Examples
#         --------
#         >>> import sigalg as sa
#         >>> import pandas as pd
#         >>> def X_function(row: pd.Series) -> float:
#         ...     numerical_row = row.apply(lambda x: 0 if x == "T" else 1)
#         ...     return numerical_row.sum()
#         >>> state_space = ["T", "H"]
#         >>> time = sa.DiscreteTime.from_list([0, 1, 2])
#         >>> sample_space = sa.SampleSpace.create_sequence_space(state_space, time)
#         >>> X = sa.RandomVariable(sample_space=sample_space, rv_function=X_function)
#         >>> print(pd.concat([sample_space, X.values.rename("X")], axis=1))
#                   0  1  2  X
#         sequence
#         omega1    T  T  T  0
#         omega2    T  T  H  1
#         omega3    T  H  T  1
#         omega4    T  H  H  2
#         omega5    H  T  T  1
#         omega6    H  T  H  2
#         omega7    H  H  T  2
#         omega8    H  H  H  3
#         """
#         if not isinstance(sample_space, SampleSpace):
#             raise ValueError("sample_space must be an instance of SampleSpace")
#         if not callable(rv_function):
#             raise ValueError("rv_function must be a callable function")

#         self.sample_space = sample_space
#         self.rv_function = rv_function
#         self.values = sample_space.apply(rv_function, axis=1)
#         self.values.name = "rv_values"

#     def expectation(self, given: SigmaAlgebra | None = None):
#         """
#         Computes the expectation of a random variable given an optional sigma-algebra.

#         Parameters
#         -------
#         given : SigmaAlgebra | None
#             If provided, computes the conditional expectation of the random variable
#             given the provided sigma algebra. Otherwise, computes the unconditional expectation.

#         Returns
#         -------
#         float | RandomVariable
#             The expectation value as a float if no sigma algebra is provided,
#             or a RandomVariable representing the conditional expectation if a sigma-algebra is provided.

#         Raises
#         ------
#         ValueError
#             If the sample space of the random variable is not a ProbabilitySpace,
#             or if the sample space of the given sigma algebra does not match
#             the sample space of the random variable.

#         Examples
#         --------
#         >>> import sigalg as sa
#         >>> import pandas as pd
#         >>> def prob_measure(row: pd.Series) -> float:
#         ...     numerical_row = row.apply(lambda x: 0 if x == "T" else 1)
#         ...     num_heads = numerical_row.sum()
#         ...     num_tails = len(row) - num_heads
#         ...     return 0.75**num_heads * 0.25**num_tails
#         >>> state_space = ["T", "H"]
#         >>> time = sa.DiscreteTime.from_list([0, 1, 2])
#         >>> probability_space = sa.ProbabilitySpace.create_sequence_space(
#         ...     state_space, time
#         ... ).add_prob_measure(prob_measure)
#         >>> def X_function(row: pd.Series) -> float:
#         ...     numerical_row = row.apply(lambda x: 0 if x == "T" else 1)
#         ...     return numerical_row.sum()
#         >>> X = sa.RandomVariable(sample_space=probability_space, rv_function=X_function)
#         >>> atom_labels = pd.Series(
#         ...     index=probability_space.index,
#         ...     data=[0, 0, 1, 1, 2, 2, 3, 3],
#         ...     name="atom_id",
#         ... )
#         >>> F = sa.SigmaAlgebra(probability_space, atom_labels=atom_labels)
#         >>> X.values.name = "X"
#         >>> cond_exp = X.expectation(given=F).values
#         >>> cond_exp.name = "E[X|F]"
#         >>> print(
#         ...     pd.concat(
#         ...         [
#         ...             probability_space,
#         ...             probability_space.probabilities,
#         ...             F.atom_labels,
#         ...             X.values,
#         ...             cond_exp,
#         ...         ],
#         ...         axis=1,
#         ...     )
#         ... )
#                   0  1  2  probability  atom_id  X  E[X|F]
#         sequence
#         omega1    T  T  T     0.015625        0  0    0.75
#         omega2    T  T  H     0.046875        0  1    0.75
#         omega3    T  H  T     0.046875        1  1    1.75
#         omega4    T  H  H     0.140625        1  2    1.75
#         omega5    H  T  T     0.046875        2  1    1.75
#         omega6    H  T  H     0.140625        2  2    1.75
#         omega7    H  H  T     0.140625        3  2    2.75
#         omega8    H  H  H     0.421875        3  3    2.75
#         """
#         from ..operators.expectation import expectation

#         return expectation(of=self, given=given)

#     def variance(self) -> float:
#         """
#         Computes the variance of the random variable.

#         Returns
#         -------
#         float
#             The variance of the random variable.

#         Raises
#         ------
#         ValueError
#             If sample_space is not an instance of ProbabilitySpace.

#         Examples
#         --------
#         >>> import sigalg as sa
#         >>> import pandas as pd
#         >>> def prob_measure(row: pd.Series) -> float:
#             numerical_row = row.apply(lambda x: 0 if x == "T" else 1)
#             num_heads = numerical_row.sum()
#             num_tails = len(row) - num_heads
#             return 0.75**num_heads * 0.25**num_tails
#         >>> state_space = ["T", "H"]
#         >>> time = sa.DiscreteTime.from_list([0, 1, 2])
#         >>> probability_space = sa.ProbabilitySpace.create_sequence_space(
#         ...     state_space, time
#         ... ).add_prob_measure(prob_measure)
#         >>> def X_function(row: pd.Series) -> float:
#         ...     numerical_row = row.apply(lambda x: 0 if x == "T" else 1)
#         ...     return numerical_row.sum()
#         >>> X = sa.RandomVariable(
#                 sample_space=probability_space,
#         ...     rv_function=X_function
#         ... )
#         >>> print(
#         ...     pd.concat(
#         ...         [
#         ...             probability_space,
#         ...             probability_space.probabilities,
#         ...             X.values.rename("X")
#         ...         ],
#         ...         axis=1,
#         ...     )
#         ... )
#                 0  1  2  probability  X
#         sequence
#         omega1    T  T  T     0.015625  0
#         omega2    T  T  H     0.046875  1
#         omega3    T  H  T     0.046875  1
#         omega4    T  H  H     0.140625  2
#         omega5    H  T  T     0.046875  1
#         omega6    H  T  H     0.140625  2
#         omega7    H  H  T     0.140625  2
#         omega8    H  H  H     0.421875  3
#         >>> print("Variance of X:", X.variance())
#         Variance of X: 0.5625
#         """
#         from ..operators.variance import variance

#         return variance(of=self)

#     def sd(self) -> float:
#         """
#         Computes the standard deviation of the random variable.

#         Returns
#         -------
#         float
#             The standard deviation of the random variable.
        
#         Raises
#         ------
#         ValueError
#             If sample_space is not an instance of ProbabilitySpace.
        
#         Examples
#         --------
#         >>> import sigalg as sa
#         >>> import pandas as pd
#         >>> def prob_measure(row: pd.Series) -> float:
#             numerical_row = row.apply(lambda x: 0 if x == "T" else 1)
#             num_heads = numerical_row.sum()
#             num_tails = len(row) - num_heads
#             return 0.75**num_heads * 0.25**num_tails
#         >>> state_space = ["T", "H"]
#         >>> time = sa.DiscreteTime.from_list([0, 1, 2])
#         >>> probability_space = sa.ProbabilitySpace.create_sequence_space(
#         ...     state_space, time
#         ... ).add_prob_measure(prob_measure)
#         >>> def X_function(row: pd.Series) -> float:
#         ...     numerical_row = row.apply(lambda x: 0 if x == "T" else 1)
#         ...     return numerical_row.sum()
#         >>> X = sa.RandomVariable(
#                 sample_space=probability_space,
#         ...     rv_function=X_function
#         ... )
#         >>> print(
#         ...     pd.concat(
#         ...         [
#         ...             probability_space,
#         ...             probability_space.probabilities,
#         ...             X.values.rename("X")
#         ...         ],
#         ...         axis=1,
#         ...     )
#         ... )
#                 0  1  2  probability  X
#         sequence
#         omega1    T  T  T     0.015625  0
#         omega2    T  T  H     0.046875  1
#         omega3    T  H  T     0.046875  1
#         omega4    T  H  H     0.140625  2
#         omega5    H  T  T     0.046875  1
#         omega6    H  T  H     0.140625  2
#         omega7    H  H  T     0.140625  2
#         omega8    H  H  H     0.421875  3
#         >>> print("Standard Deviation of X:", sa.sd(of=X))
#         Standard Deviation of X: 0.75
#         """
#         from ..operators.sd import sd

#         return sd(of=self)

#     @property
#     def sigma_algebra(self) -> SigmaAlgebra:
#         """
#         Generates the sigma algebra induced by the random variable.

#         Returns
#         -------
#         SigmaAlgebra
#             The sigma algebra induced by the random variable.
#         """
#         return SigmaAlgebra(self.sample_space, self.values)

#     def is_measurable(self, sigma_algebra: SigmaAlgebra | None = None) -> bool:
#         """
#         Checks if the random variable is measurable with respect to a given sigma algebra.

#         The random variable is measurable with respect to a sigma algebra if (and only if) the sigma algebra induced by the random variable is a sub-algebra of the given sigma algebra. This is the same as the random variable being constant on
#         the atoms of the given sigma-algebra.

#         If no sigma algebra is provided, checks measurability with respect to the sample space's sigma algebra.

#         Parameters
#         ----------
#         sigma_algebra : SigmaAlgebra | None
#             The sigma algebra to check measurability against. If None, checks measurability with respect to the sample space's sigma algebra.

#         Returns
#         -------
#         bool
#             True if the random variable is measurable, False otherwise.

#         Examples
#         --------
#         >>> import sigalg as sa
#         >>> import pandas as pd
#         >>> state_space = ["T", "H"]
#         >>> time = sa.DiscreteTime.from_list([0, 1, 2])
#         >>> sample_space = sa.SampleSpace.create_sequence_space(state_space, time)
#         >>> atom_labels = pd.Series(
#         ...     index=sample_space.index,
#         ...     data=[0, 0, 1, 1, 2, 2, 3, 3],
#         ...     name="atom_id",
#         ... )
#         >>> sigma_algebra = sa.SigmaAlgebra(sample_space, atom_labels=atom_labels)
#         >>> sample_space.set_sigma_algebra(sigma_algebra)
#         >>> def X_function(row: pd.Series) -> float:
#         ...    numerical_row = row.apply(lambda x: 0 if x == "T" else 1)
#         ...    return numerical_row.sum()
#         >>> def Y_function(row: pd.Series) -> float:
#         ...    numerical_row = row.apply(lambda x: 0 if x == "T" else 1)
#         ...    return numerical_row[1]
#         >>> X = sa.RandomVariable(sample_space=sample_space, rv_function=X_function)
#         >>> Y = sa.RandomVariable(sample_space=sample_space, rv_function=Y_function)
#         >>> print(
#         ...     pd.concat(
#         ...         [
#         ...             sample_space,
#         ...             sample_space.sigma_algebra.atom_labels,
#         ...             X.values.rename("X"),
#         ...             Y.values.rename("Y"),
#         ...         ],
#         ...         axis=1,
#         ...     )
#         ... )
#         >>> print("Is X measurable?", "yes" if X.is_measurable() else "no")
#         >>> print("Is Y measurable?", "yes" if Y.is_measurable() else "no")
#                   0  1  2  atom_id  X  Y
#         sequence
#         omega1    T  T  T        0  0  0
#         omega2    T  T  H        0  1  0
#         omega3    T  H  T        1  1  1
#         omega4    T  H  H        1  2  1
#         omega5    H  T  T        2  1  0
#         omega6    H  T  H        2  2  0
#         omega7    H  H  T        3  2  1
#         omega8    H  H  H        3  3  1
#         Is X measurable? no
#         Is Y measurable? yes
#         """
#         if sigma_algebra is None:
#             sigma_algebra = self.sample_space.sigma_algebra

#         return self.sigma_algebra.is_subalgebra_of(self.sample_space.sigma_algebra)
