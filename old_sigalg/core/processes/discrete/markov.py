# from ...spaces import SequentialSampleSpace, ProbabilitySpace
# from ..base import StochasticProcess
# from ...time import DiscreteTime
# import numpy as np
# import pandas as pd
# from typing import Self


# class FirstOrderMarkovChain(StochasticProcess):
#     """
#     Represents a first-order Markov chain stochastic process.

#     Contains methods to generate the probability space and simulate trajectories.

#     Attributes
#     ----------
#     init_prob : np.ndarray
#         The initial probability distribution over the state space.
#     transition_matrix : np.ndarray
#         The state transition probability matrix.
#     sample_space : SequentialSampleSpace, None
#         The sample space of all possible state sequences.
#     probability_measure : ProbabilityMeasure, None
#         The probability measure defined over the sample space.
#     """

#     def __init__(
#         self, state_space: list, init_prob: np.ndarray, transition_matrix: np.ndarray
#     ) -> None:
#         """
#         Initializes the FirstOrderMarkovChain with the given state space, initial
#         probabilities, and transition matrix.

#         Initializes sample_space and probability_measure to None. The former is
#         generated via the gen_probability_space method, while the latter is
#         created using the joint_prob method.

#         Parameters
#         ----------
#         state_space : list
#             A list of possible states in the Markov chain.
#         init_prob : np.ndarray
#             A 1D numpy array representing the initial probability distribution
#             over the state space.
#         transition_matrix : np.ndarray
#             A 2D numpy array representing the state transition probability matrix.

#         Raises
#         ------
#         ValueError
#             If state_space is not a non-empty list or if init_prob and
#             transition_matrix do not conform to the expected dimensions and
#             properties.
#         """
#         if not isinstance(state_space, list) or len(state_space) == 0:
#             raise ValueError("state_space must be a non-empty list")
#         self.validate_probabilities(len(state_space), init_prob, transition_matrix)

#         self._state_space = state_space
#         self.init_prob = init_prob
#         self.transition_matrix = transition_matrix
#         self.sample_space = None
#         self.probability_measure = None
#         self._state_space_dict = {state: idx for idx, state in enumerate(state_space)}

#     def joint_prob(self, X: pd.Series) -> float:
#         """
#         Computes the joint probability of a given state sequence X.

#         Parameters
#         ----------
#         X : pd.Series
#             A pandas Series representing a sequence of states.

#         Returns
#         -------
#         float
#             The joint probability of the state sequence X.

#         Raises
#         ------
#         ValueError
#             If X is not a pandas Series.
#         """
#         if not isinstance(X, pd.Series):
#             raise ValueError("X must be a pandas Series")

#         init_state = X.iloc[0]
#         init_idx = self._state_space_dict[init_state]
#         prob = self.init_prob[init_idx]

#         for t in range(1, len(X)):
#             prev_state = X.iloc[t - 1]
#             curr_state = X.iloc[t]
#             prev_idx = self._state_space_dict[prev_state]
#             curr_idx = self._state_space_dict[curr_state]
#             prob *= self.transition_matrix[prev_idx, curr_idx]

#         return prob

#     def gen_probability_space(self, time_index: DiscreteTime) -> Self:
#         """
#         Generates the sample space and probability measure for the Markov chain.

#         Parameters
#         ----------
#         time_index : DiscreteTimeIndex
#             The time index corresponding to the sequence length.

#         Returns
#         -------
#         Self
#             The FirstOrderMarkovChain instance with updated sample_space and
#             probability_measure attributes.

#         Raises
#         ------
#         ValueError
#             If time_index is not an instance of DiscreteTimeIndex.
#         """
#         if not isinstance(time_index, DiscreteTime):
#             raise ValueError("time_index must be an instance of DiscreteTimeIndex")

#         sample_space = SequentialSampleSpace(self._state_space, time_index)
#         self.probability_space = ProbabilitySpace(sample_space, self.joint_prob)

#         return self

#     def simulate(
#         self,
#         num_trajectories: int,
#         time_index: DiscreteTime,
#         seed: int | None = None,
#         name: str = "traj_num",
#     ) -> Self:
#         """
#         Simulates trajectories of the Markov chain.

#         Parameters
#         ----------
#         num_trajectories : int
#             The number of trajectories to simulate.
#         time_index : DiscreteTimeIndex
#             The time index for the simulated trajectories.
#         seed : int | None, optional
#             An optional random seed for reproducibility (default is None).
#         name : str, optional
#             The name of the trajectory index (default is "traj_num").

#         Returns
#         -------
#         Self
#             The FirstOrderMarkovChain instance with updated time_index and
#             trajectories attributes.

#         Raises
#         ------
#         ValueError
#             If num_trajectories is not a positive integer or if time_index is
#             not an instance of DiscreteTimeIndex.
#         """
#         if not isinstance(num_trajectories, int) or num_trajectories <= 0:
#             raise ValueError("num_trajectories must be a positive integer")
#         if not isinstance(time_index, DiscreteTime):
#             raise ValueError("time_index must be an instance of DiscreteTimeIndex")

#         rng = np.random.default_rng(seed)
#         n_steps = len(time_index)
#         trajectories = []

#         for _ in range(num_trajectories):
#             trajectory = []
#             current_state = rng.choice(self._state_space, p=self.init_prob)
#             trajectory.append(current_state)
#             for _ in range(1, n_steps):
#                 current_idx = self._state_space_dict[current_state]
#                 next_state = rng.choice(
#                     self._state_space, p=self.transition_matrix[current_idx]
#                 )
#                 trajectory.append(next_state)
#                 current_state = next_state
#             trajectories.append(trajectory)

#         trajectory_index = pd.Index(range(num_trajectories), name=name)
#         df = pd.DataFrame(
#             trajectories, columns=time_index.index, index=trajectory_index
#         )
#         self.time_index = time_index
#         self.trajectories = df

#         return self

#     @staticmethod
#     def validate_probabilities(n_states, init_prob, transition_matrix) -> None:
#         """
#         Validates the initial probabilities and transition matrix.

#         Parameters
#         ----------
#         n_states : int
#             The number of states in the Markov chain.
#         init_prob : np.ndarray
#             A 1D numpy array representing the initial probability distribution.
#         transition_matrix : np.ndarray
#             A 2D numpy array representing the state transition probability matrix.

#         Raises
#         ------
#         ValueError
#             If init_prob and transition_matrix do not conform to the expected
#             dimensions and properties.
#         """
#         if not isinstance(init_prob, np.ndarray):
#             raise ValueError("init_prob must be a numpy array")
#         if not isinstance(transition_matrix, np.ndarray):
#             raise ValueError("transition_matrix must be a numpy array")
#         if len(init_prob.shape) != 1:
#             raise ValueError("init_prob must be a 1D array")
#         if len(transition_matrix.shape) != 2:
#             raise ValueError("transition_matrix must be a 2D array")
#         if init_prob.shape[0] != n_states:
#             raise ValueError("init_prob length must match number of states")
#         if (
#             transition_matrix.shape[0] != n_states
#             or transition_matrix.shape[1] != n_states
#         ):
#             raise ValueError(
#                 "transition_matrix must be square with size equal to number of states"
#             )
#         if not np.isclose(np.sum(init_prob), 1):
#             raise ValueError("init_prob must sum to 1")
#         if not np.allclose(np.sum(transition_matrix, axis=1), 1):
#             raise ValueError("Each row of transition_matrix must sum to 1")
#         if np.any(init_prob < 0):
#             raise ValueError("init_prob must have non-negative entries")
#         if np.any(transition_matrix < 0):
#             raise ValueError("transition_matrix must have non-negative entries")

#     def _get_plot_title(self) -> str:
#         """
#         Generate default plot title for the first-order Markov chain.

#         Returns
#         -------
#         str
#             A string representing the title of the plot.
#         """
#         return "first-order markov chain trajectories"
