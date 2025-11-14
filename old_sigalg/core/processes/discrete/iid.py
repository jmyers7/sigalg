# from ..base import StochasticProcess, ProcessWithProbabilitySpace
# from ..time_index import DiscreteTimeIndex
# import pandas as pd
# import numpy as np


# class DiscreteIID(StochasticProcess, ProcessWithProbabilitySpace):

#     trajectories: pd.DataFrame
#     simulation_time_index: DiscreteTimeIndex
#     state_space: list
#     probs: np.ndarray
#     omega: pd.DataFrame
#     omega_time_index: DiscreteTimeIndex
#     _state_to_index: dict

#     def __init__(self, state_space: list, probs: np.ndarray) -> None:
#         super().__init__()
#         if not isinstance(probs, np.ndarray):
#             raise ValueError("probs must be a numpy ndarray")
#         if len(state_space) != len(probs):
#             raise ValueError("Length of state_space must match length of probs")
#         if not np.isclose(np.sum(probs), 1.0):
#             raise ValueError("Probabilities must sum to 1")
#         self.probs = probs
#         self._set_state_space(state_space)

#     def simulate(
#         self, simulation_time_index, num_trajectories: int = 10
#     ) -> "DiscreteIID":
#         simulation_time_index.validate()
#         self.simulation_time_index = simulation_time_index

#         trajectories = np.random.choice(
#             self.state_space,
#             size=(num_trajectories, len(simulation_time_index)),
#             p=self.probs,
#         )

#         trajectory_index = pd.Index(range(num_trajectories), name="traj_num")
#         self.trajectories = pd.DataFrame(
#             trajectories, columns=simulation_time_index.index, index=trajectory_index
#         )
#         return self

#     def _get_plot_title(self) -> str:
#         return "discrete IID process"

#     def joint_prob(self, X: pd.Series) -> float:
#         prob = 1.0
#         for value in X:
#             prob *= self.prob_dist[value]
#         return prob


# class GaussianIID(StochasticProcess):
    
#     trajectories: pd.DataFrame
#     simulation_time_index: DiscreteTimeIndex
#     mean: float
#     std: float

#     def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
#         super().__init__()
#         self.mean = mean
#         self.std = std

#     def simulate(
#         self, simulation_time_index, num_trajectories: int = 10
#     ) -> "GaussianIID":
#         simulation_time_index.validate()
#         self.simulation_time_index = simulation_time_index

#         trajectories = np.random.normal(
#             loc=self.mean,
#             scale=self.std,
#             size=(num_trajectories, len(simulation_time_index)),
#         )

#         trajectory_index = pd.Index(range(num_trajectories), name="traj_num")
#         self.trajectories = pd.DataFrame(
#             trajectories, columns=simulation_time_index.index, index=trajectory_index
#         )
#         return self
    
#     def _get_plot_title(self) -> str:
#         return f"Gaussian IID process (mean={self.mean}, std={self.std})"