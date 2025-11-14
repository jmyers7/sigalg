# from ..base import StochasticProcess
# from ...time import ContinuousTime
# from typing import Self
# from numbers import Real
# import numpy as np
# import pandas as pd


# class PoissonProcess(StochasticProcess):
#     r"""
#     Represents a Poisson process.

#     The Poisson process is a continuous-time stochastic process that can be described as follows. Consider a stream of random events occuring over a time interval beginning at t=0, and let W_1 be the waiting time for the first event, W_2 the waiting time between the first and second event, and so on. Then T_n = W_1 + W_1 + ... + W_n represents the time of arrival of the n-th event. Then the number of events that have occured over the interval [0, t] is given by N_t = max{n : T_n <= t}. Provided that the waiting times W_i are independent and identically distributed exponential random variables with parameter (rate) λ, the stochastic process N_t is called a Poisson process with rate λ.

#     Parameters
#     ----------
#     rate : float
#         The rate (intensity) of the Poisson process, must be a positive real number.
#     """

#     # Indicates that this is a step process for plotting purposes
#     _is_step_process = True

#     def __init__(self, rate: float) -> None:
#         """
#         Initializes a Poisson process with the specified rate.

#         Parameters
#         ----------
#         rate : float
#             The rate (intensity) of the Poisson process, must be a positive real number.

#         Raises
#         ------
#         ValueError
#             If rate is not a positive real number.
#         """
#         if not isinstance(rate, Real) or rate <= 0:
#             raise ValueError("rate must be a positive real number")
#         self.rate = float(rate)
#         self._jump_times = None
#         self._jump_counts = None

#     def simulate(
#         self,
#         num_trajectories: int,
#         time: ContinuousTime,
#         seed: int | None = None,
#         name: str = "traj_num",
#     ) -> Self:
#         """
#         Simulates trajectories of the Poisson process.

#         The method begins by simulating jump times and counts for each trajectory. The jump times are generated using exponential inter-arrival times, while the counts are incremented at each jump. These data are then discretized onto the provided time index to form step function trajectories.

#         Parameters
#         ----------
#         num_trajectories : int
#             The number of trajectories to simulate.
#         time : ContinuousTime
#             The time for the simulated trajectories.
#         seed : int | None, optional
#             An optional random seed for reproducibility (default is None).
#         name : str, optional
#             The name of the trajectory index (default is "traj_num").

#         Returns
#         -------
#         Self
#             The PoissonProcess instance with updated time and
#             trajectories attributes.

#         Raises
#         ------
#         ValueError
#             If num_trajectories is not a positive integer or if time is
#             not an instance of ContinuousTime.
#         """
#         if not isinstance(num_trajectories, int) or num_trajectories <= 0:
#             raise ValueError("num_trajectories must be a positive integer")
#         if not isinstance(time, ContinuousTime):
#             raise ValueError("time must be an instance of ContinuousTime")

#         if seed is not None:
#             np.random.seed(seed)

#         initial_time = time[0]
#         time_horizon = time[-1]

#         self._jump_times = []
#         self._jump_counts = []

#         for _ in range(num_trajectories):
#             jump_times = [initial_time]
#             jump_counts = [0]
#             t = initial_time
#             count = 0

#             while t < time_horizon:
#                 dt_jump = np.random.exponential(1.0 / self.rate)
#                 t += dt_jump

#                 if t <= time_horizon:
#                     count += 1
#                     jump_times.append(t)
#                     jump_counts.append(count)

#             if jump_times[-1] < time_horizon:
#                 jump_times.append(time_horizon)
#                 jump_counts.append(count)

#             self._jump_times.append(np.array(jump_times))
#             self._jump_counts.append(np.array(jump_counts))

#         trajectories = self._discretize(
#             self._jump_times, self._jump_counts, num_trajectories, time
#         )
#         trajectory_index = pd.Index(range(num_trajectories), name=name)
#         self.trajectories = pd.DataFrame(
#             trajectories, index=trajectory_index, columns=time.index
#         )

#         return self

#     @staticmethod
#     def _discretize(
#         jump_times, jump_counts, num_trajectories, time_index
#     ) -> np.ndarray:
#         """
#         Convert internal representatinon (jump times and counts) to discretized
#         trajectories on the provided time index.

#         For each trajectory, at each time point in the time index, the count is
#         determined by finding the largest jump time that is less than or equal
#         to the time point. This effectively creates a step function representation of
#         the Poisson process.

#         Parameters
#         ----------
#         jump_times : list of np.ndarray
#             A list where each element is a numpy array of jump times for a trajectory.
#         jump_counts : list of np.ndarray
#             A list where each element is a numpy array of jump counts for a trajectory.
#         num_trajectories : int
#             The number of trajectories.
#         time_index : ContinuousTimeIndex
#             The time index for discretization.

#         Returns
#         -------
#         np.ndarray
#             A 2D numpy array where each row corresponds to a trajectory and
#             each column corresponds to a time point in the simulation time index.
#         """

#         trajectories = np.zeros((num_trajectories, len(time_index)), dtype=int)

#         for i in range(num_trajectories):
#             traj_jump_times = jump_times[i]
#             traj_jump_counts = jump_counts[i]

#             for j, t in enumerate(time_index):
#                 idx = np.searchsorted(traj_jump_times, t, side="right") - 1
#                 trajectories[i, j] = traj_jump_counts[idx]

#         return trajectories

#     def _get_plot_title(self) -> str:
#         """
#         Generates a default title for the Poisson process plot.

#         Returns
#         -------
#         str
#             A string representing the title of the plot.
#         """
#         return f"poisson process trajectories (rate={self.rate})"
