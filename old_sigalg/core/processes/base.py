# import pandas as pd
# import numpy as np
# from abc import ABC, abstractmethod
# import matplotlib.pyplot as plt
# from matplotlib.axes import Axes
# from matplotlib.ticker import MaxNLocator
# from matplotlib.colors import LinearSegmentedColormap
# from ..time import Time
# from typing import Self


# class StochasticProcess(ABC):
#     """
#     Abstract base class that contains the simulated trajectories of a stochastic process.

#     Essentially a wrapper around a Time instance and a pd.DataFrame, with additional
#     methods. If X_t is a stochastic process, then the Time instance models the time variable t, while the rows of the data frame contain the simulated trajectories of X_t. The pd.Index attribute of the Time instance indexes the columns of the data frame.

#     Contains two abstract methods that must be implemented by subclasses. The first is
#     simulate(), which contains the simulation logic used to generate trajectories and
#     is unique to each concrete subclass. The second is _get_plot_title(), which is used for custom titles for plots.

#     Also contains a plot_simulations() method, which plots the simulated trajectories
#     after simulate() is called.

#     Attributes
#     ----------
#     time : Time, None
#         The time of the simulated trajectories.
#     trajectories : pd.DataFrame, None
#         A DataFrame where each row represents a simulated trajectory of the stochastic process.
#     """

#     def __init__(self) -> None:
#         """
#         Initializes the StochasticProcess with None values for time and trajectories.
#         """
#         self.time = None
#         self.trajectories = None

#     @abstractmethod
#     def simulate(
#         self, num_trajectories: int, time: Time, seed: int | None = None
#     ) -> Self:
#         """
#         Simulates trajectories of the stochastic process.

#         Parameters
#         ----------
#         num_trajectories : int
#             The number of trajectories to simulate.
#         time : Time
#             The time for the simulated trajectories.
#         seed : int | None, optional
#             An optional random seed for reproducibility (default is None).
#         """
#         pass

#     @abstractmethod
#     def _get_plot_title(self) -> str:
#         """
#         Generate default plot title for the stochastic process.

#         Returns
#         -------
#         str
#             Human-readable title describing the stochastic process.
#         """
#         pass

#     def plot_simulations(
#         self,
#         ax: Axes = None,
#         colors: list = None,
#         plot_kwargs: dict = None,
#         x_label: str = "time",
#         y_label: str = "state",
#         title: str = None,
#     ):
#         """
#         Plots the simulated trajectories of the stochastic process.

#         Parameters
#         ----------
#         ax : Axes, optional
#             A matplotlib Axes object to plot on. If None, a new figure and axes
#             are created (default is None).
#         colors : list, optional
#             A list of colors to use for the trajectories. If a single color is
#             provided, it is used for all trajectories. If multiple colors are
#             provided, they are used in order. If None, default matplotlib colors
#             are used (default is None).
#         plot_kwargs : dict, optional
#             Additional keyword arguments to pass to the plotting function
#             (default is None).
#         x_label : str, optional
#             Label for the x-axis (default is "time").
#         y_label : str, optional
#             Label for the y-axis (default is "state").
#         title : str, optional
#             Title for the plot. If None, a default title is generated
#             (default is None).
#         """
#         if self.trajectories is None:
#             raise ValueError("simulate() must be called before plotting")

#         columns = self.trajectories.columns
#         num_trajectories = len(self.trajectories)

#         if ax is None:
#             _, ax = plt.subplots()
#         elif not isinstance(ax, Axes):
#             raise ValueError("ax must be a matplotlib Axes object")

#         if plot_kwargs is None:
#             plot_kwargs = {}

#         if colors is not None:
#             if not isinstance(colors, list):
#                 raise ValueError("colors must be a list")
#             if len(colors) == 1:
#                 colors = [colors[0]] * num_trajectories
#             else:
#                 custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
#                 if num_trajectories == 1:
#                     colors = [custom_cmap(0)]
#                 else:
#                     colors = [
#                         custom_cmap(i / (num_trajectories - 1))
#                         for i in range(num_trajectories)
#                     ]

#         is_step_process = hasattr(self, "_is_step_process") and self._is_step_process

#         if is_step_process:
#             for i in range(num_trajectories):
#                 jump_times = self._jump_times[i]
#                 jump_counts = self._jump_counts[i]

#                 if colors is not None:
#                     ax.step(
#                         jump_times,
#                         jump_counts,
#                         where="post",
#                         color=colors[i],
#                         **plot_kwargs,
#                     )
#                 else:
#                     ax.step(jump_times, jump_counts, where="post", **plot_kwargs)
#         else:
#             for i, (_, row) in enumerate(self.trajectories.iterrows()):
#                 if colors is not None:
#                     ax.plot(columns, row.values, color=colors[i], **plot_kwargs)
#                 else:
#                     ax.plot(columns, row.values, **plot_kwargs)

#         is_time_integer = self._integer_check(columns.values)
#         is_trajectory_integer = self._integer_check(self.trajectories.values.flatten())
#         if is_time_integer:
#             ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#         if is_trajectory_integer:
#             ax.yaxis.set_major_locator(MaxNLocator(integer=True))

#         ax.set_xlabel(x_label)
#         ax.set_ylabel(y_label)

#         if title is None:
#             if hasattr(self, "_get_plot_title"):
#                 title = self._get_plot_title()
#             else:
#                 title = f"{self.__class__.__name__} trajectories"
#         ax.set_title(title)

#         return ax

#     @staticmethod
#     def _integer_check(arr: np.ndarray) -> bool:
#         for x in arr:
#             if isinstance(x, (int, np.integer)):
#                 return True
#             elif isinstance(x, (float, np.floating)) and x.is_integer():
#                 return True
#             else:
#                 return False


# # class TransformedProcess(StochasticProcess, ProcessWithSampleSpace):
# #     """
# #     Represents a transformed stochastic process.

# #     A TransformedProcess applies a deterministic transformation to another
# #     stochastic process (the *source process*). The transformation may modify
# #     both the simulated trajectories and their associated time index.

# #     The transformation function must accept the source process's trajectories
# #     (a pandas DataFrame) and return a tuple:

# #         (transformed_trajectories, transformed_time_index)

# #     See the simulate() method for details.

# #     If the source process has a sample space (omega), the same transformation
# #     function will also be applied to that space (dropping any probability
# #     measure). Probability measures are not propagated.

# #     Attributes
# #     ----------
# #     src_process : StochasticProcess
# #         The source stochastic process to transform.
# #     transform_func : transform_func : Callable[[pd.DataFrame], Tuple[pd.DataFrame, TimeIndex]]
# #         The transformation function applied to trajectories and time index.
# #     """

# #     def __init__(
# #         self,
# #         src_process: StochasticProcess,
# #         transform_func: Callable[[pd.DataFrame], Tuple[pd.DataFrame, TimeIndex]],
# #     ) -> None:
# #         """
# #         Initialize transformed process.

# #         Parameters
# #         ----------
# #         src_process : StochasticProcess
# #             The source stochastic process to transform.
# #         transform_func : Callable[[pd.DataFrame], Tuple[pd.DataFrame, TimeIndex]]
# #             The transformation function applied to trajectories and their time index.
# #             Must accept a pd.DataFrame and return a tuple (transformed_trajectories, transformed_time_index).
# #         """

# #         super().__init__()
# #         self.src_process = src_process
# #         self.transform_func = transform_func
# #         self._generate_omega()

# #     def _generate_omega(self) -> "TransformedProcess":
# #         """
# #         Generate transformed sample space by applying the transformation function
# #         to the source process's sample space.

# #         Overrides ProcessWithSampleSpace._generate_omega.
# #         """
# #         # If src_process does not have sample space, pass. We allow
# #         # transformations of processes without sample spaces.
# #         if (
# #             not isinstance(self.src_process, ProcessWithSampleSpace)
# #             or self.src_process.omega is None
# #         ):
# #             pass

# #         # The transformation should also transform the time index?
# #         self.omega = self.src_process.omega.copy()
# #         self.omega.drop("p", axis=1, inplace=True, errors="ignore")

# #         result = self.transform_func(self.omega)
# #         if not (isinstance(result, tuple) and len(result) == 2):
# #             raise TypeError("transform_func must return a tuple (DataFrame, TimeIndex)")

# #         omega, omega_time_index = result
# #         if not isinstance(omega, pd.DataFrame):
# #             raise TypeError("First element of transform_func output must be a DataFrame")
# #         if not isinstance(omega_time_index, TimeIndex):
# #             raise TypeError("Second element of transform_func output must be a TimeIndex")

# #         self.omega = omega
# #         self.omega_time_index = omega_time_index
# #         return self

# #     def simulate(
# #         self, simulation_time_index=None, num_trajectories=None
# #     ) -> "TransformedProcess":
# #         """
# #         Simulate transformed trajectories by applying the transformation function
# #         to the source process's simulated trajectories. The transformation must
# #         return both the transformed trajectories and a transformed time index.

# #         Overrides StochasticProcess.simulate.

# #         Parameters
# #         ----------
# #         simulation_time_index : TimeIndex, optional
# #             Ignored. The source process's simulation_time_index is used.
# #         num_trajectories : int, optional
# #             Ignored. The source process's number of trajectories is used.

# #         Returns
# #         -------
# #         TransformedProcess
# #             The transformed process with updated trajectories and time index.

# #         Raises
# #         ------
# #         ValueError
# #             If the source process has not been simulated yet.
# #         """
# #         if self.src_process.trajectories is None:
# #             raise ValueError(
# #                 "simulate() must be called on the source process before "
# #                 "calling it on the transformed process"
# #             )
# #         result = self.transform_func(self.src_process.trajectories)
# #         if not (isinstance(result, tuple) and len(result) == 2):
# #             raise TypeError("transform_func must return a tuple (DataFrame, TimeIndex)")

# #         trajectories, time_index = result
# #         if not isinstance(trajectories, pd.DataFrame):
# #             raise TypeError(
# #                 "First element of transform_func output must be a DataFrame"
# #             )
# #         if not isinstance(time_index, TimeIndex):
# #             raise TypeError(
# #                 "Second element of transform_func output must be a TimeIndex"
# #             )
# #         self.trajectories = trajectories
# #         self.simulation_time_index = time_index
# #         return self

# #     def _get_plot_title(self) -> str:
# #         """
# #         Generate default plot title for the transformed process.

# #         Returns
# #         -------
# #         str
# #             Human-readable title describing the transformed process.
# #         """
# #         return f"Transformed Process of ({self.src_process._get_plot_title()})"
