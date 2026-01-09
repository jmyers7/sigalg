"""Stochastic process transformation module."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ...core.base.time import Time

if TYPE_CHECKING:
    from ..base.stochastic_process import StochasticProcess


class ProcessTransforms:
    """A collection of methods for transforming stochastic processes."""

    @classmethod
    def cumsum(
        cls, process: StochasticProcess, name: Hashable | None = None
    ) -> StochasticProcess:
        """Compute the cumulative sum of a stochastic process along its time index.

        Parameters
        ----------
        process : StochasticProcess
            The stochastic process for which to compute the cumulative sum.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of the input process subscripted with `cumsum`, provided that the name of the input process is a string.

        Raises
        ------
        TypeError
            If `process` is not an instance of `StochasticProcess`.

        Returns
        -------
        cumsum_process : StochasticProcess
            A new stochastic process representing the cumulative sum of the input process.
        """
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")

        data_trans = process.data.copy()
        data_trans = data_trans.cumsum(axis=1)
        if name is None:
            name = f"{process.name}_cumsum" if process.name is not None else None
        return (
            StochasticProcess(name=name, domain=process.domain, index=process.time)
            .from_pandas(data_trans)
            .with_probability_measure(probability_measure=process.probability_measure)
        )

    @classmethod
    def increments(
        cls, process: StochasticProcess, name: Hashable | None = None
    ) -> StochasticProcess:
        """Compute the increments of a stochastic process along its time index.

        Parameters
        ----------
        process : StochasticProcess
            The stochastic process for which to compute the increments.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of the input process subscripted with `increments`, provided that the name of the input process is a string.

        Raises
        ------
        TypeError
            If `process` is not an instance of `StochasticProcess`.
        ValueError
            If `process` is one-dimensional.

        Returns
        -------
        increments_process : StochasticProcess
            A new stochastic process representing the increments of the input process.
        """
        from ...core.base.time import Time
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")
        if process.dimension == 1:
            raise ValueError(
                "Increments are not defined for one-dimensional processes."
            )

        data_trans = process.data.copy()
        data_trans = data_trans.diff(axis=1).dropna(axis=1)
        if name is None:
            name = f"{process.name}_increments" if process.name is not None else None
        new_index = process.time.data[1:].copy()
        return (
            StochasticProcess(
                name=name,
                domain=process.domain,
                index=Time(
                    name=process.time.name, data_name=process.time.data.name
                ).from_pandas(new_index),
            )
            .from_pandas(data_trans)
            .with_probability_measure(probability_measure=process.probability_measure)
        )

    @classmethod
    def to_counting_process(
        cls,
        process: StochasticProcess,
        time: Time,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Convert a stochastic process of "arrival times" to a counting process.

        The trajectories in the given process are assumed to be the occurrence times of some event, while its time index represents the cumulative counts of those events. This method creates a new stochastic process where, at each time point in the provided `time` index, the value represents the total count of events that have occurred up to that time.

        Parameters
        ----------
        process : StochasticProcess
            The original stochastic process to be converted. The process trajectories must be monotonically increasing.
        time : Time
            The time index for the counting process.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of the input process subscripted with `counting`, provided that the name of the input process is a string.

        Raises
        ------
        TypeError
            If `process` is not an instance of `StochasticProcess`.
        ValueError
            If the trajectories in `process` are not monotonically increasing.

        Returns
        -------
        counting_process : StochasticProcess
            A new stochastic process representing the counting process.

        Examples
        --------
        >>> from scipy.stats import expon
        >>> from sigalg.core import Index, Time
        >>> from sigalg.processes import IIDProcess
        >>> # Parameters for a Poisson process
        >>> rate = 2.0
        >>> max_trajectories = 5
        >>> random_state = 42
        >>> max_count = 5
        >>> # Create an index for the counts
        >>> counts = Index(data_name="count").from_sequence(size=max_count, initial_index=1)
        >>> # Exponential interarrival times with given rate
        >>> interarrival_times = IIDProcess(
        ...     distribution=expon(scale=1 / rate),
        ...     name="interarrival_times",
        ...     index=counts,
        ... ).from_simulation(max_trajectories=max_trajectories, random_state=random_state)
        >>> interarrival_times # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'interarrival_times':
        count         1         2         3         4         5
        0      1.202104  1.168095  1.192380  0.139897  0.043219
        1      0.726330  0.704980  1.562148  0.039647  0.523280
        2      0.035218  0.544512  0.865664  0.193447  0.615793
        3      0.076887  0.045789  0.157590  0.450600  0.206493
        4      0.623693  0.111788  0.918985  0.613543  0.327898
        >>> # Compute arrival times by cumulative sum of interarrival times
        >>> arrival_times = interarrival_times.cumsum().with_name("arrival_times")
        >>> arrival_times # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'arrival_times':
        count         1         2         3         4         5
        0      1.202104  2.370199  3.562580  3.702477  3.745695
        1      0.726330  1.431311  2.993459  3.033106  3.556386
        2      0.035218  0.579730  1.445394  1.638841  2.254634
        3      0.076887  0.122675  0.280265  0.730864  0.937357
        4      0.623693  0.735481  1.654466  2.268009  2.595907
        >>> # Determine time grid for Poisson process
        >>> longest_trajectory = arrival_times.max_value()
        >>> time = Time.continuous(
        ...     start=0.0,
        ...     stop=longest_trajectory + 0.1,
        ...     num_points=6,
        ... )
        >>> # Convert to Poisson counting process
        >>> poisson = arrival_times.to_counting_process(
        ...     time=time,
        ... ).with_name("poisson")
        >>> poisson # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Stochastic process 'poisson':
        time        0.000000  0.769139  1.538278  2.307417  3.076556  3.845695
        0                0.0       0.0       1.0       1.0       2.0       5.0
        1                0.0       1.0       2.0       2.0       4.0       5.0
        2                0.0       2.0       3.0       5.0       5.0       5.0
        3                0.0       4.0       5.0       5.0       5.0       5.0
        4                0.0       2.0       2.0       4.0       5.0       5.0
        """
        from ...core.base.time import Time
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")
        if not isinstance(time, Time):
            raise TypeError("time must be an instance of Time.")
        if not process.is_monotonic():
            raise ValueError(
                "The input process must be monotonic to convert to a counting process."
            )

        data_trans = process.data.copy()

        df_process_stacked = data_trans.stack().reset_index()
        df_process_stacked.columns = [
            "trajectory",
            "count",
            "process_values",
        ]

        df_time = pd.DataFrame(
            {
                "time": np.tile(time.data, len(data_trans)),
                "trajectory": np.repeat(data_trans.index, len(time.data)),
            }
        )

        merged_df = pd.merge_asof(
            left=df_time.sort_values(["time"]),
            right=df_process_stacked.sort_values(["process_values"]),
            left_on="time",
            right_on="process_values",
            by="trajectory",
            direction="backward",
        )

        data_trans = merged_df.pivot(
            index="trajectory",
            columns="time",
            values="count",
        ).fillna(0.0)

        if name is None:
            name = f"{process.name}_counting" if process.name is not None else None
        return (
            StochasticProcess(name=name, domain=process.domain)
            .from_pandas(data_trans)
            .with_probability_measure(probability_measure=process.probability_measure)
        )

    @classmethod
    def pointwise_map(
        cls,
        process: StochasticProcess,
        function: Callable[[Hashable], Hashable],
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Apply a function pointwise to the values of a stochastic process.

        Parameters
        ----------
        process : StochasticProcess
            The stochastic process to which the function will be applied.
        function : Callable[[Hashable], Hashable]
            A function that takes a single value and returns a transformed value. This function will be applied to each value in the stochastic process.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of the input process subscripted with `mapped`, provided that the name of the input process is a string.

        Raises
        ------
        TypeError
            If `process` is not an instance of `StochasticProcess`, or if `function` is not callable.
        ValueError
            If `process` does not have data to apply the function to.

        Returns
        -------
        mapped_process : StochasticProcess
            A new stochastic process with the function applied pointwise to its values.
        """
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")
        if not isinstance(function, Callable):
            raise TypeError("function must be a callable object.")

        data_trans = process.data.copy()
        data_trans = data_trans.map(function)
        if name is None:
            name = f"{process.name}_mapped" if process.name is not None else None
        return (
            StochasticProcess(name=name, domain=process.domain, index=process.time)
            .from_pandas(data_trans)
            .with_probability_measure(probability_measure=process.probability_measure)
        )

    @classmethod
    def timewise_map(
        cls,
        process: StochasticProcess,
        time: Real,
        function: Callable[[Hashable], Hashable],
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Apply a function to the values of a stochastic process at a specific time point.

        Parameters
        ----------
        process : StochasticProcess
            The stochastic process to which the function will be applied.
        time : Real
            The specific time point at which to apply the function.
        function : Callable[[Hashable], Hashable]
            A function that takes a single value and returns a transformed value. This function will be applied to the values of the stochastic process at the specified time point.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of the input process subscripted with `mapped`, provided that the name of the input process is a string.

        Raises
        ------
        TypeError
            If `process` is not an instance of `StochasticProcess`, if `function` is not callable, or if `time` is not a real number.
        ValueError
            If `process` does not have data to apply the function to, or if `time` is not a valid time point in the process.

        Returns
        -------
        mapped_process : StochasticProcess
            A new stochastic process with the function applied to its values at the specified time point.
        """
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")
        if not isinstance(function, Callable):
            raise TypeError("function must be a callable object.")
        if time not in process.time.data:
            raise ValueError("time must be a valid time point in the process.")

        data_trans = process.data.copy()
        data_trans[time] = data_trans[time].map(function)
        if name is None:
            name = f"{process.name}_mapped" if process.name is not None else None
        return (
            StochasticProcess(name=name, domain=process.domain, index=process.time)
            .from_pandas(data_trans)
            .with_probability_measure(probability_measure=process.probability_measure)
        )

    @staticmethod
    def max_value(process: StochasticProcess) -> Real:
        """Get the maximum value across all trajectories and time points of a stochastic process.

        Parameters
        ----------
        process : StochasticProcess
            The stochastic process for which to find the maximum value.

        Raises
        ------
        TypeError
            If `process` is not an instance of `StochasticProcess`.

        Returns
        -------
        max_value : Real
            The maximum value found in the stochastic process.
        """
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")
        return process.data.values.max()

    @staticmethod
    def is_monotonic(process: StochasticProcess, increasing: bool = True) -> bool:
        """Check if the trajectories of a stochastic process are monotonic.

        Parameters
        ----------
        process : StochasticProcess
            The stochastic process to check for monotonicity.
        increasing : bool, default=True
            If `True`, check for monotonically increasing trajectories; if `False`, check for monotonically decreasing trajectories.

        Raises
        ------
        TypeError
            If `process` is not an instance of `StochasticProcess`, or if `increasing` is not a boolean value.

        Returns
        -------
        is_monotonic : bool
            `True` if all trajectories are monotonic in the specified direction, `False` otherwise.
        """
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")
        if not isinstance(increasing, bool):
            raise TypeError("increasing must be a boolean value.")
        diffs = process.data.diff(axis=1).dropna(axis=1)
        if increasing:
            return bool((diffs >= 0).all().all())
        else:
            return bool((diffs <= 0).all().all())

    # @staticmethod
    # def time_shift(process: StochasticProcess, shift: int) -> StochasticProcess:
    # shifted_data = process._data.shift(periods=shift, axis=1)
    # return SampleSpaceFeatures(
    #     features=shifted_data,
    #     sample_space=process.sample_space,
    #     feature_index=list(process._data.columns),
    # )
    # pass

    # @staticmethod
    # def running_maximum(process: StochasticProcess) -> StochasticProcess:
    # max_data = process._data.cummax(axis=1)
    # return SampleSpaceFeatures(
    #     features=max_data,
    #     sample_space=process.sample_space,
    #     feature_index=list(process._data.columns),
    # )
    # pass

    # @staticmethod
    # def moving_average(process: StochasticProcess, window: int) -> StochasticProcess:
    # ma_data = process._data.rolling(window=window, axis=1).mean()
    # return SampleSpaceFeatures(
    #     features=ma_data,
    #     sample_space=process.sample_space,
    #     feature_index=list(process._data.columns),
    # )
    # pass

    # @staticmethod
    # def compose(
    #     process1: StochasticProcess,
    #     process2: StochasticProcess,
    #     op: Callable[[float, float], float],
    # ) -> StochasticProcess:
    # if not process1.sample_space == process2.sample_space:
    #     raise ValueError("Processes must have the same sample space")
    # result_data = op(process1._data, process2._data)
    # return SampleSpaceFeatures(
    #     features=result_data,
    #     sample_space=process1.sample_space,
    #     feature_index=list(process1._data.columns),
    # )
    # pass

    # @staticmethod
    # def stopped_process(
    #     process: StochasticProcess, stopping_times: dict[Hashable, int]
    # ) -> StochasticProcess:
    # stopped_data = process._data.copy()

    # for omega in process.sample_space.index:
    #     if omega in stopping_times:
    #         tau = stopping_times[omega]
    #         # Get column positions
    #         cols = list(process._data.columns)
    #         tau_idx = cols.index(tau) if tau in cols else len(cols) - 1
    #         # After tau, keep the value constant
    #         for j in range(tau_idx + 1, len(cols)):
    #             stopped_data.loc[omega, cols[j]] = stopped_data.loc[
    #                 omega, cols[tau_idx]
    #             ]

    # return SampleSpaceFeatures(
    #     features=stopped_data,
    #     sample_space=process.sample_space,
    #     feature_index=list(process._data.columns),
    # )
    # pass


class ProcessTransformMethods:
    """Mixin class providing transformation methods for `StochasticProcess`."""

    def cumsum(self, name: Hashable | None = None) -> StochasticProcess:
        """Compute the cumulative sum of the stochastic process along its time index.

        Parameters
        ----------
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of `self` subscripted with `cumsum`, provided that the name of `self` is a string.

        Returns
        -------
        cumsum_process : StochasticProcess
            A new stochastic process representing the cumulative sum of the input process.
        """
        return ProcessTransforms.cumsum(self, name=name)

    def increments(self, name: Hashable | None = None) -> StochasticProcess:
        """Compute the increments of the stochastic process along its time index.

        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of `self` subscripted with `increments`, provided that the name of `self` is a string.

        Returns
        -------
        increments_process : StochasticProcess
            A new stochastic process representing the increments of the input process.
        """
        return ProcessTransforms.increments(self, name=name)

    def to_counting_process(
        self,
        time: Time,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Convert the stochastic process of "arrival times" to a counting process.

        The trajectories in the process are assumed to be the occurrence times of some event, while its time index represents the cumulative counts of those events. This method creates a new stochastic process where, at each time point in the provided `time` index, the value represents the total count of events that have occurred up to that time.

        Parameters
        ----------
        time : Time
            The time index for the counting process.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of `self` subscripted with `counting`, provided that the name of `self` is a string.

        Returns
        -------
        counting_process : StochasticProcess
            A new stochastic process representing the counting process.

        Examples
        --------
        >>> from scipy.stats import expon
        >>> from sigalg.core import Index, Time
        >>> from sigalg.processes import IIDProcess
        >>> # Parameters for a Poisson process
        >>> rate = 2.0
        >>> max_trajectories = 5
        >>> random_state = 42
        >>> max_count = 5
        >>> # Create an index for the counts
        >>> counts = Index(data_name="count").from_sequence(size=max_count, initial_index=1)
        >>> # Exponential interarrival times with given rate
        >>> interarrival_times = IIDProcess(
        ...     distribution=expon(scale=1 / rate),
        ...     name="interarrival_times",
        ...     index=counts,
        ... ).from_simulation(max_trajectories=max_trajectories, random_state=random_state)
        >>> interarrival_times # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'interarrival_times':
        count         1         2         3         4         5
        0      1.202104  1.168095  1.192380  0.139897  0.043219
        1      0.726330  0.704980  1.562148  0.039647  0.523280
        2      0.035218  0.544512  0.865664  0.193447  0.615793
        3      0.076887  0.045789  0.157590  0.450600  0.206493
        4      0.623693  0.111788  0.918985  0.613543  0.327898
        >>> # Compute arrival times by cumulative sum of interarrival times
        >>> arrival_times = interarrival_times.cumsum().with_name("arrival_times")
        >>> arrival_times # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'arrival_times':
        count         1         2         3         4         5
        0      1.202104  2.370199  3.562580  3.702477  3.745695
        1      0.726330  1.431311  2.993459  3.033106  3.556386
        2      0.035218  0.579730  1.445394  1.638841  2.254634
        3      0.076887  0.122675  0.280265  0.730864  0.937357
        4      0.623693  0.735481  1.654466  2.268009  2.595907
        >>> # Determine time grid for Poisson process
        >>> longest_trajectory = arrival_times.max_value()
        >>> time = Time.continuous(
        ...     start=0.0,
        ...     stop=longest_trajectory + 0.1,
        ...     num_points=6,
        ... )
        >>> # Convert to Poisson counting process
        >>> poisson = arrival_times.to_counting_process(
        ...     time=time,
        ... ).with_name("poisson")
        >>> poisson # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Stochastic process 'poisson':
        time        0.000000  0.769139  1.538278  2.307417  3.076556  3.845695
        0                0.0       0.0       1.0       1.0       2.0       5.0
        1                0.0       1.0       2.0       2.0       4.0       5.0
        2                0.0       2.0       3.0       5.0       5.0       5.0
        3                0.0       4.0       5.0       5.0       5.0       5.0
        4                0.0       2.0       2.0       4.0       5.0       5.0
        """
        return ProcessTransforms.to_counting_process(self, time=time, name=name)

    def pointwise_map(
        self,
        function: Callable[[Hashable], Hashable],
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Apply a function pointwise to the values of the stochastic process.

        Parameters
        ----------
        function : Callable[[Hashable], Hashable]
            A function that takes a single value and returns a transformed value. This function will be applied to each value in the stochastic process.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of `self` subscripted with `mapped`, provided that the name of `self` is a string.

        Returns
        -------
        mapped_process : StochasticProcess
            A new stochastic process with the function applied pointwise to its values.
        """
        return ProcessTransforms.pointwise_map(self, function=function, name=name)

    def timewise_map(
        self,
        time: Real,
        function: Callable[[Hashable], Hashable],
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Apply a function to the values of the stochastic process at a specific time point.

        Parameters
        ----------
        time : Real
            The specific time point at which to apply the function.
        function : Callable[[Hashable], Hashable]
            A function that takes a single value and returns a transformed value. This function will be applied to the values of the stochastic process at the specified time point.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of `self` subscripted with `mapped`, provided that the name of `self` is a string.

        Returns
        -------
        mapped_process : StochasticProcess
            A new stochastic process with the function applied to its values at the specified time point.
        """
        return ProcessTransforms.timewise_map(
            self, time=time, function=function, name=name
        )

    def max_value(self) -> Real:
        """Get the maximum value across all trajectories and time points of the stochastic process.

        Returns
        -------
        max_value : Real
            The maximum value found in the stochastic process.
        """
        return ProcessTransforms.max_value(self)

    def is_monotonic(self, increasing: bool = True) -> bool:
        """Check if the trajectories of the stochastic process are monotonic.

        Parameters
        ----------
        increasing : bool, default=True
            If `True`, check for monotonically increasing trajectories; if `False`, check for monotonically decreasing trajectories.

        Returns
        -------
        is_monotonic : bool
            `True` if all trajectories are monotonic in the specified direction, `False` otherwise.
        """
        return ProcessTransforms.is_monotonic(self, increasing)
