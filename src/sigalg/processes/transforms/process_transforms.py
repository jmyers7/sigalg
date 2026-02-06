"""Stochastic process transformation module."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ...core.base.time import Time

if TYPE_CHECKING:
    from ...core.random_objects.random_variable import RandomVariable
    from ..base.stochastic_process import StochasticProcess


class ProcessTransforms:
    """A collection of methods for transforming stochastic processes."""

    @staticmethod
    def transform(
        process: StochasticProcess,
        functions: list[Callable[[StochasticProcess], RandomVariable]],
        new_time: Time,
        name: Hashable | None = None,
    ):
        """Apply a transformation to a stochastic process.

        Parameters
        ----------
        process : StochasticProcess
            The stochastic process to transform.
        functions : list[Callable[[StochasticProcess], RandomVariable]]
            A list of functions to apply to the stochastic process.
        new_time : Time
            The new time index for the transformed process.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be `function(process.name)` if `process.name` is not `None`.

        Returns
        -------
        StochasticProcess
            The transformed stochastic process.
        """
        from ...core.base.time import Time
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")
        if not isinstance(functions, list):
            raise TypeError("functions must be a list.")
        if not isinstance(new_time, Time):
            raise TypeError("new_time must be an instance of Time.")
        if len(functions) != len(new_time):
            raise ValueError(
                "The number of functions must match the length of new_time."
            )
        for f in functions:
            if not isinstance(f, Callable):
                raise TypeError("Each element in functions must be callable.")

        transformed_rvs = {}

        for f, t in zip(functions, new_time, strict=False):
            transformed_rvs[t] = f(process).data

        data = pd.DataFrame(transformed_rvs, index=process.domain)

        if name is None:
            name = f"function({process.name})" if process.name is not None else None
        result = StochasticProcess(
            domain=process.domain, time=new_time, name=name
        ).from_pandas(data)
        result._probability_measure = process.probability_measure

        return result

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
            StochasticProcess(name=name, domain=process.domain, time=process.time)
            .from_pandas(data_trans)
            .with_probability_measure(probability_measure=process.probability_measure)
        )

    @classmethod
    def cumprod(
        cls, process: StochasticProcess, name: Hashable | None = None
    ) -> StochasticProcess:
        """Compute the cumulative product of a stochastic process along its time index.

        Parameters
        ----------
        process : StochasticProcess
            The stochastic process for which to compute the cumulative product.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of the input process subscripted with `cumprod`, provided that the name of the input process is a string.

        Raises
        ------
        TypeError
            If `process` is not an instance of `StochasticProcess`.

        Returns
        -------
        cumprod_process : StochasticProcess
            A new stochastic process representing the cumulative product of the input process.
        """
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")
        data_trans = process.data.copy()
        data_trans = data_trans.cumprod(axis=1)
        if name is None:
            name = f"{process.name}_cumprod" if process.name is not None else None
        return (
            StochasticProcess(name=name, domain=process.domain, time=process.time)
            .from_pandas(data_trans)
            .with_probability_measure(probability_measure=process.probability_measure)
        )

    @classmethod
    def sum(
        cls, process: StochasticProcess, name: Hashable | None = None
    ) -> RandomVariable:
        """Compute the sum of a stochastic process across its time index.

        Parameters
        ----------
        process : StochasticProcess
            The stochastic process for which to compute the sum.
        name : Hashable | None, default=None
            The name of the transformed random variable. If `None`, the new name will be the name of the input process subscripted with `sum`, provided that the name of the input process is a string.

        Raises
        ------
        TypeError
            If `process` is not an instance of `StochasticProcess`.

        Returns
        -------
        sum_variable : RandomVariable
            A new random variable representing the sum of the input process across its time index.
        """
        from ...core.random_objects.random_variable import RandomVariable
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")

        data_trans = process.data.copy()
        data_trans = data_trans.sum(axis=1)

        if name is None:
            name = f"{process.name}_sum" if process.name is not None else None

        return (
            RandomVariable(name=name, domain=process.domain)
            .from_pandas(data_trans)
            .with_probability_measure(probability_measure=process.probability_measure)
        )

    @classmethod
    def increments(
        cls,
        process: StochasticProcess,
        forward: bool = False,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Compute the increments of a stochastic process along its time index.

        Parameters
        ----------
        process : StochasticProcess
            The stochastic process for which to compute the increments.
        forward : bool, default=False
            If `True`, compute forward increments, i.e., X(t) is replaced with X(t+1) - X(t). If `False`, compute backward increments, i.e., X(t) is replaced with X(t) - X(t-1).
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

        if forward:
            data_trans = -1 * data_trans.diff(periods=-1, axis=1).dropna(axis=1)
            new_time = Time(
                name=process.time.name, data_name=process.time.data.name
            ).from_pandas(process.time.data[:-1])
        else:
            data_trans = data_trans.diff(axis=1).dropna(axis=1)
            new_time = Time(
                name=process.time.name, data_name=process.time.data.name
            ).from_pandas(process.time.data[1:])
        new_time.is_discrete = process.time.is_discrete

        if name is None:
            name = f"{process.name}_increments" if process.name is not None else None

        return (
            StochasticProcess(
                name=name,
                domain=process.domain,
                time=new_time,
            )
            .from_pandas(data_trans)
            .with_probability_measure(probability_measure=process.probability_measure)
        )

    @staticmethod
    def ito_integral(
        integrand: StochasticProcess,
        integrator: StochasticProcess,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Compute the Itô integral of a stochastic process with respect to another stochastic process.

        Parameters
        ----------
        integrand : StochasticProcess
            The stochastic process to be integrated.
        integrator : StochasticProcess
            The stochastic process with respect to which the integral is computed.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be `int X dW`, where `X` is the name of the integrand and `W` is the name of the integrator.

        Returns
        -------
        ito_integral_process : StochasticProcess
            A new stochastic process representing the Itô integral of the input process with respect to the integrator.
        """
        name = f"int {integrand.name} d{integrator.name}" if name is None else None

        if integrand.time == integrator.time:
            return (
                (integrand.drop_initial_state() * integrator.increments())
                .sum()
                .with_name(name)
            )
        elif np.all(integrand.time.data == integrator.time.data[1:]):
            return (integrand * integrator.increments()).sum().with_name(name)
        else:
            raise ValueError("Incompatible time indices for Itô integral.")

    @staticmethod
    def drop_initial_state(
        process: StochasticProcess, name: Hashable | None = None
    ) -> StochasticProcess:
        """Drop the initial state of a stochastic process.

        Parameters
        ----------
        process : StochasticProcess
            The stochastic process from which to drop the initial state.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of the input process subscripted with `dropped_initial`, provided that the name of the input process is a string.

        Raises
        ------
        TypeError
            If `process` is not an instance of `StochasticProcess`.
        ValueError
            If `process` is one-dimensional.

        Returns
        -------
        dropped_initial_process : StochasticProcess
            A new stochastic process with the initial state dropped.
        """
        from ...core.base.time import Time
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")
        if process.dimension == 1:
            raise ValueError(
                "Cannot drop the initial state of a one-dimensional process."
            )

        new_time = Time(
            name=process.time.name, data_name=process.time.data.name
        ).from_pandas(process.time.data[1:])
        new_time.is_discrete = process.time.is_discrete

        data_trans = process.data.iloc[:, 1:]

        name = f"{process.name}_dropped_initial" if process.name is not None else None

        return (
            StochasticProcess(
                name=name,
                domain=process.domain,
                time=new_time,
            )
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
    def min_value(process: StochasticProcess) -> Real:
        """Get the minimum value across all trajectories and time points of a stochastic process.

        Parameters
        ----------
        process : StochasticProcess
            The stochastic process for which to find the minimum value.

        Raises
        ------
        TypeError
            If `process` is not an instance of `StochasticProcess`.

        Returns
        -------
        min_value : Real
            The minimum value found in the stochastic process.
        """
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")
        return process.data.values.min()

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
            StochasticProcess(name=name, domain=process.domain, time=process.time)
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
            StochasticProcess(
                name=name,
                domain=process.domain,
                time=process.time,
                is_discrete_time=process.is_discrete_time,
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
        >>> n_trajectories = 5
        >>> random_state = 42
        >>> max_count = 5
        >>> # Create an index for the counts
        >>> counts = Time.discrete(start=1, stop=max_count, data_name="count", name=None)
        >>> # Exponential interarrival times with given rate
        >>> interarrival_times = IIDProcess(
        ...     distribution=expon(scale=1 / rate),
        ...     name="interarrival_times",
        ...     time=counts,
        ... ).from_simulation(n_trajectories=n_trajectories, random_state=random_state)
        >>> interarrival_times # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'interarrival_times':
        count         1         2         3         4         5
        trajectory
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
        trajectory
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
        trajectory
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
            StochasticProcess(
                name=name,
                domain=process.domain,
                is_discrete_time=False,
            )
            .from_pandas(data_trans)
            .with_probability_measure(probability_measure=process.probability_measure)
        )


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

    def cumprod(self, name: Hashable | None = None) -> StochasticProcess:
        """Compute the cumulative product of the stochastic process along its time index.

        Parameters
        ----------
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of `self` subscripted with `cumprod`, provided that the name of `self` is a string.

        Returns
        -------
        cumprod_process : StochasticProcess
            A new stochastic process representing the cumulative product of the input process.
        """
        return ProcessTransforms.cumprod(self, name=name)

    def sum(self, name: Hashable | None = None) -> RandomVariable:
        """Compute the sum of the stochastic process across its time index.

        Parameters
        ----------
        name : Hashable | None, default=None
            The name of the transformed random variable. If `None`, the new name will be the name of `self` subscripted with `sum`, provided that the name of `self` is a string.

        Returns
        -------
        sum_random_variable : RandomVariable
            A new random variable representing the sum of the input stochastic process.
        """
        return ProcessTransforms.sum(self, name=name)

    def increments(
        self, forward: bool = False, name: Hashable | None = None
    ) -> StochasticProcess:
        """Compute the increments of the stochastic process along its time index.

        Parameters
        ----------
        forward : bool, default=False
            If `True`, compute forward increments, i.e., X(t) is replaced with X(t+1) - X(t). If `False`, compute backward increments, i.e., X(t) is replaced with X(t) - X(t-1).
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of `self` subscripted with `increments`, provided that the name of `self` is a string.

        Returns
        -------
        increments_process : StochasticProcess
            A new stochastic process representing the increments of the input process.
        """
        return ProcessTransforms.increments(self, forward=forward, name=name)

    def ito_integral(
        self,
        integrator: StochasticProcess,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Compute the Itô integral of the stochastic process with respect to another stochastic process.

        Parameters
        ----------
        integrator : StochasticProcess
            The stochastic process with respect to which the integral is computed.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be `int X dW`, where `X` is the name of the integrand and `W` is the name of the integrator.

        Returns
        -------
        ito_integral_process : StochasticProcess
            A new stochastic process representing the Itô integral of the input process with respect to the integrator.
        """
        return ProcessTransforms.ito_integral(self, integrator=integrator, name=name)

    def drop_initial_state(self, name: Hashable | None = None) -> StochasticProcess:
        """Drop the initial state of the stochastic process.

        Parameters
        ----------
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of `self` subscripted with `dropped_initial`, provided that the name of `self` is a string.

        Returns
        -------
        dropped_initial_process : StochasticProcess
            A new stochastic process with the initial state dropped.

        Raises
        ------
        TypeError
            If `self` is not an instance of `StochasticProcess`.
        ValueError
            If `self` is one-dimensional.
        """
        return ProcessTransforms.drop_initial_state(self, name=name)

    def max_value(self) -> Real:
        """Get the maximum value across all trajectories and time points of the stochastic process.

        Returns
        -------
        max_value : Real
            The maximum value found in the stochastic process.
        """
        return ProcessTransforms.max_value(self)

    def min_value(self) -> Real:
        """Get the minimum value across all trajectories and time points of the stochastic process.

        Returns
        -------
        min_value : Real
            The minimum value found in the stochastic process.
        """
        return ProcessTransforms.min_value(self)

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
        >>> n_trajectories = 5
        >>> random_state = 42
        >>> max_count = 5
        >>> # Create an index for the counts
        >>> counts = Time.discrete(start=1, stop=max_count, data_name="count", name=None)
        >>> # Exponential interarrival times with given rate
        >>> interarrival_times = IIDProcess(
        ...     distribution=expon(scale=1 / rate),
        ...     name="interarrival_times",
        ...     time=counts,
        ... ).from_simulation(n_trajectories=n_trajectories, random_state=random_state)
        >>> interarrival_times # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'interarrival_times':
        count         1         2         3         4         5
        trajectory
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
        trajectory
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
        trajectory
        0                0.0       0.0       1.0       1.0       2.0       5.0
        1                0.0       1.0       2.0       2.0       4.0       5.0
        2                0.0       2.0       3.0       5.0       5.0       5.0
        3                0.0       4.0       5.0       5.0       5.0       5.0
        4                0.0       2.0       2.0       4.0       5.0       5.0
        """
        return ProcessTransforms.to_counting_process(self, time=time, name=name)
