"""A module containing transforms of stochastic processes.

Classes
-------
ProcessTransforms
    A class containing methods for transforming stochastic processes.
ProcessTransformsMethods
    A mixin class that adds transformation methods to the StochasticProcess class.
"""

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
    from ..stopping_times.stopping_time import StoppingTime


# TODO: Update docstrings
# TODO: Check that transforms preserve things like probability measures, the `_is_enumerate` attribute, and the `is_discrete_state` attribute
class ProcessTransforms:
    """A collection of methods for transforming stochastic processes."""

    # TODO: Update docstrings
    @staticmethod
    def transform(
        process: StochasticProcess,
        functions: list[Callable[[StochasticProcess], RandomVariable]],
        time: Time | None = None,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Apply a transformation to a stochastic process.

        Parameters
        ----------
        process : StochasticProcess
            The stochastic process to transform.
        functions : list[Callable[[StochasticProcess], RandomVariable]]
            A list of functions to apply to the stochastic process.
        time : Time | None, default=None
            The new time index for the transformed process. If `None`, the original time index of `process` will be used.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be `function(process.name)` if `process.name` is not `None`.

        Raises
        ------
        TypeError
            If `process` is not an instance of `StochasticProcess`, or `functions` is not a list of callables, or `time` is not an instance of `Time`.
        ValueError
            If the length of `functions` does not match the length of `time`.

        Returns
        -------
        transformed_process : StochasticProcess
            The transformed stochastic process.

        Examples
        --------
        >>> from scipy.stats import bernoulli
        >>> from sigalg.core import RandomVariable, Time
        >>> from sigalg.processes import IIDProcess, StochasticProcess
        >>> T = Time().discrete(start=0, length=2)
        >>> X = IIDProcess(distribution=bernoulli(p=0.5), support=[0, 1], time=T).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        0  1  2
        trajectory
        0           0  0  0
        1           0  0  1
        2           0  1  0
        3           0  1  1
        4           1  0  0
        5           1  0  1
        6           1  1  0
        7           1  1  1
        >>> S = Time().discrete(start=4, stop=5)
        >>> def f4(process: StochasticProcess) -> RandomVariable:
        ...     X0, X1, _ = X
        ...     return X0 + X1
        >>> def f5(process: StochasticProcess) -> RandomVariable:
        ...     _, X1, X2 = X
        ...     return X1 + X2
        >>> print(X.transform(functions=[f4, f5], time=S)) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'function(X)':
        time        4  5
        trajectory
        0           0  0
        1           0  1
        2           1  1
        3           1  2
        4           1  0
        5           1  1
        6           2  1
        7           2  2
        """
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")
        if not isinstance(functions, list):
            raise TypeError("functions must be a list.")
        if time is not None and not isinstance(time, Time):
            raise TypeError("time must be an instance of Time.")
        if time is not None and len(functions) != len(time):
            raise ValueError("The number of functions must match the length of time.")
        for f in functions:
            if not isinstance(f, Callable):
                raise TypeError("Each element in functions must be callable.")

        if time is None:
            time = process.time

        transformed_rvs = {}

        for f, t in zip(functions, time, strict=False):
            transformed_rvs[t] = f(process).data

        data = pd.DataFrame(
            transformed_rvs, index=process.domain.data, columns=time.data
        )

        if name is None:
            name = f"function({process.name})" if process.name is not None else None

        result = StochasticProcess(
            sample_space=process.domain, time=time, name=name
        ).from_pandas(data)

        result._probability_measure = process.prob_measure
        result._is_discrete_state = process.is_discrete_state
        result._is_discrete_time = process.is_discrete_time

        return result

    # TODO: Update docstrings
    @staticmethod
    def pointwise_map(
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

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk(p=0.5, time=T, initial_state=3).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        0  1  2
        trajectory
        0           3  2  1
        1           3  2  3
        2           3  4  3
        3           3  4  5
        >>> def f(x):
        ...     return x + 1
        >>> print(X.pointwise_map(function=f)) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_mapped':
        time        0  1  2
        trajectory
        0           4  3  2
        1           4  3  4
        2           4  5  4
        3           4  5  6
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
            StochasticProcess(name=name, sample_space=process.domain, time=process.time)
            .from_pandas(data_trans)
            .with_probability_measure(prob_measure=process.prob_measure)
        )

    # TODO: Update docstrings
    @staticmethod
    def insert_rv(
        process: StochasticProcess,
        time: Real,
        rv: RandomVariable | None = None,
        state: Hashable | None = None,
        name: Hashable | None = None,
        in_place: bool = False,
    ) -> StochasticProcess:
        """Insert a random variable to a stochastic process at a specific time.

        Parameters
        ----------
        process : StochasticProcess
            The stochastic process to which the random variable will be inserted.
        time : Real
            The time at which to insert the random variable.
        rv : RandomVariable | None, default=None
            The random variable to insert. One or the other of `rv` or `state` must be provided, but not both.
        state: Hashable | None, default=None
            A constant state to assign to the inserted random variable for all trajectories. One or the other of `rv` or `state` must be provided, but not both.
        name : Hashable | None, default=None
            The name of the new stochastic process. If `None`, the new name will be `insert(process.name)` if `process.name` is not `None`.
        in_place : bool, default=False
            If `True`, modify the input process in place and return it. If `False`, return a new stochastic process with the random variable inserted, leaving the input process unchanged.

        Raises
        ------
        TypeError
            If `process` is not an instance of `StochasticProcess`, or `rv` is not an instance of `RandomVariable`, or `time` is not a real number.
        ValueError
            If `process` has no data, or if `process` and `rv` do not have the same domain.

        Returns
        -------
        inserted_process : StochasticProcess
            A new stochastic process with the random variable inserted at the specified time.

        Examples
        --------
        >>> from scipy.stats import bernoulli
        >>> from sigalg.core import RandomVariable, Time
        >>> from sigalg.processes import IIDProcess
        >>> T = Time().discrete(start=1, length=2)
        >>> X = IIDProcess(distribution=bernoulli(p=0.5), support=[0, 1], time=T).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        1  2  3
        trajectory
        0           0  0  0
        1           0  0  1
        2           0  1  0
        3           0  1  1
        4           1  0  0
        5           1  0  1
        6           1  1  0
        7           1  1  1
        >>> X0 = RandomVariable(domain=X.domain).from_constant(0)
        >>> print(X.insert_rv(rv=X0, time=0)) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'insert(X)':
        time        0  1  2  3
        trajectory
        0           0  0  0  0
        1           0  0  0  1
        2           0  0  1  0
        3           0  0  1  1
        4           0  1  0  0
        5           0  1  0  1
        6           0  1  1  0
        7           0  1  1  1
        """
        from ...core.random_objects.random_variable import RandomVariable
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")
        if not isinstance(time, Real):
            raise TypeError("time must be a real number.")
        if process.data is None:
            raise ValueError("process has no data.")
        if rv is not None and not isinstance(rv, RandomVariable):
            raise TypeError(
                "If rv is provided, it must be an instance of RandomVariable."
            )
        if state is not None and not isinstance(state, Hashable):
            raise TypeError("If state is provided, it must be a hashable object.")
        if (rv is None and state is None) or (rv is not None and state is not None):
            raise ValueError(
                "Exactly one of rv or state must be provided, but not both."
            )
        if rv is not None and process.domain != rv.domain:
            raise ValueError("process and rv must have the same domain.")

        new_time = process.time.insert_time(time)

        if rv is None:
            rv = RandomVariable(sample_space=process.domain).from_constant(state)

        if in_place:
            pos = process.data.columns.searchsorted(time)
            process.data.insert(pos, time, rv.data)
            process._index = new_time
            if name is not None:
                process.name = name
            return process
        else:
            new_data = process.data.copy()
            pos = new_data.columns.searchsorted(time)
            new_data.insert(pos, time, rv.data)
            if name is None:
                name = f"insert({process.name})" if process.name is not None else None

            return (
                StochasticProcess(
                    sample_space=process.domain,
                    time=new_time,
                    name=name,
                    is_discrete_state=process.is_discrete_state,
                )
                .from_pandas(new_data)
                .with_probability_measure(prob_measure=process.prob_measure)
            )

    # TODO: Update docstrings
    @staticmethod
    def remove_rv(
        process: StochasticProcess,
        time: Real | None = None,
        pos: int | None = None,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Remove a random variable from a stochastic process at a specified time.

        Parameters
        ----------
        process : StochasticProcess
            The stochastic process from which to remove the random variable.
        time : Real | None, default=None
            The time point at which to remove the random variable. If `None`, `pos` must be specified.
        pos : int | None, default=None
            The position at which to remove the random variable. If `None`, `time` must be specified.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of the input process subscripted with `remove`, provided that the name of the input process is a string.

        Raises
        ------
        TypeError
            If `process` is not an instance of `StochasticProcess`, if `time` is not a real number, or if `pos` is not an integer.
        ValueError
            If `process` has no data, if `time` is not in the process time index, if both `time` and `pos` are specified, or if neither is specified.

        Returns
        -------
        removed_process : StochasticProcess
            A new stochastic process with the random variable removed at the specified time.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(start=1, length=2)
        >>> X = RandomWalk(p=0.6, time=T).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        1  2  3
        trajectory
        0           0 -1 -2
        1           0 -1  0
        2           0  1  0
        3           0  1  2
        >>> print(X.remove_rv(time=2)) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'remove(X)':
        time        1  3
        trajectory
        0           0 -2
        1           0  0
        2           0  0
        3           0  2
        >>> S = Time.continuous(start=0, stop=0.3, dt=0.101)
        >>> Y = RandomWalk(p=0.6, time=S, name="Y").from_enumeration()
        >>> print(Y) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'Y':
        time        0.0  0.1  0.2  0.3
        trajectory
        0             0   -1   -2   -3
        1             0   -1   -2   -1
        2             0   -1    0   -1
        3             0   -1    0    1
        4             0    1    0   -1
        5             0    1    0    1
        6             0    1    2    1
        7             0    1    2    3
        >>> print(Y.remove_rv(pos=2)) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'remove(Y)':
        time        0.0  0.1  0.3
        trajectory
        0             0   -1   -3
        1             0   -1   -1
        2             0   -1   -1
        3             0   -1    1
        4             0    1   -1
        5             0    1    1
        6             0    1    1
        7             0    1    3
        """
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")
        if time is not None and not isinstance(time, Real):
            raise TypeError("If provided, time must be a real number.")
        if pos is not None and not isinstance(pos, int):
            raise TypeError("If provided, pos must be an integer.")
        if time is not None and pos is not None:
            raise ValueError("Cannot specify both time and pos.")
        if time is None and pos is None:
            raise ValueError("Must specify exactly one of time or pos.")
        if process.data is None:
            raise ValueError("process has no data.")
        if time is not None and time not in process.time:
            raise ValueError("time must be in the process time index.")

        new_time = process.time.remove_time(time=time, pos=pos)
        new_data = process.data.copy()
        if time is None:
            time = new_data.columns[pos]
        new_data.drop(columns=[time], inplace=True)
        if name is None:
            name = f"remove({process.name})" if process.name is not None else None

        return (
            StochasticProcess(sample_space=process.domain, time=new_time, name=name)
            .from_pandas(new_data)
            .with_probability_measure(prob_measure=process.prob_measure)
        )

    # TODO: Update docstrings
    @staticmethod
    def cumsum(
        process: StochasticProcess, name: Hashable | None = None
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

        Examples
        --------
        >>> from scipy.stats import bernoulli
        >>> from sigalg.core import Time
        >>> from sigalg.processes import IIDProcess
        >>> T = Time.discrete(start=1, length=2)
        >>> X = IIDProcess(distribution=bernoulli(p=0.6), support=[0, 1], time=T).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        1  2  3
        trajectory
        0           0  0  0
        1           0  0  1
        2           0  1  0
        3           0  1  1
        4           1  0  0
        5           1  0  1
        6           1  1  0
        7           1  1  1
        >>> print(X.cumsum()) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_cumsum':
        time        1  2  3
        trajectory
        0           0  0  0
        1           0  0  1
        2           0  1  1
        3           0  1  2
        4           1  1  1
        5           1  1  2
        6           1  2  2
        7           1  2  3
        """
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")

        data_trans = process.data.copy()
        data_trans = data_trans.cumsum(axis=1)
        if name is None:
            name = f"{process.name}_cumsum" if process.name is not None else None
        result = (
            StochasticProcess(
                name=name,
                sample_space=process.domain,
                time=process.time,
                is_discrete_state=process.is_discrete_state,
            )
            .from_pandas(data_trans)
            .with_probability_measure(prob_measure=process.prob_measure)
        )

        return result

    # TODO: Update docstrings
    @staticmethod
    def cumprod(
        process: StochasticProcess, name: Hashable | None = None
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

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=3)
        >>> X = RandomWalk(p=0.5, time=T, initial_state=3).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        0  1  2  3
        trajectory
        0           3  2  1  0
        1           3  2  1  2
        2           3  2  3  2
        3           3  2  3  4
        4           3  4  3  2
        5           3  4  3  4
        6           3  4  5  4
        7           3  4  5  6
        >>> print(X.cumprod()) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_cumprod':
        time        0   1   2    3
        trajectory
        0           3   6   6    0
        1           3   6   6   12
        2           3   6  18   36
        3           3   6  18   72
        4           3  12  36   72
        5           3  12  36  144
        6           3  12  60  240
        7           3  12  60  360
        """
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")

        data_trans = process.data.copy()
        data_trans = data_trans.cumprod(axis=1)
        if name is None:
            name = f"{process.name}_cumprod" if process.name is not None else None
        return (
            StochasticProcess(name=name, sample_space=process.domain, time=process.time)
            .from_pandas(data_trans)
            .with_probability_measure(prob_measure=process.prob_measure)
        )

    # TODO: Update docstrings
    @staticmethod
    def sum(process: StochasticProcess, name: Hashable | None = None) -> RandomVariable:
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

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk(p=0.5, time=T, initial_state=3).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        0  1  2
        trajectory
        0           3  2  1
        1           3  2  3
        2           3  4  3
        3           3  4  5
        >>> print(X.sum()) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_sum':
                    X_sum
        trajectory
        0               6
        1               8
        2              10
        3              12
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
            RandomVariable(name=name, sample_space=process.domain)
            .from_pandas(data_trans)
            .with_probability_measure(prob_measure=process.prob_measure)
        )

    # TODO: Update docstrings
    @staticmethod
    def mean(
        process: StochasticProcess, name: Hashable | None = None
    ) -> RandomVariable:
        """Compute the mean of a stochastic process across its time index.

        Parameters
        ----------
        process : StochasticProcess
            The stochastic process for which to compute the mean.
        name : Hashable | None, default=None
            The name of the transformed random variable. If `None`, the new name will be the name of the input process subscripted with `mean`, provided that the name of the input process is a string.

        Raises
        ------
        TypeError
            If `process` is not an instance of `StochasticProcess`.

        Returns
        -------
        mean_variable : RandomVariable
            A new random variable representing the mean of the input process across its time index.
        """
        from ...core.random_objects.random_variable import RandomVariable
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")

        data_trans = process.data.copy()
        data_trans = data_trans.mean(axis=1)

        if name is None:
            name = f"{process.name}_mean" if process.name is not None else None

        return (
            RandomVariable(name=name, sample_space=process.domain)
            .from_pandas(data_trans)
            .with_probability_measure(prob_measure=process.prob_measure)
        )

    @staticmethod
    def discount(
        process: StochasticProcess, rate: Real, name: Hashable | None = None
    ) -> StochasticProcess:
        r"""Return the discounted process of a given stochastic process.

        The discounted process is given by

        $$
        \tilde{S}_t = \frac{S_t}{(1+r)^t},
        $$

        where $S_t$ is the original process and $r$ is the discount rate.

        Parameters
        ----------
        process : StochasticProcess
            The original process to be discounted.
        rate : Real
            The discount rate, which must be a positive real number.
        name : Hashable | None, default=None
            The name of the discounted process. If `None`, the new name will be the name of the input process subscripted with `discount`, provided that the name of the input process is a string.

        Raises
        ------
        TypeError
            If `process` is not an instance of `StochasticProcess`, if `rate` is not a positive real number, or if `name` is not a hashable object or `None`.

        Returns
        -------
        discounted_process : StochasticProcess
            The discounted process.
        """
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess")
        if not isinstance(rate, Real) or rate <= 0:
            raise TypeError("rate must be a positive real number")
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("name must be a hashable object or None")

        discount_factors = (1 + rate) ** (-process.time.data)
        discounted_data = process.data.multiply(discount_factors, axis=1)
        if name is None:
            name = f"{process.name}_discount" if process.name is not None else None

        result = StochasticProcess(
            sample_space=process.domain,
            time=process.time,
            is_discrete_state=process.is_discrete_state,
            name=name,
        ).from_pandas(discounted_data)
        result._probability_measure = process.prob_measure

        return result

    @staticmethod
    def increments(
        process: StochasticProcess,
        forward: bool = True,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        r"""Compute the increments of a stochastic process along its time index.

        Given a stochastic process $X_t$ with index set $\{t_0,t_0+1,\ldots,T\}$ there are two types of increments that can be computed: The first are *forward* increments, which results in a stochastic process $\Delta X_t$ defined as

        $$
        \Delta X_t = X_{t+1} - X_t,
        $$

        for each $t=t_0,\ldots,T-1$. The second type are *backward* increments, which results in a stochastic process $\Delta X_t$ where

        $$
        \Delta X_t = X_t - X_{t-1},
        $$

        for each $t=t_0+1,\ldots,T$.

        Parameters
        ----------
        process : StochasticProcess
            The stochastic process for which to compute the increments.
        forward : bool, default=True
            If `True`, compute forward increments; otherwise, compute backward increments.
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

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk(p=0.5, time=T, initial_state=3).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        0  1  2
        trajectory
        0           3  2  1
        1           3  2  3
        2           3  4  3
        3           3  4  5
        >>> print(X.increments(forward=True)) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_increments':
        time        0  1
        trajectory
        0          -1 -1
        1          -1  1
        2           1 -1
        3           1  1
        >>> print(X.increments(forward=False)) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_increments':
        time        1  2
        trajectory
        0          -1 -1
        1          -1  1
        2           1 -1
        3           1  1
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

        output = (
            StochasticProcess(
                name=name,
                sample_space=process.domain,
                time=new_time,
            )
            .from_pandas(data_trans)
            .with_probability_measure(prob_measure=process.prob_measure)
        )
        output.is_discrete_state = process.is_discrete_state

        return output

    # TODO: Update docstrings
    @staticmethod
    def stopped(
        process: StochasticProcess,
        stopping_time: StoppingTime,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Pass."""
        from ..base.stochastic_process import StochasticProcess
        from ..stopping_times.stopping_time import StoppingTime

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess")
        if not isinstance(stopping_time, StoppingTime):
            raise TypeError("stopping_time must be an instance of StoppingTime")
        if process.time != stopping_time.time:
            raise ValueError(
                "The time indices of the process and stopping time must match"
            )

        data = process.data
        time_arr = process.time.data.values
        stopping_arr = stopping_time.data.values.reshape(-1, 1)

        col_idx = np.minimum(time_arr[None, :], stopping_arr).astype(int)
        row_idx = np.arange(len(data))[:, None]

        stopped_data = pd.DataFrame(
            data.to_numpy()[row_idx, col_idx],
        )

        if name is None:
            name = (
                f"{process.name}^{stopping_time.name}"
                if process.name is not None and stopping_time.name is not None
                else None
            )

        stopped_process = StochasticProcess(
            name=name,
            sample_space=process.domain,
            time=process.time,
        ).from_pandas(stopped_data)
        stopped_process.prob_measure = process.prob_measure

        return stopped_process

    # TODO: Update docstrings
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

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk, StochasticProcess
        >>> time = Time().discrete(length=2)
        >>> X = RandomWalk(p=0.6, time=time).from_enumeration()
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        0  1  2
        trajectory
        0           0 -1 -2
        1           0 -1  0
        2           0  1  0
        3           0  1  2
        >>> T = StochasticProcess(domain=X.domain, time=time, name="T").from_time()
        >>> print(T)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'T':
        time        0  1  2
        trajectory
        0           0  1  2
        1           0  1  2
        2           0  1  2
        3           0  1  2
        >>> integral = X.increments().ito_integral(T)
        >>> print(integral)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'int X_increments dT':
        time        0  1  2
        trajectory
        0           0 -1 -2
        1           0 -1  0
        2           0  1  0
        3           0  1  2
        """
        from ..base.stochastic_process import StochasticProcess

        if name is None:
            name = (
                f"int {integrand.name} d{integrator.name}"
                if integrand.name is not None and integrator.name is not None
                else None
            )

        data = (
            (integrand.data * integrator.increments().data)
            .cumsum(axis=1)
            .dropna(axis=1, how="all")
        )
        data.columns = data.columns + 1
        data.insert(0, 0, 0)

        integral = StochasticProcess(
            name=name,
            sample_space=integrand.domain,
            is_discrete_time=integrand.time.is_discrete,
            is_discrete_state=integrand.is_discrete_state,
        ).from_pandas(data)
        integral.prob_measure = integrand.prob_measure

        return integral

    # TODO: Update docstrings
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

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk(p=0.5, time=T, initial_state=3).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        0  1  2
        trajectory
        0           3  2  1
        1           3  2  3
        2           3  4  3
        3           3  4  5
        >>> print(X.max_value())
        5
        """
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")
        return process.data.values.max()

    # TODO: Update docstrings
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

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk(p=0.5, time=T, initial_state=3).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        0  1  2
        trajectory
        0           3  2  1
        1           3  2  3
        2           3  4  3
        3           3  4  5
        >>> print(X.min_value())
        1
        """
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")
        return process.data.values.min()

    # TODO: Update docstrings
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

        Examples
        --------
        >>> from scipy.stats import bernoulli
        >>> from sigalg.core import Time
        >>> from sigalg.processes import IIDProcess
        >>> T = Time.discrete(start=1, length=2)
        >>> X = IIDProcess(distribution=bernoulli(p=0.6), support=[0, 1], time=T).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        1  2  3
        trajectory
        0           0  0  0
        1           0  0  1
        2           0  1  0
        3           0  1  1
        4           1  0  0
        5           1  0  1
        6           1  1  0
        7           1  1  1
        >>> print(X.cumsum()) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_cumsum':
        time        1  2  3
        trajectory
        0           0  0  0
        1           0  0  1
        2           0  1  1
        3           0  1  2
        4           1  1  1
        5           1  1  2
        6           1  2  2
        7           1  2  3
        >>> print(X.cumsum().is_monotonic())
        True
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

    # TODO: Update docstrings
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
                sample_space=process.domain,
                is_discrete_time=False,
            )
            .from_pandas(data_trans)
            .with_probability_measure(prob_measure=process.prob_measure)
        )


# TODO: Update docstrings
class ProcessTransformMethods:
    """Mixin class providing transformation methods for `StochasticProcess`."""

    # TODO: Update docstrings
    def transform(
        self,
        functions: list[Callable[[StochasticProcess], RandomVariable]],
        time: Time | None = None,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Apply a transformation to the stochastic process.

        Parameters
        ----------
        functions : list[Callable[[StochasticProcess], RandomVariable]]
            A list of functions to apply to the stochastic process.
        time : Time | None, default=None
            The new time index for the transformed process. If `None`, the original time index of the process will be used.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be `function(process.name)` if `process.name` is not `None`.

        Returns
        -------
        transformed_process : StochasticProcess
            The transformed stochastic process.

        Examples
        --------
        >>> from scipy.stats import bernoulli
        >>> from sigalg.core import RandomVariable, Time
        >>> from sigalg.processes import IIDProcess, StochasticProcess
        >>> T = Time().discrete(start=0, length=2)
        >>> X = IIDProcess(distribution=bernoulli(p=0.5), support=[0, 1], time=T).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        0  1  2
        trajectory
        0           0  0  0
        1           0  0  1
        2           0  1  0
        3           0  1  1
        4           1  0  0
        5           1  0  1
        6           1  1  0
        7           1  1  1
        >>> S = Time().discrete(start=4, stop=5)
        >>> def f4(process: StochasticProcess) -> RandomVariable:
        ...     X0, X1, _ = X
        ...     return X0 + X1
        >>> def f5(process: StochasticProcess) -> RandomVariable:
        ...     _, X1, X2 = X
        ...     return X1 + X2
        >>> print(X.transform(functions=[f4, f5], time=S)) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'function(X)':
        time        4  5
        trajectory
        0           0  0
        1           0  1
        2           1  1
        3           1  2
        4           1  0
        5           1  1
        6           2  1
        7           2  2
        """
        return ProcessTransforms.transform(
            self, functions=functions, time=time, name=name
        )

    # TODO: Update docstrings
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

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk(p=0.5, time=T, initial_state=3).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        0  1  2
        trajectory
        0           3  2  1
        1           3  2  3
        2           3  4  3
        3           3  4  5
        >>> def f(x):
        ...     return x + 1
        >>> print(X.pointwise_map(function=f)) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_mapped':
        time        0  1  2
        trajectory
        0           4  3  2
        1           4  3  4
        2           4  5  4
        3           4  5  6
        """
        return ProcessTransforms.pointwise_map(self, function=function, name=name)

    # TODO: Update docstrings
    def insert_rv(
        self,
        time: Real,
        rv: RandomVariable | None = None,
        state: Hashable | None = None,
        name: Hashable | None = None,
        in_place: bool = False,
    ) -> StochasticProcess:
        """Insert a random variable to the stochastic process at a specific time.

        Parameters
        ----------
        time : Real
            The time at which to insert the random variable.
        rv : RandomVariable | None, default=None
            The random variable to insert. One or the other of `rv` or `state` must be provided, but not both.
        state: Hashable | None, default=None
            A constant state to assign to the inserted random variable for all trajectories. One or the other of `rv` or `state` must be provided, but not both.
        name : Hashable | None, default=None
            The name of the new stochastic process. If `None`, the new name will be `insert(process.name)` if `process.name` is not `None`.
        in_place : bool, default=False
            If `True`, modify the input process in place and return it. If `False`, return a new stochastic process with the random variable inserted, leaving the input process unchanged.

        Returns
        -------
        inserted_process : StochasticProcess
            A new stochastic process with the random variable inserted at the specified time.

        Examples
        --------
        >>> from scipy.stats import bernoulli
        >>> from sigalg.core import RandomVariable, Time
        >>> from sigalg.processes import IIDProcess
        >>> T = Time().discrete(start=1, length=2)
        >>> X = IIDProcess(distribution=bernoulli(p=0.5), support=[0, 1], time=T).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        1  2  3
        trajectory
        0           0  0  0
        1           0  0  1
        2           0  1  0
        3           0  1  1
        4           1  0  0
        5           1  0  1
        6           1  1  0
        7           1  1  1
        >>> X0 = RandomVariable(domain=X.domain).from_constant(0)
        >>> print(X.insert_rv(rv=X0, time=0)) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'insert(X)':
        time        0  1  2  3
        trajectory
        0           0  0  0  0
        1           0  0  0  1
        2           0  0  1  0
        3           0  0  1  1
        4           0  1  0  0
        5           0  1  0  1
        6           0  1  1  0
        7           0  1  1  1
        """
        return ProcessTransforms.insert_rv(
            self, rv=rv, state=state, time=time, name=name, in_place=in_place
        )

    # TODO: Update docstrings
    def remove_rv(
        self,
        time: Real | None = None,
        pos: int | None = None,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Remove a random variable from the stochastic process at a specified time.

        Parameters
        ----------
        time : Real | None, default=None
            The time point at which to remove the random variable. If `None`, `pos` must be specified.
        pos : int | None, default=None
            The position at which to remove the random variable. If `None`, `time` must be specified.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of the input process subscripted with `remove`, provided that the name of the input process is a string.

        Returns
        -------
        removed_process : StochasticProcess
            A new stochastic process with the random variable removed at the specified time.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(start=1, length=2)
        >>> X = RandomWalk(p=0.6, time=T).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        1  2  3
        trajectory
        0           0 -1 -2
        1           0 -1  0
        2           0  1  0
        3           0  1  2
        >>> print(X.remove_rv(time=2)) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'remove(X)':
        time        1  3
        trajectory
        0           0 -2
        1           0  0
        2           0  0
        3           0  2
        >>> S = Time.continuous(start=0, stop=0.3, dt=0.101)
        >>> Y = RandomWalk(p=0.6, time=S, name="Y").from_enumeration()
        >>> print(Y) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'Y':
        time        0.0  0.1  0.2  0.3
        trajectory
        0             0   -1   -2   -3
        1             0   -1   -2   -1
        2             0   -1    0   -1
        3             0   -1    0    1
        4             0    1    0   -1
        5             0    1    0    1
        6             0    1    2    1
        7             0    1    2    3
        >>> print(Y.remove_rv(pos=2)) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'remove(Y)':
        time        0.0  0.1  0.3
        trajectory
        0             0   -1   -3
        1             0   -1   -1
        2             0   -1   -1
        3             0   -1    1
        4             0    1   -1
        5             0    1    1
        6             0    1    1
        7             0    1    3
        """
        return ProcessTransforms.remove_rv(self, time=time, pos=pos, name=name)

    # TODO: Update docstrings
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

        Examples
        --------
        >>> from scipy.stats import bernoulli
        >>> from sigalg.core import Time
        >>> from sigalg.processes import IIDProcess
        >>> T = Time.discrete(start=1, length=2)
        >>> X = IIDProcess(distribution=bernoulli(p=0.6), support=[0, 1], time=T).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        1  2  3
        trajectory
        0           0  0  0
        1           0  0  1
        2           0  1  0
        3           0  1  1
        4           1  0  0
        5           1  0  1
        6           1  1  0
        7           1  1  1
        >>> print(X.cumsum()) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_cumsum':
        time        1  2  3
        trajectory
        0           0  0  0
        1           0  0  1
        2           0  1  1
        3           0  1  2
        4           1  1  1
        5           1  1  2
        6           1  2  2
        7           1  2  3
        """
        return ProcessTransforms.cumsum(self, name=name)

    # TODO: Update docstrings
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

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=3)
        >>> X = RandomWalk(p=0.5, time=T, initial_state=3).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        0  1  2  3
        trajectory
        0           3  2  1  0
        1           3  2  1  2
        2           3  2  3  2
        3           3  2  3  4
        4           3  4  3  2
        5           3  4  3  4
        6           3  4  5  4
        7           3  4  5  6
        >>> print(X.cumprod()) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_cumprod':
        time        0   1   2    3
        trajectory
        0           3   6   6    0
        1           3   6   6   12
        2           3   6  18   36
        3           3   6  18   72
        4           3  12  36   72
        5           3  12  36  144
        6           3  12  60  240
        7           3  12  60  360
        """
        return ProcessTransforms.cumprod(self, name=name)

    # TODO: Update docstrings
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

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk(p=0.5, time=T, initial_state=3).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        0  1  2
        trajectory
        0           3  2  1
        1           3  2  3
        2           3  4  3
        3           3  4  5
        >>> print(X.sum()) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_sum':
                    X_sum
        trajectory
        0               6
        1               8
        2              10
        3              12
        """
        return ProcessTransforms.sum(self, name=name)

    def mean(self, name: Hashable | None = None) -> RandomVariable:
        """Compute the mean of the stochastic process across its time index.

        Parameters
        ----------
        name : Hashable | None, default=None
            The name of the transformed random variable. If `None`, the new name will be the name of `self` subscripted with `mean`, provided that the name of `self` is a string.

        Returns
        -------
        mean_random_variable : RandomVariable
            A new random variable representing the mean of the input stochastic process.
        """
        return ProcessTransforms.mean(self, name=name)

    def discount(self, rate: float, name: Hashable | None = None) -> StochasticProcess:
        """Apply a discount factor to the stochastic process.

        Parameters
        ----------
        rate : float
            The discount rate to apply to the process.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of `self` subscripted with `discount`, provided that the name of `self` is a string.

        Returns
        -------
        discounted_process : StochasticProcess
            A new stochastic process representing the discounted values of the input process.
        """
        return ProcessTransforms.discount(self, rate=rate, name=name)

    # TODO: Update docstrings
    def stopped(
        self,
        stopping_time: StoppingTime,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Pass."""
        return ProcessTransforms.stopped(
            process=self, stopping_time=stopping_time, name=name
        )

    def increments(
        self, forward: bool = True, name: Hashable | None = None
    ) -> StochasticProcess:
        r"""Compute the increments of the stochastic process along its time index.

        Given a stochastic process $X_t$ with index set $\{t_0,t_0+1,\ldots,T\}$ there are two types of increments that can be computed: The first are *forward* increments, which results in a stochastic process $\Delta X_t$ defined as

        $$
        \Delta X_t = X_{t+1} - X_t,
        $$

        for each $t=t_0,\ldots,T-1$. The second type are *backward* increments, which results in a stochastic process $\Delta X_t$ where

        $$
        \Delta X_t = X_t - X_{t-1},
        $$

        for each $t=t_0+1,\ldots,T$.

        Parameters
        ----------
        forward : bool, default=True
            If `True`, compute forward increments; otherwise, compute backward increments.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of `self` subscripted with `increments`, provided that the name of `self` is a string.

        Returns
        -------
        increments_process : StochasticProcess
            A new stochastic process representing the increments of the input process.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk(p=0.5, time=T, initial_state=3).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        0  1  2
        trajectory
        0           3  2  1
        1           3  2  3
        2           3  4  3
        3           3  4  5
        >>> print(X.increments(forward=True)) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_increments':
        time        0  1
        trajectory
        0          -1 -1
        1          -1  1
        2           1 -1
        3           1  1
        >>> print(X.increments(forward=False)) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_increments':
        time        1  2
        trajectory
        0          -1 -1
        1          -1  1
        2           1 -1
        3           1  1
        """
        return ProcessTransforms.increments(self, forward=forward, name=name)

    # TODO: Update docstrings
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

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk, StochasticProcess
        >>> time = Time().discrete(length=2)
        >>> X = RandomWalk(p=0.6, time=time).from_enumeration()
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        0  1  2
        trajectory
        0           0 -1 -2
        1           0 -1  0
        2           0  1  0
        3           0  1  2
        >>> T = StochasticProcess(domain=X.domain, time=time, name="T").from_time()
        >>> print(T)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'T':
        time        0  1  2
        trajectory
        0           0  1  2
        1           0  1  2
        2           0  1  2
        3           0  1  2
        >>> integral = X.increments().ito_integral(T)
        >>> print(integral)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'int X_increments dT':
        time        0  1  2
        trajectory
        0           0 -1 -2
        1           0 -1  0
        2           0  1  0
        3           0  1  2
        """
        return ProcessTransforms.ito_integral(self, integrator=integrator, name=name)

    # TODO: Update docstrings
    def max_value(self) -> Real:
        """Get the maximum value across all trajectories and time points of the stochastic process.

        Returns
        -------
        max_value : Real
            The maximum value found in the stochastic process.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk(p=0.5, time=T, initial_state=3).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        0  1  2
        trajectory
        0           3  2  1
        1           3  2  3
        2           3  4  3
        3           3  4  5
        >>> print(X.max_value())
        5
        """
        return ProcessTransforms.max_value(self)

    # TODO: Update docstrings
    def min_value(self) -> Real:
        """Get the minimum value across all trajectories and time points of the stochastic process.

        Returns
        -------
        min_value : Real
            The minimum value found in the stochastic process.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk(p=0.5, time=T, initial_state=3).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        0  1  2
        trajectory
        0           3  2  1
        1           3  2  3
        2           3  4  3
        3           3  4  5
        >>> print(X.min_value())
        1
        """
        return ProcessTransforms.min_value(self)

    # TODO: Update docstrings
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

        Examples
        --------
        >>> from scipy.stats import bernoulli
        >>> from sigalg.core import Time
        >>> from sigalg.processes import IIDProcess
        >>> T = Time.discrete(start=1, length=2)
        >>> X = IIDProcess(distribution=bernoulli(p=0.6), support=[0, 1], time=T).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        1  2  3
        trajectory
        0           0  0  0
        1           0  0  1
        2           0  1  0
        3           0  1  1
        4           1  0  0
        5           1  0  1
        6           1  1  0
        7           1  1  1
        >>> print(X.cumsum()) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_cumsum':
        time        1  2  3
        trajectory
        0           0  0  0
        1           0  0  1
        2           0  1  1
        3           0  1  2
        4           1  1  1
        5           1  1  2
        6           1  2  2
        7           1  2  3
        >>> print(X.cumsum().is_monotonic())
        True
        """
        return ProcessTransforms.is_monotonic(self, increasing)

    # TODO: Update docstrings
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
