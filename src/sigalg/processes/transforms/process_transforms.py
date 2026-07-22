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

from ...core.indices.time import Time

if TYPE_CHECKING:
    from ...core.functions.random_variable import RandomVariable
    from ..base.stochastic_process import StochasticProcess
    from ..stopping_times.stopping_time import StoppingTime


class ProcessTransforms:
    """A collection of methods for transforming stochastic processes."""

    @classmethod
    def transform(
        cls,
        process: StochasticProcess,
        functions: list[Callable[[StochasticProcess], RandomVariable]],
        index: Time | None = None,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Apply a transformation to a stochastic process.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        process : StochasticProcess
            The stochastic process to transform.
        functions : list[Callable[[StochasticProcess], RandomVariable]]
            A list of functions to apply to the stochastic process.
        index : Time | None, default=None
            The new time index for the transformed process. If `None`, the original time index of `process` will be used.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be `(process.name)_transform`.

        Raises
        ------
        TypeError
            If `process` is not an instance of `StochasticProcess`, or `functions` is not a list of callables, or `index` is not an instance of `Time`.
        ValueError
            If the length of `functions` does not match the length of `index`.

        Returns
        -------
        transformed_process : StochasticProcess
            The transformed stochastic process.

        Examples
        --------
        >>> from scipy.stats import bernoulli
        >>> from sigalg.core import RandomVariable, Time
        >>> from sigalg.processes import IIDProcess, StochasticProcess
        >>> T = Time.discrete(start=0, length=2)
        >>> X = IIDProcess.generate(
        ...     mode="enum",
        ...     distribution=bernoulli(p=0.5),
        ...     support=[0, 1],
        ...     index=T,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        time    0  1  2
        sample
        0       0  0  0
        1       0  0  1
        2       0  1  0
        3       0  1  1
        4       1  0  0
        5       1  0  1
        6       1  1  0
        7       1  1  1
        >>> S = Time.discrete(start=4, stop=5)
        >>> def f4(process: StochasticProcess) -> RandomVariable:
        ...     X0, X1, _ = X
        ...     return X0 + X1
        >>> def f5(process: StochasticProcess) -> RandomVariable:
        ...     _, X1, X2 = X
        ...     return X1 + X2
        >>> X_transform = X.transform(functions=[f4, f5], index=S)
        >>> print(X_transform)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_transform':
        time    4  5
        sample
        0       0  0
        1       0  1
        2       1  1
        3       1  2
        4       1  0
        5       1  1
        6       2  1
        7       2  2
        """
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")
        if not isinstance(functions, list):
            raise TypeError("functions must be a list.")
        if not all(isinstance(f, Callable) for f in functions):
            raise TypeError("Each element in functions must be callable.")
        if index is not None and not isinstance(index, Time):
            raise TypeError("index must be an instance of Time.")
        if index is not None and len(functions) != len(index):
            raise ValueError("The number of functions must match the length of index.")

        if index is None:
            index = process.time

        transformed_rvs = {}

        for f, t in zip(functions, index):
            transformed_rvs[t] = f(process).data

        data = pd.DataFrame(
            transformed_rvs, index=process.sample_space.data, columns=index.data
        )

        if name is None:
            name = f"{process.name}_transform"

        return StochasticProcess(
            *process.prob_space, mapping=data, index=index, name=name
        )

    @classmethod
    def pointwise_map(
        cls,
        process: StochasticProcess,
        function: Callable[[Hashable], Hashable],
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Apply a function pointwise to the values of a stochastic process.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        process : StochasticProcess
            The stochastic process to which the function will be applied.
        function : Callable[[Hashable], Hashable]
            A function that takes a single value and returns a transformed value. This function will be applied to each value in the stochastic process.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of the input process subscripted with `mapped`.

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
        >>> X = RandomWalk.generate(mode="enum", p=0.5, index=T, initial_state=3)
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time    0  1  2
        sample
        0       3  2  1
        1       3  2  3
        2       3  4  3
        3       3  4  5
        >>> def f(x):
        ...     return x + 1
        >>> X_mapped = X.pointwise_map(function=f)
        >>> print(X_mapped)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_mapped':
        time    0  1  2
        sample
        0       4  3  2
        1       4  3  4
        2       4  5  4
        3       4  5  6
        """
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")
        if not isinstance(function, Callable):
            raise TypeError("function must be a callable object.")

        data_trans = process.data.copy()
        data_trans = data_trans.map(function)
        if name is None:
            name = f"{process.name}_mapped"
        return StochasticProcess(
            *process.prob_space, mapping=data_trans, name=name, index=process.time
        )

    @classmethod
    def insert_rv(
        cls,
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
            The name of the new stochastic process. If `None`, the new name will be `process.name_insert`.
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
        >>> T = Time.discrete(start=1, length=2)
        >>> X = IIDProcess.generate(mode="enum", distribution=bernoulli(p=0.5), support=[0, 1], index=T)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        time        1  2  3
        sample
        0           0  0  0
        1           0  0  1
        2           0  1  0
        3           0  1  1
        4           1  0  0
        5           1  0  1
        6           1  1  0
        7           1  1  1
        >>> X0 = RandomVariable.from_constant(sample_space=X.sample_space, constant=0)
        >>> X_insert = X.insert_rv(rv=X0, time=0)
        >>> print(X_insert) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_insert':
        time        0  1  2  3
        sample
        0           0  0  0  0
        1           0  0  0  1
        2           0  0  1  0
        3           0  0  1  1
        4           0  1  0  0
        5           0  1  0  1
        6           0  1  1  0
        7           0  1  1  1
        """
        from ...core.functions.random_variable import RandomVariable
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
        if rv is not None and process.sample_space != rv.sample_space:
            raise ValueError("process and rv must have the same sample_space.")

        new_time = process.time.insert_time(time)

        if rv is None:
            rv = RandomVariable.from_constant(*process.prob_space, constant=state)

        if in_place:
            pos = process.data.columns.searchsorted(time)
            process.data.insert(pos, time, rv.data)
            process._index = new_time
            if name is not None:
                process.name = f"{process.name}_insert"
            return process
        else:
            new_data = process.data.copy()
            pos = new_data.columns.searchsorted(time)
            new_data.insert(pos, time, rv.data)
            if name is None:
                name = f"{process.name}_insert"

            return StochasticProcess(
                *process.prob_space,
                index=new_time,
                name=name,
                mapping=new_data,
            )

    @classmethod
    def remove_rv(
        cls,
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
            The name of the transformed process. If `None`, the new name will be the name of the input process subscripted with `remove`.

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
        >>> X = RandomWalk.generate(mode="enum", p=0.6, initial_state=0, index=T)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time        1  2  3
        sample
        0           0 -1 -2
        1           0 -1  0
        2           0  1  0
        3           0  1  2
        >>> X_remove = X.remove_rv(time=2)
        >>> print(X_remove) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_remove':
        time        1  3
        sample
        0           0 -2
        1           0  0
        2           0  0
        3           0  2
        >>> S = Time.continuous(start=0, stop=0.3, dt=0.101)
        >>> Y = RandomWalk.generate(mode="enum", p=0.6, initial_state=0, index=S, name="Y")
        >>> print(Y) # doctest: +NORMALIZE_WHITESPACE
        Random walk 'Y':
        time        0.0  0.1  0.2  0.3
        sample
        0             0   -1   -2   -3
        1             0   -1   -2   -1
        2             0   -1    0   -1
        3             0   -1    0    1
        4             0    1    0   -1
        5             0    1    0    1
        6             0    1    2    1
        7             0    1    2    3
        >>> Y_remove = Y.remove_rv(pos=2)
        >>> print(Y_remove) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'Y_remove':
        time        0.0  0.1  0.3
        sample
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
            name = f"{process.name}_remove"

        return StochasticProcess(
            *process.prob_space, index=new_time, name=name, mapping=new_data
        )

    @classmethod
    def cumsum(
        cls,
        process: StochasticProcess,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Compute the cumulative sum of a stochastic process along its time index.

        See the Notes section below for the mathematical details.

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
        >>> X = IIDProcess.generate(mode="enum", distribution=bernoulli(p=0.6), support=[0, 1], index=T)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        time        1  2  3
        sample
        0           0  0  0
        1           0  0  1
        2           0  1  0
        3           0  1  1
        4           1  0  0
        5           1  0  1
        6           1  1  0
        7           1  1  1
        >>> X_cumsum = X.cumsum()
        >>> print(X_cumsum) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_cumsum':
        time        1  2  3
        sample
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
            name = f"{process.name}_cumsum"
        return StochasticProcess(
            *process.prob_space,
            mapping=data_trans,
            index=process.time,
            name=name,
        )

    @classmethod
    def cumprod(
        cls,
        process: StochasticProcess,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Compute the cumulative product of a stochastic process along its time index.

        See the Notes section below for the mathematical details.

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
        >>> X = RandomWalk.generate(mode="enum", p=0.5, initial_state=3, index=T)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time        0  1  2  3
        sample
        0           3  2  1  0
        1           3  2  1  2
        2           3  2  3  2
        3           3  2  3  4
        4           3  4  3  2
        5           3  4  3  4
        6           3  4  5  4
        7           3  4  5  6
        >>> X_cumprod = X.cumprod()
        >>> print(X_cumprod) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_cumprod':
        time        0   1   2    3
        sample
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
            name = f"{process.name}_cumprod"
        return StochasticProcess(
            *process.prob_space, name=name, mapping=data_trans, index=process.time
        )

    @classmethod
    def sum(
        cls,
        process: StochasticProcess,
        name: Hashable | None = None,
    ) -> RandomVariable:
        """Compute the sum of a stochastic process across its time index.

        See the Notes section below for the mathematical details.

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
        >>> X = RandomWalk.generate(mode="enum", p=0.5, initial_state=3, index=T)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time        0  1  2
        sample
        0           3  2  1
        1           3  2  3
        2           3  4  3
        3           3  4  5
        >>> X_sum = X.sum()
        >>> print(X_sum) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_sum':
                    X_sum
        sample
        0               6
        1               8
        2              10
        3              12
        """
        from ...core.functions.random_variable import RandomVariable
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")

        data_trans = process.data.copy()
        data_trans = data_trans.sum(axis=1)

        if name is None:
            name = f"{process.name}_sum"

        return RandomVariable(
            *process.prob_space,
            name=name,
            mapping=data_trans,
        )

    @classmethod
    def mean(
        cls,
        process: StochasticProcess,
        name: Hashable | None = None,
    ) -> RandomVariable:
        """Compute the mean of a stochastic process across its time index.

        See the Notes section below for the mathematical details.

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
        from ...core.functions.random_variable import RandomVariable
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")

        data_trans = process.data.copy()
        data_trans = data_trans.mean(axis=1)

        if name is None:
            name = f"{process.name}_mean"

        return RandomVariable(
            *process.prob_space,
            name=name,
            mapping=data_trans,
        )

    @classmethod
    def discount(
        cls,
        process: StochasticProcess,
        rate: Real,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        r"""Return the discounted process of a given stochastic process.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        process : StochasticProcess
            The original process to be discounted.
        rate : Real
            The discount rate, which must be a positive real number.
        name : Hashable | None, default=None
            The name of the discounted process. If `None`, the new name will be the name of the input process subscripted with `discount`.

        Raises
        ------
        TypeError
            If `process` is not an instance of `StochasticProcess`, if `rate` is not a real number, or if `name` is not a hashable object or `None`.
        ValueError
            If `rate` is not positive.

        Returns
        -------
        discounted_process : StochasticProcess
            The discounted process.

        Notes
        -----
        The discounted process is given by

        $$
        \tilde{S}_t = \frac{S_t}{(1+r)^t},
        $$

        where $S_t$ is the original process and $r$ is the discount rate.

        Examples
        --------
        >>> from sigalg.finance import BinomialPricingModel
        >>> S_0 = 4
        >>> u = 1.2
        >>> d = 0.9
        >>> p = 0.6
        >>> r = 0.01
        >>> S = BinomialPricingModel.generate(
        ...     mode="enum",
        ...     initial_price=S_0,
        ...     up_factor=u,
        ...     down_factor=d,
        ...     up_prob=p,
        ...     risk_free_rate=r,
        ...     length=2,
        ... )
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Binomial price process 'S':
        time    0    1     2
        sample
        0       4  4.8  5.76
        1       4  4.8  4.32
        2       4  3.6  4.32
        3       4  3.6  3.24
        >>> Q = S.EMMs
        >>> print(Q)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'Q':
                probability
        sample
        0          0.134444
        1          0.232222
        2          0.232222
        3          0.401111
        >>> is_martingale_wrt_real_world_measure = S.discount(r).is_martingale()
        >>> print(is_martingale_wrt_real_world_measure)
        False
        >>> is_martingale_wrt_EMM = S.discount(r).is_martingale(prob_measure=Q)
        >>> print(is_martingale_wrt_EMM)
        True
        """
        from ..base.stochastic_process import StochasticProcess

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")
        if not isinstance(rate, Real):
            raise TypeError("rate must be a real number.")
        if rate <= 0:
            raise ValueError("rate must be positive.")
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("name must be a hashable object or None.")

        discount_factors = (1 + rate) ** (-process.time.data)
        discounted_data = process.data.multiply(discount_factors, axis=1)

        result = StochasticProcess(
            *process.prob_space,
            index=process.time,
            name=f"{process.name}_discount",
            mapping=discounted_data,
        )

        return result

    @classmethod
    def increments(
        cls,
        process: StochasticProcess,
        forward: bool = True,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        r"""Compute the increments of a stochastic process along its time index.

        See the Notes section below for the mathematical details.

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
        >>> X = RandomWalk.generate(mode="enum", p=0.5, index=T, initial_state=3)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time        0  1  2
        sample
        0           3  2  1
        1           3  2  3
        2           3  4  3
        3           3  4  5
        >>> X_increments = X.increments(forward=True)
        >>> print(X_increments) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_increments':
        time        0  1
        sample
        0          -1 -1
        1          -1  1
        2           1 -1
        3           1  1
        >>> X_increments = X.increments(forward=False)
        >>> print(X_increments) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_increments':
        time        1  2
        sample
        0          -1 -1
        1          -1  1
        2           1 -1
        3           1  1

        Notes
        -----
        Given a stochastic process $X_t$ with index set $\{t_0,t_0+1,\ldots,T\}$ there are two types of increments that can be computed: The first are *forward* increments, which results in a stochastic process $\Delta X_t$ defined as

        $$
        \Delta X_t = X_{t+1} - X_t,
        $$

        for each $t=t_0,\ldots,T-1$. The second type are *backward* increments, which results in a stochastic process $\Delta X_t$ where

        $$
        \Delta X_t = X_t - X_{t-1},
        $$

        for each $t=t_0+1,\ldots,T$.
        """
        from ...core.indices.time import Time
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
                name=process.time.name,
                variable_names=[process.time.data.name],
                indices=process.time.data[:-1],
            )
        else:
            data_trans = data_trans.diff(axis=1).dropna(axis=1)
            new_time = Time(
                name=process.time.name,
                variable_names=[process.time.data.name],
                indices=process.time.data[1:],
            )

        if name is None:
            name = f"{process.name}_increments"

        return StochasticProcess(
            *process.prob_space, name=name, index=new_time, mapping=data_trans
        )

    @classmethod
    def stopped(
        cls,
        process: StochasticProcess,
        stopping_time: StoppingTime,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        r"""Get the stopped process from a stopping time.

        See the Notes section below for the mathematical details.

        Examples
        --------
        >>> from math import inf
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk, StoppingTime
        >>> T = Time.discrete(start=1, stop=10)
        >>> S = RandomWalk.generate(
        ...     mode="sim",
        ...     p=0.4,
        ...     initial_state=10,
        ...     index=T,
        ...     n_trajectories=8,
        ...     random_state=42,
        ...     name="S",
        ... )
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Random walk 'S':
        time    1   2   3   4   5   6   7   8   9   10
        sample
        0       10  11  10  11  12  11  12  13  14  13
        1       10   9   8   9  10  11  10   9   8   7
        2       10  11  12  13  12  13  14  15  14  13
        3       10   9   8   9  10  11  10   9   8   7
        4       10   9   8   7   8   7   8   9   8   9
        5       10  11  10   9  10   9   8   7   8   9
        6       10  11  12  11  10   9   8   9   8   7
        7       10  11  12  11  10   9   8   7   6   5
        >>> tau = StoppingTime.from_filtration(
        ...     process=S,
        ...     mapping={
        ...         0: inf,
        ...         1: 3,
        ...         2: inf,
        ...         3: 3,
        ...         4: 3,
        ...         5: 7,
        ...         6: 7,
        ...         7: 7,
        ...     },
        ... )
        >>> print(tau)  # doctest: +NORMALIZE_WHITESPACE
        Stopping time 'tau':
                tau
        sample
        0       inf
        1       3.0
        2       inf
        3       3.0
        4       3.0
        5       7.0
        6       7.0
        7       7.0
        >>> S_stopped = S.stopped(stopping_time=tau)
        >>> print(S_stopped)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'S^tau':
        time    1   2   3   4   5   6   7   8   9   10
        sample
        0       10  11  10  11  12  11  12  13  14  13
        1       10   9   8   8   8   8   8   8   8   8
        2       10  11  12  13  12  13  14  15  14  13
        3       10   9   8   8   8   8   8   8   8   8
        4       10   9   8   8   8   8   8   8   8   8
        5       10  11  10   9  10   9   8   8   8   8
        6       10  11  12  11  10   9   8   8   8   8
        7       10  11  12  11  10   9   8   8   8   8

        Notes
        -----
        Let $X$ be a $T$-indexed stochastic process on a probability space $(\Omega, \mathcal{F},P)$, and let $\tau: \Omega \to T$ be a stopping time. The *stopped process*, denoted $X^\tau$, is defined for all $t\in T$ by

        $$
        X^\tau_t(\omega) = X_{\min\{t, \tau(\omega)\}}(\omega).
        $$
        """
        from ..base.stochastic_process import StochasticProcess
        from ..stopping_times.stopping_time import StoppingTime

        if not isinstance(process, StochasticProcess):
            raise TypeError("process must be an instance of StochasticProcess.")
        if not isinstance(stopping_time, StoppingTime):
            raise TypeError("stopping_time must be an instance of StoppingTime.")
        if process.time != stopping_time.time:
            raise ValueError(
                "The time indices of the process and stopping time must match."
            )

        def _fill_value(s: pd.Series) -> pd.Series:
            idx = s.searchsorted(value=np.nan)
            s.iloc[idx:] = s.iloc[idx - 1]
            return s

        mask = pd.DataFrame(
            process.data.columns.values.reshape(1, -1)
            <= stopping_time.data.values.reshape(-1, 1),
            index=process.sample_space.data,
            columns=process.index.data,
        )

        mapping = (
            process.data[mask].apply(_fill_value, axis=1).astype(process.data.dtypes)
        )

        if name is None:
            name = f"{process.name}^{stopping_time.name}"

        return StochasticProcess(
            *process.prob_space,
            name=name,
            index=process.time,
            mapping=mapping,
        )

    @classmethod
    def ito_integral(
        cls,
        integrand: StochasticProcess,
        integrator: StochasticProcess,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Compute the Itô integral of a stochastic process with respect to another stochastic process.

        See the Notes section below for the mathematical details.

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
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk.generate(mode="enum", p=0.6, initial_state=0, index=T)
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time    0  1  2
        sample
        0       0 -1 -2
        1       0 -1  0
        2       0  1  0
        3       0  1  2
        >>> time = StochasticProcess.from_time(*X.prob_space, index=T, name="time")
        >>> print(time)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'time':
        time    0  1  2
        sample
        0       0  1  2
        1       0  1  2
        2       0  1  2
        3       0  1  2
        >>> integral = X.increments().ito_integral(time)
        >>> print(integral)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'int X_increments dtime':
        time    0  1  2
        sample
        0       0 -1 -2
        1       0 -1  0
        2       0  1  0
        3       0  1  2
        """
        from ..base.stochastic_process import StochasticProcess

        if name is None:
            name = f"int {integrand.name} d{integrator.name}"

        data = (
            (integrand.data * integrator.increments().data)
            .cumsum(axis=1)
            .dropna(axis=1, how="all")
        )
        data.columns = data.columns + 1
        data.insert(0, 0, 0)

        return StochasticProcess(
            *integrand.prob_space,
            name=name,
            mapping=data,
        )

    @classmethod
    def max_value(cls, process: StochasticProcess) -> Real:
        """Get the maximum value across all trajectories and time points of a stochastic process.

        See the Notes section below for the mathematical details.

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
        >>> X = RandomWalk.generate(mode="enum", p=0.5, index=T, initial_state=3)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time        0  1  2
        sample
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

    @classmethod
    def min_value(cls, process: StochasticProcess) -> Real:
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
        >>> X = RandomWalk.generate(mode="enum", p=0.5, index=T, initial_state=3)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time        0  1  2
        sample
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

    @classmethod
    def is_monotonic(cls, process: StochasticProcess, increasing: bool = True) -> bool:
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
        >>> X = IIDProcess.generate(mode="enum", distribution=bernoulli(p=0.6), support=[0, 1], index=T)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        time        1  2  3
        sample
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
        sample
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
        >>> from sigalg.core import Time
        >>> from sigalg.processes import IIDProcess

        Parameters for a Poisson process.

        >>> rate = 2.0
        >>> n_trajectories = 5
        >>> random_state = 42
        >>> max_count = 5

        Create an index for the counts.

        >>> counts = Time.discrete(start=1, stop=max_count, variable_name="count")

        Exponential interarrival times with given rate

        >>> interarrival_times = IIDProcess.generate(
        ...     mode="sim",
        ...     distribution=expon(scale=1 / rate),
        ...     name="interarrival_times",
        ...     index=counts,
        ...     n_trajectories=n_trajectories,
        ...     random_state=random_state,
        ... )
        >>> print(interarrival_times)  # doctest: +NORMALIZE_WHITESPACE
        IID process 'interarrival_times':
        count          1         2         3         4         5
        sample
        0       1.202104  1.168095  1.192380  0.139897  0.043219
        1       0.726330  0.704980  1.562148  0.039647  0.523280
        2       0.035218  0.544512  0.865664  0.193447  0.615793
        3       0.076887  0.045789  0.157590  0.450600  0.206493
        4       0.623693  0.111788  0.918985  0.613543  0.327898

        Compute arrival times by cumulative sum of interarrival times.

        >>> arrival_times = interarrival_times.cumsum().with_name("arrival_times")
        >>> print(arrival_times)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'arrival_times':
        count          1         2         3         4         5
        sample
        0       1.202104  2.370199  3.562580  3.702477  3.745695
        1       0.726330  1.431311  2.993459  3.033106  3.556386
        2       0.035218  0.579730  1.445394  1.638841  2.254634
        3       0.076887  0.122675  0.280265  0.730864  0.937357
        4       0.623693  0.735481  1.654466  2.268009  2.595907

        Determine time grid for Poisson process.

        >>> longest_trajectory = arrival_times.max_value()
        >>> T = Time.continuous(
        ...     start=0.0,
        ...     stop=longest_trajectory + 0.1,
        ...     num_points=6,
        ... )

        Convert to Poisson counting process.

        >>> poisson = arrival_times.to_counting_process(
        ...     time=T,
        ... ).with_name("poisson")
        >>> print(poisson)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Stochastic process 'poisson':
        time    0.000000  0.769139  1.538278  2.307417  3.076556  3.845695
        sample
        0            0.0       0.0       1.0       1.0       2.0       5.0
        1            0.0       1.0       2.0       2.0       4.0       5.0
        2            0.0       2.0       3.0       5.0       5.0       5.0
        3            0.0       4.0       5.0       5.0       5.0       5.0
        4            0.0       2.0       2.0       4.0       5.0       5.0
        """
        from ...core.indices.time import Time
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
        df_time["time"] = df_time["time"].astype("float64")

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
        data_trans.index = process.sample_space.data

        if name is None:
            name = f"{process.name}_counting" if process.name is not None else None
        return StochasticProcess(
            *process.prob_space,
            name=name,
            mapping=data_trans,
        )


class ProcessTransformMethods:
    """Mixin class providing transformation methods for `StochasticProcess`."""

    def transform(
        self,
        functions: list[Callable[[StochasticProcess], RandomVariable]],
        index: Time | None = None,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Apply a transformation to a stochastic process.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        functions : list[Callable[[StochasticProcess], RandomVariable]]
            A list of functions to apply to the stochastic process.
        index : Time | None, default=None
            The new time index for the transformed process. If `None`, the original time index of `process` will be used.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be `(process.name)_transform`.

        Returns
        -------
        transformed_process : StochasticProcess
            The transformed stochastic process.

        Examples
        --------
        >>> from scipy.stats import bernoulli
        >>> from sigalg.core import RandomVariable, Time
        >>> from sigalg.processes import IIDProcess, StochasticProcess
        >>> T = Time.discrete(start=0, length=2)
        >>> X = IIDProcess.generate(
        ...     mode="enum",
        ...     distribution=bernoulli(p=0.5),
        ...     support=[0, 1],
        ...     index=T,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        time    0  1  2
        sample
        0       0  0  0
        1       0  0  1
        2       0  1  0
        3       0  1  1
        4       1  0  0
        5       1  0  1
        6       1  1  0
        7       1  1  1
        >>> S = Time.discrete(start=4, stop=5)
        >>> def f4(process: StochasticProcess) -> RandomVariable:
        ...     X0, X1, _ = X
        ...     return X0 + X1
        >>> def f5(process: StochasticProcess) -> RandomVariable:
        ...     _, X1, X2 = X
        ...     return X1 + X2
        >>> X_transform = X.transform(functions=[f4, f5], index=S)
        >>> print(X_transform)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_transform':
        time    4  5
        sample
        0       0  0
        1       0  1
        2       1  1
        3       1  2
        4       1  0
        5       1  1
        6       2  1
        7       2  2
        """
        return ProcessTransforms.transform(
            self, functions=functions, index=index, name=name
        )

    def pointwise_map(
        self,
        function: Callable[[Hashable], Hashable],
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Apply a function pointwise to the values of a stochastic process.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        function : Callable[[Hashable], Hashable]
            A function that takes a single value and returns a transformed value. This function will be applied to each value in the stochastic process.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of the input process subscripted with `mapped`.

        Returns
        -------
        mapped_process : StochasticProcess
            A new stochastic process with the function applied pointwise to its values.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk.generate(mode="enum", p=0.5, index=T, initial_state=3)
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time    0  1  2
        sample
        0       3  2  1
        1       3  2  3
        2       3  4  3
        3       3  4  5
        >>> def f(x):
        ...     return x + 1
        >>> X_mapped = X.pointwise_map(function=f)
        >>> print(X_mapped)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_mapped':
        time    0  1  2
        sample
        0       4  3  2
        1       4  3  4
        2       4  5  4
        3       4  5  6
        """
        return ProcessTransforms.pointwise_map(self, function=function, name=name)

    def insert_rv(
        self,
        time: Real,
        rv: RandomVariable | None = None,
        state: Hashable | None = None,
        name: Hashable | None = None,
        in_place: bool = False,
    ) -> StochasticProcess:
        """Insert a random variable to a stochastic process at a specific time.

        Parameters
        ----------
        time : Real
            The time at which to insert the random variable.
        rv : RandomVariable | None, default=None
            The random variable to insert. One or the other of `rv` or `state` must be provided, but not both.
        state: Hashable | None, default=None
            A constant state to assign to the inserted random variable for all trajectories. One or the other of `rv` or `state` must be provided, but not both.
        name : Hashable | None, default=None
            The name of the new stochastic process. If `None`, the new name will be `process.name_insert`.
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
        >>> T = Time.discrete(start=1, length=2)
        >>> X = IIDProcess.generate(mode="enum", distribution=bernoulli(p=0.5), support=[0, 1], index=T)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        time        1  2  3
        sample
        0           0  0  0
        1           0  0  1
        2           0  1  0
        3           0  1  1
        4           1  0  0
        5           1  0  1
        6           1  1  0
        7           1  1  1
        >>> X0 = RandomVariable.from_constant(sample_space=X.sample_space, constant=0)
        >>> X_insert = X.insert_rv(rv=X0, time=0)
        >>> print(X_insert) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_insert':
        time        0  1  2  3
        sample
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

    def remove_rv(
        self,
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
            The name of the transformed process. If `None`, the new name will be the name of the input process subscripted with `remove`.

        Returns
        -------
        removed_process : StochasticProcess
            A new stochastic process with the random variable removed at the specified time.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(start=1, length=2)
        >>> X = RandomWalk.generate(mode="enum", p=0.6, initial_state=0, index=T)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time        1  2  3
        sample
        0           0 -1 -2
        1           0 -1  0
        2           0  1  0
        3           0  1  2
        >>> X_remove = X.remove_rv(time=2)
        >>> print(X_remove) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_remove':
        time        1  3
        sample
        0           0 -2
        1           0  0
        2           0  0
        3           0  2
        >>> S = Time.continuous(start=0, stop=0.3, dt=0.101)
        >>> Y = RandomWalk.generate(mode="enum", p=0.6, initial_state=0, index=S, name="Y")
        >>> print(Y) # doctest: +NORMALIZE_WHITESPACE
        Random walk 'Y':
        time        0.0  0.1  0.2  0.3
        sample
        0             0   -1   -2   -3
        1             0   -1   -2   -1
        2             0   -1    0   -1
        3             0   -1    0    1
        4             0    1    0   -1
        5             0    1    0    1
        6             0    1    2    1
        7             0    1    2    3
        >>> Y_remove = Y.remove_rv(pos=2)
        >>> print(Y_remove) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'Y_remove':
        time        0.0  0.1  0.3
        sample
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

    def cumsum(self, name: Hashable | None = None) -> StochasticProcess:
        """Compute the cumulative sum of the stochastic process along its time index.

        See the Notes section below for the mathematical details.

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
        >>> X = IIDProcess.generate(mode="enum", distribution=bernoulli(p=0.6), support=[0, 1], index=T)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        time        1  2  3
        sample
        0           0  0  0
        1           0  0  1
        2           0  1  0
        3           0  1  1
        4           1  0  0
        5           1  0  1
        6           1  1  0
        7           1  1  1
        >>> X_cumsum = X.cumsum()
        >>> print(X_cumsum) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_cumsum':
        time        1  2  3
        sample
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

    def cumprod(self, name: Hashable | None = None) -> StochasticProcess:
        """Compute the cumulative product of a stochastic process along its time index.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of the input process subscripted with `cumprod`, provided that the name of the input process is a string.

        Returns
        -------
        cumprod_process : StochasticProcess
            A new stochastic process representing the cumulative product of the input process.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=3)
        >>> X = RandomWalk.generate(mode="enum", p=0.5, initial_state=3, index=T)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time        0  1  2  3
        sample
        0           3  2  1  0
        1           3  2  1  2
        2           3  2  3  2
        3           3  2  3  4
        4           3  4  3  2
        5           3  4  3  4
        6           3  4  5  4
        7           3  4  5  6
        >>> X_cumprod = X.cumprod()
        >>> print(X_cumprod) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_cumprod':
        time        0   1   2    3
        sample
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

    def sum(self, name: Hashable | None = None) -> RandomVariable:
        """Compute the sum of a stochastic process across its time index.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        name : Hashable | None, default=None
            The name of the transformed random variable. If `None`, the new name will be the name of the input process subscripted with `sum`, provided that the name of the input process is a string.

        Returns
        -------
        sum_variable : RandomVariable
            A new random variable representing the sum of the input process across its time index.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk.generate(mode="enum", p=0.5, initial_state=3, index=T)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time        0  1  2
        sample
        0           3  2  1
        1           3  2  3
        2           3  4  3
        3           3  4  5
        >>> X_sum = X.sum()
        >>> print(X_sum) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_sum':
                    X_sum
        sample
        0               6
        1               8
        2              10
        3              12
        """
        return ProcessTransforms.sum(self, name=name)

    def mean(self, name: Hashable | None = None) -> RandomVariable:
        """Compute the mean of a stochastic process across its time index.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        name : Hashable | None, default=None
            The name of the transformed random variable. If `None`, the new name will be the name of the input process subscripted with `mean`, provided that the name of the input process is a string.

        Returns
        -------
        mean_variable : RandomVariable
            A new random variable representing the mean of the input process across its time index.
        """
        return ProcessTransforms.mean(self, name=name)

    def discount(self, rate: float, name: Hashable | None = None) -> StochasticProcess:
        r"""Return the discounted process of the stochastic process.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        rate : Real
            The discount rate, which must be a positive real number.
        name : Hashable | None, default=None
            The name of the discounted process. If `None`, the new name will be the name of the input process subscripted with `discount`.

        Raises
        ------
        TypeError
            If `process` is not an instance of `StochasticProcess`, if `rate` is not a real number, or if `name` is not a hashable object or `None`.
        ValueError
            If `rate` is not positive.

        Returns
        -------
        discounted_process : StochasticProcess
            The discounted process.

        Notes
        -----
        The discounted process is given by

        $$
        \tilde{S}_t = \frac{S_t}{(1+r)^t},
        $$

        where $S_t$ is the original process and $r$ is the discount rate.

        Examples
        --------
        >>> from sigalg.finance import BinomialPricingModel
        >>> S_0 = 4
        >>> u = 1.2
        >>> d = 0.9
        >>> p = 0.6
        >>> r = 0.01
        >>> S = BinomialPricingModel.generate(
        ...     mode="enum",
        ...     initial_price=S_0,
        ...     up_factor=u,
        ...     down_factor=d,
        ...     up_prob=p,
        ...     risk_free_rate=r,
        ...     length=2,
        ... )
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Binomial price process 'S':
        time    0    1     2
        sample
        0       4  4.8  5.76
        1       4  4.8  4.32
        2       4  3.6  4.32
        3       4  3.6  3.24
        >>> Q = S.EMMs
        >>> print(Q)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'Q':
                probability
        sample
        0          0.134444
        1          0.232222
        2          0.232222
        3          0.401111
        >>> is_martingale_wrt_real_world_measure = S.discount(r).is_martingale()
        >>> print(is_martingale_wrt_real_world_measure)
        False
        >>> is_martingale_wrt_EMM = S.discount(r).is_martingale(prob_measure=Q)
        >>> print(is_martingale_wrt_EMM)
        True
        """
        return ProcessTransforms.discount(self, rate=rate, name=name)

    def stopped(
        self,
        stopping_time: StoppingTime,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        r"""Get the stopped process from a stopping time.

        Examples
        --------
        >>> from math import inf
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk, StoppingTime
        >>> T = Time.discrete(start=1, stop=10)
        >>> S = RandomWalk.generate(
        ...     mode="sim",
        ...     p=0.4,
        ...     initial_state=10,
        ...     index=T,
        ...     n_trajectories=8,
        ...     random_state=42,
        ...     name="S",
        ... )
        >>> print(S)  # doctest: +NORMALIZE_WHITESPACE
        Random walk 'S':
        time    1   2   3   4   5   6   7   8   9   10
        sample
        0       10  11  10  11  12  11  12  13  14  13
        1       10   9   8   9  10  11  10   9   8   7
        2       10  11  12  13  12  13  14  15  14  13
        3       10   9   8   9  10  11  10   9   8   7
        4       10   9   8   7   8   7   8   9   8   9
        5       10  11  10   9  10   9   8   7   8   9
        6       10  11  12  11  10   9   8   9   8   7
        7       10  11  12  11  10   9   8   7   6   5
        >>> tau = StoppingTime.from_filtration(
        ...     process=S,
        ...     mapping={
        ...         0: inf,
        ...         1: 3,
        ...         2: inf,
        ...         3: 3,
        ...         4: 3,
        ...         5: 7,
        ...         6: 7,
        ...         7: 7,
        ...     },
        ... )
        >>> print(tau)  # doctest: +NORMALIZE_WHITESPACE
        Stopping time 'tau':
                tau
        sample
        0       inf
        1       3.0
        2       inf
        3       3.0
        4       3.0
        5       7.0
        6       7.0
        7       7.0
        >>> S_stopped = S.stopped(stopping_time=tau)
        >>> print(S_stopped)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'S^tau':
        time    1   2   3   4   5   6   7   8   9   10
        sample
        0       10  11  10  11  12  11  12  13  14  13
        1       10   9   8   8   8   8   8   8   8   8
        2       10  11  12  13  12  13  14  15  14  13
        3       10   9   8   8   8   8   8   8   8   8
        4       10   9   8   8   8   8   8   8   8   8
        5       10  11  10   9  10   9   8   8   8   8
        6       10  11  12  11  10   9   8   8   8   8
        7       10  11  12  11  10   9   8   8   8   8

        Notes
        -----
        Let $X$ be a $T$-indexed stochastic process on a probability space $(\Omega, \mathcal{F},P)$, and let $\tau: \Omega \to T$ be a stopping time. The *stopped process*, denoted $X^\tau$, is defined for all $t\in T$ by

        $$
        X^\tau_t(\omega) = X_{\min\{t, \tau(\omega)\}}(\omega).
        $$
        """
        return ProcessTransforms.stopped(
            process=self, stopping_time=stopping_time, name=name
        )

    def increments(
        self, forward: bool = True, name: Hashable | None = None
    ) -> StochasticProcess:
        r"""Compute the increments of a stochastic process along its time index.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        forward : bool, default=True
            If `True`, compute forward increments; otherwise, compute backward increments.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of the input process subscripted with `increments`, provided that the name of the input process is a string.

        Returns
        -------
        increments_process : StochasticProcess
            A new stochastic process representing the increments of the input process.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk.generate(mode="enum", p=0.5, index=T, initial_state=3)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time        0  1  2
        sample
        0           3  2  1
        1           3  2  3
        2           3  4  3
        3           3  4  5
        >>> X_increments = X.increments(forward=True)
        >>> print(X_increments) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_increments':
        time        0  1
        sample
        0          -1 -1
        1          -1  1
        2           1 -1
        3           1  1
        >>> X_increments = X.increments(forward=False)
        >>> print(X_increments) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X_increments':
        time        1  2
        sample
        0          -1 -1
        1          -1  1
        2           1 -1
        3           1  1

        Notes
        -----
        Given a stochastic process $X_t$ with index set $\{t_0,t_0+1,\ldots,T\}$ there are two types of increments that can be computed: The first are *forward* increments, which results in a stochastic process $\Delta X_t$ defined as

        $$
        \Delta X_t = X_{t+1} - X_t,
        $$

        for each $t=t_0,\ldots,T-1$. The second type are *backward* increments, which results in a stochastic process $\Delta X_t$ where

        $$
        \Delta X_t = X_t - X_{t-1},
        $$

        for each $t=t_0+1,\ldots,T$.
        """
        return ProcessTransforms.increments(self, forward=forward, name=name)

    def ito_integral(
        self,
        integrator: StochasticProcess,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Compute the Itô integral of a stochastic process with respect to another stochastic process.

        See the Notes section below for the mathematical details.

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
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk.generate(mode="enum", p=0.6, initial_state=0, index=T)
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time    0  1  2
        sample
        0       0 -1 -2
        1       0 -1  0
        2       0  1  0
        3       0  1  2
        >>> time = StochasticProcess.from_time(*X.prob_space, index=T, name="time")
        >>> print(time)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'time':
        time    0  1  2
        sample
        0       0  1  2
        1       0  1  2
        2       0  1  2
        3       0  1  2
        >>> integral = X.increments().ito_integral(time)
        >>> print(integral)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'int X_increments dtime':
        time    0  1  2
        sample
        0       0 -1 -2
        1       0 -1  0
        2       0  1  0
        3       0  1  2
        """
        return ProcessTransforms.ito_integral(self, integrator=integrator, name=name)

    def max_value(self) -> Real:
        """Get the maximum value across all trajectories and time points of a stochastic process.

        See the Notes section below for the mathematical details.

        Returns
        -------
        max_value : Real
            The maximum value found in the stochastic process.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk.generate(mode="enum", p=0.5, index=T, initial_state=3)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time        0  1  2
        sample
        0           3  2  1
        1           3  2  3
        2           3  4  3
        3           3  4  5
        >>> print(X.max_value())
        5
        """
        return ProcessTransforms.max_value(self)

    def min_value(self) -> Real:
        """Get the minimum value across all trajectories and time points of a stochastic process.

        Parameters
        ----------
        process : StochasticProcess
            The stochastic process for which to find the minimum value.

        Returns
        -------
        min_value : Real
            The minimum value found in the stochastic process.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(length=2)
        >>> X = RandomWalk.generate(mode="enum", p=0.5, index=T, initial_state=3)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time        0  1  2
        sample
        0           3  2  1
        1           3  2  3
        2           3  4  3
        3           3  4  5
        >>> print(X.min_value())
        1
        """
        return ProcessTransforms.min_value(self)

    def is_monotonic(self, increasing: bool = True) -> bool:
        """Check if the trajectories of a stochastic process are monotonic.

        Parameters
        ----------
        process : StochasticProcess
            The stochastic process to check for monotonicity.
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
        >>> X = IIDProcess.generate(mode="enum", distribution=bernoulli(p=0.6), support=[0, 1], index=T)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        time        1  2  3
        sample
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
        sample
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

    def to_counting_process(
        self,
        time: Time,
        name: Hashable | None = None,
    ) -> StochasticProcess:
        """Convert a stochastic process of "arrival times" to a counting process.

        The trajectories in the given process are assumed to be the occurrence times of some event, while its time index represents the cumulative counts of those events. This method creates a new stochastic process where, at each time point in the provided `time` index, the value represents the total count of events that have occurred up to that time.

        Parameters
        ----------
        time : Time
            The time index for the counting process.
        name : Hashable | None, default=None
            The name of the transformed process. If `None`, the new name will be the name of the input process subscripted with `counting`, provided that the name of the input process is a string.

        Returns
        -------
        counting_process : StochasticProcess
            A new stochastic process representing the counting process.

        Examples
        --------
        >>> from scipy.stats import expon
        >>> from sigalg.core import Time
        >>> from sigalg.processes import IIDProcess

        Parameters for a Poisson process.

        >>> rate = 2.0
        >>> n_trajectories = 5
        >>> random_state = 42
        >>> max_count = 5

        Create an index for the counts.

        >>> counts = Time.discrete(start=1, stop=max_count, variable_name="count")

        Exponential interarrival times with given rate

        >>> interarrival_times = IIDProcess.generate(
        ...     mode="sim",
        ...     distribution=expon(scale=1 / rate),
        ...     name="interarrival_times",
        ...     index=counts,
        ...     n_trajectories=n_trajectories,
        ...     random_state=random_state,
        ... )
        >>> print(interarrival_times)  # doctest: +NORMALIZE_WHITESPACE
        IID process 'interarrival_times':
        count          1         2         3         4         5
        sample
        0       1.202104  1.168095  1.192380  0.139897  0.043219
        1       0.726330  0.704980  1.562148  0.039647  0.523280
        2       0.035218  0.544512  0.865664  0.193447  0.615793
        3       0.076887  0.045789  0.157590  0.450600  0.206493
        4       0.623693  0.111788  0.918985  0.613543  0.327898

        Compute arrival times by cumulative sum of interarrival times.

        >>> arrival_times = interarrival_times.cumsum().with_name("arrival_times")
        >>> print(arrival_times)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'arrival_times':
        count          1         2         3         4         5
        sample
        0       1.202104  2.370199  3.562580  3.702477  3.745695
        1       0.726330  1.431311  2.993459  3.033106  3.556386
        2       0.035218  0.579730  1.445394  1.638841  2.254634
        3       0.076887  0.122675  0.280265  0.730864  0.937357
        4       0.623693  0.735481  1.654466  2.268009  2.595907

        Determine time grid for Poisson process.

        >>> longest_trajectory = arrival_times.max_value()
        >>> T = Time.continuous(
        ...     start=0.0,
        ...     stop=longest_trajectory + 0.1,
        ...     num_points=6,
        ... )

        Convert to Poisson counting process.

        >>> poisson = arrival_times.to_counting_process(
        ...     time=T,
        ... ).with_name("poisson")
        >>> print(poisson)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Stochastic process 'poisson':
        time    0.000000  0.769139  1.538278  2.307417  3.076556  3.845695
        sample
        0            0.0       0.0       1.0       1.0       2.0       5.0
        1            0.0       1.0       2.0       2.0       4.0       5.0
        2            0.0       2.0       3.0       5.0       5.0       5.0
        3            0.0       4.0       5.0       5.0       5.0       5.0
        4            0.0       2.0       2.0       4.0       5.0       5.0
        """
        return ProcessTransforms.to_counting_process(self, time=time, name=name)
