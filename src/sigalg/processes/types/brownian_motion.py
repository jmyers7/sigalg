"""A class representing a Brownian motion."""

from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ..base.stochastic_process import StochasticProcess

if TYPE_CHECKING:
    from collections.abc import Hashable

    from ...typing.index_like import IndexLike


# TODO: add Notes section to class docstring
# TODO: add custom __repr__ method
class BrownianMotion(StochasticProcess):
    """A class representing a Brownian motion.

    The constructor is not intended for direct usage. Instead, user's should call the `generate` class method. See the Examples section below.

    Examples
    --------
    >>> import numpy as np
    >>> from sigalg.core import Time
    >>> from sigalg.processes import BrownianMotion

    Simulate four trajectories from a standard Brownian motion.

    >>> T = Time.continuous(start=1, stop=2, dt=0.2)
    >>> X = BrownianMotion.generate(n_trajectories=4, index=T, random_state=42)
    >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
    Brownian motion 'X':
    t      1.0       1.2       1.4       1.6       1.8       2.0
    omega
    0      0.0  0.136274 -0.328821  0.006791  0.427424 -0.445106
    1      0.0 -0.582352 -0.525180 -0.666608 -0.674122 -1.055615
    2      0.0  0.393279  0.741118  0.770648  1.274765  1.483842
    3      0.0 -0.384287 -0.219377 -0.648202 -0.255347 -0.277675

    Simulate four trajectories from a geometric Brownian motion with explicit drift and diffusion parameters, and initial state.

    >>> X = BrownianMotion.generate(
    ...     drift=0.9,
    ...     diffusion=0.2,
    ...     initial_state=10,
    ...     n_trajectories=4,
    ...     index=T,
    ...     random_state=42,
    ... )
    >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
    Brownian motion 'X':
    t       1.0        1.2        1.4        1.6        1.8        2.0
    omega
    0      10.0  10.207255  10.294236  10.541358  10.805485  10.810979
    1      10.0  10.063530  10.254964  10.406678  10.585176  10.688877
    2      10.0  10.258656  10.508224  10.694130  10.974953  11.196768
    3      10.0  10.103143  10.316125  10.410360  10.668931  10.844465

    Simulate four trajectories each from three correlated geometric Brownian motions. Notice that the `corr` and `process_params` parameters are explicitly given.

    >>> X_1, X_2, X_3 = BrownianMotion.generate(
    ...     process_params={
    ...         "process_1": {
    ...             "drift": 0.2,
    ...             "diffusion": 0.2,
    ...             "initial_state": 5,
    ...         },
    ...         "process_2": {
    ...             "drift": 0.3,
    ...             "diffusion": 0.1,
    ...             "initial_state": 10,
    ...         },
    ...         "process_3": {
    ...             "drift": 0.1,
    ...             "diffusion": 0.3,
    ...             "initial_state": 8,
    ...         },
    ...     },
    ...     corr=np.array(
    ...         [
    ...             [1, 0.8, 0.2],
    ...             [0.8, 1, 0.7],
    ...             [0.2, 0.7, 1],
    ...         ]
    ...     ),
    ...     kind="geo",
    ...     n_trajectories=4,
    ...     index=T,
    ...     random_state=42,
    ... )
    >>> print(X_1)  # doctest: +NORMALIZE_WHITESPACE
    Brownian motion 'X_1':
    t      1.0       1.2       1.4       1.6       1.8       2.0
    omega
    0        5  4.831503  4.180535  4.222883  4.905912  5.390442
    1        5  5.577524  5.405173  6.266389  6.605897  6.840311
    2        5  4.330638  4.962413  4.929643  5.092950  5.611992
    3        5  5.350632  5.313522  4.863619  5.032385  4.512444
    >>> print(X_2)  # doctest: +NORMALIZE_WHITESPACE
    Brownian motion 'X_2':
    t      1.0        1.2        1.4        1.6        1.8        2.0
    omega
    0       10  10.408150  10.648137  11.228736  12.325913  13.017113
    1       10  11.085709  11.321918  12.405477  13.365085  13.920751
    2       10   9.671646  10.573562  11.317016  11.635994  12.707257
    3       10  10.501574  10.792152  11.135433  12.072122  11.906228
    >>> print(X_3)  # doctest: +NORMALIZE_WHITESPACE
    Brownian motion 'X_3':
    t      1.0       1.2       1.4       1.6       1.8       2.0
    omega
    0        8  8.692539  9.440641  9.698771  9.922823  9.040637
    1        8  8.429620  7.838853  7.572118  8.288721  7.821338
    2        8  6.764620  7.086107  7.754669  6.912697  7.314529
    3        8  7.330884  6.911280  7.472078  8.361373  7.973022
    """

    _repr_name = "BrownianMotion"
    _str_name = "Brownian motion"

    # --------------------- constructors --------------------- #

    @StochasticProcess.generator
    def generate(
        cls,
        drift: Real = 0.0,
        diffusion: Real = 1.0,
        initial_state: Real = 0.0,
        kind: Literal["arith", "geo"] = "arith",
        corr: np.ndarray | None = None,
        process_params: dict[str, dict] | None = None,
        mode: Literal["enum", "sim"] = "sim",
        n_trajectories: int | None = None,
        domain_name: Hashable | None = None,
        index: IndexLike | None = None,
        name: Hashable = "X",
        random_state: int | np.random.Generator | None = None,
    ) -> BrownianMotion | tuple[BrownianMotion, ...]:
        """Simulate trajectories of a Brownian motion.

        Parameters
        ----------
        drift : Real, default=0.0
            The drift coefficient of the Brownian motion.
        diffusion : Real, default=0.0
            The diffusion coefficient of the Brownian motion.
        initial_state : Real, default=0.0
            The initial state of the Brownian motion.
        kind : Literal["arith", "geo"], default="arith"
            The type of Brownian motion, either arithmetic or geometric.
        corr : np.ndarray | None, default=None
            A correlation matrix if multiple correlated Brownian motions are desired. See the Examples.
        process_params : dict[str, dict] | None, default=None
            If multiple correlated Brownian motions are desired, this is a dictionary with keys `'process_1'`, ... `'process_n'` and values the parameters listed above. If passed, a correlation matrix must also be passed. See the Examples.
        mode : Literal["enum", "sim"], default="sim"
            This parameter is included only for consistency with the parent class `StochasticProcess`. It is ignored in this implementation, as the Brownian motion is always generated by Monte Carlo simulation.
        n_trajectories : int | None, default=None
            The number of trajectories to simulate.
        domain_name : Hashable | None, default=None
            The name of the domain. If `None`, a default will be generated.
        index : IndexLike | None, default=None
            The index of the stochastic process.
        name : Hashable, default="X"
            The name of the stochastic process.
        random_state : int | np.random.Generator | None, default=None
            An optional random state for reproducibility.

        Returns
        -------
        processes : BrownianMotion | tuple[BrownianMotion, ...]
            Either a single instance of Brownian motion, or a tuple of correlated Brownian motions.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import Time
        >>> from sigalg.processes import BrownianMotion

        Simulate four trajectories from a standard Brownian motion.

        >>> T = Time.continuous(start=1, stop=2, dt=0.2)
        >>> X = BrownianMotion.generate(n_trajectories=4, index=T, random_state=42)
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Brownian motion 'X':
        t      1.0       1.2       1.4       1.6       1.8       2.0
        omega
        0      0.0  0.136274 -0.328821  0.006791  0.427424 -0.445106
        1      0.0 -0.582352 -0.525180 -0.666608 -0.674122 -1.055615
        2      0.0  0.393279  0.741118  0.770648  1.274765  1.483842
        3      0.0 -0.384287 -0.219377 -0.648202 -0.255347 -0.277675

        Simulate four trajectories from a geometric Brownian motion with explicit drift and diffusion parameters, and initial state.

        >>> X = BrownianMotion.generate(
        ...     drift=0.9,
        ...     diffusion=0.2,
        ...     initial_state=10,
        ...     n_trajectories=4,
        ...     index=T,
        ...     random_state=42,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Brownian motion 'X':
        t       1.0        1.2        1.4        1.6        1.8        2.0
        omega
        0      10.0  10.207255  10.294236  10.541358  10.805485  10.810979
        1      10.0  10.063530  10.254964  10.406678  10.585176  10.688877
        2      10.0  10.258656  10.508224  10.694130  10.974953  11.196768
        3      10.0  10.103143  10.316125  10.410360  10.668931  10.844465

        Simulate four trajectories each from three correlated geometric Brownian motions. Notice that the `corr` and `process_params` parameters are explicitly given.

        >>> X_1, X_2, X_3 = BrownianMotion.generate(
        ...     process_params={
        ...         "process_1": {
        ...             "drift": 0.2,
        ...             "diffusion": 0.2,
        ...             "initial_state": 5,
        ...         },
        ...         "process_2": {
        ...             "drift": 0.3,
        ...             "diffusion": 0.1,
        ...             "initial_state": 10,
        ...         },
        ...         "process_3": {
        ...             "drift": 0.1,
        ...             "diffusion": 0.3,
        ...             "initial_state": 8,
        ...         },
        ...     },
        ...     corr=np.array(
        ...         [
        ...             [1, 0.8, 0.2],
        ...             [0.8, 1, 0.7],
        ...             [0.2, 0.7, 1],
        ...         ]
        ...     ),
        ...     kind="geo",
        ...     n_trajectories=4,
        ...     index=T,
        ...     random_state=42,
        ... )
        >>> print(X_1)  # doctest: +NORMALIZE_WHITESPACE
        Brownian motion 'X_1':
        t      1.0       1.2       1.4       1.6       1.8       2.0
        omega
        0        5  4.831503  4.180535  4.222883  4.905912  5.390442
        1        5  5.577524  5.405173  6.266389  6.605897  6.840311
        2        5  4.330638  4.962413  4.929643  5.092950  5.611992
        3        5  5.350632  5.313522  4.863619  5.032385  4.512444
        >>> print(X_2)  # doctest: +NORMALIZE_WHITESPACE
        Brownian motion 'X_2':
        t      1.0        1.2        1.4        1.6        1.8        2.0
        omega
        0       10  10.408150  10.648137  11.228736  12.325913  13.017113
        1       10  11.085709  11.321918  12.405477  13.365085  13.920751
        2       10   9.671646  10.573562  11.317016  11.635994  12.707257
        3       10  10.501574  10.792152  11.135433  12.072122  11.906228
        >>> print(X_3)  # doctest: +NORMALIZE_WHITESPACE
        Brownian motion 'X_3':
        t      1.0       1.2       1.4       1.6       1.8       2.0
        omega
        0        8  8.692539  9.440641  9.698771  9.922823  9.040637
        1        8  8.429620  7.838853  7.572118  8.288721  7.821338
        2        8  6.764620  7.086107  7.754669  6.912697  7.314529
        3        8  7.330884  6.911280  7.472078  8.361373  7.973022
        """
        if not isinstance(drift, Real):
            raise TypeError("The drift coefficient must be a real number.")
        if not isinstance(diffusion, Real):
            raise TypeError("The diffusion coefficient must be a real number.")
        if not isinstance(initial_state, Real):
            raise TypeError("The initial state must be a real number.")
        if kind not in {"arith", "geo"}:
            raise ValueError("The 'kind' parameter must be either 'arith' or 'geo'.")

        return {
            "drift": drift,
            "diffusion": diffusion,
            "initial_state": initial_state,
            "corr": corr,
            "kind": kind,
        }

    # --------------------- simulation methods --------------------- #

    def _simulation_subclass_hook(self) -> pd.DataFrame:
        """Generate simulated data for the Brownian motion.

        Returns
        -------
        trajectories : pd.DataFrame
            A DataFrame containing the simulated trajectories as rows and time points as columns.
        """
        dt = self._index.data[1] - self._index.data[0]
        initial_time = self._index.data[0]

        increments_arr = self._random_state.normal(
            scale=np.sqrt(dt), size=(self._n_trajectories, len(self._index) - 1)
        )
        increments = pd.DataFrame(increments_arr, columns=self._index.data[1:])

        if self._kind == "arith":
            trajectories = (self._drift * dt + self._diffusion * increments).cumsum(
                axis=1
            )
            trajectories.insert(loc=0, column=initial_time, value=0.0)
            return trajectories + self._initial_state

        else:
            log_incr = (
                self._drift - 0.5 * self._diffusion**2
            ) * dt + self._diffusion * increments
            trajectories = np.exp(log_incr.cumsum(axis=1)) * self._initial_state
            trajectories.insert(loc=0, column=initial_time, value=self._initial_state)
            return trajectories

    @classmethod
    def _multiple_simulation_subclass_hook(
        cls, processes: list[StochasticProcess]
    ) -> list[pd.DataFrame]:
        """Generate simulated data for multiple correlated Brownian motions.

        Returns
        -------
        trajectories : list[pd.DataFrame]
            A list of DataFrames containing the simulated trajectories as rows and time points as columns.
        """
        dt = processes[0]._index.data[1] - processes[0]._index.data[0]
        initial_time = processes[0]._index.data[0]
        rng = processes[0]._random_state
        kind = processes[0]._kind
        n_trajectories = processes[0]._n_trajectories
        length_increments = len(processes[0]._index) - 1
        n_processes = len(processes)

        mean = np.zeros(shape=n_processes)
        corr = processes[0]._corr
        sigmas = np.array([np.sqrt(dt)] * n_processes)
        cov = np.outer(sigmas, sigmas) * corr
        increments_arr = rng.multivariate_normal(
            mean=mean, cov=cov, size=(n_trajectories, length_increments)
        )

        trajectories = []
        if kind == "arith":
            for k, process in enumerate(processes):
                increments = pd.DataFrame(
                    increments_arr[..., k], columns=process._index.data[1:]
                )
                trajectories_k = (
                    process._drift * dt + process._diffusion * increments
                ).cumsum(axis=1)
                trajectories_k.insert(loc=0, column=initial_time, value=0.0)
                trajectories.append(trajectories_k + process._initial_state)

            return trajectories

        else:
            for k, process in enumerate(processes):
                increments = pd.DataFrame(
                    increments_arr[..., k], columns=process._index.data[1:]
                )
                log_incr = (
                    process._drift - 0.5 * process._diffusion**2
                ) * dt + process._diffusion * increments
                trajectories_k = (
                    np.exp(log_incr.cumsum(axis=1)) * process._initial_state
                )
                trajectories_k.insert(
                    loc=0, column=initial_time, value=process._initial_state
                )
                trajectories.append(trajectories_k)

            return trajectories
