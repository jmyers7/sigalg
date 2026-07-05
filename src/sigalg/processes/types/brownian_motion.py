"""A class representing a Brownian motion."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..base.stochastic_process import StochasticProcess

if TYPE_CHECKING:
    from ...core.base.index import Index


class BrownianMotion(StochasticProcess):
    """A class representing a Brownian motion.

    The constructor is not intended for direct usage. Instead, user's should call the class method `from_simulation`. See the Examples section below.

    See also the Notes section below for the mathematical details.

    Parameters
    ----------
    sample_space : SampleSpace | None, default=None
        The sample space of the underlying probability space.
    sig_alg : SigmaAlgebra | None, default=None
        The sigma algebra of the underlying probability space.
    prob_measure : ProbabilityMeasure | None, default=None
        The probability measure of the underlying probability space.
    index : Index | None, default=None
        The index of the random vector.
    name : Hashable, default="X"
        The name of the random vector.
    **kwargs
        Additional keyword arguments for subclass constructors.

    Examples
    --------
    >>> from sigalg.core import Time
    >>> from sigalg.processes import BrownianMotion
    >>> T = Time.continuous(start=0.1, stop=1.1, dt=0.35)
    >>> X = BrownianMotion.from_simulation(n_trajectories=4, index=T, random_state=42)
    >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
    Brownian motion 'X':
    time        0.100000  0.433333  0.766667  1.100000
    sample
    0                0.0  0.175928 -0.424507  0.008767
    1                0.0  0.543035 -0.583395 -1.335209
    2                0.0  0.073809 -0.108774 -0.118474
    3                0.0 -0.492505  0.015216  0.464274
    """

    _repr_name = "Brownian motion"

    # --------------------- simulation methods --------------------- #

    @classmethod
    def from_simulation(
        cls,
        n_trajectories: int,
        index: Index | None = None,
        random_state: int | np.random.Generator | None = None,
        name: Hashable = "X",
    ) -> StochasticProcess:
        """Simulate trajectories of the Brownian motion.

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories to simulate.
        index : Index | None, default=None
            The index of the stochastic process.
        random_state : int | np.random.Generator | None, default=None
            An optional seed (`int`) for the random number generator, or a `np.random.Generator` instance to use directly. If an integer is provided, a new generator is created with that seed. If a `Generator` is provided, it is used directly and its state is advanced. If `None`, the random number generator is not seeded.
        name : Hashable | None, default="X"
            The name of the stochastic process.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import BrownianMotion
        >>> T = Time.continuous(start=0.1, stop=1.1, dt=0.35)
        >>> X = BrownianMotion.from_simulation(n_trajectories=4, index=T, random_state=42)
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Brownian motion 'X':
        time        0.100000  0.433333  0.766667  1.100000
        sample
        0                0.0  0.175928 -0.424507  0.008767
        1                0.0  0.543035 -0.583395 -1.335209
        2                0.0  0.073809 -0.108774 -0.118474
        3                0.0 -0.492505  0.015216  0.464274
        """
        index = cls._validate_and_return_index(index=index, length=None)
        random_state = cls._validate_simulation_parameters_and_return_random_state(
            n_trajectories=n_trajectories, random_state=random_state
        )
        process = cls(index=index, name=name)

        process.n_trajectories = n_trajectories
        process.random_state = random_state

        return process._simulation_logic()

    def _simulation_hook(self) -> pd.DataFrame:
        """Generate simulated data for the Brownian motion.

        Returns
        -------
        trajectories : pd.DataFrame
            A DataFrame containing the simulated trajectories as rows and time points as columns.
        """
        from scipy.stats import norm

        from ...core.random_objects.random_variable import RandomVariable
        from .iid_process import IIDProcess

        dt = self.time.data[1] - self.time.data[0]
        initial_time = self.time.data[0]
        increments_time = self.time.remove_time(pos=0)

        increments = IIDProcess.from_simulation(
            distribution=norm(loc=0.0, scale=np.sqrt(dt)),
            index=increments_time,
            n_trajectories=self.n_trajectories,
            random_state=self.random_state,
        )

        initial_value = RandomVariable.from_constant(
            sample_space=increments.sample_space,
            name=initial_time,
            constant=0.0,
        )

        return increments.insert_rv(rv=initial_value, time=initial_time).cumsum().data
