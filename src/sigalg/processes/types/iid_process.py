"""A class representing an independent and identically distributed (IID) stochasti process."""

from __future__ import annotations

from collections.abc import Hashable
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats._multivariate import multinomial_frozen

from ..base.stochastic_process import StochasticProcess

if TYPE_CHECKING:
    from ...core.base.index import Index
    from ...core.probability_measures.probability_measure import ProbabilityMeasure


# TODO: Update docstrings—be sure to add description of `support` parameter
class IIDProcess(StochasticProcess):
    """A class representing an independent and identically distributed (IID) stochastic process.

    The `is_discrete_state` attribute from the parent class `StochasticProcess` is automatically determined based on whether the provided distribution is discrete or continuous.

    Parameters
    ----------
    distribution : rv_frozen
        A frozen random variable from scipy.stats representing the common distribution of the IID process.
    support: list | dict | None, default=None
        Add description later.
    time : Time | None, default=None
        The time index of the stochastic process. If `None`, then the `is_discrete_time` property must be provided.
    is_discrete_time : bool | None, default=None
        Whether the stochastic process is a discrete-time process. If `None`, then `time` parameter must be provided.
    domain : SampleSpace | None, default=None
        The sample space representing the domain of the stochastic process. If `None`, it will be generated later through data generation methods.
    name : Hashable | None, default="X"
        The name of the stochastic process.

    Raises
    ------
    TypeError
        If `rv` is not an instance of `rv_frozen` or `multinomial_frozen`.

    Examples
    --------
    >>> from scipy.stats import bernoulli
    >>> from sigalg.core import SampleSpace, Time
    >>> from sigalg.processes import IIDProcess
    >>> domain = SampleSpace().from_sequence(size=3, prefix="omega")
    >>> time = Time.discrete(length=2)
    >>> # Construct Bernoulli IID process via exhaustive enumeration
    >>> X = IIDProcess(distribution=bernoulli(p=0.25), support=[0, 1], time=time).from_enumeration()
    >>> X # doctest: +NORMALIZE_WHITESPACE
    Stochastic process 'X':
    time  0  1  2
    trajectory
    0     0  0  0
    1     0  0  1
    2     0  1  0
    3     0  1  1
    4     1  0  0
    5     1  0  1
    6     1  1  0
    7     1  1  1
    >>> # Generate the exact probability measure associated with the enumerated process
    >>> P = X.prob_measure
    >>> P # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'P':
            probability
    trajectory
    0        0.421875
    1        0.140625
    2        0.140625
    3        0.046875
    4        0.140625
    5        0.046875
    6        0.046875
    7        0.015625
    >>> # Construct Poisson IID process via simulation, with non-specified domain and time index
    >>> from scipy.stats import poisson
    >>> time = Time.discrete(length=2)
    >>> Y = IIDProcess(distribution=poisson(mu=1.0), time=time, name="Y").from_simulation(
    ...     n_trajectories=10_000, random_state=42
    ... )
    >>> Y # doctest: +NORMALIZE_WHITESPACE
    Stochastic process 'Y':
    time  0  1  2
    trajectory
    0     1  2  3
    1     1  3  0
    2     1  3  3
    3     1  0  3
    4     1  0  0
    ...  .. .. ..
    9995  1  2  2
    9996  0  3  0
    9997  0  2  1
    9998  1  3  2
    9999  1  2  2
    <BLANKLINE>
    [10000 rows x 3 columns]
    """

    _repr_name = "IID process"

    # --------------------- enumeration methods --------------------- #

    @classmethod
    def from_enumeration(
        cls,
        distribution: rv_frozen | multinomial_frozen,
        support: list,
        index: Index | None = None,
        length: int | None = None,
        name: Hashable = "X",
    ) -> StochasticProcess:
        """Later."""
        if not (
            isinstance(distribution, rv_frozen)
            or isinstance(distribution, multinomial_frozen)
        ):
            raise TypeError(
                "distribution must be an instance of rv_frozen or multinomial_frozen from scipy.stats."
            )
        if support is not None and not (
            isinstance(support, list) or isinstance(support, dict)
        ):
            raise TypeError("If given, support must be a list or dict.")

        index = cls._validate_and_return_index(index=index, length=length)
        process = cls(index=index, name=name)

        process.distribution = distribution
        process.support = support

        return process._enumeration_logic()

    def _enumeration_hook(self) -> pd.DataFrame:
        """Generate the enumerated trajectories for the IID process based on the provided support and trajectory length.

        Returns
        -------
        trajectories : pd.DataFrame
            A DataFrame containing the enumerated trajectories as rows and time points as columns.
        """
        if isinstance(self.support, dict):
            support = self.support.values()
        else:
            support = self.support

        trajectories = list(product(support, repeat=len(self.time)))
        return pd.DataFrame(data=trajectories, columns=self.time.data)

    def _generate_exact_prob_measure(self) -> ProbabilityMeasure:
        """Generate the exact probability measure for the IID process based on its distribution and domain.

        Parameters
        ----------
        name : Hashable | None, default="P"
            The name of the generated probability measure.

        Returns
        -------
        prob_measure : ProbabilityMeasure
            The generated probability measure.
        """
        from ...core.probability_measures.probability_measure import ProbabilityMeasure
        from ...core.sigma_algebras.sigma_algebra import SigmaAlgebra

        if isinstance(self.support, dict):
            inverse_support = {y: x for x, y in self.support.items()}
            values = self.data.map(lambda x: inverse_support[x]).values
        else:
            values = self.data.values

        if isinstance(self.distribution, multinomial_frozen):
            element_wise_probabilities = self.distribution.p[values]
        else:
            element_wise_probabilities = self.distribution.pmf(values)
        probabilities = pd.Series(
            data=np.prod(element_wise_probabilities, axis=1),
            index=self.sample_space.data,
        )

        probabilities /= probabilities.sum()
        return ProbabilityMeasure(
            sig_alg=SigmaAlgebra.power_set(self.sample_space),
            mapping=probabilities,
        )

    # --------------------- simulation methods --------------------- #

    @classmethod
    def from_simulation(
        cls,
        distribution: rv_frozen | multinomial_frozen,
        support: list,
        n_trajectories: int,
        index: Index | None = None,
        length: int | None = None,
        random_state: int | np.random.Generator | None = None,
        name: Hashable = "X",
    ) -> StochasticProcess:
        """Later."""
        if not (
            isinstance(distribution, rv_frozen)
            or isinstance(distribution, multinomial_frozen)
        ):
            raise TypeError(
                "distribution must be an instance of rv_frozen or multinomial_frozen from scipy.stats."
            )
        if support is not None and not (
            isinstance(support, list) or isinstance(support, dict)
        ):
            raise TypeError("If given, support must be a list or dict.")

        index = cls._validate_and_return_index(index=index, length=length)
        process = cls(index=index, name=name)

        process.n_trajectories = n_trajectories
        process.random_state = random_state
        process.distribution = distribution
        process.support = support

        return process._simulation_logic()

    def _simulation_hook(self) -> pd.DataFrame:
        """Generate simulated data for the IID process.

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories to simulate.
        random_state : int | None
            An optional random seed for reproducibility.

        Returns
        -------
        trajectories : pd.DataFrame
            A DataFrame containing the simulated trajectories as rows and time points as columns.
        """
        trajectories = self.distribution.rvs(
            size=(self.n_trajectories, len(self.time)),
            random_state=self.random_state,
        )
        trajectories_df = pd.DataFrame(data=trajectories, columns=self.time.data)

        if isinstance(self.support, dict):
            return trajectories_df.map(lambda x: self.support[x])
        else:
            return trajectories_df
