"""A class representing an independent and identically distributed (IID) process.

Classes
------
IIDProcess
    A class representing an independent and identically distributed (IID) stochastic process.
"""

from collections.abc import Hashable
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats._distn_infrastructure import rv_discrete, rv_frozen

from ...core.base.sample_space import SampleSpace
from ...core.base.time import Time
from ...core.probability_measures.probability_measure import ProbabilityMeasure
from ..base.stochastic_process import StochasticProcess


# TODO: Update docstrings—be sure to add description of `support` parameter
class IIDProcess(StochasticProcess):
    """A class representing an Independent and Identically Distributed (IID) stochastic process.

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
        If `rv` is not an instance of `rv_frozen`.

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
    >>> P = X.probability_measure
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
    >>> Y = IIDProcess(distribution=poisson(mu=1.0), is_discrete_time=True, name="Y").from_simulation(
    ...     n_trajectories=10_000, random_state=42, length=2
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

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        distribution: rv_frozen,
        support: list | None = None,
        time: Time | None = None,
        is_discrete_time: bool | None = None,
        domain: SampleSpace | None = None,
        name: Hashable | None = "X",
    ) -> None:
        if not isinstance(distribution, rv_frozen):
            raise TypeError(
                "distribution must be an instance of rv_frozen from scipy.stats."
            )
        if (
            support is not None
            and not isinstance(support, list)
            and not isinstance(support, dict)
        ):
            raise TypeError("support must be a list or dict if provided.")
        self.distribution = distribution
        self.support = support

        super().__init__(
            domain=domain,
            time=time,
            is_discrete_state=isinstance(distribution.dist, rv_discrete),
            is_discrete_time=is_discrete_time,
            name=name,
        )

    # --------------------- data generation methods --------------------- #

    def _enumeration_logic(self, **kwargs) -> pd.DataFrame:
        """Generate the enumerated trajectories for the IID process based on the provided support and trajectory length.

        Returns
        -------
        trajectories : pd.DataFrame
            A DataFrame containing the enumerated trajectories as rows and time points as columns.
        """
        if self.is_discrete_state is False:
            raise ValueError("Enumeration is only supported for discrete state spaces.")
        if self.support is None:
            raise ValueError("Support must be provided for enumeration.")

        if isinstance(self.support, dict):
            support = self.support.values()
        else:
            support = self.support

        trajectories = list(product(support, repeat=len(self.time)))
        return pd.DataFrame(data=trajectories, columns=self.time.data)

    def _simulation_logic(
        self,
        n_trajectories: int,
        random_state: int | None,
    ) -> pd.DataFrame:
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
            size=(n_trajectories, len(self.time)),
            random_state=np.random.default_rng(random_state),
        )
        trajectories_df = pd.DataFrame(data=trajectories, columns=self.time.data)

        if isinstance(self.support, dict):
            return trajectories_df.map(lambda x: self.support[x])
        else:
            return trajectories_df

    # --------------------- probability methods --------------------- #

    def _generate_exact_prob_measure(
        self, name: Hashable | None = "P"
    ) -> ProbabilityMeasure:
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
        if isinstance(self.support, dict):
            inverse_support = {y: x for x, y in self.support.items()}
            values = self.data.map(lambda x: inverse_support[x]).values
        else:
            values = self.data.values

        element_wise_probabilities = self.distribution.pmf(values)
        probabilities = pd.Series(
            data=np.prod(element_wise_probabilities, axis=1),
            index=self.domain.data,
        )
        probabilities /= probabilities.sum()
        return ProbabilityMeasure(sample_space=self.domain, name=name).from_pandas(
            probabilities
        )

    # --------------------- plotting methods --------------------- #

    def _plot_title(self):
        prefix = "Enumerated IID" if self._is_enumerated else "IID"
        return (
            f"{prefix} {self.distribution.dist.name.capitalize()} process '{self.name}'"
        )
