"""Independent and identically distributed (IID) process module."""

from collections.abc import Hashable

import numpy as np
import pandas as pd
from scipy.stats._distn_infrastructure import rv_discrete, rv_frozen

from ...core.base.index import Index
from ...core.base.sample_space import SampleSpace
from ...core.probability_measures.probability_measure import ProbabilityMeasure
from ..base.stochastic_process import StochasticProcess


class IIDProcess(StochasticProcess):
    """
    A class representing an Independent and Identically Distributed (IID) stochastic process.

    Each random variable in the process follows the same probability distribution, and the joint distribution of any finite collection of these variables is the product of their individual distributions.

    Parameters
    ----------
    distribution : rv_frozen
        A frozen random variable from scipy.stats representing the common distribution of the IID process.
    domain : SampleSpace | None, default=None
        The sample space representing the domain of the stochastic process. If `None`, it will be generated later through data generation methods.
    index : Index | None, default=None
        The index of the stochastic process. If `None`, it will be generated later through data generation methods.
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
    >>> time = Time.discrete(length=3)
    >>> # Construct Bernoulli IID process via exhaustive enumeration
    >>> X = IIDProcess(distribution=bernoulli(p=0.25), index=time).from_enumeration(support=[0, 1])
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
    >>> Y = IIDProcess(distribution=poisson(mu=1.0), name="Y").from_simulation(
    ...     max_trajectories=10_000, random_state=42, length=3
    ... )
    >>> Y # doctest: +NORMALIZE_WHITESPACE
    Stochastic process 'Y':
    time  0  1  2
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
        domain: SampleSpace | None = None,
        index: Index | None = None,
        name: Hashable | None = "X",
    ) -> None:
        super().__init__(
            domain=domain,
            index=index,
            name=name,
        )
        if not isinstance(distribution, rv_frozen):
            raise TypeError(
                "distribution must be an instance of rv_frozen from scipy.stats."
            )

        self.distribution = distribution
        self._is_discrete_state = isinstance(distribution.dist, rv_discrete)

    # --------------------- data generation methods --------------------- #

    def _simulation_logic(
        self,
        max_trajectories: int,
        random_state: int | None,
    ) -> pd.DataFrame:
        """Generate simulated data for the IID process.

        Parameters
        ----------
        max_trajectories : int
            The maximum number of trajectories to simulate.
        random_state : int | None
            An optional random seed for reproducibility.

        Returns
        -------
        trajectories : pd.DataFrame
            A DataFrame containing the simulated trajectories as rows and time points as columns.
        """
        trajectories = self.distribution.rvs(
            size=(max_trajectories, len(self.time)),
            random_state=np.random.default_rng(random_state),
        )
        return pd.DataFrame(data=trajectories, columns=self.time.data)

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
        element_wise_probabilities = self.distribution.pmf(self.data.values)
        probabilities = pd.Series(
            data=np.prod(element_wise_probabilities, axis=1),
            index=self.domain.data,
        )
        return ProbabilityMeasure(sample_space=self.domain, name=name).from_pandas(
            probabilities
        )

    # --------------------- plotting methods --------------------- #

    def _plot_title(self):
        prefix = "Enumerated IID" if self.is_enumerated else "IID"
        return (
            f"{prefix} {self.distribution.dist.name.capitalize()} process '{self.name}'"
        )
