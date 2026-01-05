"""Independent and identically distributed (IID) process module."""

from collections.abc import Hashable

import numpy as np
import pandas as pd
from scipy.stats._distn_infrastructure import rv_frozen

from ...core.base.sample_space import SampleSpace
from ...core.base.time import Time
from ...core.probability_measures.probability_measure import ProbabilityMeasure
from ..base.stochastic_process import StochasticProcess


class IIDProcess(StochasticProcess):
    """
    A class representing an Independent and Identically Distributed (IID) stochastic process.

    Each random variable in the process follows the same probability distribution, and the joint distribution of any finite collection of these variables is the product of their individual distributions.

    Parameters
    ----------
    rv : rv_frozen
        A frozen random variable from scipy.stats representing the common distribution of the IID process.
    support : list | None, default=None
        An optional list of values representing the support of the random variable. If provided, it is used for validation and enumeration of trajectories.
    domain : SampleSpace | None, default=None
        An optional SampleSpace object defining the domain of the process. If not provided, it will be generated during data generation.
    index : Time | None, default=None
        An optional Time object defining the time index of the process. If not provided, it will be generated during data generation.
    name : Hashable | None, default="X"
        An optional name for the process.
    is_enumerated : bool, default=False
        A flag indicating whether the process is enumerated.

    Raises
    ------
    TypeError
        If `rv` is not an instance of `rv_frozen`.
    ValueError
        If `support` contains values incompatible with the provided `rv`.

    Examples
    --------
    >>> from scipy.stats import bernoulli
    >>> from sigalg.core import SampleSpace, Time
    >>> from sigalg.processes import IIDProcess
    >>> domain = SampleSpace().from_sequence(size=3, prefix="omega")
    >>> time = Time.discrete(length=3)
    >>> # Construct Bernoulli IID process via exhaustive enumeration
    >>> rv = bernoulli(p=0.25)
    >>> X = IIDProcess(rv=rv, support=[0, 1], index=time).from_enumeration()
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
    >>> P = X.generate_prob_measure()
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
    >>> rv = poisson(mu=1.0)
    >>> Y = IIDProcess(rv=rv, name="Y").from_simulation(
    ...     max_trajectories=10_000, random_state=42, length=3
    ... )
    >>> Y # doctest: +NORMALIZE_WHITESPACE
    Stochastic process 'Y':
    time  0  1  2
    trajectory
    0     0  0  0
    1     0  0  1
    2     0  0  2
    3     0  0  3
    4     0  0  4
    ...  .. .. ..
    163   5  3  2
    164   6  0  0
    165   6  0  1
    166   6  1  2
    167   6  4  0
    <BLANKLINE>
    [168 rows x 3 columns]
    >>> # Generate the empirical probability measure associated with the simulated process
    >>> Q = Y.generate_prob_measure(name="Q")
    >>> Q # doctest: +NORMALIZE_WHITESPACE
    Probability measure 'Q':
            probability
    trajectory
    0          0.0461
    1          0.0521
    2          0.0238
    3          0.0089
    4          0.0019
    ...           ...
    163        0.0001
    164        0.0001
    165        0.0001
    166        0.0001
    167        0.0001
    <BLANKLINE>
    [168 rows x 1 columns]
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        rv: rv_frozen,
        support: list | None = None,
        domain: SampleSpace | None = None,
        index: Time | None = None,
        name: Hashable | None = "X",
        is_enumerated: bool = False,
    ) -> None:
        super().__init__(
            domain=domain,
            index=index,
            name=name,
            is_enumerated=is_enumerated,
        )

        if not isinstance(rv, rv_frozen):
            raise TypeError("rv must be an instance of rv_frozen from scipy.stats.")

        if support is not None:
            try:
                _ = rv.pmf(support)
            except Exception as e:
                raise ValueError(
                    "support contains values incompatible with the provided rv."
                ) from e

        self.rv = rv
        self.support = support

    # --------------------- data generation methods --------------------- #

    def _simulation_logic(
        self, max_trajectories: int, length: int | None, random_state: int | None
    ):
        """Generate simulated data for the IID process.

        Parameters
        ----------
        max_trajectories : int
            The maximum number of trajectories to simulate.
        length : int | None
            The length of each trajectory. If `None`, the length of the existing time index is used.
        random_state : int | None
            An optional random seed for reproducibility.

        Returns
        -------
        all_data : pd.DataFrame
            A DataFrame containing the simulated trajectories.
        """
        all_trajectories = self.rv.rvs(
            size=(max_trajectories, len(self.time)),
            random_state=np.random.default_rng(random_state),
        )
        all_data = pd.DataFrame(data=all_trajectories, columns=self.time.data)
        return all_data

    # --------------------- probability methods --------------------- #

    def _generate_exact_prob_measure(
        self, name: Hashable | None = "P"
    ) -> ProbabilityMeasure:
        raw_trajectories = self.data
        element_wise_probabilities = self.rv.pmf(raw_trajectories.values)
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
        return f"{prefix} {self.rv.dist.name.capitalize()} process '{self.name}'"
