"""Stochastic process module.

Classes
-------
StochasticProcess
    A class representing a stochastic process.
"""

from collections.abc import Hashable
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from ...core.base.index import Index
from ...core.base.sample_space import SampleSpace
from ...core.probability_measures.probability_measure import ProbabilityMeasure
from ...core.random_objects.random_variable import RandomVariable
from ...core.random_objects.random_vector import RandomVector
from ...core.sigma_algebras.filtration import Filtration
from ..transforms.process_transforms import ProcessTransformMethods


class StochasticProcess(RandomVector, ProcessTransformMethods):
    """A class representing a stochastic process.

    Parameters
    ----------
    domain : SampleSpace | None, default=None
        The sample space representing the domain of the stochastic process. If `None`, it will be generated later through data generation methods.
    index : Index | None, default=None
        The index of the stochastic process. If `None`, it will be generated later through data generation methods.
    name : Hashable | None, default="X"
        The name of the stochastic process.
    **kwargs
        Additional keyword arguments for subclasses.

    Examples
    --------
    >>> from sigalg.core import SampleSpace, Time
    >>> from sigalg.processes import StochasticProcess
    >>> domain = SampleSpace().from_sequence(size=3, prefix="omega")
    >>> time = Time.discrete(length=3)
    >>> X = StochasticProcess(domain=domain, index=time).from_dict(
    ...     {
    ...         "omega_0": (1, 2, 3),
    ...         "omega_1": (4, 5, 6),
    ...         "omega_2": (7, 8, 9),
    ...     }
    ... )
    >>> X # doctest: +NORMALIZE_WHITESPACE
    Stochastic process 'X':
    time      0  1  2
    sample
    omega_0   1  2  3
    omega_1   4  5  6
    omega_2   7  8  9
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        domain: SampleSpace | None = None,
        index: Index | None = None,
        name: Hashable | None = "X",
        **kwargs,
    ) -> None:
        super().__init__(
            domain=domain,
            index=index,
            name=name,
        )

        # caches
        self._trajectory_counts: pd.Series | None = None
        self._probability_measure: ProbabilityMeasure | None = None

    # --------------------- properties --------------------- #

    @property
    def time(self) -> Index | None:
        """Get the time index.

        This attribute is an alias for public attribute `index` of the superclass `RandomVector`.

        Returns
        -------
        time : Index | None
            The time index of the stochastic process.
        """
        return self.index

    @time.setter
    def time(self, time: Index) -> None:
        """Set the time index.

        Parameters
        ----------
        time : Index
            The time index to set.
        """
        self.index = time

    @property
    def n_trajectories(self) -> int | None:
        """Get the number of trajectories in the stochastic process.

        Returns
        -------
        n_trajectories : int | None
            The number of trajectories in the stochastic process. `None` if data has not been generated.
        """
        return len(self._data) if self._data is not None else None

    @property
    def trajectory_counts(self) -> pd.Series | None:
        """Get the counts of each unique trajectory in the stochastic process.

        Returns
        -------
        trajectory_counts : pd.Series | None
            A Series containing the counts of each unique trajectory, indexed by the domain.
        """
        return self._trajectory_counts

    @trajectory_counts.setter
    def trajectory_counts(self, counts: pd.Series) -> None:
        """Set the trajectory counts.

        This attribute is not meant to be set directly by users. It is intended to be set internally during process transforms and data generation methods.

        Parameters
        ----------
        counts : pd.Series
            A Series containing the counts of each unique trajectory, indexed by the domain.
        """
        if self._data is None:
            raise ValueError("Data must be generated before setting trajectory counts.")
        if not isinstance(counts, pd.Series):
            raise TypeError("counts must be a pandas Series.")
        if self.domain is not None and not counts.index.equals(self.domain.data):
            raise ValueError(
                "The index of counts must match the domain of the process."
            )
        self._trajectory_counts = counts

    @property
    def probability_measure(self) -> ProbabilityMeasure:
        """Generate a probability measure on the domain of the stochastic process.

        Raises a ValueError if data has not been generated for the stochastic process. Data generation must be implemented in subclasses.

        Raises
        ------
        ValueError
            If data has not been generated for the stochastic process.

        Returns
        -------
        prob_measure : ProbabilityMeasure
            The generated probability measure.
        """
        if self._probability_measure is None:
            try:
                if self.is_enumerated:
                    self._probability_measure = self._generate_exact_prob_measure()
                else:
                    self._probability_measure = self._generate_empirical_prob_measure()
            except ValueError:
                return None
        return self._probability_measure

    @probability_measure.setter
    def probability_measure(self, probability_measure: ProbabilityMeasure) -> None:
        """Set the probability measure.

        This attribute is not meant to be set directly by users. It is intended to be set internally during process transforms and data generation methods.

        Parameters
        ----------
        probability_measure : ProbabilityMeasure
            The probability measure to set.
        """
        if self._data is None:
            raise ValueError(
                "Data must be generated before setting a probability measure."
            )
        if not isinstance(probability_measure, ProbabilityMeasure):
            raise TypeError(
                "probability_measure must be an instance of ProbabilityMeasure."
            )
        if (
            self.domain is not None
            and not probability_measure.sample_space.data.equals(self.domain.data)
        ):
            raise ValueError(
                "The sample space of the probability measure must match the domain of the process."
            )
        self._probability_measure = probability_measure

    @property
    def is_enumerated(self) -> bool:
        """Check if the stochastic process is enumerated.

        Raises
        ------
        ValueError
            If the `is_enumerated` property is not set.

        Returns
        -------
        is_enumerated : bool
            `True` if the stochastic process is enumerated, `False` otherwise.
        """
        if not hasattr(self, "_is_enumerated") or not isinstance(
            self._is_enumerated, bool
        ):
            raise ValueError("The is_enumerated property is not set.")
        else:
            return self._is_enumerated

    @property
    def is_discrete_time(self) -> bool:
        """Check if the stochastic process is a discrete-time process.

        Raises
        ------
        TypeError
            If the time index is not an instance of `Time`.

        Returns
        -------
        is_discrete_time : bool
            `True` if the stochastic process is a discrete-time process, `False` otherwise.
        """
        from ...core.base.time import Time

        if self.time is None or not isinstance(self.time, Time):
            raise TypeError("Time index must be an instance of Time.")
        return self.time.is_discrete

    @property
    def is_discrete_state(self) -> bool:
        """Check if the stochastic process is a discrete-state process.

        Raises
        ------
        ValueError
            If the `is_discrete_state` property is not set.

        Returns
        -------
        is_discrete_state : bool
            `True` if the stochastic process is a discrete-state process, `False` otherwise.
        """
        if not hasattr(self, "_is_discrete_state") or not isinstance(
            self._is_discrete_state, bool
        ):
            raise ValueError("The is_discrete_state property is not set.")
        else:
            return self._is_discrete_state

    @property
    def natural_filtration(self) -> Filtration | None:
        """Get the natural filtration of the stochastic process.

        Raises
        ------
        ValueError
            If `name_prefix` is not a string.

        Returns
        -------
        natural_filtration : Filtration | None
            The natural filtration of the stochastic process, or `None` if data has not been generated for the stochastic process.
        """
        if self.data is None:
            return None

        df = pd.DataFrame(
            data={
                t: self.data.iloc[:, : t + 1].apply(tuple, axis=1)
                for t in range(len(self))
            }
        )
        return Filtration(time=self.time).from_pandas(df)

    # --------------------- methods --------------------- #

    def __len__(self) -> int:
        """Get the length of the stochastic process, defined as the number of time points.

        Returns
        -------
        length : int
            The length of the stochastic process.
        """
        return len(self.time) if self.time is not None else None

    # --------------------- data generation methods --------------------- #

    def from_enumeration(self, support: list | None = None, length: int | None = None):
        """Generate data by exhaustively enumerating all possible trajectories.

        If we assume that each random variable in the stochastic process has the same support, then we can generate data for the stochastic process by exhaustively enumerating all possible trajectories of a given length.

        Beware that the number of trajectories grows exponentially with the length of the trajectories and the size of the support. Use this method with caution for large trajectory lengths or supports.

        Parameters
        ----------
        support : list | None, default=None
            A list of values representing the support or states of the stochastic process. If `None`, the support must be set via some other method in the subclass constructors.
        length : int | None, default=None
            The length of each trajectory. If `None`, the length of the existing index is used.

        Returns
        -------
        self : StochasticProcess
            The stochastic process with enumerated trajectories.
        """
        self._validate_and_initialize_time(length)
        if support is None:
            if hasattr(self, "support") and isinstance(self.support, list):
                support = self.support
            else:
                raise ValueError(
                    "Support must be provided to enumerate trajectories. If support is not provided, it must be set through some other method in the subclass constructors."
                )
        all_trajectories = list(product(support, repeat=len(self.time)))
        n_trajectories = len(all_trajectories)
        self._validate_and_initialize_domain(n_trajectories)
        self._trajectory_counts = pd.Series(1, index=self.domain.data, name="counts")
        data = pd.DataFrame(
            data=all_trajectories, index=self.domain.data, columns=self.time.data
        )
        self._is_enumerated = True
        return self.from_pandas(data)

    def from_simulation(
        self,
        max_trajectories: int,
        length: int | None = None,
        random_state: int | None = None,
    ):
        """Generate data by simulating trajectories.

        For this method to be used, a subclass must implement the `_simulation_logic` method, which defines how to simulate trajectories for the specific type of stochastic process.

        Parameters
        ----------
        max_trajectories : int
            The maximum number of trajectories to simulate.
        length : int | None, default=None
            The length of each trajectory. If `None`, the length of the existing time index is used.
        random_state : int | None, default=None
            An optional random seed for reproducibility.

        Raises
        ------
        ValueError
            If `max_trajectories` is not a positive integer, or if a user-specified domain is provided for simulation.

        Returns
        -------
        self : StochasticProcess
            The stochastic process with simulated trajectories.
        """
        if not isinstance(max_trajectories, int) or max_trajectories <= 0:
            raise ValueError("max_trajectories must be a positive integer.")
        if self.domain is not None:
            raise ValueError(
                "A user-specified domain cannot be provided for simulation. A domain will be generated automatically."
            )

        self._validate_and_initialize_time(length)
        trajectories = self._simulation_logic(
            max_trajectories=max_trajectories, random_state=random_state
        )
        # data, self._trajectory_counts = self._group_and_count_simulated_data(
        #     trajectories
        # )
        # self.is_enumerated = False
        # return self.from_pandas(data)

        self._is_enumerated = False
        self._trajectory_counts = pd.Series(
            1, index=range(len(trajectories)), name="counts"
        )
        return self.from_pandas(trajectories)

    def _simulation_logic(
        self, max_trajectories: int, random_state: int | None
    ) -> pd.DataFrame:
        """Abstract method for simulation logic.

        This method must be implemented in subclasses to define how to simulate trajectories.

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
        raise NotImplementedError("Not implemented.")

    def _group_and_count_simulated_data(
        self, trajectories: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Group simulated data by unique trajectories and count occurrences.

        It is possible that the same trajectory is simulated multiple times, especially if the number of trajectories to simulate is large relative to the size of the support. This method groups the simulated data by unique trajectories and counts how many times each unique trajectory was simulated.

        Parameters
        ----------
        trajectories : pd.DataFrame
            A DataFrame containing the simulated trajectories as rows and time points as columns.

        Returns
        -------
        data_and_counts : tuple[pd.DataFrame, pd.Series]
            A tuple containing a DataFrame with unique trajectories and their counts.
        """
        df_counts = trajectories.value_counts(sort=False).reset_index(name="counts")
        counts = df_counts["counts"]
        data = df_counts.drop(columns="counts")
        data.columns = self.time.data
        self._validate_and_initialize_domain(n_trajectories=len(data))
        counts.index = self.domain.data
        data.index = self.domain.data
        return data, counts

    def _validate_and_initialize_time(self, length: int | None = None):
        """Validate and initialize the time index.

        The process may be constructed either with an explicit `Index` instance or `None`. If `None`, this method initializes the index based on the provided `length`. If both an `Index` instance and `length` are provided, this method checks for consistency between them.

        Parameters
        ----------
        length : int | None, default=None
            The length of each trajectory. If `None`, the length of the existing time index is used.

        Raises
        ------
        ValueError
            If neither time index nor length is provided, or if the lengths are inconsistent.
        """
        from ...core.base.time import Time

        if length is not None and (not isinstance(length, int) or length <= 0):
            raise ValueError("If provided, length must be a positive integer.")
        if self.time is None and length is None:
            raise ValueError(
                "Either time index or length must be provided to enumerate the IID process."
            )
        if self.time is not None and length is not None:
            if len(self.time) != length:
                raise ValueError(
                    "Provided length does not match the length of the time index."
                )
        if self.time is None:
            self._index = Time.discrete(length=length)

    def _validate_and_initialize_domain(self, n_trajectories: int):
        """Validate and initialize the domain.

        The process may be constructed either with a `SampleSpace` instance or `None`. If `None`, this method initializes the domain based on the number of trajectories. If a `SampleSpace` instance is provided, this method checks for consistency between its size and the number of trajectories.

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories.

        Raises
        ------
        ValueError
            If neither domain nor number of trajectories is provided, or if sizes are inconsistent.
        """
        if self.domain is None:
            self.domain = SampleSpace(data_name="trajectory").from_sequence(
                size=n_trajectories
            )
        elif len(self.domain) != n_trajectories:
            raise ValueError(
                "The size of the provided domain does not match the number of trajectories."
            )

    # --------------------- probability methods --------------------- #

    def _generate_exact_prob_measure(
        self, name: Hashable | None = "P"
    ) -> ProbabilityMeasure:
        """Generate the exact probability measure for an enumerated stochastic process.

        Subclasses that support enumeration should implement this method to generate the exact probability measure based on the enumerated trajectories.

        Parameters
        ----------
        name : Hashable | None, default="P"
            The name of the generated probability measure.

        Returns
        -------
        prob_measure : ProbabilityMeasure
            The exact probability measure for the enumerated stochastic process.
        """
        raise NotImplementedError(
            "Method to generate exact probability measure not implemented."
        )

    def _generate_empirical_prob_measure(
        self, name: Hashable | None = "P"
    ) -> ProbabilityMeasure:
        """Generate the empirical probability measure for a simulated stochastic process.

        For a simulated stochastic process, we can generate an empirical probability measure by calculating the relative frequencies of the unique trajectories in the simulated data.

        Parameters
        ----------
        name : Hashable | None, default="P"
            The name of the generated probability measure.

        Raises
        ------
        ValueError
            If the process is enumerated, since an empirical probability measure cannot be generated for an enumerated process.

        Returns
        -------
        prob_measure : ProbabilityMeasure
            The empirical probability measure for the simulated stochastic process.

        """
        if self.is_enumerated:
            raise ValueError(
                "Empirical probability measure cannot be generated for an enumerated process."
            )
        # counts_series = self._trajectory_counts
        # probabilities = counts_series / sum(counts_series)
        # return ProbabilityMeasure(sample_space=self.domain, name=name).from_pandas(
        #     probabilities
        # )
        return ProbabilityMeasure.uniform(sample_space=self.domain, name=name)

    # --------------------- data access methods --------------------- #

    def __getitem__(self, time_idx: Hashable) -> RandomVariable:
        """Get the random variable corresponding to a specific time index.

        Parameters
        ----------
        time_idx : Hashable
            The time index to access.

        Returns
        -------
        rv : RandomVariable
            The random variable corresponding to the specified time index.
        """
        from sigalg.core.base.time import Time

        if self.time is None:
            raise ValueError("Time index is not defined for this stochastic process.")

        if not isinstance(self.time, Time) or self.time.is_discrete:
            if time_idx not in self.time:
                raise ValueError(f"Time {time_idx} not in process time index")
        else:
            time_idx = self.time.find_nearest_time(time_idx)

        name = f"{self.name}_{time_idx}" if self.name is not None else None
        return self.get_component_rv(time_idx).with_name(name)

    @property
    def at(self):
        """Get an indexer for accessing component random variables at specific times.

        Returns
        -------
        at : _RVAtIndexer
            An indexer for accessing component random variables at specific times.
        """
        return self._RVAtIndexer(self)

    class _RVAtIndexer:
        def __init__(self, stochastic_process):
            self.stochastic_process = stochastic_process

        def __getitem__(self, time_idx) -> RandomVariable:

            if self.stochastic_process.time.is_discrete:
                if time_idx not in self.stochastic_process.time:
                    raise ValueError(f"Time {time_idx} not in process time index")
                else:
                    name = (
                        f"{self.stochastic_process.name}_{time_idx}"
                        if self.stochastic_process.name is not None
                        else None
                    )
                    return self.stochastic_process.get_component_rv(time_idx).with_name(
                        name
                    )
            else:
                nearest_time = self.stochastic_process.time.find_nearest_time(time_idx)
                name = (
                    f"{self.stochastic_process.name}_{nearest_time}"
                    if self.stochastic_process.name is not None
                    else None
                )
                return self.stochastic_process.get_component_rv(nearest_time).with_name(
                    name
                )

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Get the string representation of the stochastic process.

        Returns
        -------
        repr_str : str
            The string representation of the stochastic process.
        """
        if self.dimension == 1:
            data = self.data.to_frame()
            data.columns = [self.name]
        else:
            data = self.data
        if self.name is None:
            return f"Stochastic process:\n{data}"
        else:
            return f"Stochastic process '{self.name}':\n{data}"

    # --------------------- equality --------------------- #

    def __eq__(self, other) -> bool:
        """Check equality between two stochastic processes.

        Parameters
        ----------
        other : StochasticProcess
            The other stochastic process to compare with.

        Returns
        -------
        is_equal : bool
            True if the stochastic processes are equal, False otherwise.
        """
        if not isinstance(other, StochasticProcess):
            return False
        return super().__eq__(other)

    # --------------------- plotting methods --------------------- #

    def plot_trajectories(
        self,
        ax: Axes = None,
        colors: list = None,
        plot_kwargs: dict = None,
        x_label: str = "time",
        y_label: str = "state",
        title: str = None,
    ):
        """Plot the trajectories of the stochastic process.

        Requires the data to be generated for the stochastic process. Only subclasses that implement data generation methods can use this method.

        Parameters
        ----------
        ax : Axes, default=None
            A matplotlib Axes object to plot on. If `None`, a new figure and axes will be created.
        colors : list, default=None
            A list of colors to use for the trajectories. If `None`, default matplotlib colors will be used.
        plot_kwargs : dict, default=None
            Additional keyword arguments to pass to the plotting function.
        x_label : str, default="time"
            Label for the x-axis.
        y_label : str, default="state"
            Label for the y-axis.
        title : str, default=None
            Title of the plot. If `None`, a default title will be generated.

        Raises
        ------
        ValueError
            If data has not been generated for the stochastic process.

        Returns
        -------
        ax : Axes
            The matplotlib Axes object with the plot.
        """
        if self._data is None:
            raise ValueError("Data must be generated before plotting trajectories.")

        columns = self.time.data
        n_trajectories = self.n_trajectories

        if ax is None:
            _, ax = plt.subplots()
        elif not isinstance(ax, Axes):
            raise ValueError("ax must be a matplotlib Axes object")

        if plot_kwargs is None:
            plot_kwargs = {}

        if colors is not None:
            if not isinstance(colors, list):
                raise ValueError("colors must be a list")
            if len(colors) == 1:
                colors = [colors[0]] * n_trajectories
            else:
                custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
                if n_trajectories == 1:
                    colors = [custom_cmap(0)]
                else:
                    colors = [
                        custom_cmap(i / (n_trajectories - 1))
                        for i in range(n_trajectories)
                    ]

        for i, (_, row) in enumerate(self.data.iterrows()):
            if colors is not None:
                ax.plot(columns, row, color=colors[i], **plot_kwargs)
            else:
                ax.plot(columns, row, **plot_kwargs)

        is_time_integer = self._integer_check(columns.values)
        is_trajectory_integer = self._integer_check(self.data.values.flatten())
        if is_time_integer:
            time_values = columns.values.astype(int)
            if len(time_values) <= 20:
                ax.set_xticks(time_values)
            else:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if is_trajectory_integer:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if title is None:
            title = self._plot_title()
        ax.set_title(title)

        return ax

    def _integer_check(self, values):
        try:
            return np.allclose(values, np.round(values))
        except (TypeError, AttributeError):
            return False

    def _plot_title(self):
        """Generate a default plot title based on the name of the stochastic process.

        Subclasses can override this method to provide more specific default titles for different types of stochastic processes.
        """
        return f"Stochastic process '{self.name}'"
