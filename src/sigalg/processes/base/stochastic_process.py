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
    is_enumerated : bool, default=False
        A boolean indicating whether the trajectories of the stochastic process have been exhaustively enumerated.
    **kwargs
        Additional keyword arguments for subclasses.

    Raises
    ------
    TypeError
        If `is_enumerated` is not a boolean.

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
        is_enumerated: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            domain=domain,
            index=index,
            name=name,
        )
        if not isinstance(is_enumerated, bool):
            raise TypeError("is_enumerated must be a boolean.")
        self.is_enumerated = is_enumerated

        # cache for trajectory counts in case of enumeration
        self._trajectory_counts: pd.Series | None = None

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

    # --------------------- data generation methods --------------------- #

    def from_enumeration(self, length: int | None = None):
        """Generate data by exhaustively enumerating all possible trajectories.

        This method only works for subclasses of `StochasticProcess` that have a `support` attribute.

        Parameters
        ----------
        length : int | None, default=None
            The length of each trajectory. If `None`, the length of the existing time index is used.

        Raises
        ------
        AttributeError
            If the process does not have a `support` attribute.

        Returns
        -------
        self : StochasticProcess
            The stochastic process with enumerated trajectories.
        """
        if not hasattr(self, "support"):
            raise AttributeError(
                "The process must be a type with a 'support' attribute for enumeration of trajectories."
            )
        if self.support is None:
            raise AttributeError(
                "The 'support' attribute must be defined for enumeration of trajectories."
            )

        self._validate_and_initialize_time(length)

        all_trajectories = list(product(self.support, repeat=len(self.time)))
        n_trajectories = len(all_trajectories)

        self._validate_and_initialize_domain(n_trajectories)

        self._trajectory_counts = pd.Series(1, index=self.domain)

        data = pd.DataFrame(
            data=all_trajectories, index=self.domain.data, columns=self.time.data
        )

        self.is_enumerated = True
        return self.from_pandas(data)

    def from_simulation(
        self,
        max_trajectories: int,
        length: int | None = None,
        random_state: int | None = None,
    ):
        """Generate data by simulating trajectories.

        This is an abstract method that must be implemented in subclasses.

        Parameters
        ----------
        max_trajectories : int
            The maximum number of trajectories to simulate.
        length : int | None, default=None
            The length of each trajectory. If `None`, the length of the existing time index is used.
        random_state : int | None, default=None
            An optional random seed for reproducibility.
        """
        if not isinstance(max_trajectories, int) or max_trajectories <= 0:
            raise ValueError("max_trajectories must be a positive integer.")

        self._validate_and_initialize_time(length)

        all_data = self._simulation_logic(
            max_trajectories=max_trajectories, length=length, random_state=random_state
        )
        grouped_data = all_data.value_counts(sort=False).reset_index(name="counts")
        n_trajectories = len(grouped_data)

        if self.domain is not None:
            raise ValueError(
                "A user-specified domain cannot be provided for simulation. A domain will be generated automatically."
            )

        self._validate_and_initialize_domain(n_trajectories)

        self._trajectory_counts = grouped_data["counts"]
        self._trajectory_counts.index = self.domain.data

        data = grouped_data.drop(columns="counts")
        data.index = self.domain.data
        data.columns = self.time.data

        self.is_enumerated = False
        return self.from_pandas(data)

    def _simulation_logic(
        self, max_trajectories: int, length: int | None, random_state: int | None
    ):
        """Abstract method for simulation logic.

        This method must be implemented in subclasses to define how to simulate trajectories.

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
        raise NotImplementedError(
            "Subclasses must implement the _simulation_logic method."
        )

    def _validate_and_initialize_time(self, length: int | None = None):
        """Validate and initialize the time index.

        The process may be constructed either with a `Time` instance or `None`. If `None`, this method initializes the time index based on the provided `length`. If both a `Time` instance and `length` are provided, this method checks for consistency between them.

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
            self.time = Time.discrete(length=length)

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

    # --------------------- data access methods --------------------- #

    @property
    def rv_at(self):
        """Get an indexer for accessing component random variables at specific times.

        Returns
        -------
        rv_at : _RVAtIndexer
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

    # --------------------- probability methods --------------------- #

    def generate_prob_measure(self, name: Hashable | None = "P") -> ProbabilityMeasure:
        """Generate a probability measure on the domain of the stochastic process.

        Raises a ValueError if data has not been generated for the stochastic process. Data generation must be implemented in subclasses.

        Parameters
        ----------
        name : Hashable | None, default="P"
            The name of the probability measure.

        Raises
        ------
        ValueError
            If data has not been generated for the stochastic process.
        TypeError
            If `name` is not a hashable type or `None`.

        Returns
        -------
        prob_measure : ProbabilityMeasure
            The generated probability measure.
        """
        if self._data is None:
            raise ValueError(
                "Data must be generated before generating a probability measure."
            )
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("name must be a hashable type or None.")

        if self.is_enumerated:
            return self._generate_exact_prob_measure(name=name)
        else:
            return self._generate_empirical_prob_measure(name=name)

    def _generate_exact_prob_measure(
        self, name: Hashable | None = "P"
    ) -> ProbabilityMeasure:
        raise NotImplementedError(
            "Subclasses must implement the _generate_exact_prob_measure method."
        )

    def _generate_empirical_prob_measure(
        self, name: Hashable | None = "P"
    ) -> ProbabilityMeasure:
        counts_series = self._trajectory_counts
        probabilities = counts_series / sum(counts_series)
        return ProbabilityMeasure(sample_space=self.domain, name=name).from_pandas(
            probabilities
        )

    # --------------------- Representation --------------------- #

    def __repr__(self) -> str:
        """Get the string representation of the random vector.

        Returns
        -------
        repr_str : str
            The string representation of the random vector.
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
        return f"Stochastic process '{self.name}'"
