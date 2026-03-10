"""Base class for stochastic processes.

Classes
-------
StochasticProcess
    A class representing a stochastic process.
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable
from numbers import Real

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from ...core.base.sample_space import SampleSpace
from ...core.base.time import Time
from ...core.probability_measures.probability_measure import ProbabilityMeasure
from ...core.random_objects.random_variable import RandomVariable
from ...core.random_objects.random_vector import RandomVector
from ...core.sigma_algebras.filtration import Filtration
from ..transforms.process_transforms import ProcessTransformMethods


# TODO: Update docstrings
class StochasticProcess(RandomVector, ProcessTransformMethods):
    r"""A class representing a stochastic process.

    The types of stochastic processes modeled by SigAlg are the following: A collection of random variables $X_t$, indexed by a finite set $T$ usually refered to as *time*.

    Parameters
    ----------
    time : Time | None, default=None
        The time index of the stochastic process. If `None`, then the `is_discrete_time` property must be provided and the time index will be generated later through data generation methods.
    is_discrete_time : bool | None, default=None
        Whether the stochastic process is a discrete-time process. If `None`, then `time` parameter must be provided and the `is_discrete_time` attribute will be determined based on the discreteness of the provided time index.
    domain : SampleSpace | None, default=None
        The sample space representing the domain of the stochastic process. If `None`, it will be generated later through data generation methods.
    is_discrete_state : bool | None, default=None
        Whether the stochastic process is a discrete-state process. If `None`, then the `is_discrete_state` attribute will be set later through data generation methods.
    name : Hashable | None, default="X"
        The name of the stochastic process.
    **kwargs
        Additional keyword arguments for subclasses.

    Examples
    --------
    >>> from sigalg.core import SampleSpace, Time
    >>> from sigalg.processes import StochasticProcess
    >>> domain = SampleSpace().from_sequence(size=3, prefix="omega")
    >>> time = Time.discrete(length=2)
    >>> X = StochasticProcess(domain=domain, time=time).from_dict(
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
        time: Time | None = None,
        is_discrete_time: bool | None = None,
        domain: SampleSpace | None = None,
        is_discrete_state: bool | None = None,
        name: Hashable | None = "X",
        **kwargs,
    ) -> None:
        super().__init__(
            domain=domain,
            index=time,
            name=name,
        )

        if time is not None and not isinstance(time, Time):
            raise TypeError("time must be an instance of Time or None.")
        if is_discrete_time is not None and not isinstance(is_discrete_time, bool):
            raise TypeError("is_discrete_time must be a boolean or None.")
        if (
            time is not None
            and is_discrete_time is not None
            and time.is_discrete != is_discrete_time
        ):
            raise ValueError(
                "The is_discrete_time property must be consistent with the discreteness of the provided time index."
            )
        if time is None and is_discrete_time is None:
            raise ValueError(
                "At least one of time or is_discrete_time must be provided."
            )
        if is_discrete_time is None:
            is_discrete_time = time.is_discrete
        self.is_discrete_time = is_discrete_time
        self.is_discrete_state = is_discrete_state

    def from_pandas(self, data: pd.DataFrame) -> StochasticProcess:
        """Later."""
        data_name = data.columns.name if data.columns.name is not None else "time"
        time = Time(data_name=data_name).from_list(
            data.columns.to_list(), is_discrete=self.is_discrete_time
        )
        self = super().from_pandas(data)
        self._index = time
        self._data.columns = time.data
        return self

    def from_numpy(self, array: np.ndarray, initial_time: int = 0) -> StochasticProcess:
        """Later."""
        time = Time().discrete(
            start=initial_time, stop=initial_time + array.shape[1] - 1
        )
        self = super().from_numpy(array)
        self._index = time
        self._data.columns = time.data
        return self

    def from_randint(
        self,
        low: int,
        high: int,
        length: int,
        num_trajectories: int,
        initial_time: int = 0,
        random_state: int | None = None,
    ) -> StochasticProcess:
        """Later."""
        # rng = np.random.default_rng(random_state)
        # arr = rng.integers(low, high, size=(num_trajectories, length + 1))
        # data = pd.DataFrame(arr)
        # return self
        pass

    # TODO: Update docstrings
    # TODO: Write unit tests
    # TODO: Change from class method to instance method
    @classmethod
    def from_time(
        cls,
        domain: SampleSpace,
        time: Time,
        is_discrete_state: bool = True,
        name: Hashable | None = "T",
    ) -> StochasticProcess:
        r"""Define a stochastic process whose trajectories are the time index itself.

        Parameters
        ----------
        domain: SampleSpace
            The sample space representing the domain of the stochastic process.
        time : Time
            The time index to set.
        is_discrete_state : bool, default=True
            Whether the stochastic process is a discrete-state process.
        name : Hashable | None, default="T"
            The name of the stochastic process.

        Returns
        -------
        time_process : StochasticProcess
            A stochastic process whose trajectories are the time index itself.
        """
        if not isinstance(time, Time):
            raise TypeError("time must be an instance of Time.")
        if not isinstance(domain, SampleSpace):
            raise TypeError("domain must be an instance of SampleSpace.")
        if not isinstance(is_discrete_state, bool):
            raise TypeError("is_discrete_state must be a boolean.")
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("name must be hashable or None.")

        time_process = cls(
            time=time,
            is_discrete_time=time.is_discrete,
            domain=domain,
            is_discrete_state=is_discrete_state,
            name=name,
        )
        data = pd.DataFrame(
            {t: [t] * len(domain) for t in time.data}, index=domain.data
        )
        time_process.from_pandas(data)
        time_process._is_enumerated = True
        time_process._probability_measure = ProbabilityMeasure.uniform(
            sample_space=domain
        )
        return time_process

    # --------------------- properties --------------------- #

    # TODO: Update docstrings
    @property
    def time(self) -> Time | None:
        """Get the time index.

        This attribute is an alias for public attribute `index` of the superclass `RandomVector`.

        Returns
        -------
        time : Time | None
            The time index of the stochastic process.
        """
        return self.index

    @time.setter
    def time(self, time: Time) -> None:
        """Set the time index.

        If the time index is changed, any existing data, index, and domain are cleared to ensure consistency.

        Parameters
        ----------
        time : Time
            The time index to set.
        """
        if not isinstance(time, Time):
            raise TypeError("time must be an instance of Time.")

        if self._data is not None:
            self._data = None
            self._index = None
            self._probability_measure = None
            self.domain = None
        self._index = time

    # TODO: Update docstrings
    @property
    def n_trajectories(self) -> int | None:
        """Get the number of trajectories in the stochastic process.

        Returns
        -------
        n_trajectories : int | None
            The number of trajectories in the stochastic process. `None` if data has not been generated.
        """
        return len(self.data) if self.data is not None else None

    # TODO: Update docstrings
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
            except ValueError as e:
                if self.data is None:
                    raise ValueError(
                        "Data must be generated for the stochastic process before accessing the probability measure."
                    ) from e
                else:
                    self._probability_measure = ProbabilityMeasure.uniform(self.domain)
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
        if self.domain is not None and not probability_measure.sample_space.data.equals(
            self.domain.data
        ):
            raise ValueError(
                "The sample space of the probability measure must match the domain of the process."
            )
        self._probability_measure = probability_measure

    # TODO: Update docstrings
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
        if hasattr(self, "_is_enumerated"):
            return self._is_enumerated
        else:
            raise ValueError("The is_enumerated property is not set.")

    # TODO: Update docstrings
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
                t: (
                    self.data.iloc[:, : t + 1].apply(tuple, axis=1)
                    if t != 0
                    else self.data.iloc[:, :1].squeeze()
                )
                for t in range(len(self.time))
            }
        )
        df.columns = self.time.data
        return Filtration(time=self.time).from_pandas(df)

    # TODO: Update docstrings
    @property
    def last_rv(self) -> RandomVariable:
        """Get the random variable corresponding to the last time point.

        Raises
        ------
        ValueError
            If data has not been generated for the stochastic process.

        Returns
        -------
        last_rv : RandomVariable
            The random variable corresponding to the last time point.
        """
        if self._data is None:
            raise ValueError(
                "Data must be generated before accessing the last random variable."
            )
        rounded_time = round(self.time[-1], 2)
        return self.get_component_rv(self.time[-1]).with_name(
            f"{self.name}_{rounded_time}" if self.name is not None else None
        )

    # TODO: Update docstrings
    @property
    def random_variables(self) -> dict[RandomVariable]:
        """Get the dictionary of random variables corresponding to each time point.

        Raises
        ------
        ValueError
            If data has not been generated for the stochastic process.

        Returns
        -------
        random_variables : dict[RandomVariable]
            The dictionary of random variables corresponding to each time point.
        """
        if self._data is None:
            raise ValueError(
                "Data must be generated before accessing the random variables."
            )
        return {t: self.get_component_rv(t) for t in self.time}

    # TODO: Update docstrings
    @property
    def range(self) -> RandomVector:
        """Get the range of the stochastic process.

        Overrides the `range` property of the superclass `RandomVector` to return a `StochasticProcess` instance representing the range of the process, with its own domain and probability measure derived from the trajectories of the original process.
        """
        if self._range is None:
            if isinstance(self.data, pd.Series):
                data = self.data.to_frame().copy()
            else:
                data = self.data.copy()
            cols = data.columns.tolist()
            ones = pd.Series(1, index=self.domain.data, name="count")
            outputs_probs_counts = (
                pd.concat([data, self.probability_measure.data, ones], axis=1)
                .groupby(cols)
                .sum()
            ).reset_index()

            range_name = f"range({self.name})" if isinstance(self.name, str) else None
            range_sample_space = SampleSpace.generate_sequence(
                size=len(outputs_probs_counts),
                prefix=None,
                name=range_name,
                data_name="trajectory",
            )
            outputs_probs_counts.index = range_sample_space.data

            self._range_counts = outputs_probs_counts["count"]
            outputs_probs = outputs_probs_counts.drop(columns=["count"])

            prob_measure_name = f"P_{self.name}" if isinstance(self.name, str) else None
            range_probability_measure = ProbabilityMeasure(
                sample_space=range_sample_space,
                name=prob_measure_name,
            ).from_pandas(outputs_probs["probability"])
            outputs = outputs_probs.drop(columns=["probability"])

            if outputs.shape[1] == 1:
                outputs = outputs.iloc[:, 0].rename(self.name)

            range_name = f"{self.name}_range" if isinstance(self.name, str) else None

            self._range = (
                StochasticProcess(
                    domain=range_sample_space,
                    name=range_name,
                    time=self.time,
                )
                .from_pandas(data=outputs)
                .with_probability_measure(probability_measure=range_probability_measure)
            )
            self._range._is_enumerated = self._is_enumerated

        return self._range

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

    # TODO: Update docstrings
    def from_enumeration(
        self, length: int | None = None, **kwargs
    ) -> StochasticProcess:
        """Generate data by exhaustively enumerating all possible trajectories.

        For this method to be used, a subclass must implement the `_enumeration_logic` method, which defines how to enumerate trajectories for the specific type of stochastic process.

        Parameters
        ----------
        length : int | None, default=None
            The length of each trajectory. If `None`, the length of the existing index is used.
        **kwargs
            Additional keyword arguments for subclasses, which may include parameters needed for the enumeration logic.

        Returns
        -------
        self : StochasticProcess
            The stochastic process with enumerated trajectories.
        """
        if length is not None and (not isinstance(length, int) or length <= 0):
            raise ValueError("If provided, length must be a positive integer.")

        self.domain = None

        self._validate_and_initialize_time(length)
        trajectories = self._enumeration_logic(**kwargs)
        self._validate_and_initialize_domain(len(trajectories))
        self._is_enumerated = True
        return self.from_pandas(trajectories)

    # TODO: Update docstrings
    def from_simulation(
        self,
        n_trajectories: int,
        length: int | None = None,
        random_state: int | None = None,
    ) -> StochasticProcess:
        """Generate data by simulating trajectories.

        For this method to be used, a subclass must implement the `_simulation_logic` method, which defines how to simulate trajectories for the specific type of stochastic process.

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories to simulate.
        length : int | None, default=None
            The length of each trajectory. If `None`, the length of the existing time index is used.
        random_state : int | None, default=None
            An optional random seed for reproducibility.

        Raises
        ------
        ValueError
            If `n_trajectories` is not a positive integer.

        Returns
        -------
        self : StochasticProcess
            The stochastic process with simulated trajectories.
        """
        if not isinstance(n_trajectories, int) or n_trajectories <= 0:
            raise ValueError("n_trajectories must be a positive integer.")
        if length is not None and (not isinstance(length, int) or length <= 0):
            raise ValueError("If provided, length must be a positive integer.")

        self.domain = None

        self._validate_and_initialize_time(length)
        trajectories = self._simulation_logic(
            n_trajectories=n_trajectories, random_state=random_state
        )
        self._validate_and_initialize_domain(n_trajectories)
        self._is_enumerated = False
        return self.from_pandas(trajectories)

    # TODO: Update docstrings
    def from_constant(
        self, value: Real, length: int | None = None
    ) -> StochasticProcess:
        """Create a stochastic process with all trajectories equal to a constant value.

        Parameters
        ----------
        value : Real
            The constant value for all trajectories.
        length : int | None, default=None
            The length of each trajectory. If `None`, the length of the existing time index is used.

        Raises
        ------
        ValueError
            If `length` is not a positive integer or if the domain is not initialized.
        TypeError
            If `value` is not a real number.

        Returns
        -------
        self : StochasticProcess
            The stochastic process with constant trajectories.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, Time
        >>> from sigalg.processes import StochasticProcess
        >>> Omega = SampleSpace().from_sequence(size=2)
        >>> T = Time().discrete(length=3)
        >>> X = StochasticProcess(domain=Omega, time=T).from_constant(2)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time    0  1  2  3
        sample
        0       2  2  2  2
        1       2  2  2  2
        """
        if length is not None and (not isinstance(length, int) or length <= 0):
            raise ValueError("If provided, length must be a positive integer.")
        if self.domain is None:
            raise ValueError(
                "Domain must be initialized before creating a constant process."
            )
        if not isinstance(value, Real):
            raise TypeError("Value must be a real number.")

        self._validate_and_initialize_time(length)
        data = dict.fromkeys(self.domain, len(self.time) * [value])
        trajectories = pd.DataFrame.from_dict(data, orient="index")
        trajectories.columns = self.time.data
        self.from_pandas(trajectories)
        self._is_enumerated = True
        self._probability_measure = ProbabilityMeasure.uniform(sample_space=self.domain)
        return self

    def _enumeration_logic(self, **kwargs) -> pd.DataFrame:
        """Abstract method for enumeration logic.

        This method must be implemented in subclasses to define how to enumerate trajectories.

        Parameters
        ----------
        **kwargs
            Keyword arguments for subclasses, which includes parameters needed for the enumeration logic.

        Returns
        -------
        trajectories : pd.DataFrame
            A DataFrame containing the enumerated trajectories as rows and time points as columns.
        """
        raise NotImplementedError("Not implemented.")

    def _simulation_logic(
        self, n_trajectories: int, random_state: int | None
    ) -> pd.DataFrame:
        """Abstract method for simulation logic.

        This method must be implemented in subclasses to define how to simulate trajectories.

        Parameters
        ----------
        n_trajectories : int
            The maximum number of trajectories to simulate.
        random_state : int | None
            An optional random seed for reproducibility.

        Returns
        -------
        trajectories : pd.DataFrame
            A DataFrame containing the simulated trajectories as rows and time points as columns.
        """
        raise NotImplementedError("Not implemented.")

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
            raise ValueError("Either time index or length must be provided.")
        if self.time is None and not self.is_discrete_time:
            raise ValueError(
                "Time index must be provided for a non-discrete-time process."
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
        return ProbabilityMeasure.uniform(sample_space=self.domain, name=name)

    # --------------------- martingale methods --------------------- #

    # TODO: Update docstrings
    def is_martingale(
        self,
        filtration: Filtration | None = None,
        rtol: float = 1e-05,
        atol: float = 1e-08,
    ) -> bool:
        """Check if the stochastic process is a martingale (with respect to an optional filtration).

        Take care when using this method for non-enumerated processes, as the check may be inaccurate due to probabilities being approximated. Adjusting the `rtol` and `atol` parameters may help in such cases. Also, even for enumerated processes, this method is very computationally intensive, as it requires calculating conditional expectations at each time step.

        Parameters
        ----------
        filtration : Filtration | None, default=None
            The filtration with respect to which the martingale property is checked. If None, the natural filtration of the process is used.
        rtol : float, default=1e-05
            The relative tolerance parameter for numerical comparison. Internally passed to `numpy.allclose`.
        atol : float, default=1e-08
            The absolute tolerance parameter for numerical comparison. Internally passed to `numpy.allclose`.

        Raises
        ------
        ValueError
            If data has not been generated for the stochastic process, or if the process is not discrete-state.
        TypeError
            If the provided filtration is not an instance of Filtration, or its sample space does not match the domain of the process, or its time index does not match the time index of the process.

        Returns
        -------
        is_martingale : bool
            `True` if the stochastic process is a martingale, `False` otherwise.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(start=1, length=2)
        >>> # Symmetric random walks are martingales
        >>> X = RandomWalk(p=0.5, time=T).from_enumeration()
        >>> print(X.is_martingale())
        True
        >>> # Non-symmetric random walks are not martingales
        >>> Y = RandomWalk(p=0.7, time=T).from_enumeration()
        >>> print(Y.is_martingale())
        False
        """
        if self.data is None:
            raise ValueError(
                "Data must be generated before checking martingale property."
            )
        if not self.is_discrete_state:
            raise ValueError(
                "Martingale check is only implemented for discrete-state processes."
            )
        if not self.is_enumerated:
            warnings.warn(
                "The process is not enumerated. The martingale check may be inaccurate. Adjusting tolerances may help.",
                UserWarning,
                stacklevel=2,
            )
        if filtration is not None:
            if not isinstance(filtration, Filtration):
                raise TypeError(
                    "If filtration is provided, it must be an instance of Filtration."
                )
            if filtration.sample_space != self.domain:
                raise TypeError(
                    "If filtration is provided, its sample space must match the domain of the process."
                )
            if filtration.time != self.time:
                raise TypeError(
                    "If filtration is provided, its time index must match the time index of the process."
                )

        if filtration is None:
            filtration = self.natural_filtration

        for t_prev, t_curr in zip(self.time[:-1], self.time[1:], strict=False):
            df = pd.DataFrame(
                {
                    "atom ID": filtration[t_prev].data,
                    "rv": self[t_curr].data,
                    "probability": self.probability_measure.data,
                }
            )
            weighted_sum = (
                (df["probability"] * df["rv"]).groupby(df["atom ID"]).transform("sum")
            )
            group_probs = df["probability"].groupby(df["atom ID"]).transform("sum")
            expectation = weighted_sum / group_probs
            if not np.allclose(expectation, self[t_prev].data, rtol=rtol, atol=atol):
                return False

        return True

    # TODO: Update docstrings
    def is_submartingale(
        self,
        filtration: Filtration | None = None,
        rtol: float = 1e-05,
        atol: float = 1e-08,
    ) -> bool:
        """Check if the stochastic process is a submartingale (with respect to an optional filtration).

        Parameters
        ----------
        filtration : Filtration | None, default=None
            The filtration with respect to which the submartingale property is checked. If None, the natural filtration of the process is used.
        rtol : float, default=1e-05
            The relative tolerance parameter for numerical comparison. Internally passed to `numpy.allclose`.
        atol : float, default=1e-08
            The absolute tolerance parameter for numerical comparison. Internally passed to `numpy.allclose`.

        Raises
        ------
        ValueError
            If data has not been generated for the stochastic process, or if the process is not discrete-state.
        TypeError
            If the provided filtration is not an instance of Filtration, or its sample space does not match the domain of the process, or its time index does not match the time index of the process.

        Returns
        -------
        is_submartingale : bool
            `True` if the stochastic process is a submartingale, `False` otherwise.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(start=1, length=2)
        >>> # A random walk with upward drift is a submartingale
        >>> X = RandomWalk(p=0.6, time=T).from_enumeration()
        >>> print(X.is_submartingale())
        True
        """
        if self.data is None:
            raise ValueError(
                "Data must be generated before checking submartingale property."
            )
        if filtration is not None:
            if not isinstance(filtration, Filtration):
                raise TypeError(
                    "If filtration is provided, it must be an instance of Filtration."
                )
            if filtration.sample_space != self.domain:
                raise TypeError(
                    "If filtration is provided, its sample space must match the domain of the process."
                )
            if filtration.time != self.time:
                raise TypeError(
                    "If filtration is provided, its time index must match the time index of the process."
                )

        if filtration is None:
            filtration = self.natural_filtration

        for t_prev, t_curr in zip(self.time[:-1], self.time[1:], strict=False):
            df = pd.DataFrame(
                {
                    "atom ID": filtration[t_prev].data,
                    "rv": self[t_curr].data,
                    "probability": self.probability_measure.data,
                }
            )
            weighted_sum = (
                (df["probability"] * df["rv"]).groupby(df["atom ID"]).transform("sum")
            )
            group_probs = df["probability"].groupby(df["atom ID"]).transform("sum")
            expectation = weighted_sum / group_probs
            if np.allclose(expectation, self[t_prev].data, rtol=rtol, atol=atol):
                continue
            if all(expectation > self[t_prev].data):
                continue
            else:
                return False

        return True

    # TODO: Update docstrings
    def is_supermartingale(
        self,
        filtration: Filtration | None = None,
        rtol: float = 1e-05,
        atol: float = 1e-08,
    ) -> bool:
        """Check if the stochastic process is a supermartingale (with respect to an optional filtration).

        Parameters
        ----------
        filtration : Filtration | None, default=None
            The filtration with respect to which the supermartingale property is checked. If None, the natural filtration of the process is used.
        rtol : float, default=1e-05
            The relative tolerance parameter for numerical comparison. Internally passed to `numpy.allclose`.
        atol : float, default=1e-08
            The absolute tolerance parameter for numerical comparison. Internally passed to `numpy.allclose`.

        Raises
        ------
        ValueError
            If data has not been generated for the stochastic process, or if the process is not discrete-state.
        TypeError
            If the provided filtration is not an instance of Filtration, or its sample space does not match the domain of the process, or its time index does not match the time index of the process.

        Returns
        -------
        is_supermartingale : bool
            True if the stochastic process is a supermartingale, False otherwise.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(start=1, length=2)
        >>> # A random walk with downward drift is a supermartingale
        >>> X = RandomWalk(p=0.4, time=T).from_enumeration()
        >>> print(X.is_supermartingale())
        True
        """
        if self.data is None:
            raise ValueError(
                "Data must be generated before checking supermartingale property."
            )
        if not self.is_discrete_state:
            raise ValueError(
                "Supermartingale check is only implemented for discrete-state processes."
            )
        if not self.is_enumerated:
            warnings.warn(
                "The process is not enumerated. The supermartingale check may be inaccurate.",
                UserWarning,
                stacklevel=2,
            )
        if filtration is not None:
            if not isinstance(filtration, Filtration):
                raise TypeError(
                    "If filtration is provided, it must be an instance of Filtration."
                )
            if filtration.sample_space != self.domain:
                raise TypeError(
                    "If filtration is provided, its sample space must match the domain of the process."
                )
            if filtration.time != self.time:
                raise TypeError(
                    "If filtration is provided, its time index must match the time index of the process."
                )

        if filtration is None:
            filtration = self.natural_filtration

        for t_prev, t_curr in zip(self.time[:-1], self.time[1:], strict=False):
            df = pd.DataFrame(
                {
                    "atom ID": filtration[t_prev].data,
                    "rv": self[t_curr].data,
                    "probability": self.probability_measure.data,
                }
            )
            weighted_sum = (
                (df["probability"] * df["rv"]).groupby(df["atom ID"]).transform("sum")
            )
            group_probs = df["probability"].groupby(df["atom ID"]).transform("sum")
            expectation = weighted_sum / group_probs
            if np.allclose(expectation, self[t_prev].data, rtol=rtol, atol=atol):
                continue
            if all(expectation < self[t_prev].data):
                continue
            else:
                return False

        return True

    # TODO: Update docstrings
    def is_adapted(self, filtration: Filtration):
        """Check if the stochastic process is adapted to a given filtration.

        Parameters
        ----------
        filtration : Filtration
            The filtration to check adaptation against.

        Raises
        ------
        ValueError
            If data has not been generated for the stochastic process.
        TypeError
            If the provided filtration is not an instance of Filtration, or its sample space does not match the domain of the process, or its time index does not match the time index of the process.

        Returns
        -------
        is_adapted : bool
            True if the stochastic process is adapted to the given filtration, False otherwise.

        Examples
        --------
        >>> from sigalg.core import RandomVariable, Time
        >>> from sigalg.processes import RandomWalk, StochasticProcess
        >>> T = Time.discrete(start=0, stop=2)
        >>> X = RandomWalk(p=0.7, time=T).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        0  1  2
        trajectory
        0           0 -1 -2
        1           0 -1  0
        2           0  1  0
        3           0  1  2
        >>> def f0(X: StochasticProcess) -> RandomVariable:
        ...     return X[0]
        >>> def f1(X: StochasticProcess) -> RandomVariable:
        ...     return 2 * X[0] + X[1]
        >>> def f2(X: StochasticProcess) -> RandomVariable:
        ...     return X[2] - X[1] + X[0]
        >>> Y = X.transform(functions=[f0, f1, f2], name="Y")
        >>> print(Y) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'Y':
        time        0  1  2
        trajectory
        0           0 -1 -1
        1           0 -1  1
        2           0  1 -1
        3           0  1  1
        >>> print(Y.is_adapted(filtration=X.natural_filtration))
        True
        """
        if self.data is None:
            raise ValueError("Data must be generated before checking adaptation.")
        if filtration is not None:
            if not isinstance(filtration, Filtration):
                raise TypeError(
                    "If filtration is provided, it must be an instance of Filtration."
                )
            if filtration.sample_space != self.domain:
                raise TypeError(
                    "If filtration is provided, its sample space must match the domain of the process."
                )
            if filtration.time != self.time:
                raise TypeError(
                    "If filtration is provided, its time index must match the time index of the process."
                )

        for t in self.time:
            if self[t].is_measurable(filtration[t]):
                continue
            else:
                return False

        return True

    # TODO: Update docstrings
    def is_predictable(self, filtration: Filtration):
        """Check if the stochastic process is predictable with respect to a given filtration.

        The time index of `self` must match all but the first time indices of the filtration.

        Parameters
        ----------
        filtration : Filtration
            The filtration to check predictability against.

        Raises
        ------
        ValueError
            If data has not been generated for the stochastic process.
        TypeError
            If the provided filtration is not an instance of `Filtration`, or its sample space does not match the domain of the process, or if the time indices of the process do not match all but the first time indices of the filtration.

        Returns
        -------
        is_predictable : bool
            `True` if the stochastic process is predictable with respect to the given filtration, `False` otherwise.

        Examples
        --------
        >>> from sigalg.core import RandomVariable, Time
        >>> from sigalg.processes import RandomWalk, StochasticProcess
        >>> T = Time.discrete(start=0, stop=3)
        >>> X = RandomWalk(p=0.7, time=T).from_enumeration()
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time        0  1  2  3
        trajectory
        0           0 -1 -2 -3
        1           0 -1 -2 -1
        2           0 -1  0 -1
        3           0 -1  0  1
        4           0  1  0 -1
        5           0  1  0  1
        6           0  1  2  1
        7           0  1  2  3
        >>> def f1(X: StochasticProcess) -> RandomVariable:
        ...     return 2 * X[0]
        >>> def f2(X: StochasticProcess) -> RandomVariable:
        ...     return X[1] + X[0]
        >>> def f3(X: StochasticProcess) -> RandomVariable:
        ...     return X[2] - 5 * X[1]
        >>> S = Time.discrete(start=1, stop=3)
        >>> Y = X.transform(functions=[f1, f2, f3], time=S, name="Y")
        >>> print(Y) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'Y':
        time        1  2  3
        trajectory
        0           0 -1  3
        1           0 -1  3
        2           0 -1  5
        3           0 -1  5
        4           0  1 -5
        5           0  1 -5
        6           0  1 -3
        7           0  1 -3
        >>> print(Y.is_predictable(filtration=X.natural_filtration))
        True
        """
        if self.data is None:
            raise ValueError("Data must be generated before checking predictability.")
        if filtration is not None:
            if not isinstance(filtration, Filtration):
                raise TypeError(
                    "If filtration is provided, it must be an instance of Filtration."
                )
            if filtration.sample_space != self.domain:
                raise TypeError(
                    "If filtration is provided, its sample space must match the domain of the process."
                )
            if len(filtration.time) == len(self.time) + 1:
                if not np.all(filtration.time.data[1:] == self.time.data):
                    raise TypeError(
                        "The time indices of self must match all but the first time indices of the filtration."
                    )
            elif len(filtration.time) == len(self.time):
                if not np.all(filtration.time.data[1:] == self.time.data[1:]):
                    raise TypeError(
                        "The time indices of self must match all but the first time indices of the filtration."
                    )
            else:
                raise TypeError(
                    "The time indices of self must match all but the first time indices of the filtration."
                )

        for t in self.time[1:]:
            if self[t].is_measurable(filtration[t - 1]):
                continue
            else:
                return False

        return True

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

    # TODO: Update docstrings
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

    def __iter__(self):
        """Iterate over the component random variables of the stochastic process."""
        for t in self.time:
            yield self[t]

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

    # TODO: Update docstrings
    def print_trajectories_and_probabilities(self):
        """Print the trajectories and their corresponding probabilities."""
        if self._data is None:
            raise ValueError(
                "Data must be generated before printing trajectories and probabilities."
            )

        trajectories_and_probs = pd.concat(
            [self.data, self.probability_measure.data], axis=1
        )
        print(trajectories_and_probs)

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

    # TODO: Update docstrings
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
        TypeError
            If ax is not a matplotlib Axes object.

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
            raise TypeError("ax must be a matplotlib Axes object")

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
