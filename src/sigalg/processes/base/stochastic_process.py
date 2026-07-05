"""Base class for stochastic processes."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from ...core.random_objects.random_vector import RandomVector
from ..transforms.process_transforms import ProcessTransformMethods

if TYPE_CHECKING:
    from ...core.base.index import Index
    from ...core.base.sample_space import SampleSpace
    from ...core.probability_measures.probability_measure import ProbabilityMeasure
    from ...core.random_objects.random_variable import RandomVariable
    from ...core.sigma_algebras.filtration import Filtration
    from ...core.sigma_algebras.sigma_algebra import SigmaAlgebra


class StochasticProcess(RandomVector, ProcessTransformMethods):
    """Base class for stochastic processes.

    The constructor is not intended for direct usage. Instead, user's should call one of either class methods `from_enumeration` or `from_simulation` on a subclass. See the Examples section below.

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
    Exhaustively generate all trajectories of length 2 from an independent and identically distributed (IID) process by calling the `from_enumeration` class method of the class `IIDProcess`.

    >>> from scipy.stats import bernoulli
    >>> from sigalg.processes import IIDProcess, RandomWalk
    >>> X = IIDProcess.from_enumeration(
    ...     distribution=bernoulli(p=0.25),
    ...     support=[0, 1],
    ...     length=2,
    ... )

    Print all trajectories.

    >>> print(X) # doctest: +NORMALIZE_WHITESPACE
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

    Print the underlying probability space, showing the probability associated with each trajectory.

    >>> print(X.prob_space)  # doctest: +NORMALIZE_WHITESPACE
    Probability space (Omega, power_set, P)
    =======================================
    <BLANKLINE>
    * Sample space 'Omega':
     sample
          0
          1
          2
          3
          4
          5
          6
          7
    <BLANKLINE>
    * Sigma algebra 'power_set':
            atom_ID
    sample
    0             0
    1             1
    2             2
    3             3
    4             4
    5             5
    6             6
    7             7
    <BLANKLINE>
    * Probability measure 'P':
            probability
    sample
    0          0.421875
    1          0.140625
    2          0.140625
    3          0.046875
    4          0.140625
    5          0.046875
    6          0.046875
    7          0.015625

    Simulate ten trajectories of length 2 from a random walk stochastic process by calling the `from_simulation` class method of the class `RandomWalk`.

    >>> Y = RandomWalk.from_simulation(
    ...     p=0.75,
    ...     initial_state=2,
    ...     length=2,
    ...     n_trajectories=10,
    ...     random_state=42,
    ...     name="Y",
    ... )

    Print the simulated trajectories.

    >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
    Random walk 'Y':
    time    0  1  2
    sample
    0       2  1  2
    1       2  1  2
    2       2  3  2
    3       2  1  0
    4       2  3  4
    5       2  3  2
    6       2  3  2
    7       2  3  4
    8       2  3  4
    9       2  1  2

    Print the range of the random walk process, which (in the present) case yields only four unique trajectories. Notice the probability measure on the range reflects these different counts of trajectories.

    >>> print(Y.range)  # doctest: +NORMALIZE_WHITESPACE
    Probability space (Y_range, power_set, P_Y)
    ===========================================
    <BLANKLINE>
    * Sample space 'Y_range':
     Y_0  Y_1  Y_2
       2    1    0
       2    1    2
       2    3    2
       2    3    4
    <BLANKLINE>
    * Sigma algebra 'power_set':
                atom_ID
    Y_0 Y_1 Y_2
    2   1   0    (2, 1, 0)
            2    (2, 1, 2)
        3   2    (2, 3, 2)
            4    (2, 3, 4)
    <BLANKLINE>
    * Probability measure 'P_Y':
                probability
    Y_0 Y_1 Y_2
    2   1   0            0.1
            2            0.3
        3   2            0.3
            4            0.3

    """

    _properties = RandomVector._properties + [
        "_n_trajectories",
        "_natural_filtration",
        "_last_rv",
        "_random_state",
    ]
    _repr_name = "Stochastic process"

    # --------------------- constructors --------------------- #

    @classmethod
    def from_time(
        cls,
        sample_space: SampleSpace | None = None,
        sig_alg: SigmaAlgebra | None = None,
        prob_measure: ProbabilityMeasure | None = None,
        index: Index | None = None,
        name: Hashable = "X",
    ) -> StochasticProcess:
        """Define a stochastic process whose trajectories are the time index itself.

        Parameters
        ----------
        sample_space : SampleSpace | None, default=None
            The sample space of the underlying probability space.
        sig_alg : SigmaAlgebra | None, default=None
            The sigma algebra of the underlying probability space.
        prob_measure : ProbabilityMeasure | None, default=None
            The probability measure of the underlying probability space.
        index : Index | None, default=None
            The index of the stochastic process.
        name : Hashable, default="X"
            The name of the stochastic process.

        Raises
        ------
        TypeError
            If `sample_space` is not an instance of `SampleSpace` (if given), or if `sig_alg` is not an instance of `SigmaAlgebra` (if given).
        ValueError
            If both `sample_space` and `sig_alg` are `None`, or if both are given and the sample space of `sig_alg` does not match `sample_space`.

        Returns
        -------
        self : StochasticProcess
            A stochastic process whose trajectories are the time index itself.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, Time
        >>> from sigalg.processes import StochasticProcess
        >>> Omega = SampleSpace.from_sequence(size=2)
        >>> T = Time.discrete(length=3)
        >>> X = StochasticProcess.from_time(sample_space=Omega, index=T)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time    0  1  2  3
        sample
        0       0  1  2  3
        1       0  1  2  3
        """
        from ...core.base.sample_space import SampleSpace
        from ...core.sigma_algebras.sigma_algebra import SigmaAlgebra

        if sample_space is not None and not isinstance(sample_space, SampleSpace):
            raise TypeError(
                "sample space must be an instance of SampleSpace, if given."
            )
        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be an instance of SigmaAlgebra, if given.")
        if sample_space is None and sig_alg is None:
            raise ValueError("One of sample_space or sig_alg must be given.")
        if (sample_space is not None and sig_alg is not None) and (
            sig_alg.sample_space != sample_space
        ):
            raise ValueError(
                "The sample space of the given sigma-algebra does not match the given sample space."
            )

        if sample_space is None:
            sample_space = sig_alg.sample_space

        mapping = pd.DataFrame(
            {t: [t] * len(sample_space) for t in index},
            index=sample_space.data,
        )
        mapping.columns = index.data

        return cls(
            sample_space=sample_space,
            sig_alg=sig_alg,
            prob_measure=prob_measure,
            mapping=mapping,
            name=name,
        )

    # --------------------- enumeration methods --------------------- #

    @classmethod
    def from_enumeration(
        cls,
        index: Index | None = None,
        length: int | None = None,
        name: Hashable = "X",
    ) -> StochasticProcess:
        """Abstract method to be overriden by subclasses that implement data generation by exhaustive enumeration.

        The signature of the overriden method should include all parameters necessary for the subclass to generate trajectories, along with the `index` and `length` parameters listed in the signature of the current abstract method.

        The overriden method should do the following, in order:

        1. Validate all input parameters, besides `index` and `length`.
        2. Include the line `index = cls._validate_and_return_index(index=index, length=length)` to check that `index` and `length` are compatible and return the correct index.
        3. Include the line `process = cls(index=index, name=name)` to generate an empty process.
        4. Populate any subclass-specific attributes of `process`.
        5. Return with the line `return process._enumeration_logic()`.

        The implementation of `from_enumeration` in the subclass `IIDProcess` is a good template to review to help understand the above steps.

        Parameters
        ----------
        index : Index | None, default=None
            The index of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        length : int | None, default=None
            The length of the trajectories of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        name : Hashable | None, default="X"
            The name of the stochastic process.

        Returns
        -------
        self : StochasticProcess
            This method should ultimately return `self`, which can be guaranteed by returning with the line `return process._enumeration_logic()` as mentioned above.
        """
        raise NotImplementedError("Not implemented.")

    def _enumeration_logic(self) -> StochasticProcess:
        """Calls the subclass-specific overriden methods `_enumeration_hook` and `_generate_exact_prob_measure` to generate trajectories and the underlying probability space, and returns `self`.

        This method is not meant to be overriden by subclasses. Leave it as is.
        """  # noqa: D401
        from ...core.base.sample_space import SampleSpace
        from ...validation.mapping_validator import MappingValidator

        mapping = self._enumeration_hook()

        v = MappingValidator(
            mapping=mapping,
            output_name=self.name,
            index=self.index,
            name=self.name,
        )

        self._data = v.data
        self._index = v.index
        self._name = v.name
        sample_space = SampleSpace.from_domain(v.domain)
        sample_space.name = "Omega"
        sample_space.variable_names = ["sample"]
        self.sample_space = sample_space
        self.prob_measure = self._generate_exact_prob_measure()

        return self

    def _enumeration_hook(self) -> pd.DataFrame:
        """Abstract hook for enumeration logic.

        This method must be implemented in subclasses to define how to enumerate trajectories. It will use any subclass-specific attributes of `self` to generate a `pd.DataFrame` whose rows are trajectories of the stochastic process.

        The implementation of `_enumeration_hook` in the subclass `IIDProcess` is a good template to review to help understand the needed steps.

        Returns
        -------
        trajectories : pd.DataFrame
            A data frame containing the trajectories of the stochastic process.
        """
        raise NotImplementedError("Not implemented.")

    def _generate_exact_prob_measure(self) -> ProbabilityMeasure:
        """Generate the exact probability measure for an enumerated stochastic process.

        Subclasses that support enumeration should implement this method to generate the exact probability measure based on the enumerated trajectories.

        The implementation of `_generate_exact_prob_measure` in the subclass `IIDProcess` is a good template to review to help understand the needed steps.

        Returns
        -------
        prob_measure : ProbabilityMeasure
            The exact probability measure for the enumerated stochastic process.
        """
        raise NotImplementedError(
            "Method to generate exact probability measure not implemented."
        )

    # --------------------- simulation methods --------------------- #

    @classmethod
    def from_simulation(
        self,
        n_trajectories: int,
        index: Index | None,
        length: int | None,
        random_state: int | np.random.Generator | None = None,
        name: Hashable = "X",
    ) -> StochasticProcess:
        """Abstract method to be overriden by subclasses that implement data generation by Monte Carlo simulation.

        The signature of the overriden method should include all parameters necessary for the subclass to generate trajectories, along with the parameters listed in the signature of the current abstract method.

        The overriden method should do the following, in order:

        1. Validate all input parameters, besides `index` and `length`.
        2. Include the line `index = cls._validate_and_return_index(index=index, length=length)` to check that `index` and `length` are compatible and return the correct index.
        3. Include the line `process = cls(index=index, name=name)` to generate an empty process.
        4. Include the lines `process.n_trajectories = n_trajectories` and `process.random_state = random_state` to set these attributes.
        5. Populate any further subclass-specific attributes of `process`
        6. Return with the line `return process._simulation_logic()`.

        The implementation of `from_simulation` in the subclass `IIDProcess` is a good template to review to help understand the above steps.

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories to simulate.
        index : Index | None, default=None
            The index of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        length : int | None, default=None
            The length of the trajectories of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        random_state : int | np.random.Generator | None, default=None
            An optional seed (`int`) for the random number generator, or a `np.random.Generator` instance to use directly. If an integer is provided, a new generator is created with that seed. If a `Generator` is provided, it is used directly and its state is advanced. If `None`, the random number generator is not seeded.
        name : Hashable | None, default="X"
            The name of the stochastic process.

        Returns
        -------
        self : StochasticProcess
            This method should ultimately return `self`, which can be guaranteed by returning with the line `return process._simulation_logic()` as mentioned above.
        """
        raise NotImplementedError("Not implemented.")

    def _simulation_logic(self) -> StochasticProcess:
        """Calls the subclass-specific overriden methods `_simulation_hook` to generate trajectories and the underlying probability space, and returns `self`.

        This method is not meant to be overriden by subclasses. Leave it as is.
        """  # noqa: D401
        from ...core.base.sample_space import SampleSpace
        from ...validation.mapping_validator import MappingValidator

        mapping = self._simulation_hook()

        v = MappingValidator(
            mapping=mapping,
            output_name=self.name,
            index=self.index,
            name=self.name,
        )

        self._data = v.data
        self._index = v.index
        self._name = v.name
        sample_space = SampleSpace.from_domain(v.domain)
        sample_space.name = "Omega"
        sample_space.variable_names = ["sample"]
        self.sample_space = sample_space
        self.prob_measure.name = "P"

        return self

    def _simulation_hook(self) -> pd.DataFrame:
        """Abstract hook for simulation logic.

        This method must be implemented in subclasses to define how to simulation trajectories. It will use any subclass-specific attributes of `self` to generate a `pd.DataFrame` whose rows are trajectories of the stochastic process.

        The implementation of `_simulation_hook` in the subclass `IIDProcess` is a good template to review to help understand the needed steps.

        Returns
        -------
        trajectories : pd.DataFrame
            A data frame containing the trajectories of the stochastic process.
        """
        raise NotImplementedError("Not implemented.")

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_and_return_index(
        index: Index | None = None,
        length: int | None = None,
    ) -> Index:
        from ...core.base.index import Index
        from ...core.base.time import Time

        if length is not None:
            if not isinstance(length, int):
                raise TypeError("If given, length must be an integer.")
            if length <= 0:
                raise ValueError("If given, length must be positive.")
        if index is not None and not isinstance(index, Index):
            raise TypeError("If given, index must be an instance of index.")
        if length is None and index is None:
            raise ValueError("One or the other of length or index must be given.")
        if (length is not None and index is not None) and (length != len(index)):
            raise ValueError(
                "If both length and index are given, the lengths must be consistent."
            )

        if index is None:
            return Time.discrete(length=length)
        else:
            return index

    @staticmethod
    def _validate_simulation_parameters_and_return_random_state(
        n_trajectories: int,
        random_state: int | np.random.Generator | None = None,
    ) -> np.random.Generator:
        if not isinstance(n_trajectories, int):
            raise TypeError("n_trajectories must be an integer.")
        if n_trajectories <= 0:
            raise ValueError("n_trajectories must be positive.")
        if random_state is not None and not isinstance(
            random_state, (int, np.random.Generator)
        ):
            raise TypeError(
                "random_state must be an integer, np.random.Generator, or None."
            )

        return (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

    # --------------------- properties --------------------- #

    @property
    def is_discrete_time(self) -> bool | None:
        """Whether the stochastic process is a discrete-time process.

        Returns
        -------
        is_discrete_time : bool | None
            `True` if the stochastic process is discrete-time, `False` if it is continuous-time, and `None` if the time index is not set.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import StochasticProcess
        >>> T = Time.discrete(length=3)
        >>> X = StochasticProcess(index=T)
        >>> print(X.is_discrete_time)
        True
        >>> S = Time.continuous(start=0, stop=1, dt=0.25)
        >>> Y = StochasticProcess(index=S, name="Y")
        >>> print(Y.is_discrete_time)
        False
        """
        return self.time.is_discrete if self.time is not None else None

    @property
    def time(self) -> Index | None:
        """Get the time index.

        This attribute is an alias for the public attribute `index` of the superclass `RandomVector`.

        Returns
        -------
        time : Time | None
            The time index of the stochastic process.
        """
        return self.index

    @property
    def n_trajectories(self) -> int | None:
        """Get the number of trajectories in the stochastic process.

        The n_trajectories property is settable, but this functionality is only intended for internal use by subclasses.

        Returns
        -------
        n_trajectories : int | None
            The number of trajectories in the stochastic process. `None` if data has not been generated.
        """
        if self.data is not None:
            self._n_trajectories = len(self.data)

        return self._n_trajectories

    @n_trajectories.setter
    def n_trajectories(self, num: int) -> None:
        """Set the number of trajectories of the stochstic process.

        Intended for internal use only.
        """
        if not isinstance(num, int):
            return TypeError("n_trajectories must be an integer.")
        if num <= 0:
            return ValueError("n_trajectories must be positive.")

        self._n_trajectories = num

    @property
    def random_state(self) -> int | np.random.Generator | None:
        """Get the random state of the stochastic process.

        The random_process property is settable, but this functionality is only intended for internal use by subclasses.

        Returns
        -------
        random_state : int | np.random.Generator | None
            The random state of the stochstic process.
        """
        return self._random_state

    @random_state.setter
    def random_state(self, state: int | np.random.Generator | None) -> None:
        """Set the random state of the stochastic process.

        Intended for internal use only.
        """
        if state is not None and not isinstance(state, (int, np.random.Generator)):
            raise TypeError(
                "random_state must be an integer, np.random.Generator, or None."
            )

        self._random_state = (
            state
            if isinstance(state, np.random.Generator)
            else np.random.default_rng(state)
        )

    @property
    def natural_filtration(self) -> Filtration | None:
        r"""Get the natural filtration of the stochastic process.

        See the Notes section below for the mathematical details.

        Returns
        -------
        natural_filtration : Filtration | None
            The natural filtration of the stochastic process.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, Time
        >>> from sigalg.processes import StochasticProcess
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> T = Time.discrete(length=3)
        >>> X = StochasticProcess.from_randint(
        ...     sample_space=Omega,
        ...     low=0,
        ...     high=2,
        ...     random_state=42,
        ...     index=T,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time    0  1  2  3
        sample
        0       0  1  1  0
        1       0  1  0  1
        2       0  0  1  1
        >>> print(X.natural_filtration)  # doctest: +NORMALIZE_WHITESPACE
        Filtration 'F'
        ==============
        <BLANKLINE>
        * Time 'T':
         time
            0
            1
            2
            3
        <BLANKLINE>
        * At index 0:
        Sigma algebra 'F_0':
                atom_ID
        sample
        0             0
        1             0
        2             0
        <BLANKLINE>
        * At index 1:
        Sigma algebra 'F_1':
            atom_ID
        sample
        0       (0, 1)
        1       (0, 1)
        2       (0, 0)
        <BLANKLINE>
        * At index 2:
        Sigma algebra 'F_2':
                atom_ID
        sample
        0       (0, 1, 1)
        1       (0, 1, 0)
        2       (0, 0, 1)
        <BLANKLINE>
        * At index 3:
        Sigma algebra 'F_3':
                    atom_ID
        sample
        0       (0, 1, 1, 0)
        1       (0, 1, 0, 1)
        2       (0, 0, 1, 1)

        Notes
        -----
        Given a stochastic process $X_t$ indexed by $T$, the natural filtration is defined as the collection of $\sigma$-algebras $\mathcal{F}_t$ where $\mathcal{F}_t = \sigma(X_s : s \leq t)$ for each $t \in T$.
        """
        from ...core.sigma_algebras.filtration import Filtration

        if self._natural_filtration is None and self.data is not None:
            data = pd.DataFrame(
                {
                    t: (
                        self.data.loc[:, :t].apply(tuple, axis=1)
                        if t != self.time.data[0]
                        else self.data.loc[:, self.time.data[0]].squeeze()
                    )
                    for t in self.time
                },
                columns=self.time.data,
            )
            self._natural_filtration = Filtration(sig_algs=data, index=self.time)

        return self._natural_filtration

    @property
    def last_rv(self) -> RandomVariable:
        """Get the random variable corresponding to the last time point.

        Returns
        -------
        last_rv : RandomVariable
            The random variable corresponding to the last time point.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, Time
        >>> from sigalg.processes import StochasticProcess
        >>> Omega = SampleSpace.from_sequence(size=4)
        >>> T = Time.discrete(length=3)
        >>> X = StochasticProcess.from_randint(
        ...     sample_space=Omega,
        ...     index=T,
        ...     low=0,
        ...     high=6,
        ...     random_state=42,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time    0  1  2  3
        sample
        0       0  4  3  2
        1       2  5  0  4
        2       1  0  3  5
        3       4  4  4  4
        >>> print(X.last_rv)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_3':
                X_3
        sample
        0         2
        1         4
        2         5
        3         4
        """
        if self._last_rv is None and self.data is not None:
            name = f"{self.name}_{self.time[-1]}".replace(".", "_")
            self._last_rv = self.components[-1].with_name(name)

        return self._last_rv

    # --------------------- methods --------------------- #

    def __len__(self) -> int:
        """Get the length of the stochastic process, defined as the number of time points.

        Returns
        -------
        length : int
            The length of the stochastic process.
        """
        return len(self.time) if self.time is not None else None

    # --------------------- martingale methods --------------------- #

    def is_martingale(
        self,
        filtration: Filtration | None = None,
        prob_measure: ProbabilityMeasure | None = None,
        rtol: float = 1e-05,
        atol: float = 1e-08,
    ) -> bool:
        r"""Check if the stochastic process is a martingale with respect to an optional filtration.

        Beware that the check is computationally intensive, as it requires calculating conditional expectations at each time step.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        filtration : Filtration | None, default=None
            The filtration with respect to which the martingale property is checked. If None, the natural filtration of the process is used.
        prob_measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which the martingale property is checked. If `None`, the probability measure of the process is used.
        rtol : float, default=1e-05
            The relative tolerance parameter for numerical comparison. Internally passed to `numpy.allclose`.
        atol : float, default=1e-08
            The absolute tolerance parameter for numerical comparison. Internally passed to `numpy.allclose`.

        Raises
        ------
        ValueError
            If data has not been generated for the stochastic process.
        TypeError
            If the provided filtration is not an instance of Filtration, or its sample space does not match the sample space of the process, or its time index does not match the time index of the process, or if the provided probability measure is not an instance of ProbabilityMeasure, or its sample space does not match the sample space of the process.

        Returns
        -------
        is_martingale : bool
            `True` if the stochastic process is a martingale, `False` otherwise.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(start=1, length=2)
        >>> X = RandomWalk.from_enumeration(p=0.5, initial_state=0, index=T)
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time    1  2  3
        sample
        0       0 -1 -2
        1       0 -1  0
        2       0  1  0
        3       0  1  2
        >>> print(X.is_martingale())
        True
        >>> Y = RandomWalk.from_enumeration(p=0.7, initial_state=1, index=T, name="Y")
        >>> print(Y.is_martingale())
        False

        Notes
        -----
        A stochastic process $X_t$ with index set $T$ is a *martingale* relative to a filtration $\mathcal{F}_t$ if

        $$
        E(X_{t+1} | \mathcal{F}_t) = X_t
        $$

        for all $t\in T$ for which $t+1 \in T$.
        """
        if self.data is None:
            raise ValueError(
                "Data must be generated before checking martingale property."
            )
        if filtration is not None:
            if not isinstance(filtration, Filtration):
                raise TypeError(
                    "If filtration is provided, it must be an instance of Filtration."
                )
            if filtration.sample_space != self.sample_space:
                raise TypeError(
                    "If filtration is provided, its sample space must match the sample_space of the process."
                )
            if filtration.index != self.time:
                raise TypeError(
                    "If filtration is provided, its index must match the index of the process."
                )
        if prob_measure is not None:
            if not isinstance(prob_measure, ProbabilityMeasure):
                raise TypeError(
                    "If prob_measure is provided, it must be an instance of ProbabilityMeasure."
                )
            if prob_measure.sample_space != self.sample_space:
                raise TypeError(
                    "If prob_measure is provided, its sample space must match the sample_space of the process."
                )

        if filtration is None:
            filtration = self.natural_filtration

        if prob_measure is None:
            prob_measure = self.prob_measure

        for t_prev, t_curr in zip(self.time[:-1], self.time[1:]):
            df = pd.DataFrame(
                {
                    "atom ID": filtration[t_prev].data,
                    "rv": self[t_curr].data,
                    "probability": prob_measure.data,
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

    def is_submartingale(
        self,
        filtration: Filtration | None = None,
        prob_measure: ProbabilityMeasure | None = None,
        rtol: float = 1e-05,
        atol: float = 1e-08,
    ) -> bool:
        r"""Check if the stochastic process is a submartingale with respect to an optional filtration.

        Beware that the check is computationally intensive, as it requires calculating conditional expectations at each time step.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        filtration : Filtration | None, default=None
            The filtration with respect to which the submartingale property is checked. If None, the natural filtration of the process is used.
        prob_measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which the submartingale property is checked. If `None`, the probability measure of the process is used.
        rtol : float, default=1e-05
            The relative tolerance parameter for numerical comparison. Internally passed to `numpy.allclose`.
        atol : float, default=1e-08
            The absolute tolerance parameter for numerical comparison. Internally passed to `numpy.allclose`.

        Raises
        ------
        ValueError
            If data has not been generated for the stochastic process.
        TypeError
            If the provided filtration is not an instance of Filtration, or its sample space does not match the sample space of the process, or its time index does not match the time index of the process, or if the provided probability measure is not an instance of ProbabilityMeasure, or its sample space does not match the sample space of the process.

        Returns
        -------
        is_submartingale : bool
            `True` if the stochastic process is a submartingale, `False` otherwise.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(start=1, length=2)
        >>> X = RandomWalk.from_enumeration(p=0.6, initial_state=0, index=T)
        >>> print(X.is_submartingale())
        True

        Notes
        -----
        A stochastic process $X_t$ with index set $T$ is a *submartingale* relative to a filtration $\mathcal{F}_t$ if

        $$
        E(X_{t+1} | \mathcal{F}_t) \geq X_t
        $$

        for all $t\in T$ for which $t+1 \in T$.
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
            if filtration.sample_space != self.sample_space:
                raise TypeError(
                    "If filtration is provided, its sample space must match the sample space of the process."
                )
            if filtration.time != self.time:
                raise TypeError(
                    "If filtration is provided, its time index must match the time index of the process."
                )
        if prob_measure is not None:
            if not isinstance(prob_measure, ProbabilityMeasure):
                raise TypeError(
                    "If prob_measure is provided, it must be an instance of ProbabilityMeasure."
                )
            if prob_measure.sample_space != self.sample_space:
                raise TypeError(
                    "If prob_measure is provided, its sample space must match the sample space of the process."
                )

        if filtration is None:
            filtration = self.natural_filtration

        if prob_measure is None:
            prob_measure = self.prob_measure

        for t_prev, t_curr in zip(self.time[:-1], self.time[1:]):
            df = pd.DataFrame(
                {
                    "atom ID": filtration[t_prev].data,
                    "rv": self[t_curr].data,
                    "probability": prob_measure.data,
                }
            )
            weighted_sum = (
                (df["probability"] * df["rv"]).groupby(df["atom ID"]).transform("sum")
            )
            group_probs = df["probability"].groupby(df["atom ID"]).transform("sum")
            expectation = weighted_sum / group_probs
            is_close = np.isclose(expectation, self[t_prev].data, rtol=rtol, atol=atol)
            is_greater = expectation > self[t_prev].data
            if not np.all(is_close | is_greater):
                return False

        return True

    def is_supermartingale(
        self,
        filtration: Filtration | None = None,
        prob_measure: ProbabilityMeasure | None = None,
        rtol: float = 1e-05,
        atol: float = 1e-08,
    ) -> bool:
        r"""Check if the stochastic process is a supermartingale with respect to an optional filtration.

        Beware that the check is computationally intensive, as it requires calculating conditional expectations at each time step.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        filtration : Filtration | None, default=None
            The filtration with respect to which the supermartingale property is checked. If None, the natural filtration of the process is used.
        prob_measure : ProbabilityMeasure | None, default=None
            The probability measure with respect to which the supermartingale property is checked. If `None`, the probability measure of the process is used.
        rtol : float, default=1e-05
            The relative tolerance parameter for numerical comparison. Internally passed to `numpy.allclose`.
        atol : float, default=1e-08
            The absolute tolerance parameter for numerical comparison. Internally passed to `numpy.allclose`.

        Raises
        ------
        ValueError
            If data has not been generated for the stochastic process.
        TypeError
            If the provided filtration is not an instance of Filtration, or its sample space does not match the sample space of the process, or its time index does not match the time index of the process, or if the provided probability measure is not an instance of ProbabilityMeasure, or its sample space does not match the sample space of the process.

        Returns
        -------
        is_supermartingale : bool
            True if the stochastic process is a supermartingale, False otherwise.

        Examples
        --------
        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(start=1, length=2)
        >>> X = RandomWalk.from_enumeration(p=0.4, initial_state=0, index=T)
        >>> print(X.is_supermartingale())
        True

        Notes
        -----
        A stochastic process $X_t$ with index set $T$ is a *supermartingale* relative to a filtration $\mathcal{F}_t$ if

        $$
        E(X_{t+1} | \mathcal{F}_t) \leq X_t
        $$

        for all $t\in T$ for which $t+1 \in T$.
        """
        if self.data is None:
            raise ValueError(
                "Data must be generated before checking supermartingale property."
            )
        if filtration is not None:
            if not isinstance(filtration, Filtration):
                raise TypeError(
                    "If filtration is provided, it must be an instance of Filtration."
                )
            if filtration.sample_space != self.sample_space:
                raise TypeError(
                    "If filtration is provided, its sample space must match the sample space of the process."
                )
            if filtration.time != self.time:
                raise TypeError(
                    "If filtration is provided, its time index must match the time index of the process."
                )
        if prob_measure is not None:
            if not isinstance(prob_measure, ProbabilityMeasure):
                raise TypeError(
                    "If prob_measure is provided, it must be an instance of ProbabilityMeasure."
                )
            if prob_measure.sample_space != self.sample_space:
                raise TypeError(
                    "If prob_measure is provided, its sample space must match the sample space of the process."
                )

        if filtration is None:
            filtration = self.natural_filtration

        if prob_measure is None:
            prob_measure = self.prob_measure

        for t_prev, t_curr in zip(self.time[:-1], self.time[1:], strict=False):
            df = pd.DataFrame(
                {
                    "atom ID": filtration[t_prev].data,
                    "rv": self[t_curr].data,
                    "probability": prob_measure.data,
                }
            )
            weighted_sum = (
                (df["probability"] * df["rv"]).groupby(df["atom ID"]).transform("sum")
            )
            group_probs = df["probability"].groupby(df["atom ID"]).transform("sum")
            expectation = weighted_sum / group_probs
            is_close = np.isclose(expectation, self[t_prev].data, rtol=rtol, atol=atol)
            is_less = expectation < self[t_prev].data
            if not np.all(is_close | is_less):
                return False

        return True

    def is_adapted(self, filtration: Filtration):
        """Check if the stochastic process is adapted to a given filtration.

        See the Notes section below for more details.

        Parameters
        ----------
        filtration : Filtration
            The filtration to check adaptation against.

        Raises
        ------
        ValueError
            If data has not been generated for the stochastic process.
        TypeError
            If the provided filtration is not an instance of `Filtration`, or its sample space does not match the domain of the process, or if the time indices of the process and the filtration do not have a non-empty intersection.

        Returns
        -------
        is_adapted : bool
            True if the stochastic process is adapted to the given filtration, False otherwise.

        Examples
        --------
        >>> from sigalg.core import RandomVariable, Time
        >>> from sigalg.processes import RandomWalk, StochasticProcess
        >>> T = Time.discrete(start=0, stop=2)
        >>> X = RandomWalk.from_enumeration(p=0.7, initial_state=0, index=T)
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time    0  1  2
        sample
        0       0 -1 -2
        1       0 -1  0
        2       0  1  0
        3       0  1  2
        >>> def f0(X: StochasticProcess) -> RandomVariable:
        ...     return X[0] + 1
        >>> def f1(X: StochasticProcess) -> RandomVariable:
        ...     return 2 * X[0] + X[1]
        >>> def f2(X: StochasticProcess) -> RandomVariable:
        ...     return X[2] - X[1] + X[0]
        >>> Y = X.transform(functions=[f0, f1, f2], name="Y")
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'Y':
        time    0  1  2
        sample
        0       1 -1 -1
        1       1 -1  1
        2       1  1 -1
        3       1  1  1
        >>> print(Y.is_adapted(filtration=X.natural_filtration))
        True
        """
        from ...core.sigma_algebras.filtration import Filtration

        if self.data is None:
            raise ValueError("Data must be generated before checking adaptation.")
        if not isinstance(filtration, Filtration):
            raise TypeError("filtration must be an instance of Filtration.")
        if filtration.sample_space != self.sample_space:
            raise TypeError(
                "The sample space of the filtration must match the sample_space of the process."
            )

        times = self.time & filtration.index

        if times is None:
            raise TypeError(
                "The time indices of the process and the filtration must have a non-empty intersection."
            )

        for t in times:
            if self[t].is_measurable(filtration[t]):
                continue
            else:
                return False

        return True

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
            title = f"{type(self)._repr_name} '{self.name}'"
        ax.set_title(title)

        return ax

    def _integer_check(self, values):
        try:
            return np.allclose(values, np.round(values))
        except (TypeError, AttributeError):
            return False
