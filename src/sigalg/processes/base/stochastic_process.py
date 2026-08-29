"""Base class for stochastic processes."""

from __future__ import annotations

import inspect
from collections.abc import Hashable
from functools import wraps
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from ...core.functions.random_vector import RandomVector
from ..transforms.process_transforms import ProcessTransformMethods

if TYPE_CHECKING:
    from ...core.functions.random_variable import RandomVariable
    from ...core.indices.time import Time
    from ...core.measures.probability_measure import ProbabilityMeasure
    from ...core.sigma_algebras.filtration import Filtration
    from ...core.sigma_algebras.sigma_algebra import SigmaAlgebra
    from ...core.spaces.domain import Domain
    from ...typing.index_like import IndexLike
    from ...typing.mapping_like import MappingLike


def generator(func):
    """Decorator that handles common setup and execution for generate methods."""  # noqa: D401

    @wraps(func)
    def wrapper(
        cls,
        *args,
        mode: Literal["enum", "sim"] = "sim",
        n_trajectories: int | None = None,
        index: Time | IndexLike | None = None,
        length: int | None = None,
        random_state: int | np.random.Generator | None = None,
        name: Hashable = "X",
        **kwargs,
    ):
        subclass_kwargs = func(cls, *args, mode=mode, **kwargs)

        index, random_state = cls._validate_and_return_generation_params(
            index=index,
            length=length,
            n_trajectories=n_trajectories,
            mode=mode,
            random_state=random_state,
        )

        process = cls()
        process._index = index
        process._name = name
        process._mode = mode
        process._n_trajectories = n_trajectories
        process._random_state = random_state

        for key, value in subclass_kwargs.items():
            setattr(process, "_" + key, value)

        if mode == "enum":
            return process._enumeration_logic()
        else:
            return process._simulation_logic()

    return classmethod(wrapper)


class StochasticProcess(RandomVector, ProcessTransformMethods):
    r"""Base class for stochastic processes.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    domain : Domain | IndexLike | None, default=None
        The domain of the underlying probability space.
    sig_alg : SigmaAlgebra | None, default=None
        The sigma algebra of the underlying probability space.
    measure : ProbabilityMeasure | None, default=None
        The probability measure of the underlying probability space. This is a required argument. The default `None` is only provided to maintain consistency with the
        parent class `MeasurableVector`, which does not require a probability measure.
    mapping : MappingLike | None, default=None
        The mapping defining the stochastic process.
    index : Time | Index | None, default=None
        The time index of the stochastic process.
    name : Hashable, default="X"
        The name of the stochastic process.

    Examples
    --------
    >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
    >>> from sigalg.processes import StochasticProcess
    >>> Omega = SampleSpace.from_sequence(size=3, variable_name="omega")
    >>> F = SigmaAlgebra(
    ...     domain=Omega,
    ...     mapping={
    ...         0: 0,
    ...         1: 0,
    ...         2: 1,
    ...     },
    ... )
    >>> P = ProbabilityMeasure.from_rand(domain=F, random_state=42)
    >>> X = StochasticProcess(
    ...     domain=Omega,
    ...     sig_alg=F,
    ...     measure=P,
    ...     mapping={
    ...         0: (0, 1),
    ...         1: (0, 1),
    ...         2: (1, 0),
    ...     },
    ... )
    >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
    Stochastic process 'X':
    time   0  1
    omega
    0      0  1
    1      0  1
    2      1  0

    Notes
    -----
    Let $(\Omega, \mathcal{F}, P)$ be a probability space and let $T$ be a linearly-ordered set. A (*$T$-indexed*) *stochastic process* is a collection of random variables $X_t$ defined on the probability space, one for each $t\in T$. The index set $T$ is often interpreted as "time".
    """

    # _properties = RandomVector._properties + [
    #     "_n_trajectories",
    #     "_natural_filtration",
    #     "_last_rv",
    #     "_random_state",
    #     "_mode",
    #     "_length",
    #     "_time",
    # ]
    _repr_name = "StochasticProcess"
    _str_name = "Stochastic process"
    _default_name = "X"

    # --------------------- constructors --------------------- #

    # TODO: add parameters for domain, sig-alg, and measure names
    def __init__(
        self,
        domain: Domain | IndexLike | None = None,
        sig_alg: SigmaAlgebra | None = None,
        measure: ProbabilityMeasure | None = None,
        mapping: MappingLike | None = None,
        index: Time | IndexLike | None = None,
        name: Hashable = "X",
    ) -> None:
        from ...core.indices.time import Time

        if not isinstance(index, Time):
            index = Time(index) if index is not None else None

        super().__init__(
            domain=domain,
            sig_alg=sig_alg,
            measure=measure,
            mapping=mapping,
            index=index,
            name=name,
        )

    @classmethod
    def from_time(
        cls,
        domain: Domain | IndexLike,
        sig_alg: SigmaAlgebra | None = None,
        measure: ProbabilityMeasure | None = None,
        index: Time | IndexLike | None = None,
        name: Hashable = "X",
    ) -> StochasticProcess:
        """Define a stochastic process whose trajectories are the time index itself.

        Parameters
        ----------
        domain : Domain | IndexLike
            The domain of the underlying probability space.
        sig_alg : SigmaAlgebra | None, default=None
            The sigma algebra of the underlying probability space.
        measure : ProbabilityMeasure | None, default=None
            The probability measure of the underlying probability space. This is a required argument. The default `None` is only provided to maintain consistency with the
            parent class `MeasurableVector`, which does not require a probability measure.
        index : Time | IndexLike | None, default=None
            The time index of the stochastic process.
        name : Hashable, default="X"
            The name of the stochastic process.

        Returns
        -------
        process : StochasticProcess
            A stochastic process whose trajectories are the time index itself.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, Time
        >>> from sigalg.processes import StochasticProcess
        >>> Omega = SampleSpace.from_sequence(size=2)
        >>> T = Time.discrete(length=3)
        >>> X = StochasticProcess.from_time(domain=Omega, index=T)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Stochastic process 'X':
        time    0  1  2  3
        sample
        0       0  1  2  3
        1       0  1  2  3
        """
        from ...core.indices.time import Time
        from ...core.measures.probability_measure import ProbabilityMeasure
        from ...core.sigma_algebras.sigma_algebra import SigmaAlgebra
        from ...core.spaces.sample_space import SampleSpace

        if not isinstance(domain, SampleSpace):
            domain = SampleSpace(domain)

        if sig_alg is None:
            sig_alg = SigmaAlgebra.power_set(domain)
        if measure is None:
            measure = ProbabilityMeasure.uniform(sig_alg)

        if not isinstance(index, Time):
            index = Time(index)

        mapping = pd.DataFrame(
            {t: [t] * len(domain) for t in index},
            index=domain.data,
        )
        mapping.columns = index.data

        return cls(
            domain=domain,
            sig_alg=sig_alg,
            measure=measure,
            mapping=mapping,
            index=index,
            name=name,
        )

    @generator
    def generate(
        cls,
        *args,  # Subclass-specific parameters should be defined in the subclass's generate method signature
        mode: Literal["enum", "sim"] = "sim",
        n_trajectories: int | None = None,
        index: Time | IndexLike | None = None,
        length: int | None = None,
        random_state: int | np.random.Generator | None = None,
        name: Hashable = "X",
    ) -> dict[str, object]:
        """Generate trajectories of the stochastic process by either exhaustive enumeration or Monte Carlo simulation.

        This is an abstract method that must be implemented in subclasses. It should validate any subclass-specific parameters, pack them into a dictionary, and return the dictionary. Provided that the user decorates with the `@generator` decorator, the base class will handle the rest of the generation logic. The implementation of the `generate` method in the subclass `IIDProcess` is a good template to review to help understand the needed steps.

        Parameters
        ----------
        mode : Literal["enum", "sim"], default="sim"
            The generation mode of the process, either `enum` for exhaustive enumeration or `sim` for Monte Carlo simulation.
        n_trajectories : int | None, default=None
            The number of trajectories to simulate. If `mode` is set to `enum`, this parameter is ignored.
        index : Time | IndexLike | None, default=None
            The index of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        length : int | None, default=None
            The length of the trajectories of the stochastic process. One of `index` or `length` must be provided; if both are provided, the length of `index` must match `length`.
        random_state : int | np.random.Generator | None, default=None
            An optional random state for reproducibility.
        name : Hashable, default="X"
            The name of the stochastic process.

        Returns
        -------
        result : dict[str, object]
            A dictionary containing the names of the subclass-specific parameters and their values.
        """
        # Validate subclass-specific parameters here, pack them into a dictionary, and return the dictionary.

        raise NotImplementedError("Not implemented.")

    def regenerate(self) -> None:
        """Regenerate the trajectories of the stochastic process using the current parameters.

        Examples
        --------
        Generate a symmetric random walk (i.e., with `p=0.5`) and print its trajectories and probability measure.

        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(start=1, length=2)
        >>> X = RandomWalk.generate(mode="enum", p=0.5, initial_state=0, index=T)
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        Random walk 'X':
        time    1  2  3
        sample
        0       0 -1 -2
        1       0 -1  0
        2       0  1  0
        3       0  1  2
        >>> print(X.measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        sample
        0              0.25
        1              0.25
        2              0.25
        3              0.25

        Set the `p` parameter to `0.7`, regenerate the trajectories, and print the new probability measure.

        >>> X.p = 0.7
        >>> X.regenerate()
        >>> print(X.measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        sample
        0              0.09
        1              0.21
        2              0.21
        3              0.49
        """
        self_dict = {
            key[1:] if key.startswith("_") else key: value
            for key, value in self.__dict__.items()
        }
        generate_params = list(inspect.signature(type(self).generate).parameters)
        params = {
            key: value for key, value in self_dict.items() if key in generate_params
        }

        try:
            new_process = type(self).generate(**params)
        except Exception as e:
            raise RuntimeError(
                "Failed to regenerate the stochastic process. Perhaps you have not initially called the 'generate' method."
            ) from e

        self.__dict__.update(new_process.__dict__)

    # --------------------- enumeration methods --------------------- #

    def _enumeration_logic(self) -> StochasticProcess:
        """Calls the subclass-specific overriden methods `_enumeration_hook` and `_generate_exact_prob_measure` to generate trajectories and the underlying probability space, and returns `self`.

        This method is not meant to be overriden by subclasses. Leave it as is.
        """  # noqa: D401
        from ...validation.mapping_validator import MappingValidator

        mapping = self._enumeration_subclass_hook()

        v = MappingValidator(
            mapping=mapping,
            output_name=self.name,
            index=self.index,
            name=self.name,
            domain_kind="sample_space",
        )

        self._data = v.data
        self._index = v.index
        self._name = v.name
        self.measure = self._generate_exact_prob_measure(v.domain)

        return self

    def _enumeration_subclass_hook(self) -> pd.DataFrame:
        """Abstract hook for enumeration logic.

        This method must be implemented in subclasses to define how to enumerate trajectories. It will use any subclass-specific attributes of `self` to generate a `pd.DataFrame` whose rows are trajectories of the stochastic process.

        The implementation of `_enumeration_hook` in the subclass `IIDProcess` is a good template to review to help understand the needed steps.

        Returns
        -------
        trajectories : pd.DataFrame
            A data frame containing the trajectories of the stochastic process.
        """
        raise NotImplementedError("Not implemented.")

    def _generate_exact_prob_measure(self, domain: Domain) -> ProbabilityMeasure:
        """Generate the exact probability measure for an enumerated stochastic process.

        Subclasses that support enumeration should implement this method to generate the exact probability measure based on the enumerated trajectories.

        The implementation of `_generate_exact_prob_measure` in the subclass `IIDProcess` is a good template to review to help understand the needed steps.

        Parameters
        ----------
        domain : Domain
            The domain of the underlying probability space.

        Returns
        -------
        prob_measure : ProbabilityMeasure
            The exact probability measure for the enumerated stochastic process.
        """
        raise NotImplementedError(
            "Method to generate exact probability measure not implemented."
        )

    # --------------------- simulation methods --------------------- #

    def _simulation_logic(self) -> StochasticProcess:
        """Calls the subclass-specific overriden methods `_simulation_hook` to generate trajectories and the underlying probability space, and returns `self`.

        This method is not meant to be overriden by subclasses. Leave it as is.
        """  # noqa: D401
        from ...core.measures.probability_measure import ProbabilityMeasure
        from ...validation.mapping_validator import MappingValidator

        mapping = self._simulation_subclass_hook()

        v = MappingValidator(
            mapping=mapping,
            output_name=self.name,
            index=self.time,
            name=self.name,
            domain_kind="sample_space",
        )

        self._data = v.data
        self._index = v.index
        self._name = v.name
        self.measure = ProbabilityMeasure.uniform(v.domain)

        return self

    def _simulation_subclass_hook(self) -> pd.DataFrame:
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
    def _validate_and_return_generation_params(
        mode: Literal["enum", "sim"] | None = None,
        index: Time | IndexLike | None = None,
        length: int | None = None,
        n_trajectories: int | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> tuple[Time, np.random.Generator]:
        from ...core.indices.time import Time

        if mode is not None:
            if not isinstance(mode, str):
                raise TypeError("mode must be a string.")
            if mode not in ["enum", "sim"]:
                raise ValueError("mode must be either 'enum' or 'sim'.")

        if length is not None:
            if not isinstance(length, int):
                raise TypeError("If given, length must be an integer.")
            if length <= 0:
                raise ValueError("If given, length must be positive.")
        if not isinstance(index, Time):
            index = Time(index) if index is not None else None
        if index is not None and not isinstance(index, Time):
            raise TypeError("If given, index must be an instance of Time.")
        if length is None and index is None:
            raise ValueError("One or the other of length or index must be given.")
        if (length is not None and index is not None) and (length != len(index)):
            raise ValueError(
                "If both length and index are given, the lengths must be consistent."
            )
        if n_trajectories is not None:
            if not isinstance(n_trajectories, int):
                raise TypeError("If given, n_trajectories must be an integer.")
            if n_trajectories <= 0:
                raise ValueError("If given, n_trajectories must be positive.")

        if random_state is not None and not isinstance(
            random_state, (int, np.random.Generator)
        ):
            raise TypeError(
                "random_state must be an integer, np.random.Generator, or None."
            )

        return (
            Time.discrete(length=length) if index is None else index,
            (
                random_state
                if isinstance(random_state, np.random.Generator)
                else np.random.default_rng(random_state)
            ),
        )

    def _erase_generated_data(self) -> None:
        from ...core.functions.measurable_vector import MeasurableVector

        empty = MeasurableVector(index=self._index, name=self._name)
        self.__dict__.update(empty.__dict__)

    # --------------------- properties --------------------- #

    @property
    def mode(self) -> Literal["enum", "sim"] | None:
        """Get the generation mode of the process.

        The `mode` property is settable. If the trajectories of a process were first generated by calling `from_enumeration`, be sure to set the `n_trajectories` and `random_state` properties before setting the `mode` property to `sim`. See the Examples section below for usage.

        Returns
        -------
        mode : Literal["enum", "sim"] | None:
            The generation mode of the process, or `None` if it has not yet been set.

        Examples
        --------
        Exhaustively enumerate all trajectories of an IID Bernoulli process.

        >>> from scipy.stats import bernoulli
        >>> from sigalg.core import Time
        >>> from sigalg.processes import IIDProcess, RandomWalk
        >>> T = Time.discrete(length=2)
        >>> X = IIDProcess.generate(
        ...     mode="enum",
        ...     distribution=bernoulli(0.75),
        ...     support=[0, 1],
        ...     index=T,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
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

        Set the `n_trajectories` and `random_state` properties before setting `mode` to `sim`. Regenerate the trajectories and print them.

        >>> X.n_trajectories = 10
        >>> X.random_state = 42
        >>> X.mode = "sim"
        >>> X.regenerate()
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        time    0  1  2
        sample
        0       0  1  0
        1       1  1  0
        2       0  0  1
        3       1  1  0
        4       1  0  1
        5       1  1  1
        6       0  1  0
        7       1  0  0
        8       0  1  1
        9       1  1  1

        Now, we go the other way. First simulate trajectories of a random walk, and then switch the generation mode to `enumeration`.

        >>> Y = RandomWalk.generate(
        ...     mode="sim",
        ...     p=0.7,
        ...     initial_state=0,
        ...     n_trajectories=10,
        ...     index=T,
        ...     random_state=42,
        ...     name="Y",
        ... )
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random walk 'Y':
        time    0  1  2
        sample
        0       0 -1  0
        1       0 -1  0
        2       0  1  0
        3       0 -1 -2
        4       0  1  2
        5       0  1  0
        6       0  1  0
        7       0  1  2
        8       0  1  2
        9       0 -1  0
        >>> Y.mode = "enum"
        >>> Y.regenerate()
        >>> print(Y)  # doctest: +NORMALIZE_WHITESPACE
        Random walk 'Y':
        time    0  1  2
        sample
        0       0 -1 -2
        1       0 -1  0
        2       0  1  0
        3       0  1  2
        """
        return self._mode

    @mode.setter
    def mode(self, value: Literal["enum", "sim"]) -> None:
        """Set the generation mode of the process.

        If the trajectories of a process were first generated by calling `from_enumeration`, be sure to set the `n_trajectories` and `random_state` properties before setting the `mode` property to `sim`. See the Examples section below for usage.

        Parameters
        ----------
        value : Literal["enum", "sim"]
            The new value for `mode`.
        """
        if value not in ["enum", "sim"]:
            raise ValueError("mode must be either 'enum' or 'sim'.")

        if self.mode == value:
            return

        if self.mode == "enum" and value == "sim":
            if self.n_trajectories is None:
                raise ValueError(
                    "To switch from enumeration to simulation, the n_trajectories property must be set."
                )
            if self.random_state is None:
                raise ValueError(
                    "To switch from enumeration to simulation, the random_state property must be set."
                )

        self._mode = value
        self._erase_generated_data()

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
    def time(self) -> Time | None:
        """Get the time index of the process.

        This property is an alias for the `index` property of the superclass `RandomVector`.

        The `time` property is settable. See the Examples section below for usage.

        Returns
        -------
        time : Time | None
            The time index of the stochastic process.

        Examples
        --------
        Generate trajectories of an IID process with an explicit time index.

        >>> from scipy.stats import bernoulli
        >>> from sigalg.core import Time
        >>> from sigalg.processes import IIDProcess
        >>> T = Time.discrete(length=2)
        >>> X = IIDProcess.generate(
        ...     mode="sim",
        ...     distribution=bernoulli(0.75),
        ...     support=[0, 1],
        ...     n_trajectories=10,
        ...     index=T,
        ...     random_state=42,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        time    0  1  2
        sample
        0       0  1  0
        1       1  1  0
        2       0  0  1
        3       1  1  0
        4       1  0  1
        5       1  1  1
        6       0  1  0
        7       1  0  0
        8       0  1  1
        9       1  1  1

        Set the time index and regenerate the trajectories.

        >>> X.time = Time.discrete(length=3)
        >>> X.regenerate()
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        time    0  1  2  3
        sample
        0       1  0  1  1
        1       1  1  1  1
        2       1  1  1  0
        3       1  1  0  0
        4       1  1  1  1
        5       1  1  0  1
        6       1  0  1  1
        7       1  1  1  1
        8       1  0  1  1
        9       1  1  1  1
        """
        return self.index

    @time.setter
    def time(self, value: Time | IndexLike) -> None:
        """Set the time index of the process.

        Parameters
        ----------
        value : Time | IndexLike
            The new time index of the process.
        """
        from ...core.indices.time import Time

        if self.time == value:
            return
        if not isinstance(value, Time):
            value = Time(value)
        self._index = value
        self._erase_generated_data()

    @property
    def length(self) -> int | None:
        """Get the length of the trajectories of the process, defined as the length of the underlying time interval.

        The `length` property is settable. See the Examples section below for usage.

        Returns
        -------
        length : int | None
            The length of the trajectories of the process, or `None` if the trajectories have not been generated.

        Examples
        --------
        Generate trajectories of an IID process with a specified length.

        >>> from scipy.stats import bernoulli
        >>> from sigalg.processes import IIDProcess
        >>> X = IIDProcess.generate(
        ...     mode="sim",
        ...     distribution=bernoulli(0.75),
        ...     support=[0, 1],
        ...     n_trajectories=10,
        ...     length=2,
        ...     random_state=42,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        time    0  1  2
        sample
        0       0  1  0
        1       1  1  0
        2       0  0  1
        3       1  1  0
        4       1  0  1
        5       1  1  1
        6       0  1  0
        7       1  0  0
        8       0  1  1
        9       1  1  1

        Set the length of the trajectories and regenerate the process.

        >>> X.length = 3
        >>> X.regenerate()
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        time    0  1  2  3
        sample
        0       1  0  1  1
        1       1  1  1  1
        2       1  1  1  0
        3       1  1  0  0
        4       1  1  1  1
        5       1  1  0  1
        6       1  0  1  1
        7       1  1  1  1
        8       1  0  1  1
        9       1  1  1  1
        """
        return len(self.time) - 1 if self.time is not None else None

    @length.setter
    def length(self, value: int) -> None:
        """Set the length of the trajectories of the process.

        Parameters
        ----------
        value : int
            The new length of the trajectories.
        """
        from ...core.indices.time import Time

        if self.length == value:
            return
        self._index = Time.discrete(length=value)
        self._length = None
        self._erase_generated_data()

    @property
    def n_trajectories(self) -> int | None:
        """Get the number of trajectories in the stochastic process.

        The `n_trajectories` property is settable. See the Examples section below for usage.

        Returns
        -------
        n_trajectories : int | None
            The number of trajectories in the stochastic process, or `None` if trajectories have not been simulated.

        Examples
        --------
        Simulate ten trajectories of an IID process.

        >>> from scipy.stats import bernoulli
        >>> from sigalg.core import Time
        >>> from sigalg.processes import IIDProcess
        >>> T = Time.discrete(length=2)
        >>> X = IIDProcess.generate(
        ...     mode="sim",
        ...     distribution=bernoulli(0.75),
        ...     support=[0, 1],
        ...     n_trajectories=10,
        ...     index=T,
        ...     random_state=42,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        time    0  1  2
        sample
        0       0  1  0
        1       1  1  0
        2       0  0  1
        3       1  1  0
        4       1  0  1
        5       1  1  1
        6       0  1  0
        7       1  0  0
        8       0  1  1
        9       1  1  1

        Set the number of trajectories to 5 and regenerate the process.

        >>> X.n_trajectories = 5
        >>> X.regenerate()
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        time    0  1  2
        sample
        0       1  0  1
        1       1  1  1
        2       1  1  1
        3       1  1  0
        4       1  1  0
        """
        if self.data is not None:
            self._n_trajectories = len(self.data)

        return self._n_trajectories

    @n_trajectories.setter
    def n_trajectories(self, value: int) -> None:
        """Set the number of trajectories of the stochastic process.

        Parameters
        ----------
        value : int
            The new value for `n_trajectories`.
        """
        if not isinstance(value, int):
            raise TypeError("n_trajectories must be an integer.")
        if value <= 0:
            raise ValueError("n_trajectories must be positive.")

        if self.n_trajectories == value:
            return
        self._n_trajectories = value
        self._erase_generated_data()

    @property
    def random_state(self) -> int | np.random.Generator | None:
        """Get the random state of the stochastic process.

        The `random_state` property is settable. See the Examples section below for usage.

        Returns
        -------
        random_state : int | np.random.Generator | None
            The random state of the stochstic process.

        Examples
        --------
        Simulate ten trajectories of an IID process with a specified random state.

        >>> from scipy.stats import bernoulli
        >>> from sigalg.core import Time
        >>> from sigalg.processes import IIDProcess
        >>> T = Time.discrete(length=2)
        >>> X = IIDProcess.generate(
        ...     mode="sim",
        ...     distribution=bernoulli(0.75),
        ...     support=[0, 1],
        ...     n_trajectories=10,
        ...     index=T,
        ...     random_state=42,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        time    0  1  2
        sample
        0       0  1  0
        1       1  1  0
        2       0  0  1
        3       1  1  0
        4       1  0  1
        5       1  1  1
        6       0  1  0
        7       1  0  0
        8       0  1  1
        9       1  1  1

        Set the random state to a different value and regenerate the process.

        >>> X.random_state = 101
        >>> X.regenerate()
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
        IID process 'X':
        time    0  1  2
        sample
        0       0  1  0
        1       1  1  0
        2       0  1  0
        3       1  0  1
        4       1  1  0
        5       1  1  1
        6       1  1  1
        7       1  1  1
        8       0  0  1
        9       1  0  0
        """
        return self._random_state

    @random_state.setter
    def random_state(self, value: int | np.random.Generator) -> None:
        """Set the random state of the stochastic process.

        See the docstring of the getter method for more details.

        Parameters
        ----------
        value : int | np.random.Generator
            The new value for `random_state`.
        """
        if not isinstance(value, (int, np.random.Generator)):
            raise TypeError("random_state must be an integer or np.random.Generator.")

        if self.random_state == value:
            return
        self._random_state = value
        self._erase_generated_data()

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
        Exhaustively enumerate all length-2 trajectories of an IID Bernoulli process.

        >>> from scipy.stats import bernoulli
        >>> from sigalg.processes import IIDProcess
        >>> X = IIDProcess.generate(
        ...     mode="enum",
        ...     distribution=bernoulli(p=0.5),
        ...     support=[0, 1],
        ...     length=2,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
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

        Print its natural filtration. Notice that the atom identifiers of the sigma-algebras are partial trajectories of the process.

        >>> print(X.natural_filtration)  # doctest: +NORMALIZE_WHITESPACE
        Filtration 'F'
        ==============
        <BLANKLINE>
        * Time 'T':
         time
            0
            1
            2
        <BLANKLINE>
        * At index 0:
        Sigma algebra 'F_0':
                X_0
        sample
        0         0
        1         0
        2         0
        3         0
        4         1
        5         1
        6         1
        7         1
        <BLANKLINE>
        * At index 1:
        Sigma algebra 'F_1':
                X_0  X_1
        sample
        0         0    0
        1         0    0
        2         0    1
        3         0    1
        4         1    0
        5         1    0
        6         1    1
        7         1    1
        <BLANKLINE>
        * At index 2:
        Sigma algebra 'F_2':
                X_0  X_1  X_2
        sample
        0         0    0    0
        1         0    0    1
        2         0    1    0
        3         0    1    1
        4         1    0    0
        5         1    0    1
        6         1    1    0
        7         1    1    1

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
            variable_names = {
                t: self.component_names[: s + 1]
                for t, s in zip(self.time, range(len(self.component_names)))
            }

            self._natural_filtration = Filtration(
                sig_algs=data, time=self.time, variable_names=variable_names
            )
            self._natural_filtration._domain = (
                self.domain
            )  # HACK: there should be an explicit domain parameter for Filtration

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
        Exhaustively enumerate all length-2 trajectories of an IID Bernoulli process.

        >>> from scipy.stats import bernoulli
        >>> from sigalg.processes import IIDProcess
        >>> X = IIDProcess.generate(
        ...     mode="enum",
        ...     distribution=bernoulli(p=0.5),
        ...     support=[0, 1],
        ...     length=2,
        ... )
        >>> print(X)  # doctest: +NORMALIZE_WHITESPACE
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

        Get the last random variable of the process.

        >>> print(X.last_rv)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_2':
              X_2
        sample
        0       0
        1       1
        2       0
        3       1
        4       0
        5       1
        6       0
        7       1
        """
        if self._last_rv is None and self.data is not None:
            name = f"{self.name}_{self.time[-1]}".replace(".", "_")
            self._last_rv = self.components[-1].with_name(name)

        return self._last_rv

    # --------------------- methods --------------------- #

    def __len__(self) -> int:
        """Get the length of the stochastic process, defined as the length of the underlying time interval.

        Returns
        -------
        length : int
            The length of the stochastic process.
        """
        return len(self.time) - 1 if self.time is not None else None

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
        A symmetric random walk (i.e., with `p=0.5`) is a martingale.

        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(start=1, length=2)
        >>> X = RandomWalk.generate(mode="enum", p=0.5, initial_state=0, index=T)
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

        A nonsymmetric random walk (i.e., with `p != 0.5`) is not a martingale.

        >>> X.p = 0.7
        >>> X.regenerate()
        >>> print(X.is_martingale())
        False

        Notes
        -----
        Let $X_t$ be a stochastic process with index set $T$ on a probability space $(\Omega, \mathcal{F}, P)$ and let $\mathcal{F}_t$ be a filtration of $\mathcal{F}$. The process $X_t$ is said to be a *martingale* with respect to the filtration $\mathcal{F}_t$ if it satisfies the following conditions: (1) It is adapted to the filtration, meaning that $X_t$ is $\mathcal{F}_t$-measurable for all $t \in T$, and (2) it satisfies the martingale property

        $$
        E(X_{t+1} | \mathcal{F}_t) = X_t
        $$

        for all $t\in T$ for which $t+1 \in T$.
        """
        from ...core.measures.probability_measure import ProbabilityMeasure
        from ...core.sigma_algebras.filtration import Filtration

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
            if prob_measure.domain != self.sample_space:
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
                    "atom ID": filtration.data[t_prev],
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
        A random walk with `p > 0.5` is a submartingale.

        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(start=1, length=2)
        >>> X = RandomWalk.generate(mode="enum", p=0.6, initial_state=0, index=T)
        >>> print(X.is_submartingale())
        True

        Notes
        -----
        Let $X_t$ be a stochastic process with index set $T$ on a probability space $(\Omega, \mathcal{F}, P)$ and let $\mathcal{F}_t$ be a filtration of $\mathcal{F}$. The process $X_t$ is said to be a *submartingale* with respect to the filtration $\mathcal{F}_t$ if it satisfies the following conditions: (1) It is adapted to the filtration, meaning that $X_t$ is $\mathcal{F}_t$-measurable for all $t \in T$, and (2) it satisfies the submartingale property

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
            if prob_measure.domain != self.sample_space:
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
                    "atom ID": filtration.data[t_prev],
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
        A random walk with `p < 0.5` is a supermartingale.

        >>> from sigalg.core import Time
        >>> from sigalg.processes import RandomWalk
        >>> T = Time.discrete(start=1, length=2)
        >>> X = RandomWalk.generate(mode="enum", p=0.4, initial_state=0, index=T)
        >>> print(X.is_supermartingale())
        True

        Notes
        -----
        Let $X_t$ be a stochastic process with index set $T$ on a probability space $(\Omega, \mathcal{F}, P)$ and let $\mathcal{F}_t$ be a filtration of $\mathcal{F}$. The process $X_t$ is said to be a *supermartingale* with respect to the filtration $\mathcal{F}_t$ if it satisfies the following conditions: (1) It is adapted to the filtration, meaning that $X_t$ is $\mathcal{F}_t$-measurable for all $t \in T$, and (2) it satisfies the supermartingale property

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
            if prob_measure.domain != self.sample_space:
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
                    "atom ID": filtration.data[t_prev],
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
        r"""Check if the stochastic process is adapted to a given filtration.

        See the Notes section below for the mathematical details.

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
        >>> X = RandomWalk.generate(mode="enum", p=0.7, initial_state=0, index=T)
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

        Notes
        -----
        Let $X_t$ be a stochastic process with index set $T$ on a probability space $(\Omega, \mathcal{F}, P)$  and let $\mathcal{F}_t$ be a filtration of $\mathcal{F}$. The process $X_t$ is said to be *adapted* to the filtration $\mathcal{F}_t$ if for every $t \in T$, the random variable $X_t$ is $\mathcal{F}_t$-measurable.
        """
        from ...core.sigma_algebras.filtration import Filtration

        if self.data is None:
            raise ValueError("Data must be generated before checking adaptation.")
        if not isinstance(filtration, Filtration):
            raise TypeError("filtration must be an instance of Filtration.")
        if filtration.domain != self.domain:
            raise TypeError(
                "The domain of the filtration must match the domain of the process."
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
