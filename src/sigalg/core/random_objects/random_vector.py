"""A class representing a random vector mapping between two sample spaces.

Classes
-------
RandomVector
    Represents a random vector mapping between two sample spaces.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ...validation.sample_space_mapping_in import SampleSpaceMappingIn
from .operators import OperatorsMethods

if TYPE_CHECKING:
    from ..base.event import Event
    from ..base.feature_vector import FeatureVector
    from ..base.index import Index
    from ..base.probability_space import ProbabilitySpace
    from ..base.sample_space import SampleSpace
    from ..probability_measures.probability_measure import ProbabilityMeasure
    from ..random_objects.random_variable import RandomVariable
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra


# TODO: Update docstrings
class RandomVector(OperatorsMethods):
    """A class representing a random vector mapping between two sample spaces.

    An instance of `RandomVector` represents a mapping `X: Omega -> S` from a sample space `Omega` to a feature space `S`. This means that the image `X(omega)` of a sample point `omega` is a tuple of features drawn from the component spaces, called the feature vector of `omega`. The number of component spaces (i.e., the length of the feature vector) is called the dimension of the random vector.

    Instances of `RandomVector` can be constructed directly from a `domain` sample space and a dictionary of `outputs`, whose keys are the sample points in the domain and whose values are the corresponding feature vectors (as tuples). Alternatively, other methods are provided to construct a `RandomVector` from a `pd.DataFrame` or a `np.ndarray`.

    Parameters
    ----------
    domain : SampleSpace | None, default=None
        The sample space over which the random vector is defined. The `None` value indicates that the domain will be generated later through a method like `from_dict`, `from_pandas`, or `from_numpy`.
    index : Index | None, default=None
        The index of the random vector. The `None` value indicates that the index will be generated later through a method like `from_dict`, `from_pandas`, or `from_numpy`.
    name : Hashable | None, default="X"
        The name of the random vector.
    **kwargs
        Additional keyword arguments for subclass constructors.

    Raises
    ------
    TypeError
        If `domain` is not a `SampleSpace` (if given), or if `index` is not an `Index` (if given), or if `name` is not a `Hashable` (if given).

    Examples
    --------
    >>> from sigalg.core import SampleSpace, RandomVector
    >>> domain = SampleSpace.generate_sequence(size=3, prefix="s", name="S")
    >>> outputs = {"s0": (0.1, 0.2), "s1": (0.3, 0.4), "s2": (0.5, 0.6)}
    >>> # Generate a 2-dimensional random vector from outputs dict
    >>> X = RandomVector(name="X").from_dict(outputs)
    >>> tuple(X("s0"))
    (0.1, 0.2)
    >>> X.dimension
    2
    >>> # Generate a 1-dimensional random vector from a pd.Series
    >>> import pandas as pd
    >>> data = pd.Series([10, 20, 30], index=pd.Index(["s0", "s1", "s2"], name="S"))
    >>> Y = RandomVector(name="Y").from_pandas(data)
    >>> Y # doctest: +NORMALIZE_WHITESPACE
    Random vector 'Y':
           Y
    S
    s0     10
    s1     20
    s2     30
    """

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        domain: SampleSpace | None = None,
        index: Index | None = None,
        name: Hashable | None = "X",
        **kwargs,
    ) -> None:
        from ..base.index import Index
        from ..base.sample_space import SampleSpace

        if domain is not None and not isinstance(domain, SampleSpace):
            raise TypeError("If given, domain must be a SampleSpace.")
        if index is not None and not isinstance(index, Index):
            raise TypeError("If given, index must be an Index.")
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("If given, name must be a Hashable.")

        self.domain = domain
        self._index = index
        self._name = name

        # caches for properties
        self._data: pd.Series | pd.DataFrame | None = None
        self._outputs: Mapping[Hashable, Hashable] | None = None
        self._sigma_algebra: SigmaAlgebra | None = None
        self._probability_measure: ProbabilityMeasure | None = None
        self._range: RandomVector | None = None
        self._range_counts: pd.Series | None = None
        self._components: list[RandomVariable] | None = None

    # TODO: Update docstrings
    def from_dict(self, outputs: Mapping[Hashable, Hashable]) -> RandomVector:
        """Create a `RandomVector` from a dictionary mapping sample points to output vectors.

        If the `domain` sample space is not provided at construction, it is automatically generated from the keys of the `outputs` dictionary. Similarly, if the `index` is not provided at construction and the random vector has dimension 2 or greater, a default feature index (i.e., an instance of `Index`) is also automatically generated. If the `domain` is provided at construction, the keys of the `outputs` dictionary must match the indices of the `domain`.

        Parameters
        ----------
        outputs : Mapping[Hashable, Hashable]
            A mapping from sample points in the domain to their corresponding output vectors (e.g., tuples of feature values).

        Raises
        ------
        ValueError
            If the data has dimension greater than 1 and `self` is an instance of `RandomVariable`.

        Returns
        -------
        self : RandomVector
            The constructed `RandomVector` instance.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> outputs = {"omega_0": (0.1, 0.2), "omega_1": (0.3, 0.4), "omega_2": (0.5, 0.6)}
        >>> X = RandomVector(name="X").from_dict(outputs)
        >>> tuple(X("omega_1"))
        (0.3, 0.4)
        >>> X.domain # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega':
        ['omega_0', 'omega_1', 'omega_2']
        >>> X.index # doctest: +NORMALIZE_WHITESPACE
        Index 'index':
        ['X_0', 'X_1']
        """
        from ..base.index import Index
        from ..base.sample_space import SampleSpace
        from .random_variable import RandomVariable

        v = SampleSpaceMappingIn(mapping=outputs, sample_space=self.domain)

        first_output = next(iter(v.mapping.values()))
        self.dimension = len(first_output) if isinstance(first_output, tuple) else 1

        if isinstance(self, RandomVariable) and self.dimension != 1:
            raise ValueError("A random variable must have dimension 1.")

        if self.domain is None:
            self.domain = SampleSpace().from_list(list(v.mapping.keys()))
        if self.dimension > 1:
            if self._index is None:
                self._index = Index.generate_sequence(
                    size=self.dimension,
                    prefix=self.name,
                    data_name="feature",
                    name="index",
                )
            if len(self._index) != self.dimension:
                raise ValueError(
                    "Length of index must match the dimension of the RandomVector."
                )
        else:
            self._index = None

        self._outputs = v.mapping

        return self

    # TODO: Update docstrings
    def from_pandas(self, data: pd.Series | pd.DataFrame) -> RandomVector:
        """Create a `RandomVector` from a  `pd.Series` or `pd.DataFrame`.

        If the `domain` sample space is not provided at construction, then it is automatically generated from the index of the provided `pd.DataFrame`. Similarly, if the `index` is not provided at construction and the random vector has dimension 2 or greater, a default feature index (i.e., an instance of `Index`) is also automatically generated. If either `domain` or `index` are provided at construction, they must match the index and columns of the provided `pd.DataFrame`, respectively.

        Parameters
        ----------
        data : pd.Series | pd.DataFrame
            A `pd.Series` or `pd.DataFrame` where each row corresponds to a feature vector of a sample point. If `data` is a `pd.Series`, the random vector is 1-dimensional; if `data` is a `pd.DataFrame`, the random vector's dimension equals the number of columns.

        Raises
        ------
        TypeError
            If `data` is not a `pd.Series` or `pd.DataFrame`.
        ValueError
            If the length of `index` (if provided) does not match the dimension of the random vector, or if the data has dimension greater than 1 and `self` is an instance of `RandomVariable`.

        Returns
        -------
        self : RandomVector
            The constructed `RandomVector` instance.

        Examples
        --------
        >>> from sigalg.core import RandomVector
        >>> import pandas as pd
        >>> # Create a 2-dimensional random vector
        >>> data = pd.DataFrame(
        ...     [[1, 2], [3, 4], [5, 6]],
        ...     index=pd.Index([0, 1, 2], name="numbers"),
        ...     columns=pd.Index(["feature1", "feature2"], name="features"),
        ... )
        >>> X = RandomVector(name="X").from_pandas(data)
        >>> X # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        features  feature1  feature2
        numbers
        0              1         2
        1              3         4
        2              5         6
        >>> # Create a 1-dimensional random variable from a series
        >>> data = pd.Series(
        ...     [10, 20, 30],
        ...     index=pd.Index([0, 1, 2], name="numbers"),
        ... )
        >>> Y = RandomVector(name="Y").from_pandas(data)
        >>> Y # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
               Y
        numbers
        0     10
        1     20
        2     30
        >>> # Create a 1-dimensional random variable from a single-column dataframe
        >>> data = pd.DataFrame([1, 2, 3], index=pd.Index([0, 1, 2], name="numbers"))
        >>> Z = RandomVector(name="Z").from_pandas(data)
        >>> Z # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Z':
               Z
        numbers
        0     1
        1     2
        2     3
        """
        from ..base.index import Index
        from ..base.sample_space import SampleSpace
        from .random_variable import RandomVariable

        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise TypeError("data must be a pd.Series or pd.DataFrame.")
        if self.domain is not None and not data.index.equals(self.domain.data):
            raise ValueError("If provided, domain must match the index of the data.")
        if self.index is not None and isinstance(data, pd.DataFrame):
            if not data.columns.equals(self.index.data):
                raise ValueError(
                    "If provided, index must match the columns of the data."
                )

        self.dimension = 1 if isinstance(data, pd.Series) else data.shape[1]

        if isinstance(self, RandomVariable) and self.dimension != 1:
            raise ValueError("A random variable must have dimension 1.")

        if self.domain is None:
            self.domain = SampleSpace(data_name=data.index.name).from_pandas(
                data.index.copy()
            )
        else:
            data.index = self.domain.data.copy()

        if self.dimension > 1:
            if self._index is None:
                self._index = Index().from_pandas(data.columns)
            else:
                data.columns = self._index.data.copy()
        else:
            self._index = None

        if self.dimension == 1 and isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]

        self._data = data.copy()
        return self

    # TODO: Update docstrings
    def from_numpy(self, array: np.ndarray) -> RandomVector:
        """Create a `RandomVector` from a NumPy `ndarray`.

        If the `domain` sample space is not provided at construction, then it is automatically generated as a default sample space with indices `0, 1, ..., n-1`, where `n` is the number of rows in the provided `ndarray`. Similarly, if the `index` is not provided at construction and the random vector has dimension 2 or greater, a default feature index (i.e., an instance of `Index`) is also automatically generated.

        Parameters
        ----------
        array : np.ndarray
            NumPy array where rows are feature vectors of sample points and columns are features.

        Returns
        -------
        self : RandomVector
            A random vector constructed from the array.

        Raises
        ------
        TypeError
            If `array` is not a NumPy ndarray.

        Examples
        --------
        >>> from sigalg.core import Index, RandomVector, SampleSpace
        >>> import numpy as np
        >>> domain = SampleSpace.generate_sequence(size=3)
        >>> index = Index.generate_sequence(size=2, prefix="feature")
        >>> arr = np.array([[1, 2], [3, 4], [5, 6]])
        >>> X = RandomVector(domain=domain, index=index, name="X").from_numpy(arr)
        >>> X # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
                feature_0  feature_1
        sample
        omega_0         1         2
        omega_1         3         4
        omega_2         5         6
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a numpy ndarray.")
        data = pd.DataFrame(
            array,
            index=self.domain.data if self.domain else None,
            columns=self.index.data if self.index else None,
        )
        return self.from_pandas(data=data)

    # TODO: Update docstrings
    def from_constant(self, constant: Hashable) -> RandomVector:
        """Create a `RandomVector` that maps every sample point in the domain to the same constant output vector.

        For this construction method, the `domain` must be provided at construction.

        Parameters
        ----------
        constant : Hashable
            The constant output vector that every sample point in the domain maps to.

        Returns
        -------
        self : RandomVector
            A random vector mapping every sample point in the domain to the same constant output vector.
        """
        if self.domain is None:
            raise ValueError("Domain must be provided at construction.")
        if not isinstance(constant, Hashable):
            raise TypeError("constant must be a Hashable.")
        if self.index is not None:
            outputs = dict.fromkeys(self.domain.data, [constant] * len(self.index))
            rv = self.from_dict(outputs=outputs)
            rv.index = self.index
            return rv
        else:
            outputs = dict.fromkeys(self.domain.data, constant)
            return self.from_dict(outputs=outputs)

    # TODO: Write unit tests
    def from_randint(
        self,
        low: int,
        high: int,
        dim: int | None = None,
        random_state: int | None = None,
    ) -> RandomVector:
        """Generate a random vector with integer outputs uniformly sampled from the range [low, high).

        Parameters
        ----------
        low : int
            The lower bound (inclusive) of the random integers.
        high : int
            The upper bound (exclusive) of the random integers.
        dim : int | None, default=None
            The dimension of the random vector. If `None`, then the index of the random vector must be provided at construction, and the dimension is inferred from the length of the index.
        random_state : int | None, default=None
            An optional seed for the random number generator to ensure reproducibility. If `None`, the random number generator is not seeded.

        Raises
        ------
        ValueError
            If the domain is not provided at construction, or if `dim` is `None` and the index is not provided at construction.
        TypeError
            If `low` or `high` are not integers, or if `dim` is not a positive integer or `None`, or if `random_state` is not an integer or `None`.

        Returns
        -------
        self : RandomVector
            A random vector with integer outputs uniformly sampled from the range [low, high).
        """
        if self.domain is None:
            raise ValueError("Domain must be provided at construction.")
        if not isinstance(low, int) or not isinstance(high, int):
            raise TypeError("low and high must be integers.")
        if dim is not None and (not isinstance(dim, int) or dim <= 0):
            raise TypeError("dim must be a positive integer or None.")
        if dim is None and self.index is None:
            raise ValueError("If dim is None, index must be provided at construction.")
        if random_state is not None and not isinstance(random_state, int):
            raise TypeError("random_state must be an integer or None.")

        if dim is None:
            dim = len(self.index)

        rng = np.random.default_rng(random_state)
        arr = rng.integers(low, high, size=(len(self.domain.data), dim))
        return self.from_numpy(array=arr)

    # TODO: Write unit tests
    def from_randnorm(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        dim: int | None = None,
        random_state: int | None = None,
    ) -> RandomVector:
        """Generate a random vector with outputs sampled from a normal distribution with specified mean and standard deviation.

        Parameters
        ----------
        loc : float, default=0.0
            The mean of the normal distribution.
        scale : float, default=1.0
            The standard deviation of the normal distribution.
        dim : int | None, default=None
            The dimension of the random vector. If `None`, then the index of the random vector must be provided at construction, and the dimension is inferred from the length of the index.
        random_state : int | None, default=None
            An optional seed for the random number generator to ensure reproducibility. If `None`, the random number generator is not seeded.
        """
        if not isinstance(loc, Real) or not isinstance(scale, Real):
            raise TypeError("loc and scale must be real numbers.")
        if scale <= 0:
            raise ValueError("scale must be positive.")
        if dim is not None and (not isinstance(dim, int) or dim <= 0):
            raise TypeError("dim must be a positive integer or None.")
        if dim is None and self.index is None:
            raise ValueError("If dim is None, index must be provided at construction.")
        if random_state is not None and not isinstance(random_state, int):
            raise TypeError("random_state must be an integer or None.")

        if dim is None:
            dim = len(self.index)

        rng = np.random.default_rng(random_state)
        arr = rng.normal(loc, scale, size=(len(self.domain.data), dim))
        return self.from_numpy(array=arr)

    # TODO: Update docstrings
    @classmethod
    def indicator_of(cls, event: Event, dim: int) -> RandomVector:
        """Create the indicator random vector of a given event of a given dimension.

        Parameters
        ----------
        event : Event
            The event for which the indicator random vector is to be created.
        dim : int
            The dimension of the indicator random vector.

        Raises
        ------
        TypeError
            If `event` is not an instance of `Event`, or if `dim` is not a positive integer.

        Returns
        -------
        indicator_rv : RandomVector
            The indicator random variable of the given event.
        """
        from ..base.event import Event

        if not isinstance(event, Event):
            raise TypeError("event must be an Event.")
        if not isinstance(dim, int) or dim <= 0:
            raise TypeError("dim must be a positive integer.")

        name = f"I_{event.name}" if event.name is not None else "indicator"

        outputs = {
            outcome: (1,) * dim if outcome in event else (0,) * dim
            for outcome in event.sample_space
        }
        return cls(domain=event.sample_space, name=name).from_dict(outputs)

    # --------------------- properties --------------------- #

    # TODO: Update docstrings
    @property
    def outputs(self) -> Mapping[Hashable, Hashable]:
        """Get the outputs mapping of the random vector.

        If not initialized in the `from_dict` method, lazily constructs the outputs mapping from the underlying pandas data structure.

        Returns
        -------
        outputs : Mapping[Hashable, Hashable]
            The mapping from sample points in the domain to their corresponding output vectors.
        """
        if self._outputs is None:
            if self._data is None:
                return None
            if isinstance(self.data, pd.Series):
                self._outputs = self.data.to_dict()
            else:
                self._outputs = self.data.apply(
                    lambda row: tuple(row), axis=1
                ).to_dict()
        return self._outputs

    # TODO: Update docstrings
    @property
    def data(self) -> pd.Series | pd.DataFrame:
        """Get the underlying pandas data structure of a random vector.

        If the random vector is of dimension 2 or greater, returns the underlying `pd.DataFrame`; otherwise, returns the underlying `pd.Series` for a random vector of dimension 1.

        If not initialized in the `from_pandas` method, lazily constructs the underlying pandas data structure from the outputs mapping.

        Returns
        -------
        data: pd.Series | pd.DataFrame
            The underlying `pd.Series` or `pd.DataFrame` representing the random vector.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace.generate_sequence(size=2, prefix="s")
        >>> outputs_2d = {"s_0": (1, 2), "s_1": (3, 4)}
        >>> X = RandomVector(domain=Omega, name="X").from_dict(outputs_2d)
        >>> # Dataframes underlie random vectors of dimension 2 or greater
        >>> X.data # doctest: +NORMALIZE_WHITESPACE
        feature  X_0  X_1
        sample
        s_0        1   2
        s_1        3   4
        >>> outputs_1d = {"s_0": 10, "s_1": 20}
        >>> Y = RandomVector(domain=Omega, name="Y").from_dict(outputs_1d)
        >>> # Series underlie random vectors of dimension 1
        >>> Y.data # doctest: +NORMALIZE_WHITESPACE
        sample
        s_0     10
        s_1     20
        Name: Y, dtype: int64
        """
        if self._data is None:
            if self._outputs is None:
                return None
            data = pd.DataFrame.from_dict(self._outputs, orient="index")
            dimension = data.shape[1]
            if dimension == 1:
                data = data.iloc[:, 0]
                data.name = self.name
            else:
                data.columns = self.index.data
            data.index.name = self.domain.data.name
            self._data = data
        return self._data

    # TODO: write unit tests for components property
    # TODO: Update docstrings
    @property
    def components(self) -> list[RandomVariable] | None:
        """Get the component random variables of the random vector, if the random vector has dimension 2 or greater.

        If the random vector has dimension 1, returns `None`.

        Returns
        -------
        component_random_variables : list[RandomVariable] | None
            A list of the component random variables of the random vector, or `None` if the random vector has dimension 1.
        """
        if self.dimension == 1:
            return None
        if self.data is None:
            raise ValueError(
                "Data must be initialized to get component random variables."
            )
        if self._components is None:
            self._components = [self.get_component_rv(idx) for idx in self.index.data]
        return self._components

    # TODO: Update docstrings
    @property
    def name(self) -> Hashable:
        """Get the name of the random vector.

        Returns
        -------
        name : Hashable
            The name of the random vector.
        """
        return self._name

    @name.setter
    def name(self, name: Hashable) -> None:
        if not isinstance(name, Hashable):
            raise TypeError("name must be a Hashable.")
        self._name = name
        if isinstance(self._data, pd.Series):
            self._data.name = name

    # TODO: Update docstrings
    def with_name(self, name: Hashable, modify_index: bool = False) -> RandomVector:
        """Set the name of the random vector and return self for chaining.

        Parameters
        ----------
        name : Hashable
            The new name for the random vector.
        modify_index : bool, default=True
            If `True` and the random vector has a feature index, also updates the feature index to reflect the new name of the random vector.

        Returns
        -------
        self : RandomVector
            Returns self to allow method chaining.
        """
        from ..base.index import Index

        self.name = name
        if modify_index and self.index is not None:
            prefix = name if isinstance(name, str) else None
            self.index = Index.generate_sequence(
                size=self.dimension,
                prefix=prefix,
                name="index",
                data_name="feature",
            )
        return self

    # TODO: Update docstrings
    @property
    def index(self) -> Index | None:
        """Get the index of the random vector.

        Returns
        -------
        index : Index | None
            The index of the random vector, or `None` if the random vector is 1-dimensional.
        """
        return self._index

    @index.setter
    def index(self, index: Index) -> None:
        from ..base.index import Index

        if not isinstance(index, Index):
            raise TypeError("index must be an Index.")
        if self._data is None:
            _ = self.data  # trigger lazy initialization of data
        self._index = index
        self._data.columns = index.data

    # TODO: Update docstrings
    @property
    def sigma_algebra(self) -> SigmaAlgebra:
        """Get the sigma-algebra induced by the random vector.

        Returns
        -------
        sigma_algebra : SigmaAlgebra
            The sigma-algebra induced by the random vector.

        Examples
        --------
        >>> from sigalg.core import (
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> domain = SampleSpace.generate_sequence(size=3, prefix="s")
        >>> X = RandomVector(domain=domain).from_dict(
        ...     outputs={"s_0": (1, 2), "s_1": (3, 4), "s_2": (3, 4)},
        ... )
        >>> sigma_algebra = SigmaAlgebra.from_random_vector(X)
        >>> sigma_algebra # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'sigma(X)':
               atom ID
        sample
        s_0      (1, 2)
        s_1      (3, 4)
        s_2      (3, 4)
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if self._sigma_algebra is None:
            self._sigma_algebra = SigmaAlgebra.from_random_vector(self)
        return self._sigma_algebra

    # TODO: Update docstrings
    @property
    def probability_measure(self) -> ProbabilityMeasure | None:
        """Get the probability measure on the domain of the random vector, if set.

        Returns
        -------
        probability_measure : ProbabilityMeasure | None
            The probability measure on the domain of the random vector, or `None` if not set.
        """
        from ..probability_measures.probability_measure import ProbabilityMeasure

        if self.domain is None:
            raise ValueError(
                "Cannot get probability measure without a domain sample space."
            )
        if self._probability_measure is None:
            self._probability_measure = ProbabilityMeasure.uniform(self.domain)
        return self._probability_measure

    @probability_measure.setter
    def probability_measure(self, probability_measure: ProbabilityMeasure) -> None:
        from ..probability_measures.probability_measure import ProbabilityMeasure

        if not isinstance(probability_measure, ProbabilityMeasure):
            raise TypeError("probability_measure must be a ProbabilityMeasure.")
        if self.domain is None:
            raise ValueError(
                "Cannot set probability measure without a domain sample space."
            )
        if probability_measure.sample_space != self.domain:
            raise ValueError(
                "The sample space of the probability measure must match the domain of the random vector."
            )
        self._probability_measure = probability_measure

    # TODO: Update docstrings
    @property
    def range(self) -> ProbabilitySpace:
        """Pass."""
        from ..base import SampleSpace
        from ..base.probability_space import ProbabilitySpace
        from ..probability_measures.probability_measure import ProbabilityMeasure

        if self._range is None:
            pushforward_data = pd.concat(
                [self.data, self.probability_measure.data], axis=1
            )
            pushforward_data = (
                pushforward_data.groupby(
                    pushforward_data.columns[: self.dimension].to_list()
                )
                .sum()
                .squeeze()
            )

            range_name = f"{self.name}_range" if isinstance(self.name, str) else "range"
            range_data = pushforward_data.index.to_flat_index()
            range_data.name = "output"
            range = SampleSpace(name=range_name).from_pandas(range_data)

            pushforward_name = (
                f"{self.probability_measure.name}_{self.name}"
                if (
                    isinstance(self.probability_measure.name, str)
                    and isinstance(self.name, str)
                )
                else "pushforward"
            )
            pushforward_data.index = range.data
            pushforward = ProbabilityMeasure(
                sample_space=range, name=pushforward_name
            ).from_pandas(pushforward_data)

            self._range = ProbabilitySpace(
                sample_space=range, probability_measure=pushforward
            )

        return self._range

    # # TODO: Update docstrings
    # @property
    # def range(self) -> RandomVector:
    #     """Get the range of the random vector.

    #     Mathematically, the range of a random vector `X:Omega -> S` is the set of all vectors `X(omega)`, as `omega` varies over the sample space `Omega`. In this implementation, the range is represented as another `RandomVector`, where the domain is a `SampleSpace` that indexes the unique output vectors of the original random vector, and the outputs are these unique vectors themselves.

    #     If the random vector has a string name (e.g., `X`), the range random vector is named `range(X)`, the domain of `range(X)` has indices `x0`, `x1`, etc., and the feature indices of `range(X)` match those of `X` itself. Otherwise, numerical indices are used.

    #     Returns
    #     -------
    #     range : RandomVector
    #         A `RandomVector` representing the range of the original random vector.

    #     Examples
    #     --------
    #     >>> from sigalg.core import SampleSpace, RandomVector
    #     >>> import pandas as pd
    #     >>> outputs = {"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (3, 4)}
    #     >>> domain = SampleSpace.generate_sequence(size=3)
    #     >>> X = RandomVector(domain=domain, name="X").from_dict(outputs)
    #     >>> pd.concat([X.range.data, X.range_counts.rename("counts")], axis=1) # doctest: +NORMALIZE_WHITESPACE
    #             X_0  X_1  counts
    #     output
    #     x_0       1   2       1
    #     x_1       3   4       2
    #     """
    #     from ..base import SampleSpace
    #     from ..probability_measures.probability_measure import ProbabilityMeasure

    #     if self._range is None:
    #         if isinstance(self.data, pd.Series):
    #             data = self.data.to_frame().copy()
    #         else:
    #             data = self.data.copy()
    #         cols = data.columns.tolist()
    #         ones = pd.Series(1, index=self.domain.data, name="count")
    #         outputs_probs_counts = (
    #             pd.concat([data, self.probability_measure.data, ones], axis=1)
    #             .groupby(cols)
    #             .sum()
    #         ).reset_index()

    #         range_name = f"range({self.name})" if isinstance(self.name, str) else None
    #         prefix = self.name.lower() if isinstance(self.name, str) else None
    #         range_sample_space = SampleSpace.generate_sequence(
    #             size=len(outputs_probs_counts),
    #             prefix=prefix,
    #             name=range_name,
    #             data_name="output",
    #         )
    #         outputs_probs_counts.index = range_sample_space.data

    #         self._range_counts = outputs_probs_counts["count"]
    #         outputs_probs = outputs_probs_counts.drop(columns=["count"])

    #         prob_measure_name = f"P_{self.name}" if isinstance(self.name, str) else None
    #         range_probability_measure = ProbabilityMeasure(
    #             sample_space=range_sample_space,
    #             name=prob_measure_name,
    #         ).from_pandas(outputs_probs["probability"])
    #         outputs = outputs_probs.drop(columns=["probability"])

    #         if outputs.shape[1] == 1:
    #             outputs = outputs.iloc[:, 0].rename(self.name)

    #         range_name = f"{self.name}_range" if isinstance(self.name, str) else None

    #         self._range = (
    #             RandomVector(
    #                 domain=range_sample_space,
    #                 name=range_name,
    #                 index=self.index,
    #             )
    #             .from_pandas(data=outputs)
    #             .with_probability_measure(probability_measure=range_probability_measure)
    #         )

    #     return self._range

    # # TODO: Update docstrings
    # @property
    # def range_counts(self) -> pd.Series:
    #     """Get the counts of each unique output in the range.

    #     This property pairs with the `range` property to identify and provide the frequency of each unique output vector in the random vector's mapping. The dataframe `range.data` contains the unique output vectors, while `range_counts` provides the corresponding counts as an index-aligned `pd.Series`.

    #     Returns
    #     -------
    #     range_counts : pd.Series
    #         A `pd.Series` where the index identifies the unique output vectors in the range, and the values represent the counts of each output vector in the original random vector.

    #     Examples
    #     --------
    #     >>> from sigalg.core import SampleSpace, RandomVector
    #     >>> import pandas as pd
    #     >>> outputs = {"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (3, 4)}
    #     >>> domain = SampleSpace.generate_sequence(size=3)
    #     >>> X = RandomVector(domain=domain, name="X").from_dict(outputs=outputs)
    #     >>> pd.concat([X.range.data, X.range_counts.rename("counts")], axis=1) # doctest: +NORMALIZE_WHITESPACE
    #             X_0  X_1  counts
    #     output
    #     x_0       1   2       1
    #     x_1       3   4       2
    #     """
    #     if self._range_counts is None:
    #         _ = self.range  # triggers computation of range and counts
    #     return self._range_counts

    # TODO: Update docstrings
    def iter_features(self):
        r"""Iterate over sample points and their feature vectors.

        Yields tuples of `(sample_index, FeatureVector)` for each sample point in the domain, allowing iteration over the random vector's entire domain.

        Yields
        ------
        sample_index : Hashable
            Index of the sample point.
        features : FeatureVector
            Feature vector of the sample point.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace.generate_sequence(size=2, prefix="s")
        >>> X = RandomVector(domain=Omega).from_dict(outputs={"s_0": (1, 2), "s_1": (3, 4)})
        >>> for _, features in X.iter_features():
        ...     print(features) # doctest: +NORMALIZE_WHITESPACE
        Feature vector of 's_0':
                 s_0
        feature
        X_0        1
        X_1        2
        Feature vector of 's_1':
                 s_1
        feature
        X_0        3
        X_1        4
        >>> Y = RandomVector(domain=Omega, name="Y").from_dict(outputs={"s_0": 1, "s_1": 2})
        >>> for idx, features in Y.iter_features():
        ...     print(f"Feature of {idx}: ", features)
        Feature of s_0:  1
        Feature of s_1:  2
        """
        for sample_index in self.data.index:
            yield sample_index, self(sample_index)

    # --------------------- sigma-algebra methods --------------------- #

    # TODO: Update docstrings
    def is_measurable(self, sigma_algebra: SigmaAlgebra) -> bool:
        """Check if the random vector is measurable with respect to a given sigma-algebra.

        Parameters
        ----------
        sigma_algebra : SigmaAlgebra
            The sigma-algebra on the domain sample space.

        Returns
        -------
        is_measurable : bool
            `True` if the random vector is measurable with respect to the given sigma-algebra, `False` otherwise.

        Examples
        --------
        >>> from sigalg.core import (
        ...     RandomVector,
        ...     SampleSpace,
        ...     SigmaAlgebra,
        ... )
        >>> domain = SampleSpace.generate_sequence(size=4, prefix="s", name="S")
        >>> X = RandomVector(domain=domain, name="X").from_dict(
        ...     outputs={"s_0": (1, 2), "s_1": (3, 4), "s_2": (3, 4), "s_3": (3, 4)},
        ... )
        >>> Y = RandomVector(domain=domain, name="Y").from_dict(
        ...     outputs={"s_0": "a", "s_1": "b", "s_2": "c", "s_3": "d"},
        ... )
        >>> F = SigmaAlgebra(sample_space=domain).from_dict(
        ...     {"s_0": 0, "s_1": 1, "s_2": 1, "s_3": 2},
        ... )
        >>> print(X.is_measurable(F))
        True
        >>> print(Y.is_measurable(F))
        False
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(sigma_algebra, SigmaAlgebra):
            raise TypeError("sigma_algebra must be a SigmaAlgebra.")

        return self.sigma_algebra <= sigma_algebra

    # --------------------- probability methods --------------------- #

    # TODO: Update docstrings
    def with_probability_measure(
        self,
        probabilities: Mapping[Hashable, Real] | None = None,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> RandomVector:
        """Set the probability measure on the domain of the random vector and return self for chaining.

        The user can provide either a `probability_measure` or a `probabilities` mapping, but not both. If a `probabilities` mapping is provided, it is used to construct a `ProbabilityMeasure` on the domain of the random vector.

        Parameters
        ----------
        probabilities : Mapping[Hashable, Real] | None, default=None
            A mapping from sample points in the domain to their corresponding probabilities. If given, this is used to construct a `ProbabilityMeasure` on the domain of the random vector.
        probability_measure : ProbabilityMeasure | None, default=None
            The probability measure to set on the domain of the random vector.
        """
        from ..probability_measures.probability_measure import ProbabilityMeasure

        if probabilities is not None and probability_measure is not None:
            raise ValueError(
                "Cannot specify both probabilities and probability_measure."
            )
        if probabilities is None and probability_measure is None:
            probability_measure = ProbabilityMeasure.uniform(self.domain)
        if probabilities is not None:
            probability_measure = ProbabilityMeasure(
                sample_space=self.domain
            ).from_dict(probabilities)
        self.probability_measure = probability_measure
        return self

    # --------------------- data access --------------------- #

    def __call__(
        self, key: Hashable | list[Hashable] | Event
    ) -> Hashable | FeatureVector | RandomVector:
        """Call a `RandomVector` on a sample point to get features, or call on multiple sample points to get the restrition of the `RandomVector`.

        As a function `X:Omega -> S`, a `RandomVector` can be called on a sample point `omega` in its domain `Omega` to get the corresponding feature vector `X(omega)`. If called on a list of sample points or an `Event` instance `A`, it returns a new `RandomVector` representing the restriction `X|A:A -> S`.

        Parameters
        ----------
        key : Hashable | list[Hashable] | Event
            A sample point in the domain, a list of sample points, or an `Event` instance.

        Raises
        ------
        TypeError
            If `key` is not a `Hashable`, list of `Hashable`, or `Event`.
        KeyError
            If any sample point in `key` is not found in the domain.
        ValueError
            If `key` is an `Event` whose sample space does not match the `RandomVector`'s domain.

        Returns
        -------
        features : Hashable | FeatureVector | RandomVector
            If `key` is a single sample point, returns the corresponding feature vector as a `Hashable` or `FeatureVector`. If `key` is a list of sample points or an `Event`, returns a new `RandomVector` restricted to those sample points.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, RandomVector
        >>> domain = SampleSpace.generate_sequence(size=3, prefix="s")
        >>> outputs = {"s_0": (1, 2), "s_1": (3, 4), "s_2": (5, 6)}
        >>> X = RandomVector(domain=domain, name="X").from_dict(outputs)
        >>> # Get features for a single sample point
        >>> X("s_0") # doctest: +NORMALIZE_WHITESPACE
        Feature vector of 's_0':
                s_0
        feature
        X_0       1
        X_1       2
        >>> # Get the restriction of X to an event
        >>> A = domain.get_event(["s_0", "s_2"])
        >>> X_A = X(A)
        >>> X_A # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X|A':
        feature  X_0  X_1
        sample
        s_0         1    2
        s_2         5    6
        """
        from ..base.event import Event
        from ..base.feature_vector import FeatureVector

        if not isinstance(key, (Hashable, list, Event)):
            raise TypeError("key must be a Hashable, list, or Event.")
        if isinstance(key, Hashable) and not isinstance(key, (list, Event)):
            if key not in self.domain:
                raise KeyError(f"Sample '{key}' not found in domain.")
            result = self.data.loc[key]
            if not isinstance(result, pd.Series):
                return result
            else:
                return FeatureVector(name=key).from_pandas(data=result)
        if isinstance(key, list):
            invalid_indices = [k for k in key if k not in self.domain.data]
            if invalid_indices:
                raise KeyError(f"Samples {invalid_indices} not found in domain.")
            name = f"{self.name}|event" if self.name is not None else None
            return RandomVector(name=name).from_pandas(data=self.data.loc[key])
        if isinstance(key, Event):
            if key.sample_space != self.domain:
                raise ValueError(
                    "Event's sample_space must match RandomVector's domain."
                )
            name = (
                f"{self.name}|{key.name}"
                if (self.name is not None and key.name is not None)
                else None
            )
            return RandomVector(name=name).from_pandas(data=self.data.loc[key.indices])

    # TODO: Update docstrings
    def get_sub_vector(self, feature_indices: list[Hashable]) -> RandomVector:
        """Get a sub-vector of the random vector by selecting specific feature indices.

        Parameters
        ----------
        feature_indices : list[Hashable]
            List of feature indices to select for the sub-vector.

        Returns
        -------
        sub_vector : RandomVector
            A new `RandomVector` containing only the specified feature indices.

        Raises
        ------
        ValueError
            If any feature index is not found.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> domain = SampleSpace.generate_sequence(size=2, prefix="s")
        >>> outputs = {"s_0": (1, 2, 3), "s_1": (4, 5, 6)}
        >>> X = RandomVector(domain=domain).from_dict(outputs)
        >>> X_sub = X.get_sub_vector(feature_indices=["X_0", "X_2"])
        >>> X_sub # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X_sub':
        feature  X_0  X_2
        sample
        s_0        1    3
        s_1        4    6
        """
        if self.dimension == 1:
            raise ValueError("Cannot get sub-vector of a 1-dimensional RandomVector.")
        invalid_features = [fi for fi in feature_indices if fi not in self.index]
        if invalid_features:
            raise ValueError(f"Feature indices {invalid_features} not found.")
        sub_data = self.data[feature_indices]
        return RandomVector(
            domain=self.domain,
            name=f"{self.name}_sub" if self.name is not None else None,
        ).from_pandas(data=sub_data)

    # TODO: Update docstrings
    def get_component_rv(self, index: Hashable) -> RandomVariable:
        """Get a component random variable corresponding to a specific feature index.

        Parameters
        ----------
        index : Hashable
            The feature index for which to get the component random variable.

        Returns
        -------
        component_rv : RandomVariable
            A new `RandomVariable` representing the component random variable.

        Raises
        ------
        ValueError
            If the feature index is not found.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> domain = SampleSpace.generate_sequence(size=2, prefix="s")
        >>> outputs = {"s_0": (1, 2), "s_1": (3, 4)}
        >>> X = RandomVector(domain=domain).from_dict(outputs)
        >>> X_component = X.get_component_rv("X_1")
        >>> X_component # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_1':
               X_1
        sample
        s_0     2
        s_1     4
        """
        component_rv = self.get_sub_vector([index]).to_random_variable()
        component_rv.name = index
        return component_rv.with_probability_measure(
            probability_measure=self.probability_measure
        )

    # TODO: Update docstrings
    def item(self) -> Hashable:
        """Get the single output value of a 1-dimensional `RandomVector` with exactly one sample point.

        Returns
        -------
        output : Hashable
            The single output value of the random vector.

        Raises
        ------
        ValueError
            If the random vector does not have exactly one sample point or is not 1-dimensional.
        """
        if self.dimension != 1:
            raise ValueError(
                "item() can only be called on a 1-dimensional RandomVector."
            )
        if self.data.nunique() != 1:
            raise ValueError(
                "item() can only be called on a RandomVector with exactly one output value."
            )
        return self.data.iloc[0]

    # --------------------- conversion methods --------------------- #

    # TODO: Update docstrings
    def to_random_variable(self) -> RandomVariable:
        """Convert a 1-dimensional `RandomVector` to a `RandomVariable`.

        Returns
        -------
        rv : RandomVariable
            The converted `RandomVariable`.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> domain = SampleSpace.generate_sequence(size=2, prefix="s")
        >>> outputs = {"s_0": 10, "s_1": 20}
        >>> X = RandomVector(domain=domain, name="X").from_dict(outputs=outputs)
        >>> X_var = X.to_random_variable()
        >>> X_var # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X':
                X
        sample
        s_0    10
        s_1    20
        """
        from .random_variable import RandomVariable

        if self.dimension != 1:
            raise ValueError(
                "Can only convert a 1-dimensional RandomVector to RandomVariable."
            )

        return (
            RandomVariable(domain=self.domain, name=self.name)
            .from_pandas(self.data)
            .with_probability_measure(probability_measure=self.probability_measure)
        )

    # --------------------- apply methods --------------------- #

    # TODO: Update docstrings
    def apply_to_features(
        self, function: Callable[[FeatureVector | Hashable], any]
    ) -> pd.Series:
        """Apply a function to the feature vector of each sample point.

        Applies the given function to each sample point's feature vector,
        returning a `pd.Series` of results indexed by sample points.

        Parameters
        ----------
        function : Callable[[FeatureVector | Hashable], any]
            Function that takes a `FeatureVector` object (in dimension > 1) or a `Hashable` (in dimension 1) and returns a value.

        Returns
        -------
        results : pd.Series
            Series of function results indexed by sample points.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace.generate_sequence(size=2, prefix="s")
        >>> X = RandomVector(domain=Omega, name="X").from_dict(outputs={"s_0": (1, 2), "s_1": (3, 4)})
        >>> X.apply_to_features(lambda f: f.sum() + 2) # doctest: +NORMALIZE_WHITESPACE
        sample
        s_0    5
        s_1    9
        dtype: int64
        >>> Y = RandomVector(domain=Omega, name="Y").from_dict(outputs={"s_0": 5, "s_1": 10})
        >>> Y.apply_to_features(lambda x: x * 2) # doctest: +NORMALIZE_WHITESPACE
        sample
        s_0    10
        s_1    20
        Name: Y, dtype: int64
        """
        from ..base.feature_vector import FeatureVector

        if self.dimension > 1:

            def wrapper(row):
                sp = FeatureVector().from_pandas(data=row)
                return function(sp)

            return self.data.apply(wrapper, axis=1)
        else:
            return self.data.apply(function)

    # TODO: Update docstrings
    def apply(
        self, function: Callable[[Hashable | FeatureVector], Hashable]
    ) -> RandomVector:
        """Apply a function to the feature vector of each sample point, returning a new `RandomVector`.

        Parameters
        ----------
        function : Callable[[Hashable | FeatureVector], Hashable]
            Function that takes a `FeatureVector` object (in dimension > 1) or a `Hashable` (in dimension 1) and returns a new output value.

        Returns
        -------
        new_rv : RandomVector
            A new `RandomVector` with outputs given by applying the function to each sample point's feature vector.
        """
        new_outputs = self.apply_to_features(function=function).to_dict()
        new_name = f"f({self.name})" if self.name is not None else None
        return RandomVector(domain=self.domain, name=new_name).from_dict(
            outputs=new_outputs
        )

    # --------------------- equality --------------------- #

    def __eq__(self, other: RandomVector, rtol=1e-5, atol=1e-8) -> bool:
        """Check equality with another random vector.

        Two random vectors `X` and `Y` are equal if they have the same domain and `X(omega) = Y(omega)` for all `omega` in the domain.

        Parameters
        ----------
        other : RandomVector
            Another random vector to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the other object is a `RandomVector` with the same domain, feature index, and data.
        """
        if not isinstance(other, RandomVector):
            return False
        if not self.domain == other.domain:
            return False
        return np.allclose(
            self.data.to_numpy(), other.data.to_numpy(), rtol=rtol, atol=atol
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
            return f"Random vector:\n{data}"
        else:
            return f"Random vector '{self.name}':\n{data}"

    # TODO: Update docstrings
    def print_values_and_probabilities(self):
        """Print the values of the random vector and their corresponding probabilities."""
        if self._data is None:
            raise ValueError(
                "Data must be generated before printing values and probabilities."
            )

        values_and_probs = pd.concat([self.data, self.probability_measure.data], axis=1)
        print(values_and_probs)

    # --------------------- arithmetic operations --------------------- #

    @staticmethod
    def _type(x):
        from ...processes.base.stochastic_process import StochasticProcess

        if isinstance(x, Real):
            return "Number"
        elif isinstance(x, StochasticProcess):
            return "StochasticProcess"
        else:
            return type(x).__name__

    def _apply_operation(
        self,
        other: RandomVector | Real,
        operation: Callable,
        op_symbol: str,
        reverse: bool = False,
    ) -> RandomVector:
        """Apply a binary operation to this random vector.

        Parameters
        ----------
        self : RandomVector
            The left operand (or right if reverse=True).
        other : RandomVector | Real
            The right operand (or left if reverse=True).
        operation : Callable
            The pandas operation to apply (e.g., `lambda a, b: a + b`).
        op_symbol : str
            Symbol representing the operation (e.g., '+', '-', '*').
        reverse : bool, default=False
            Whether this is a reverse operation (e.g., __radd__ vs __add__).

        Returns
        -------
        result : RandomVector
            A new random vector representing the result of the operation.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector` or a scalar.
        ValueError
            If operating on two `RandomVector` instances with different domains or dimensions.
        """
        from ...processes.base.stochastic_process import StochasticProcess
        from ..base.index import Index
        from .random_variable import RandomVariable

        types = {self._type(self), self._type(other)}

        if types == {"RandomVariable"}:
            if self.domain != other.domain:
                raise ValueError(
                    f"Cannot {op_symbol} RandomVariables with different domains."
                )

            if reverse:
                new_values = operation(other.data, self.data)
                new_name = (
                    f"({other.name}{op_symbol}{self.name})"
                    if self.name is not None
                    else None
                )
            else:
                new_values = operation(self.data, other.data)
                new_name = (
                    f"({self.name}{op_symbol}{other.name})"
                    if self.name is not None
                    else None
                )

            result = (
                RandomVariable(name=new_name)
                .from_pandas(data=new_values)
                .with_probability_measure(probability_measure=self.probability_measure)
            )
            result.data.name = new_name

            return result

        elif types == {"StochasticProcess"}:
            if self.domain != other.domain:
                raise ValueError(
                    f"Cannot {op_symbol} StochasticProcesses with different domains."
                )

            if len(self) != len(other):
                raise ValueError(
                    "The length of the StochasticProcesses must be the same."
                )

            if len(self) > 1 and not self.data.columns.equals(other.data.columns):
                raise ValueError(
                    "The time indices of the StochasticProcesses must be the same"
                )

            if reverse:
                new_values = operation(other.data, self.data)
                new_name = (
                    f"({other.name}{op_symbol}{self.name})"
                    if self.name is not None and other.name is not None
                    else None
                )
            else:
                new_values = operation(self.data, other.data)
                new_name = (
                    f"({self.name}{op_symbol}{other.name})"
                    if self.name is not None and other.name is not None
                    else None
                )

            result = (
                StochasticProcess(
                    domain=self.domain,
                    name=new_name,
                    time=self.time,
                    is_discrete_state=self.is_discrete_state,
                )
                .from_pandas(data=new_values)
                .with_probability_measure(probability_measure=self.probability_measure)
            )

            return result

        elif types == {"RandomVector"}:
            if self.domain != other.domain:
                raise ValueError(
                    f"Cannot {op_symbol} RandomVectors with different domains."
                )
            if self.dimension != other.dimension:
                raise ValueError("The dimension of the RandomVectors must be the same.")

            self_data = self.data.copy()
            other_data = other.data.copy()
            if self.dimension > 1:
                self_data.columns = pd.RangeIndex(self.dimension)
                other_data.columns = pd.RangeIndex(other.dimension)

            if reverse:
                new_values = operation(other_data, self_data)
                new_name = (
                    f"({other.name}{op_symbol}{self.name})"
                    if self.name is not None
                    else None
                )
            else:
                new_values = operation(self_data, other_data)
                new_name = (
                    f"({self.name}{op_symbol}{other.name})"
                    if self.name is not None
                    else None
                )

            result = (
                RandomVector(name=new_name)
                .from_pandas(data=new_values)
                .with_probability_measure(probability_measure=self.probability_measure)
            )

            new_index = Index(data_name="feature").from_sequence(
                size=self.dimension, prefix=new_name
            )
            result.data.columns = new_index
            result.data.columns.name = "feature"

            return result

        elif types == {"Number", "RandomVariable"}:
            if reverse:
                new_values = operation(other, self.data)
                new_name = (
                    f"({other}{op_symbol}{self.name})"
                    if self.name is not None
                    else None
                )
            else:
                new_values = operation(self.data, other)
                new_name = (
                    f"({self.name}{op_symbol}{other})"
                    if self.name is not None
                    else None
                )

            result = (
                RandomVariable(name=new_name)
                .from_pandas(data=new_values)
                .with_probability_measure(probability_measure=self.probability_measure)
            )
            result.data.name = new_name

            return result

        elif types == {"Number", "StochasticProcess"}:
            if reverse:
                new_values = operation(other, self.data)
                new_name = (
                    f"({other}{op_symbol}{self.name})"
                    if self.name is not None
                    else None
                )
            else:
                new_values = operation(self.data, other)
                new_name = (
                    f"({self.name}{op_symbol}{other})"
                    if self.name is not None
                    else None
                )

            result = (
                StochasticProcess(
                    domain=self.domain,
                    name=new_name,
                    time=self.time,
                    is_discrete_state=self.is_discrete_state,
                )
                .from_pandas(data=new_values)
                .with_probability_measure(probability_measure=self.probability_measure)
            )

            return result

        elif types == {"Number", "RandomVector"}:
            if reverse:
                new_values = operation(other, self.data)
                new_name = (
                    f"({other}{op_symbol}{self.name})"
                    if self.name is not None
                    else None
                )
            else:
                new_values = operation(self.data, other)
                new_name = (
                    f"({self.name}{op_symbol}{other})"
                    if self.name is not None
                    else None
                )

            result = (
                RandomVector(name=new_name)
                .from_pandas(data=new_values)
                .with_probability_measure(probability_measure=self.probability_measure)
            )

            new_index = Index(data_name="feature").from_sequence(
                size=self.dimension, prefix=new_name
            )
            result.data.columns = new_index
            result.data.columns.name = "feature"

            return result

        elif types == {"RandomVariable", "RandomVector"}:
            raise TypeError(f"Unsupported types for arithmetic operations: {types}")

        elif types == {"RandomVariable", "StochasticProcess"}:
            if self.domain != other.domain:
                raise ValueError(
                    f"Cannot {op_symbol} a RandomVariable with a StochasticProcess with different domains."
                )
            if self.probability_measure != other.probability_measure:
                raise ValueError(
                    f"Cannot {op_symbol} a RandomVariable with a StochasticProcess with different probability measures."
                )

            if self._type(self) == "RandomVariable":
                if reverse:
                    new_values = operation(other.data, self.data.values.reshape(-1, 1))
                    new_name = (
                        f"({other.name}{op_symbol}{self.name})"
                        if self.name is not None and other.name is not None
                        else None
                    )
                else:
                    new_values = operation(self.data.values.reshape(-1, 1), other.data)
                    new_name = (
                        f"({self.name}{op_symbol}{other.name})"
                        if self.name is not None and other.name is not None
                        else None
                    )

                result = (
                    StochasticProcess(
                        domain=self.domain, name=new_name, time=other.time
                    )
                    .from_pandas(data=new_values)
                    .with_probability_measure(
                        probability_measure=other.probability_measure
                    )
                )
            else:
                if reverse:
                    new_values = operation(other.data.values.reshape(-1, 1), self.data)
                    new_name = (
                        f"({other.name}{op_symbol}{self.name})"
                        if self.name is not None and other.name is not None
                        else None
                    )
                else:
                    new_values = operation(self.data, other.data.values.reshape(-1, 1))
                    new_name = (
                        f"({self.name}{op_symbol}{other.name})"
                        if self.name is not None and other.name is not None
                        else None
                    )

                result = (
                    StochasticProcess(domain=self.domain, name=new_name, time=self.time)
                    .from_pandas(data=new_values)
                    .with_probability_measure(
                        probability_measure=self.probability_measure
                    )
                )

            return result

        elif types == {"RandomVector", "StochasticProcess"}:
            raise TypeError(f"Unsupported types for arithmetic operations: {types}")

        else:
            raise TypeError(f"Unsupported types for arithmetic operations: {types}")

    def __add__(self, other: RandomVector | Real) -> RandomVector:
        """Add another random vector or a scalar to this random vector."""
        return self._apply_operation(other, lambda a, b: a + b, "+")

    def __radd__(self, other: RandomVector | Real) -> RandomVector:
        """Add another random vector or a scalar to this random vector (right-hand side)."""
        return self._apply_operation(other, lambda a, b: a + b, "+", reverse=True)

    def __sub__(self, other: RandomVector | Real) -> RandomVector:
        """Subtract another random vector or a scalar from this random vector."""
        return self._apply_operation(other, lambda a, b: a - b, "-")

    def __rsub__(self, other: RandomVector | Real) -> RandomVector:
        """Subtract this random vector from another random vector or a scalar (right-hand side)."""
        return self._apply_operation(other, lambda a, b: a - b, "-", reverse=True)

    def __mul__(self, other: RandomVector | Real) -> RandomVector:
        """Multiply this random vector by another random vector or a scalar."""
        return self._apply_operation(other, lambda a, b: a * b, "*")

    def __rmul__(self, other: RandomVector | Real) -> RandomVector:
        """Multiply another random vector or a scalar by this random vector (right-hand side)."""
        return self._apply_operation(other, lambda a, b: a * b, "*", reverse=True)

    def __truediv__(self, other: RandomVector | Real) -> RandomVector:
        """Divide this random vector by another random vector or a scalar."""
        return self._apply_operation(other, lambda a, b: a / b, "/")

    def __rtruediv__(self, other: RandomVector | Real) -> RandomVector:
        """Divide another random vector or a scalar by this random vector (right-hand side)."""
        return self._apply_operation(other, lambda a, b: a / b, "/", reverse=True)

    def __pow__(self, other: RandomVector | Real) -> RandomVector:
        """Exponentiate this random vector by another random vector or a scalar."""
        return self._apply_operation(other, lambda a, b: a**b, "**")

    def __rpow__(self, other: RandomVector | Real) -> RandomVector:
        """Exponentiate another random vector or a scalar by this random vector (right-hand side)."""
        return self._apply_operation(other, lambda a, b: a**b, "**", reverse=True)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Override NumPy ufuncs to operate on RandomVector instances.

        Parameters
        ----------
        ufunc : numpy.ufunc
            The ufunc object that was called.
        method : str
            A string indicating which ufunc method was called (e.g., '__call__', 'reduce', etc.).
        inputs : tuple
            A tuple of the input arguments to the ufunc.
        kwargs : dict
            A dictionary of keyword arguments passed to the ufunc.

        Returns
        -------
        result : RandomVector | StochasticProcess | RandomVariable
            A new instance of `RandomVector`, `StochasticProcess`, or `RandomVariable`
            containing the result of applying the ufunc to the inputs.
        """
        from ...processes.base.stochastic_process import StochasticProcess
        from .random_variable import RandomVariable

        if method != "__call__":
            return NotImplemented

        new_inputs = [
            input._data if isinstance(input, RandomVector) else input
            for input in inputs
        ]
        result_data = getattr(ufunc, method)(*new_inputs, **kwargs)
        new_name = f"{ufunc.__name__}({self.name})" if self.name is not None else None

        if isinstance(self, StochasticProcess):
            return (
                StochasticProcess(domain=self.domain, name=new_name, time=self.time)
                .from_pandas(data=result_data)
                .with_probability_measure(probability_measure=self.probability_measure)
            )
        elif isinstance(self, RandomVariable):
            result = (
                RandomVariable(domain=self.domain, name=new_name)
                .from_pandas(data=result_data)
                .with_probability_measure(probability_measure=self.probability_measure)
            )
            result.data.name = new_name
            return result
        else:
            return (
                RandomVector(domain=self.domain, name=new_name)
                .from_pandas(data=result_data)
                .with_probability_measure(probability_measure=self.probability_measure)
            )

    # --------------------- comparison methods --------------------- #

    def __bool__(self) -> bool:
        """Prevent ambiguous boolean conversion of RandomVector.

        Raises
        ------
        ValueError
            Always raised to prevent ambiguous boolean evaluation.
            Use explicit methods like .all() or .any() instead.
        """
        raise ValueError(
            "The truth value of a RandomVector is ambiguous. "
            "Use .all() or .any() methods, or check specific conditions explicitly."
        )

    # TODO: add examples to docstrings
    def all(self) -> bool:
        """Check if all values in the RandomVector are True.

        This method is typically used after a comparison operation to verify
        that the comparison holds for all sample points and all components.

        Returns
        -------
        all_true : bool
        True if all values across all samples and features are True.
        """
        return bool(self.data.all().all() if self.dimension > 1 else self.data.all())

    # TODO: add examples to docstrings
    def any(self) -> bool:
        """Check if any value in the RandomVector is True.

        This method is typically used after a comparison operation to verify
        that the comparison holds for at least one sample point or component.

        Returns
        -------
        any_true : bool
            True if any value across all samples and features is True.
        """
        return bool(self.data.any().any() if self.dimension > 1 else self.data.any())

    # TODO: add examples to docstrings
    # TODO: add unit tests if `other` is scalar
    def __lt__(self, other: RandomVector | Real) -> RandomVector:
        """Check if this random vector is less than another random vector.

        Two random vectors `X=(X_1,...,X_n)` and `Y=(Y_1,...,Y_n)` are compared across sample points and features, resulting in a new random vector of booleans where each component is `True` if `X_i(omega) < Y_i(omega)` and `False` otherwise.

        Parameters
        ----------
        other : RandomVector | Real
            The random vector to compare with. If `other` is a scalar, it is treated as a constant random vector with the same domain and dimension.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector`.
        ValueError
            If the random vectors do not have the same domain or dimension.

        Returns
        -------
        is_lt: RandomVector
            A new `RandomVector` of booleans indicating where this random vector is less than the other random vector.
        """
        from ...core.base.index import Index
        from ...processes.base.stochastic_process import StochasticProcess
        from .random_variable import RandomVariable

        if not isinstance(other, RandomVector) and isinstance(other, Real):
            other = RandomVector(domain=self.domain, index=self.index).from_constant(
                constant=other
            )
        elif not isinstance(other, RandomVector):
            raise TypeError("other must be a RandomVector")
        if self.domain != other.domain:
            raise ValueError("Random vectors must have the same domain")
        if self.dimension != other.dimension:
            raise ValueError("Random vectors must have the same dimension")

        comparison_arr = self.data.to_numpy() < other.data.to_numpy()
        name = f"({self.name} < {other.name})" if self.name and other.name else None

        if isinstance(self, StochasticProcess):
            return StochasticProcess(
                domain=self.domain, name=name, time=self.time
            ).from_numpy(array=comparison_arr)
        elif isinstance(self, RandomVariable):
            result = RandomVariable(domain=self.domain, name=name).from_numpy(
                array=comparison_arr.flatten()
            )
            result.data.name = name
            return result
        else:
            result = RandomVector(domain=self.domain, name=name).from_numpy(
                array=comparison_arr
            )
            if name is not None:
                index = Index.generate_sequence(
                    size=self.dimension, prefix=name, data_name="feature"
                )
                result._index = index
                result.data.columns = index.data
            return result

    # TODO: add examples to docstrings
    # TODO: add unit tests if `other` is scalar
    def __le__(self, other: RandomVector | Real) -> RandomVector:
        """Check if this random vector is less than or equal to another random vector.

        Two random vectors `X=(X_1,...,X_n)` and `Y=(Y_1,...,Y_n)` are compared across sample points and features, resulting in a new random vector of booleans where each component is `True` if `X_i(omega) <= Y_i(omega)` and `False` otherwise.

        Parameters
        ----------
        other : RandomVector | Real
            The random vector to compare with. If `other` is a scalar, it is treated as a constant random vector with the same domain and dimension.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector`.
        ValueError
            If the random vectors do not have the same domain or dimension.

        Returns
        -------
        is_le: RandomVector
            A new `RandomVector` of booleans indicating where this random vector is less than or equal to the other random vector.
        """
        from ...core.base.index import Index
        from ...processes.base.stochastic_process import StochasticProcess
        from .random_variable import RandomVariable

        if not isinstance(other, RandomVector) and isinstance(other, Real):
            other = RandomVector(domain=self.domain, index=self.index).from_constant(
                constant=other
            )
        elif not isinstance(other, RandomVector):
            raise TypeError("other must be a RandomVector")
        if self.domain != other.domain:
            raise ValueError("Random vectors must have the same domain")
        if self.dimension != other.dimension:
            raise ValueError("Random vectors must have the same dimension")

        comparison_arr = self.data.to_numpy() <= other.data.to_numpy()
        name = f"({self.name} <= {other.name})" if self.name and other.name else None

        if isinstance(self, StochasticProcess):
            return StochasticProcess(
                domain=self.domain, name=name, time=self.time
            ).from_numpy(array=comparison_arr)
        elif isinstance(self, RandomVariable):
            result = RandomVariable(domain=self.domain, name=name).from_numpy(
                array=comparison_arr.flatten()
            )
            result.data.name = name
            return result
        else:
            result = RandomVector(domain=self.domain, name=name).from_numpy(
                array=comparison_arr
            )
            if name is not None:
                index = Index.generate_sequence(
                    size=self.dimension, prefix=name, data_name="feature"
                )
                result._index = index
                result.data.columns = index.data
            return result

    # TODO: add examples to docstrings
    # TODO: add unit tests if `other` is scalar
    def __gt__(self, other: RandomVector | Real) -> RandomVector:
        """Check if this random vector is greater than another random vector.

        Two random vectors `X=(X_1,...,X_n)` and `Y=(Y_1,...,Y_n)` are compared across sample points and features, resulting in a new random vector of booleans where each component is `True` if `X_i(omega) > Y_i(omega)` and `False` otherwise.

        Parameters
        ----------
        other : RandomVector | Real
            The random vector to compare with. If `other` is a scalar, it is treated as a constant random vector with the same domain and dimension.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector`.
        ValueError
            If the random vectors do not have the same domain or dimension.

        Returns
        -------
        is_gt: RandomVector
             A new `RandomVector` of booleans indicating where this random vector is greater than the other random vector.
        """
        from ...core.base.index import Index
        from ...processes.base.stochastic_process import StochasticProcess
        from .random_variable import RandomVariable

        if not isinstance(other, RandomVector) and isinstance(other, Real):
            other = RandomVector(domain=self.domain, index=self.index).from_constant(
                constant=other
            )
        elif not isinstance(other, RandomVector):
            raise TypeError("other must be a RandomVector")
        if self.domain != other.domain:
            raise ValueError("Random vectors must have the same domain")
        if self.dimension != other.dimension:
            raise ValueError("Random vectors must have the same dimension")

        comparison_arr = self.data.to_numpy() > other.data.to_numpy()
        name = f"({self.name} > {other.name})" if self.name and other.name else None

        if isinstance(self, StochasticProcess):
            return StochasticProcess(
                domain=self.domain, name=name, time=self.time
            ).from_numpy(array=comparison_arr)
        elif isinstance(self, RandomVariable):
            result = RandomVariable(domain=self.domain, name=name).from_numpy(
                array=comparison_arr.flatten()
            )
            result.data.name = name
            return result
        else:
            result = RandomVector(domain=self.domain, name=name).from_numpy(
                array=comparison_arr
            )
            if name is not None:
                index = Index.generate_sequence(
                    size=self.dimension, prefix=name, data_name="feature"
                )
                result._index = index
                result.data.columns = index.data
            return result

    # TODO: add examples to docstrings
    # TODO: add unit tests if `other` is scalar
    def __ge__(self, other: RandomVector | Real) -> RandomVector:
        """Check if this random vector is greater than or equal to another random vector.

        Two random vectors `X=(X_1,...,X_n)` and `Y=(Y_1,...,Y_n)` are compared across sample points and features, resulting in a new random vector of booleans where each component is `True` if `X_i(omega) >= Y_i(omega)` and `False` otherwise.

        Parameters
        ----------
        other : RandomVector | Real
            The random vector to compare with. If `other` is a scalar, it is treated as a constant random vector with the same domain and dimension.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector`.
        ValueError
            If the random vectors do not have the same domain or dimension.

        Returns
        -------
        is_ge: RandomVector
            A new `RandomVector` of booleans indicating where this random vector is greater than or equal to the other random vector.
        """
        from ...core.base.index import Index
        from ...processes.base.stochastic_process import StochasticProcess
        from .random_variable import RandomVariable

        if not isinstance(other, RandomVector) and isinstance(other, Real):
            other = RandomVector(domain=self.domain, index=self.index).from_constant(
                constant=other
            )
        elif not isinstance(other, RandomVector):
            raise TypeError("other must be a RandomVector")
        if self.domain != other.domain:
            raise ValueError("Random vectors must have the same domain")
        if self.dimension != other.dimension:
            raise ValueError("Random vectors must have the same dimension")

        comparison_arr = self.data.to_numpy() >= other.data.to_numpy()
        name = f"({self.name} >= {other.name})" if self.name and other.name else None

        if isinstance(self, StochasticProcess):
            return StochasticProcess(
                domain=self.domain, name=name, time=self.time
            ).from_numpy(array=comparison_arr)
        elif isinstance(self, RandomVariable):
            result = RandomVariable(domain=self.domain, name=name).from_numpy(
                array=comparison_arr.flatten()
            )
            result.data.name = name
            return result
        else:
            result = RandomVector(domain=self.domain, name=name).from_numpy(
                array=comparison_arr
            )
            if name is not None:
                index = Index.generate_sequence(
                    size=self.dimension, prefix=name, data_name="feature"
                )
                result._index = index
                result.data.columns = index.data
            return result
