"""A class representing a random vector."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ...validation.sample_space_mapping_in import SampleSpaceMappingIn
from .operators import OperatorsMethods

if TYPE_CHECKING:
    from ...processes.base.stochastic_process import StochasticProcess
    from ..base.event import Event
    from ..base.feature_vector import FeatureVector
    from ..base.index import Index
    from ..base.probability_space import ProbabilitySpace
    from ..base.sample_space import SampleSpace
    from ..probability_measures.probability_measure import ProbabilityMeasure
    from ..random_objects.random_variable import RandomVariable
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra


class RandomVector(OperatorsMethods):
    r"""A class representing a random vector.

    See the Notes section below for the mathematical details.

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
    >>> Omega = SampleSpace().from_sequence(size=3)
    >>> outputs = dict(zip(Omega, [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]))
    >>> # Generate a 2-dimensional random vector from outputs dict
    >>> X = RandomVector(domain=Omega, name="X").from_dict(outputs)
    >>> print(X) # doctest: +NORMALIZE_WHITESPACE
    Random vector 'X':
    feature  X_0  X_1
    sample
    0        0.1  0.2
    1        0.3  0.4
    2        0.5  0.6
    >>> # Generate a 1-dimensional random vector from a pd.Series
    >>> import pandas as pd
    >>> data = pd.Series([10, 20, 30])
    >>> Y = RandomVector(domain=Omega, name="Y").from_pandas(data)
    >>> Y # doctest: +NORMALIZE_WHITESPACE
    Random vector 'Y':
             Y
    sample
    0       10
    1       20
    2       30

    Notes
    -----
    See also the [notebook]() at the docs website.

    Given a probability space $(\Omega,\mathcal{F},P)$, a *random vector* is an $\mathcal{F}$-measurable function $X: \Omega \to \mathbb{R}^d$, where $d$ is the *dimension* of the vector and $\mathbb{R}^d$ is equipped with its Borel $\sigma$-algebra. The image $X(\omega)\in \mathbb{R}^d$ of a sample point $\omega \in \Omega$ is called a *feature vector*.

    An instance `X` of `RandomVector` is SigAlg's representation of a random vector $X$. Such an instance may be constructed with a `domain` parameter representing $\Omega$, and a dictionary parameter `outputs` representing the mapping $\omega \to X(\omega)$. (Other construction methods exist besides this canonical one.)

    The probability measure $P$ may be represented by setting the `probability_measure` attribute of `X` to an instance of `ProbabilityMeasure` after construction. If not set explicitly, this measure defaults to the uniform measure on $\Omega$.

    The $\sigma$-algebra $\mathcal{F}$ is not carried by the instance `X`. In particular, SigAlg does not enforce the measurability requirement for random vectors on construction. However, `X` does carry a method `is_measurable` for checking measurability after construction relative to an instance of `SigmaAlgebra`.
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
        self._range: ProbabilitySpace | None = None
        self._components: list[RandomVariable] | None = None

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
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> outputs = dict(zip(Omega, [(1, 2), (3, 4), (5, 6)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        feature  X_0  X_1
        sample
        0          1    2
        1          3    4
        2          5    6
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
                self._index = Index(name="index", data_name="feature").from_sequence(
                    size=self.dimension,
                    prefix=self.name,
                )
            if len(self._index) != self.dimension:
                raise ValueError(
                    "Length of index must match the dimension of the RandomVector."
                )
        else:
            self._index = None

        self._outputs = v.mapping

        return self

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
        >>> # Construct a random vector with no specified domain or index
        >>> arr = np.array([[1, 2], [3, 4], [5, 6]])
        >>> X = RandomVector().from_numpy(arr)
        >>> X # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
           0  1
        0  1  2
        1  3  4
        2  5  6
        >>> # Construct a random vector with specified domain and index
        >>> Omega = SampleSpace().from_sequence(size=3, prefix="omega")
        >>> index = Index().from_sequence(size=2, prefix="feature")
        >>> Y = RandomVector(domain=Omega, index=index, name="Y").from_numpy(arr)
        >>> print(Y) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
            feature_0  feature_1
        sample
        omega_0          1          2
        omega_1          3          4
        omega_2          5          6
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a numpy ndarray.")
        data = pd.DataFrame(
            array,
            index=self.domain.data if self.domain else None,
            columns=self.index.data if self.index else None,
        )
        return self.from_pandas(data=data)

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

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> # Construct a constant 2D random vector
        >>> X = RandomVector(domain=Omega).from_constant(constant=(1, 2))
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        feature  X_0  X_1
        sample
        0          1    2
        1          1    2
        2          1    2
        >>> # Construct a constant 1D random vector
        >>> Y = RandomVector(domain=Omega, name="Y").from_constant(2)
        >>> print(Y) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
                Y
        sample
        0       2
        1       2
        2       2
        """
        if self.domain is None:
            raise ValueError("Domain must be provided at construction.")
        if not isinstance(constant, Hashable):
            raise TypeError("constant must be a Hashable.")
        if self.index is not None:
            outputs = dict.fromkeys(self.domain.data, (constant,) * len(self.index))
            rv = self.from_dict(outputs=outputs)
            rv._index = self.index
            return rv
        else:
            outputs = dict.fromkeys(self.domain.data, constant)
            return self.from_dict(outputs=outputs)

    def from_randint(
        self,
        low: int,
        high: int,
        dim: int | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> RandomVector:
        """Generate a random vector with integer outputs uniformly sampled from the range [low, high).

        For this construction method, the `domain` must be provided at construction.

        Parameters
        ----------
        low : int
            The lower bound (inclusive) of the random integers.
        high : int
            The upper bound (exclusive) of the random integers.
        dim : int | None, default=None
            The dimension of the random vector. If `None`, then the index of the random vector must be provided at construction, and the dimension is inferred from the length of the index.
        random_state : int | np.random.Generator | None, default=None
            An optional seed (int) for the random number generator, or a `np.random.Generator` instance to use directly. If an integer is provided, a new generator is created with that seed. If a Generator is provided, it is used directly and its state is advanced. If `None`, the random number generator is not seeded.

        Raises
        ------
        ValueError
            If the domain is not provided at construction, or if `dim` is `None` and the index is not provided at construction.
        TypeError
            If `low` or `high` are not integers, or if `dim` is not a positive integer or `None`, or if `random_state` is not an integer, Generator, or `None`.

        Returns
        -------
        self : RandomVector
            A random vector with integer outputs uniformly sampled from the range [low, high).

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> X = RandomVector(domain=Omega).from_randint(low=0, high=5, dim=2, random_state=42)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
                0  1
        sample
        0       0  3
        1       3  2
        2       2  4
        """
        if self.domain is None:
            raise ValueError("Domain must be provided at construction.")
        if not isinstance(low, int) or not isinstance(high, int):
            raise TypeError("low and high must be integers.")
        if dim is not None and (not isinstance(dim, int) or dim <= 0):
            raise TypeError("dim must be a positive integer or None.")
        if dim is None and self.index is None:
            raise ValueError("If dim is None, index must be provided at construction.")
        if random_state is not None and not isinstance(
            random_state, (int, np.random.Generator)
        ):
            raise TypeError(
                "random_state must be an integer, np.random.Generator, or None."
            )

        if dim is None:
            dim = len(self.index)

        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        arr = rng.integers(low, high, size=(len(self.domain.data), dim))
        return self.from_numpy(array=arr)

    def from_randnorm(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        dim: int | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> RandomVector:
        """Generate a random vector with outputs sampled from a normal distribution with specified mean and standard deviation.

        For this construction method, the `domain` must be provided at construction.

        Parameters
        ----------
        loc : float, default=0.0
            The mean of the normal distribution.
        scale : float, default=1.0
            The standard deviation of the normal distribution.
        dim : int | None, default=None
            The dimension of the random vector. If `None`, then the index of the random vector must be provided at construction, and the dimension is inferred from the length of the index.
        random_state : int | np.random.Generator | None, default=None
            An optional seed (int) for the random number generator, or a `np.random.Generator` instance to use directly. If an integer is provided, a new generator is created with that seed. If a Generator is provided, it is used directly and its state is advanced. If `None`, the random number generator is not seeded.

        Raises
        ------
        ValueError
            If the domain is not provided at construction, or if `dim` is `None` and the index is not provided at construction.
        TypeError
            If `loc` or `scale` are not real numbers, or if `dim` is not a positive integer or `None`, or if `random_state` is not an integer, Generator, or `None`.

        Returns
        -------
        self : RandomVector
            A random vector with outputs sampled from a normal distribution with specified mean and standard deviation.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> X = RandomVector(domain=Omega).from_randnorm(dim=2, random_state=42)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
                    0         1
        sample
        0       0.304717 -1.039984
        1       0.750451  0.940565
        2      -1.951035 -1.302180
        """
        if self.domain is None:
            raise ValueError("Domain must be provided at construction.")
        if not isinstance(loc, Real) or not isinstance(scale, Real):
            raise TypeError("loc and scale must be real numbers.")
        if scale <= 0:
            raise ValueError("scale must be positive.")
        if dim is not None and (not isinstance(dim, int) or dim <= 0):
            raise TypeError("dim must be a positive integer or None.")
        if dim is None and self.index is None:
            raise ValueError("If dim is None, index must be provided at construction.")
        if random_state is not None and not isinstance(
            random_state, (int, np.random.Generator)
        ):
            raise TypeError(
                "random_state must be an integer, np.random.Generator, or None."
            )

        if dim is None:
            dim = len(self.index)

        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        arr = rng.normal(loc, scale, size=(len(self.domain.data), dim))
        return self.from_numpy(array=arr)

    @classmethod
    def indicator_of(cls, event: Event, dim: int) -> RandomVector:
        r"""Create the indicator random vector of a given event of a given dimension.

        See the Notes section below for the mathematical details.

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

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> print(Omega)
        Sample space 'Omega':
        [0, 1, 2]
        >>> A = Omega.get_event([0, 1])
        >>> I_A = RandomVector.indicator_of(event=A, dim=2)
        >>> print(I_A) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'I_A':
        feature  I_A_0  I_A_1
        sample
        0            1      1
        1            1      1
        2            0      0

        Notes
        -----
        Let $X: \Omega \to \mathbb{R}^d$ be a random vector defined on the probability space $(\Omega,\mathcal{F},P)$. Given an event $A\in \mathcal{F}$ and a dimension $d$, the *indicator random vector* is the random vector $I_A: \Omega \to \mathbb{R}^d$ such that

        $$
        I_A(\omega) = \begin{cases}
        (1, 1, \ldots, 1) & : \omega \in A,\\
        (0, 0, \ldots, 0) & : \omega \notin A.
        \end{cases}
        $$

        The event $A$ is represented by the parameter `event`, while the dimension $d$ is represented by the parameter `dim`.
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

    @property
    def outputs(self) -> Mapping[Hashable, Hashable]:
        """Get the outputs mapping of the random vector.

        If not initialized in the `from_dict` method, lazily constructs the outputs mapping from the underlying pandas data structure.

        Returns
        -------
        outputs : Mapping[Hashable, Hashable]
            The mapping from sample points in the domain to their corresponding output vectors.

        Examples
        --------
        >>> from sigalg.core import Index, RandomVector, SampleSpace
        >>> import numpy as np
        >>> arr = np.array([[1, 2], [3, 4], [5, 6]])
        >>> X = RandomVector().from_numpy(arr)
        >>> print(X.outputs)
        {0: (1, 2), 1: (3, 4), 2: (5, 6)}
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
        >>> Omega = SampleSpace().from_sequence(size=2, prefix="s")
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

    @property
    def components(self) -> list[RandomVariable]:
        r"""Get the component random variables of the random vector.

        See the Notes section below for the mathematical details.

        Raises
        ------
        ValueError
            If `self` has an empty `data` attribute.

        Returns
        -------
        components : list[RandomVariable]
            A list of the component random variables of the random vector.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> # Get the components of a 2D random vector
        >>> X = RandomVector(domain=Omega).from_randint(low=0, high=3, dim=2, random_state=42)
        >>> for component in X.components:
        ...     print(component) # doctest: +NORMALIZE_WHITESPACE
        Random variable '0':
        0
        sample
        0       0
        1       1
        2       1
        Random variable '1':
                1
        sample
        0       2
        1       1
        2       2
        >>> # Get the component of a 1D random vector
        >>> Y = RandomVector(domain=Omega, name="Y").from_randint(low=0, high=3, dim=1, random_state=42)
        >>> print(Y) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
                Y
        sample
        0       0
        1       2
        2       1
        >>> for component in Y.components:
        ...     print(component) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'Y':
                Y
        sample
        0       0
        1       2
        2       1

        Notes
        -----
        If $X: \Omega \to \mathbb{R}^d$ is a random vector, then for each $\omega \in \Omega$ we may write

        $$
        X(\omega) = (X_1(\omega),X_2(\omega),\ldots, X_d(\omega))
        $$

        where $X_j: \Omega \to \mathbb{R}$ is the *$j$-th component random variable* of $X$.

        If the dimension of `self` is $1$, then this method returns a list consisting of `self` itself.
        """
        from .random_variable import RandomVariable

        if self.data is None:
            raise ValueError(
                "Data must be initialized to get component random variables."
            )

        if self._components is None:
            if self.dimension == 1:
                if isinstance(self, RandomVariable):
                    self._components = [self]
                else:
                    self._components = [self.to_random_variable()]
            else:
                self._components = [
                    self.get_component_rv(idx) for idx in self.index.data
                ]
        return self._components

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

    def with_name(self, name: Hashable, modify_index: bool = False) -> RandomVector:
        """Set the name of the random vector and return self for chaining.

        Parameters
        ----------
        name : Hashable
            The new name for the random vector.
        modify_index : bool, default=False
            If `True` and the random vector has a feature index, also updates the feature index to reflect the new name of the random vector.

        Returns
        -------
        self : RandomVector
            Returns self to allow method chaining.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> outputs = dict(zip(Omega, [(1, 2), (3, 4), (5, 6)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        feature  X_0  X_1
        sample
        0          1    2
        1          3    4
        2          5    6
        >>> print(X.with_name("Y")) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
        feature  X_0  X_1
        sample
        0          1    2
        1          3    4
        2          5    6
        >>> print(X.with_name("Y", modify_index=True)) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
        feature  Y_0  Y_1
        sample
        0          1    2
        1          3    4
        2          5    6
        """
        from ..base.index import Index

        self.name = name
        if modify_index and self.index is not None:
            prefix = name if isinstance(name, str) else None
            self._index = Index(
                name="index",
                data_name="feature",
            ).from_sequence(
                size=self.dimension,
                prefix=prefix,
            )
            self._data.columns = self._index.data
        return self

    @property
    def index(self) -> Index | None:
        """Get the index of the random vector.

        Returns
        -------
        index : Index | None
            The index of the random vector, or `None` if the random vector is 1-dimensional.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> # Print the index of a 2D random vector
        >>> outputs_2d = dict(zip(Omega, [(1, 2), (3, 4), (5, 6)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs_2d)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        feature  X_0  X_1
        sample
        0          1    2
        1          3    4
        2          5    6
        >>> print(X.index)
        Index 'index':
        ['X_0', 'X_1']
        >>> # Print the index of a 1D random vector
        >>> outputs_1d = dict(zip(Omega, [1, 2, 3]))
        >>> Y = RandomVector(domain=Omega, name="Y").from_dict(outputs_1d)
        >>> print(Y) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
                Y
        sample
        0       1
        1       2
        2       3
        >>> print(Y.index)
        None
        """
        return self._index

    @index.setter
    def index(self, index: Index) -> None:
        from ..base.index import Index

        if not isinstance(index, Index):
            raise TypeError("index must be an Index.")

        if self._data is None:
            _ = self.data
        self._index = index
        self._data.columns = index.data

    @property
    def sigma_algebra(self) -> SigmaAlgebra:
        r"""Get the sigma-algebra generated by a random vector.

        See the Notes section below for the mathematical details.

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
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> outputs = dict(zip(Omega, [(1, 2), (3, 4), (3, 4)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs)
        >>> X.sigma_algebra # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'sigma(X)':
               atom ID
        sample
        0      (1, 2)
        1      (3, 4)
        2      (3, 4)

        Notes
        -----
        A random vector $X: \Omega \to \mathbb{R}^d$ on a probability space $(\Omega, \mathcal{F},P)$ generates a $\sigma$-algebra denoted $\sigma(X)$. On a finite sample space $\Omega$, this $\sigma$-algebra is determined by its atoms, which are the nonempty level sets

        $$
        X^{-1}(x) = \{ \omega \in \Omega : X(\omega) = x\},
        $$

        for $x\in \mathbb{R}^d$. The atom identifiers may thus be taken as the vectors $x\in \mathbb{R}^d$ in the range of $X$.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if self._sigma_algebra is None:
            self._sigma_algebra = SigmaAlgebra.from_random_vector(self)
        return self._sigma_algebra

    @property
    def probability_measure(self) -> ProbabilityMeasure:
        """Get the probability measure on the domain of the random vector.

        If the measure is not explicitly set by the user, the measure defaults to the uniform measure.

        Raises
        ------
        ValueError
            If the `domain` attribute of the random vector is not set.

        Returns
        -------
        probability_measure : ProbabilityMeasure
            The probability measure on the domain of the random vector.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> outputs = dict(zip(Omega, [(1, 2), (3, 4), (5, 6)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        feature  X_0  X_1
        sample
        0          1    2
        1          3    4
        2          5    6
        >>> # The default probability measure is uniform
        >>> print(X.probability_measure) # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        sample
        0          0.333333
        1          0.333333
        2          0.333333
        >>> # Set the probability measure
        >>> probs = dict(zip(Omega, [0.1, 0.4, 0.5]))
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict(probs)
        >>> X.probability_measure = P
        >>> print(X.probability_measure) # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        sample
        0               0.1
        1               0.4
        2               0.5
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

    @property
    def range(self) -> ProbabilitySpace:
        r"""Return the range of a random vector as a probability space with the pushforward measure.

        See the Notes section below for the mathematical details.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> outputs = dict(zip(Omega, [(1, 2), (3, 4), (3, 4)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        feature  X_0  X_1
        sample
        0          1    2
        1          3    4
        2          3    4
        >>> probs = dict(zip(Omega, [0.1, 0.4, 0.5]))
        >>> X.probability_measure = ProbabilityMeasure(sample_space=Omega).from_dict(probs)
        >>> print(X.range) # doctest: +NORMALIZE_WHITESPACE
        Probability space (X_range, power_set, P_X)
        ===========================================
        <BLANKLINE>
        * Sample space 'X_range':
        [(1, 2), (3, 4)]
        <BLANKLINE>
        * Sigma algebra 'power_set':
            atom ID
        1 2        0
        3 4        1
        <BLANKLINE>
        * Probability measure 'P_X':
            probability
        1 2          0.1
        3 4          0.9

        Notes
        -----
        Let $X: \Omega \to \mathbb{R}^d$ be a random vector on a probability space $(\Omega, \mathcal{F},P)$. The range

        $$
        X(\Omega) = \{ X(\omega) \in \mathbb{R}^d : \omega \in \Omega \}
        $$

        of the random vector is a probability space when equipped with the *pushforward measure* $P_X$ given by

        $$
        P_X(A) = P \left( \{\omega \in \Omega \mid X(\omega) \in A \} \right),
        $$

        for all events $A \subset X(\Omega)$. In SigAlg, the $\sigma$-algebra on $X(\Omega)$ defaults to the power set.
        """
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

    # --------------------- probability space methods --------------------- #

    def is_measurable(self, sigma_algebra: SigmaAlgebra) -> bool:
        r"""Check if the random vector is measurable with respect to a given sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        sigma_algebra : SigmaAlgebra
            The sigma-algebra to check measurability against.

        Returns
        -------
        is_measurable : bool
            `True` if the random vector is measurable with respect to the given sigma-algebra, `False` otherwise.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> outputs_X = dict(zip(Omega, [(1, 2), (3, 4), (3, 4), (3, 4)]))
        >>> outputs_Y = dict(zip(Omega, [(1, 2), (3, 4), (5, 6), (7, 8)]))
        >>> X = RandomVector(domain=Omega, name="X").from_dict(outputs_X)
        >>> Y = RandomVector(domain=Omega, name="Y").from_dict(outputs_Y)
        >>> atom_ids = dict(zip(Omega, [0, 1, 1, 2]))
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(atom_ids)
        >>> # X is constant on the atoms of F, so it is measurable
        >>> print(X.is_measurable(F))
        True
        >>> # Y is not constant on the atoms, so it is not measurable
        >>> print(Y.is_measurable(F))
        False

        Notes
        -----
        Let $(\Omega, \mathcal{F})$ be a measurable space and $X: \Omega \to \mathbb{R}^d$ a function. In the case that $\Omega$ is finite (as in SigAlg), the $\sigma$-algebra is determined by its atoms. In this case, the function $X$ is said to be *$\mathcal{F}$-measurable* if $X$ is constant on the atoms of $\mathcal{F}$.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(sigma_algebra, SigmaAlgebra):
            raise TypeError("sigma_algebra must be a SigmaAlgebra.")

        return self.sigma_algebra <= sigma_algebra

    def with_probability_measure(
        self,
        probabilities: Mapping[Hashable, Real] | None = None,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> RandomVector:
        """Set the probability measure on the domain of the random vector and return self for chaining.

        This method is equivalent to setting the `probability_measure` attribute with an instance of `ProbabilityMeasure`. The method also accepts a dictionary of probabilities as a parameter, allowing the user to bypass constructing an instance of `ProbabilityMeasure`.

        The method takes either the `probabilities` parameter or the `probability_measure` parameter, but not both. If neither parameter is provided, the method defaults to setting the probability measure to the uniform measure.

        Parameters
        ----------
        probabilities : Mapping[Hashable, Real] | None, default=None
            A mapping from sample points in the domain to their corresponding probabilities.
        probability_measure : ProbabilityMeasure | None, default=None
            The probability measure to set on the domain of the random vector.

        Raises
        ------
        ValueError
            If both `probabilities` and `probability_measure` are provided.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> X = RandomVector(domain=Omega).from_randint(low=0, high=6, dim=2, random_state=42)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
                0  1
        sample
        0       0  4
        1       3  2
        2       2  5
        >>> probs_1 = dict(zip(Omega, [0.3, 0.2, 0.5]))
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict(probs_1)
        >>> _ = X.with_probability_measure(probability_measure=P)
        >>> print(X.probability_measure) # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        sample
        0               0.3
        1               0.2
        2               0.5
        >>> probs_2 = dict(zip(Omega, [0.5, 0.3, 0.2]))
        >>> _ = X.with_probability_measure(probabilities=probs_2)
        >>> print(X.probability_measure) # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        sample
        0               0.5
        1               0.3
        2               0.2
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
        self._probability_measure = probability_measure
        return self

    # --------------------- data methods --------------------- #

    def __call__(
        self, key: Hashable | list[Hashable] | Event
    ) -> Hashable | FeatureVector | RandomVector:
        r"""Evaluate a random vector on a sample point, or evaluate it on multiple sample points to get the restriction of the random vector.

        See the Notes section below for the mathematical details.

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
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> outputs = dict(zip(Omega, [(1, 2), (3, 4), (5, 6)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs)
        >>> # Call the random vector on a sample point to get the feature vector
        >>> print(X(0)) # doctest: +NORMALIZE_WHITESPACE
        Feature vector of '0':
                0
        feature
        X_0      1
        X_1      2
        >>> # Get the restriction of X to an event by calling on an `Event` instance
        >>> A = Omega.get_event([0, 2])
        >>> X_A = X(A)
        >>> print(X_A) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X|A':
        feature  X_0  X_1
        sample
        0          1    2
        2          5    6
        >>> # Get the restriction of X to an event by calling on a list of sample points
        >>> X_B = X([0, 1])
        >>> print(X_B) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X|event':
        feature  X_0  X_1
        sample
        0          1    2
        1          3    4

        Notes
        -----
        Let $X: \Omega \to \mathbb{R}^d$ be a random vector on a probability space $(\Omega, \mathcal{F}, P)$. As a function, we can evaluate $X$ at a sample point $\omega\in \Omega$ to obtain the feature vector $X(\omega) \in \mathbb{R}^d$. If $A\in \mathcal{F}$ is an event, then we may also restrict the random vector to obtain the function $X|_A : A \to \mathbb{R}^d$ on $A$. If $A$ is an event of nonzero probability, then $A$ carries the conditional probability distribution $P_A$, defined so that $P_A(B) = P(B) / P(A)$, for $B\subset A$.

        If `X` is an instance of `RandomVector`, then it to may be called on elements in its domain. It may also be called on either a list of sample points, or an instance of `Event`, to obtain the restricted random vector.
        """
        from ..base.event import Event
        from ..base.feature_vector import FeatureVector
        from ..base.probability_space import ProbabilitySpace

        if not isinstance(key, (Hashable, list, Event)):
            raise TypeError("key must be a Hashable, list, or Event.")

        if isinstance(key, Hashable) and not isinstance(key, (list, Event)):
            if key not in self.domain:
                raise KeyError(f"Sample '{key}' not found in domain.")

            data = self.data.loc[key]

            if not isinstance(data, pd.Series):
                result = data
            else:
                result = FeatureVector(name=key).from_pandas(data=data)

        if isinstance(key, list):
            invalid_indices = [k for k in key if k not in self.domain.data]
            if invalid_indices:
                raise KeyError(f"Samples {invalid_indices} not found in domain.")

            event = self.domain.get_event(key)
            event_prob_space = ProbabilitySpace.from_event(
                event=event, probability_measure=self.probability_measure
            )

            name = f"{self.name}|event" if self.name is not None else None

            result = (
                RandomVector(name=name)
                .from_pandas(data=self.data.loc[key])
                .with_probability_measure(
                    probability_measure=event_prob_space.probability_measure
                )
            )

        if isinstance(key, Event):
            if key.sample_space != self.domain:
                raise ValueError(
                    "Event's sample_space must match RandomVector's domain."
                )

            event_prob_space = ProbabilitySpace.from_event(
                event=key, probability_measure=self.probability_measure
            )

            name = (
                f"{self.name}|{key.name}"
                if (self.name is not None and key.name is not None)
                else None
            )

            result = (
                RandomVector(name=name)
                .from_pandas(data=self.data.loc[key.indices])
                .with_probability_measure(
                    probability_measure=event_prob_space.probability_measure
                )
            )

        if isinstance(result, RandomVector) and result.dimension == 1:
            result = result.to_random_variable()

        return result

    def get_component_rv(self, index: Hashable) -> RandomVariable:
        r"""Get a component random variable of the random vector.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        index : Hashable
            The feature index for which to get the component random variable.

        Returns
        -------
        component_rv : RandomVariable
            A new `RandomVariable` representing the component random variable.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=2)
        >>> outputs = dict(zip(Omega, [(1, 2), (3, 4)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        feature  X_0  X_1
        sample
        0          1    2
        1          3    4
        >>> X_1 = X.get_component_rv("X_1")
        >>> print(X_1)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_1':
                X_1
        sample
        0         2
        1         4

        Notes
        -----
        Given a random vector $X: \Omega \to \mathbb{R}^d$ on a probability space $(\Omega, \mathcal{F}, P)$, for each $\omega \in \Omega$ we may write

        $$
        X(\omega) = (X_1(\omega), X_2(\omega), \ldots, X_d(\omega)),
        $$

        where $X_j: \Omega \to \mathbb{R}$ are the *component random variables* of $X$.
        """
        component_rv = self.get_sub_vector([index]).to_random_variable()
        component_rv.name = index
        return component_rv.with_probability_measure(
            probability_measure=self.probability_measure
        )

    def get_sub_vector(self, feature_indices: list[Hashable]) -> RandomVector:
        r"""Get a sub-vector of the random vector by selecting a collection of component random variables.

        See the Notes section below for the mathematical details.

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
        >>> Omega = SampleSpace().from_sequence(size=2)
        >>> outputs = dict(zip(Omega, [(1, 2, 3), (4, 5, 6)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        feature  X_0  X_1  X_2
        sample
        0          1    2    3
        1          4    5    6
        >>> X_sub = X.get_sub_vector(["X_0", "X_2"])
        >>> print(X_sub) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X_sub':
        feature  X_0  X_2
        sample
        0          1    3
        1          4    6

        Notes
        -----
        Given a random vector $X: \Omega \to \mathbb{R}^d$ on a probability space $(\Omega, \mathcal{F}, P)$, for each $\omega \in \Omega$ we may write

        $$
        X(\omega) = (X_1(\omega), X_2(\omega), \ldots, X_d(\omega)),
        $$

        where $X_j: \Omega \to \mathbb{R}$ are the component random variables of $X$. We may create a *sub-vector* by choosing a collection of the component random variables to get a random vector of smaller dimension. For example, we may select the first and last random variables to create the $2$-dimensional random vector

        $$
        \omega \mapsto (X_1 (\omega), X_d(\omega)).
        $$
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

    def item(self) -> Hashable | FeatureVector:
        """Get the output value of a constant random vector.

        Returns
        -------
        output : Hashable | FeatureVector
            The single output value of the random vector. If the dimension of the random vector is > 1, then the return value is an instance of `FeatureVector`; otherwise, the return value is a `Hashable`.

        Raises
        ------
        ValueError
            If the random vector is not constant.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=2)
        >>> outputs_X = dict(zip(Omega, [(1, 2), (1, 2)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs_X)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        feature  X_0  X_1
        sample
        0          1    2
        1          1    2
        >>> print(X.item()) # doctest: +NORMALIZE_WHITESPACE
        Feature vector of 'sample_point':
                sample_point
        feature
        X_0                 1
        X_1                 2
        >>> outputs_Y = dict(zip(Omega, [1, 1]))
        >>> Y = RandomVector(domain=Omega, name="Y").from_dict(outputs_Y)
        >>> print(Y) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
                Y
        sample
        0       1
        1       1
        >>> print(Y.item())
        1
        """
        sample_point = self.domain[0]
        item = self(sample_point)

        if self.dimension != 1:
            if self.data.nunique().sum() != self.dimension:
                raise ValueError(
                    "item() can only be called on a constant random vector."
                )
            item.name = "sample_point"
            item.data.name = "sample_point"
        else:
            if self.data.nunique() != 1:
                raise ValueError(
                    "item() can only be called on a constant random vector."
                )

        return item

    def round(self, decimals: int = 0) -> RandomVector:
        """Round the feature vectors of the random vector to a specified number of decimal places.

        Parameters
        ----------
        decimals : int, default=0
            The number of decimal places to round to. Must be a non-negative integer.

        Raises
        ------
        ValueError
            If `decimals` is not a non-negative integer, or if the random vector's data is not set.

        Examples
        --------
        >>> import numpy as np
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=2)
        >>> outputs = dict(zip(Omega, [(0, np.pi), (np.pi / 2, 3 * np.pi / 2)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs)
        >>> print(np.sin(X).round()) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'sin(X)':
        feature  sin(X_0)  sin(X_1)
        sample
        0             0.0       0.0
        1             1.0      -1.0
        """
        if not isinstance(decimals, int) or decimals < 0:
            raise ValueError("decimals must be a non-negative integer.")
        if self._data is None:
            raise ValueError("Data must be set to round the random vector.")

        self._data = self.data.round(decimals=decimals)
        return self

    def iter_features(self):
        """Iterate over sample points and their feature vectors.

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
        >>> Omega = SampleSpace().from_sequence(size=2, prefix="s")
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

    # --------------------- conversion methods --------------------- #

    def to_random_variable(self) -> RandomVariable:
        """Convert a 1-dimensional random vector to an instance of `RandomVariable`.

        Raises
        ------
        ValueError
            If the random vector has dimension > 1.

        Returns
        -------
        rv : RandomVariable
            The converted `RandomVariable`.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=2)
        >>> outputs = dict(zip(Omega, [10, 20]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs=outputs)
        >>> X_var = X.to_random_variable()
        >>> X_var # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X':
                X
        sample
        0      10
        1      20
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

    def apply_to_features(
        self, function: Callable[[FeatureVector | Hashable], any]
    ) -> RandomVariable:
        """Apply a function to the feature vector of each sample point.

        Applies the given function to each sample point's feature vector,
        returning an instance of `RandomVariable`.

        Parameters
        ----------
        function : Callable[[FeatureVector | Hashable], any]
            Function that takes a `FeatureVector` object (in dimension > 1) or a `Hashable` (in dimension 1) and returns a value.

        Returns
        -------
        results : RandomVariable
            An instance of `RandomVariable` whose values are given by the function applied to each feature vector of the random vector.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=2)
        >>> outputs_X = dict(zip(Omega, [(1, 2), (3, 4)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs_X)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        feature  X_0  X_1
        sample
        0          1    2
        1          3    4
        >>> X.apply_to_features(lambda f: f.sum() + 2) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'X_apply':
                X_apply
        sample
        0             5
        1             9
        >>> outputs_Y = dict(zip(Omega, [5, 10]))
        >>> Y = RandomVector(domain=Omega, name="Y").from_dict(outputs_Y)
        >>> print(Y) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'Y':
                Y
        sample
        0        5
        1       10
        >>> Y.apply_to_features(lambda x: x * 2) # doctest: +NORMALIZE_WHITESPACE
        Random variable 'Y_apply':
                Y_apply
        sample
        0            10
        1            20
        """
        from ..base.feature_vector import FeatureVector
        from .random_variable import RandomVariable

        if self.dimension > 1:

            def wrapper(row):
                sp = FeatureVector().from_pandas(data=row)
                return function(sp)

            data = self.data.apply(wrapper, axis=1)
        else:
            data = self.data.apply(function)

        name = f"{self.name}_apply"
        rv = RandomVariable(domain=self.domain, name=name).from_pandas(data)

        return rv

    # --------------------- equality --------------------- #

    def __eq__(self, other: RandomVector, rtol=1e-5, atol=1e-8) -> bool:
        r"""Check equality with another random vector.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        other : RandomVector
            Another random vector to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the other object is a `RandomVector` with the same domain, feature index, and data.

        Notes
        -----
        Two random vector $X,Y: \Omega \to \mathbb{R}^d$ on the same probability space $(\Omega, \mathcal{F}, P)$ are equal if $X(\omega) = Y(\omega)$ for all $\omega \in \Omega$.
        """
        if not isinstance(other, RandomVector):
            return False
        if not self.domain == other.domain:
            return False
        return np.allclose(
            self.data.to_numpy(), other.data.to_numpy(), rtol=rtol, atol=atol
        )

    # --------------------- representation --------------------- #

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

    def print_values_and_probabilities(self) -> None:
        """Print the values of the random vector and their corresponding probabilities.

        Raises
        ------
        ValueError
            If the random vector does not contain data.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> X = RandomVector(domain=Omega).from_randint(low=0, high=10, dim=2, random_state=42)
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
                0  1
        sample
        0       0  7
        1       6  4
        2       4  8
        >>> probs = dict(zip(Omega, [0.2, 0.45, 0.35]))
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict(probs)
        >>> print(P) # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        sample
        0              0.20
        1              0.45
        2              0.35
        >>> X.probability_measure = P
        >>> X.print_values_and_probabilities() # doctest: +NORMALIZE_WHITESPACE
                0  1  probability
        sample
        0       0  7         0.20
        1       6  4         0.45
        2       4  8         0.35
        """
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

    def __array_ufunc__(
        self, ufunc, method, *inputs, **kwargs
    ) -> RandomVector | StochasticProcess | RandomVariable:
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
        from ..base.index import Index
        from .random_variable import RandomVariable

        if method != "__call__":
            return NotImplemented

        new_inputs = [
            input.data if isinstance(input, RandomVector) else input for input in inputs
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
            if self.dimension > 1 and self.name is not None:
                new_index = Index(name="index", data_name="feature").from_list(
                    [f"{ufunc.__name__}({idx_name})" for idx_name in self.index]
                )
                result_data.columns = new_index.data
            return (
                RandomVector(domain=self.domain, name=new_name)
                .from_pandas(data=result_data)
                .with_probability_measure(probability_measure=self.probability_measure)
            )

    # --------------------- comparison methods --------------------- #

    def __bool__(self) -> bool:
        """Prevent ambiguous boolean conversion of a random vector.

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

    def all(self) -> bool:
        """Check if all values in the random vector are True.

        This method is typically used after a comparison operation to verify
        that the comparison holds for all sample points and all components.

        Returns
        -------
        all_true : bool
            True if all values across all samples and features are True.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=2)
        >>> outputs_X = dict(zip(Omega, [(1, 1), (1, 1)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs_X)
        >>> print(X.all())
        True
        >>> outputs_Y = dict(zip(Omega, [(1, 0), (1, 0)]))
        >>> Y = RandomVector(domain=Omega, name="Y").from_dict(outputs_Y)
        >>> print(Y.all())
        False
        """
        return bool(self.data.all().all() if self.dimension > 1 else self.data.all())

    def any(self) -> bool:
        """Check if any value in the random vector is True.

        This method is typically used after a comparison operation to verify
        that the comparison holds for at least one sample point or component.

        Returns
        -------
        any_true : bool
            True if any value across all samples and features is True.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=2)
        >>> outputs_X = dict(zip(Omega, [(1, 0), (0, 0)]))
        >>> X = RandomVector(domain=Omega).from_dict(outputs_X)
        >>> print(X.any())
        True
        >>> outputs_Y = dict(zip(Omega, [(0, 0), (0, 0)]))
        >>> Y = RandomVector(domain=Omega, name="Y").from_dict(outputs_Y)
        >>> print(Y.any())
        False
        """
        return bool(self.data.any().any() if self.dimension > 1 else self.data.any())

    def _apply_comparison(
        self,
        other: RandomVector | Real,
        op: Callable,
        op_symbol: str,
    ) -> RandomVector:
        """Apply a comparison operation to this random vector.

        Parameters
        ----------
        other : RandomVector | Real
            The random vector or scalar to compare with.
        op : Callable
            The numpy comparison to apply (e.g., ``operator.lt``).
        op_symbol : str
            Symbol representing the comparison (e.g., '<', '<=', '>', '>=').

        Returns
        -------
        result : RandomVector
            A new random vector of booleans representing the comparison result.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector` or scalar.
        ValueError
            If the random vectors do not have the same domain or dimension.
        """
        from ...core.base.index import Index
        from ...processes.base.stochastic_process import StochasticProcess
        from .random_variable import RandomVariable

        if not isinstance(other, RandomVector) and isinstance(other, Real):
            other = RandomVector(
                domain=self.domain, index=self.index, name=other
            ).from_constant(constant=other)
        elif not isinstance(other, RandomVector):
            raise TypeError("other must be a RandomVector")
        if self.domain != other.domain:
            raise ValueError("Random vectors must have the same domain")
        if self.dimension != other.dimension:
            raise ValueError("Random vectors must have the same dimension")

        comparison_arr = op(self.data.to_numpy(), other.data.to_numpy())
        name = (
            f"({self.name} {op_symbol} {other.name})"
            if self.name and other.name
            else None
        )

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
                index = Index(data_name="feature").from_sequence(
                    size=self.dimension, prefix=name
                )
                result._index = index
                result.data.columns = index.data
            return result

    def __lt__(self, other: RandomVector | Real) -> RandomVector:
        r"""Check if this random vector is less than another random vector or scalar.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        other : RandomVector | Real
            The random vector or scalar to compare with.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector`.
        ValueError
            If the random vectors do not have the same domain or dimension.

        Returns
        -------
        is_lt: RandomVector
            A new `RandomVector` of booleans indicating where this random vector is less than the other random vector or scalar.

        Notes
        -----
        Let $X,Y: \Omega \to \mathbb{R}^d$ be two random vectors defined on a probability space $(\Omega, \mathcal{F},P)$, with component random variables

        $$
        X = (X_1, X_2,\ldots,X_d) \quad \text{and} \quad Y = (Y_1, Y_2, \ldots,Y_d).
        $$

        We define a third random variable $Z: \Omega \to \mathbb{R}^d$ with components

        $$
        Z = (Z_1, Z_2, \ldots, Z_d)
        $$

        such that $Z_j(\omega) = 1$ if $X_j(\omega) < Y_j(\omega)$, and $Z_j(\omega)=0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $Y$ is `other`.

        If $c$ is a scalar, then we define $Z$ by setting $Z_j(\omega) = 1$ if $X_j(\omega) < c$, and $Z_j(\omega) = 0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $c$ is `other`.
        """
        import operator

        return self._apply_comparison(other, operator.lt, "<")

    def __le__(self, other: RandomVector | Real) -> RandomVector:
        r"""Check if this random vector is less than or equal to another random vector or scalar.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        other : RandomVector | Real
            The random vector or scalar to compare with.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector`.
        ValueError
            If the random vectors do not have the same domain or dimension.

        Returns
        -------
        is_le: RandomVector
            A new `RandomVector` of booleans indicating where this random vector is less than or equal to the other random vector or scalar.

        Notes
        -----
        Let $X,Y: \Omega \to \mathbb{R}^d$ be two random vectors defined on a probability space $(\Omega, \mathcal{F},P)$, with component random variables

        $$
        X = (X_1, X_2,\ldots,X_d) \quad \text{and} \quad Y = (Y_1, Y_2, \ldots,Y_d).
        $$

        We define a third random variable $Z: \Omega \to \mathbb{R}^d$ with components

        $$
        Z = (Z_1, Z_2, \ldots, Z_d)
        $$

        such that $Z_j(\omega) = 1$ if $X_j(\omega) \leq Y_j(\omega)$, and $Z_j(\omega)=0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $Y$ is `other`.

        If $c$ is a scalar, then we define $Z$ by setting $Z_j(\omega) = 1$ if $X_j(\omega) \leq c$, and $Z_j(\omega) = 0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $c$ is `other`.
        """
        import operator

        return self._apply_comparison(other, operator.le, "<=")

    def __gt__(self, other: RandomVector | Real) -> RandomVector:
        r"""Check if this random vector is greater than another random vector or scalar.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        other : RandomVector | Real
            The random vector or scalar to compare with.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector`.
        ValueError
            If the random vectors do not have the same domain or dimension.

        Returns
        -------
        is_gt: RandomVector
            A new `RandomVector` of booleans indicating where this random vector is greater than the other random vector or scalar.

        Notes
        -----
        Let $X,Y: \Omega \to \mathbb{R}^d$ be two random vectors defined on a probability space $(\Omega, \mathcal{F},P)$, with component random variables

        $$
        X = (X_1, X_2,\ldots,X_d) \quad \text{and} \quad Y = (Y_1, Y_2, \ldots,Y_d).
        $$

        We define a third random variable $Z: \Omega \to \mathbb{R}^d$ with components

        $$
        Z = (Z_1, Z_2, \ldots, Z_d)
        $$

        such that $Z_j(\omega) = 1$ if $X_j(\omega) > Y_j(\omega)$, and $Z_j(\omega)=0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $Y$ is `other`.

        If $c$ is a scalar, then we define $Z$ by setting $Z_j(\omega) = 1$ if $X_j(\omega) > c$, and $Z_j(\omega) = 0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $c$ is `other`.
        """
        import operator

        return self._apply_comparison(other, operator.gt, ">")

    def __ge__(self, other: RandomVector | Real) -> RandomVector:
        r"""Check if this random vector is greater than or equal another random vector or scalar.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        other : RandomVector | Real
            The random vector or scalar to compare with.

        Raises
        ------
        TypeError
            If `other` is not a `RandomVector`.
        ValueError
            If the random vectors do not have the same domain or dimension.

        Returns
        -------
        is_ge: RandomVector
            A new `RandomVector` of booleans indicating where this random vector is greater than or equal the other random vector or scalar.

        Notes
        -----
        Let $X,Y: \Omega \to \mathbb{R}^d$ be two random vectors defined on a probability space $(\Omega, \mathcal{F},P)$, with component random variables

        $$
        X = (X_1, X_2,\ldots,X_d) \quad \text{and} \quad Y = (Y_1, Y_2, \ldots,Y_d).
        $$

        We define a third random variable $Z: \Omega \to \mathbb{R}^d$ with components

        $$
        Z = (Z_1, Z_2, \ldots, Z_d)
        $$

        such that $Z_j(\omega) = 1$ if $X_j(\omega) \geq Y_j(\omega)$, and $Z_j(\omega)=0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $Y$ is `other`.

        If $c$ is a scalar, then we define $Z$ by setting $Z_j(\omega) = 1$ if $X_j(\omega) \geq c$, and $Z_j(\omega) = 0$ otherwise. This method returns the random vector $Z$, in the case that $X$ is `self` and $c$ is `other`.
        """
        import operator

        return self._apply_comparison(other, operator.ge, ">=")
