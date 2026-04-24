"""A class representing a feature vector."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ..random_objects.random_vector import RandomVector


class FeatureVector:
    r"""A class representing a feature vector.

    See the Notes section below for the mathematical details.

    The `__init__` method initializes an instance, but it does not populate it with data. Instead, data is primarily set using the `from_rv` method, but may also be set using the `from_pandas` method.

    Parameters
    ----------
    name : Hashable | None, default=None
        An optional identifier for this feature vector, typically corresponding to the sample point it represents. If not provided, it can be set later or inferred from data.

    Raises
    ------
    TypeError
        If `name` is not a `Hashable` or `None`.

    Examples
    --------
    >>> from sigalg.core import FeatureVector, RandomVector, SampleSpace
    >>> Omega = SampleSpace().from_sequence(size=3)
    >>> X = RandomVector(domain=Omega).from_dict(
    ...     {
    ...         0: (1, 2),
    ...         1: (3, 4),
    ...         2: (5, 6),
    ...     }
    ... )
    >>> print(X) # doctest: +NORMALIZE_WHITESPACE
    Random vector 'X':
    feature  X_0  X_1
    sample
    0          1    2
    1          3    4
    2          5    6
    >>> # Obtain the feature vector X(1)
    >>> v = FeatureVector().from_rv(1, X)
    >>> print(v) # doctest: +NORMALIZE_WHITESPACE
    Feature vector of '1':
            1
    feature
    X_0      3
    X_1      4
    >>> # The same feature vector can also be obtained by calling the random vector
    >>> print(X(1)) # doctest: +NORMALIZE_WHITESPACE
    Feature vector of '1':
            1
    feature
    X_0      3
    X_1      4

    Notes
    -----
    Let $X: \Omega \to \mathbb{R}^d$ be a random vector on a probability space $(\Omega, \mathcal{F}, P)$. The *feature vector* of a sample point $\omega\in \Omega$ is the vector $X(\omega) \in \mathbb{R}^d$.
    """

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        name: Hashable | None = None,
    ) -> None:
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("name must be a Hashable or None.")
        self._name = name

        # caches
        self._data: pd.Series | None = None
        self._rv: RandomVector | None = None

    def from_pandas(self, data: pd.Series) -> FeatureVector:
        """Create a feature vector from a `pd.Series` object.

        Parameters
        ----------
        data : pd.Series
            A `pd.Series` containing feature values, indexed by feature names.

        Returns
        -------
        self : FeatureVector
            The created FeatureVector instance.
        """
        self.data = data
        return self

    def from_rv(self, sample_point: Hashable, rv: RandomVector) -> FeatureVector:
        """Obtain the feature vector for a specific sample point from a random vector.

        Parameters
        ----------
        sample_point : Hashable
            The sample point for which to obtain the feature vector.
        rv : RandomVector
            The random vector to associate.

        Returns
        -------
        self : FeatureVector
            The updated `FeatureVector` instance.

        Examples
        --------
        >>> from sigalg.core import FeatureVector, RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> X = RandomVector(domain=Omega).from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (5, 6),
        ...     }
        ... )
        >>> print(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        feature  X_0  X_1
        sample
        0          1    2
        1          3    4
        2          5    6
        >>> # Obtain the feature vector X(1)
        >>> v = FeatureVector().from_rv(1, X)
        >>> print(v) # doctest: +NORMALIZE_WHITESPACE
        Feature vector of '1':
                1
        feature
        X_0      3
        X_1      4
        """
        self._rv = rv
        self.data = rv.data.loc[sample_point]
        return self

    # --------------------- properties --------------------- #

    @property
    def data(self) -> pd.Series:
        """Get the underlying `pd.Series` object.

        Returns
        -------
        data : pd.Series
            The feature values as a pandas Series, indexed by feature names.
        """
        return self._data

    @data.setter
    def data(self, data: pd.Series) -> None:
        """Set the feature vector data.

        Parameters
        ----------
        data : pd.Series
            A `pd.Series` containing feature values, indexed by feature names.

        Raises
        ------
        TypeError
            If `data` is not a `pd.Series`.
        """
        if not isinstance(data, pd.Series):
            raise TypeError("data must be a `pd.Series`.")
        if data.name is None:
            data.name = self._name
        if data.name is not None and self._name is None:
            self._name = data.name
        self._data = data

    @property
    def name(self) -> Hashable:
        """Get the name of the feature vector.

        Returns
        -------
        name : Hashable
            The identifier for this sample point.
        """
        return self._name

    @name.setter
    def name(self, name: Hashable) -> None:
        """Set the name of the feature vector.

        Parameters
        ----------
        name : Hashable
            New identifier for this feature vector.

        Raises
        ------
        TypeError
            If `name` is not hashable.
        """
        if not isinstance(name, Hashable):
            raise TypeError("name must be a Hashable.")
        self._name = name
        self.data.name = name

    @property
    def random_vector(self) -> RandomVector | None:
        """Get the associated random vector.

        Returns
        -------
        random_vector : RandomVector | None
            The random vector from which these features were derived, or `None`
            if not set.

        Examples
        --------
        >>> from sigalg.core import FeatureVector, RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> X = RandomVector(domain=Omega).from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (5, 6),
        ...     }
        ... )
        >>> v = FeatureVector().from_rv(1, X)
        >>> print(v.random_vector) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'X':
        feature  X_0  X_1
        sample
        0          1    2
        1          3    4
        2          5    6
        """
        return self._rv

    # --------------------- data access methods --------------------- #

    @property
    def feature_at(self) -> _iLocIndexer:
        """Get indexer for positional access to features.

        Returns
        -------
        indexer : _iLocIndexer
            Indexer for accessing features by integer position.

        Examples
        --------
        >>> from sigalg.core import FeatureVector, RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> X = RandomVector(domain=Omega).from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (5, 6),
        ...     }
        ... )
        >>> v = FeatureVector().from_rv(1, X)
        >>> print(v) # doctest: +NORMALIZE_WHITESPACE
        Feature vector of '1':
                1
        feature
        X_0      3
        X_1      4
        >>> print(v.feature_at[0])
        3
        """
        return self._iLocIndexer(self)

    class _iLocIndexer:
        def __init__(self, parent) -> None:
            self.parent = parent

        def __getitem__(self, key: int | slice | list[int]):
            return self.parent.data.iloc[key]

    def __getitem__(self, key: Hashable) -> Any:
        """Get feature value by feature name.

        Parameters
        ----------
        key : Hashable
            The feature name to access.

        Raises
        ------
        KeyError
            If the feature name is not found.

        Returns
        -------
        value : Any
            The value of the specified feature.

        Examples
        --------
        >>> from sigalg.core import FeatureVector, RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> X = RandomVector(domain=Omega).from_dict(
        ...     {
        ...         0: (1, 2),
        ...         1: (3, 4),
        ...         2: (5, 6),
        ...     }
        ... )
        >>> v = FeatureVector().from_rv(1, X)
        >>> print(v) # doctest: +NORMALIZE_WHITESPACE
        Feature vector of '1':
                1
        feature
        X_0      3
        X_1      4
        >>> print(v["X_0"])
        3
        """
        if key not in self.data.index:
            raise KeyError(f"Feature '{key}' not found.")
        return self.data[key]

    # --------------------- sequence methods --------------------- #

    def __iter__(self) -> iter:
        """Iterate over feature values.

        Yields
        ------
        value : Any
            Each feature value in order.
        """
        return iter(self.data)

    def __len__(self) -> int:
        """Return the number of features.

        Returns
        -------
        length : int
            The number of features for this sample point.
        """
        return len(self.data)

    def sum(self) -> Any:
        """Return the sum of all feature values.

        Returns
        -------
        total : Any
            The sum of all feature values.
        """
        return self.data.sum()

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a string representation of the feature vector.

        Returns
        -------
        repr_str : str
            A string showing the sample point name and feature values.
        """
        return f"Feature vector of '{self.name}':\n{self.data.to_frame()}"

    # --------------------- equality --------------------- #

    def __eq__(self, other: object) -> bool:
        """Check equality with another feature vector.

        Two feature vectors are equal if they have the underlying data.

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the other object is a `FeatureVector` with identical
            data.
        """
        if not isinstance(other, FeatureVector):
            return False
        return self.data.equals(other.data)
