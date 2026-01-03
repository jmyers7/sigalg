"""Feature vector module.

This module provides the `FeatureVector` class, which represents the feature
vector for a single sample point. Given a random vector `X: Omega -> S`, a `FeatureVector` object represents `X(omega)` for a specific `omega` in the
sample space.

Classes
-------
FeatureVector
    Represents the feature vector for a single sample point.

Examples
--------
>>> from sigalg.core import RandomVector, SampleSpace
>>> Omega = SampleSpace.generate_sequence(size=2, prefix="s")
>>> X = RandomVector(domain=Omega).from_dict({"s_0": (1, 2), "s_1": (3, 4)})
>>> # Get the feature vector for sample point 's_0'
>>> features = X("s_0")
>>> features #doctest: +NORMALIZE_WHITESPACE
Feature vector of 's_0':
         s_0
feature
X_0        1
X_1        2
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ..random_objects.random_vector import RandomVector


class FeatureVector:
    """A class representing a feature vector for a single sample point.

    A `FeatureVector` object represents the feature values for one sample
    point from a random vector. Given a random vector `X: Omega -> S`, this represents the output `X(omega)` for a specific sample point `omega`.

    Parameters
    ----------
    data : pd.Series
        Series of feature values for the sample point.

    Raises
    ------
    TypeError
        If `data` is not a `pd.Series`.

    Examples
    --------
    >>> from sigalg.core import FeatureVector
    >>> import pandas as pd
    >>> data = pd.Series([1, 2, 3], index=["X", "Y", "Z"], name="s0")
    >>> features = FeatureVector(data=data)
    >>> features.data #doctest: +NORMALIZE_WHITESPACE
    X    1
    Y    2
    Z    3
    Name: s0, dtype: int64
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        data: pd.Series,
    ) -> None:
        if not isinstance(data, pd.Series):
            raise TypeError("data must be a pandas Series.")
        self.data = data

        # cache for associated random vector
        self._random_vector: RandomVector | None = None

    # --------------------- properties --------------------- #

    @property
    def name(self) -> Hashable:
        """Get the sample point identifier.

        Returns
        -------
        name : Hashable
            The identifier for this sample point.
        """
        return self.data.name

    @name.setter
    def name(self, name: Hashable) -> None:
        """Set the sample point identifier.

        Parameters
        ----------
        name : Hashable
            New identifier for this sample point.

        Raises
        ------
        TypeError
            If `name` is not hashable.
        """
        if not isinstance(name, Hashable):
            raise TypeError("name must be a Hashable.")
        self.data.name = name

    def random_vector(self) -> RandomVector | None:
        """Get the associated random vector, if available.

        Returns
        -------
        random_vector : RandomVector | None
            The random vector from which these features were derived, or `None`
            if not set.
        """
        return self._random_vector

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
        >>> Omega = SampleSpace.generate_sequence(size=1, prefix="s")
        >>> X = RandomVector(domain=Omega).from_dict({"s": (1, 2)})
        >>> features = FeatureVector.from_random_vector("s", X)
        >>> int(features.feature_at[0])
        1
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

        Returns
        -------
        value : Any
            The value of the specified feature.

        Raises
        ------
        KeyError
            If the feature name is not found.

        Examples
        --------
        >>> from sigalg.core import FeatureVector, RandomVector, SampleSpace
        >>> Omega = SampleSpace.generate_sequence(size=1, prefix="s")
        >>> X = RandomVector(domain=Omega).from_dict({"s": (1, 2)})
        >>> features = FeatureVector.from_random_vector("s", X)
        >>> int(features["X_0"])
        1
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

    # --------------------- class methods --------------------- #

    @classmethod
    def from_random_vector(
        cls,
        sample_index: Hashable,
        random_vector: RandomVector,
    ) -> FeatureVector:
        """Extract a feature vector from a random vector.

        Extracts the feature vector for a specific sample point from a random
        vector. This represents evaluating a random vector `X: Omega -> S` at
        a single point: `X(omega)` for some `omega` in `Omega`.

        Parameters
        ----------
        sample_index : Hashable
            Index of the sample point.
        random_vector : RandomVector
            The random vector to extract features from.

        Returns
        -------
        features : FeatureVector
            Feature values for the specified sample point.

        Examples
        --------
        >>> from sigalg.core import FeatureVector, RandomVector, SampleSpace
        >>> Omega = SampleSpace.generate_sequence(size=2, prefix="s")
        >>> X = RandomVector(domain=Omega).from_dict({"s_0": (1, 2), "s_1": (3, 4)})
        >>> features = FeatureVector.from_random_vector("s_0", X)
        >>> features #doctest: +NORMALIZE_WHITESPACE
        Feature vector of 's_0':
                 s_0
        feature
        X_0        1
        X_1        2
        """
        data = random_vector.data.loc[sample_index]
        features = cls(data=data)
        features._random_vector = random_vector
        return features

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
