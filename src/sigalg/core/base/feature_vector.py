"""Feature vector module.

Given a random vector `X: Omega -> S`, a `FeatureVector` object represents `X(omega)` for a specific `omega` in the
sample space.

Classes
-------
FeatureVector
    Represents the feature vector for a single sample point.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ..random_objects.random_vector import RandomVector


class FeatureVector:
    """A class representing a feature vector for a single sample point.

    Given a random vector `X: Omega -> S`, a `FeatureVector` represents the output `X(omega)` for a specific sample point `omega`.
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
        """Create a `FeatureVector` from a `pd.Series`.

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

    def from_rv(
        self, sample_index: Hashable, random_vector: RandomVector
    ) -> FeatureVector:
        """Associate a `RandomVector` with this `FeatureVector`.

        Parameters
        ----------
        random_vector : RandomVector
            The random vector to associate.

        Returns
        -------
        self : FeatureVector
            The updated FeatureVector instance.
        """
        self._rv = random_vector
        self.data = random_vector.data.loc[sample_index]
        return self

    # --------------------- properties --------------------- #

    @property
    def data(self) -> pd.Series:
        """Get the feature vector data.

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
        """Get the sample point identifier.

        Returns
        -------
        name : Hashable
            The identifier for this sample point.
        """
        return self._name

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
