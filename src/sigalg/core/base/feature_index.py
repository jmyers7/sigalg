"""Feature indices for representing feature spaces.

This module provides the `FeatureIndex` class, which is used to index feature embeddings.

Classes
-------
FeatureIndex
    Represents an ordered collection of feature identifiers.

Examples
--------
>>> from sigalg.core import FeatureIndex
>>> feature_index = FeatureIndex(indices=["X0", "X1", "X2"])
>>> len(feature_index)
3
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import pandas as pd

from .index import Index


class FeatureIndex(Index):
    """An ordered collection of feature identifiers.

    A feature index is used to index feature embeddings, storing
    an ordered collection of feature names or identifiers. It provides indexing
    and iteration capabilities for working with features.

    Parameters
    ----------
    indices : list[Hashable]
        `list[Hashable]` of hashable feature identifiers.
    values : pd.Index, optional
        `pd.Index` object containing feature identifiers.
        Mutually exclusive with indices.
    values_name : str, default="feature"
        Name for the index of values.

    Raises
    ------
    TypeError
        If `indices` is not a `list` or `values` is not a `pd.Index`.
    ValueError
        If both `indices` and `values` are provided, or if neither is provided.
        If `indices` contains duplicate values.

    Examples
    --------
    >>> from sigalg.core import FeatureIndex
    >>> feature_index = FeatureIndex(indices=["X0", "X1", "X2"])
    >>> len(feature_index)
    3
    >>> list(feature_index)
    ['X0', 'X1', 'X2']
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        indices: list[Hashable] | None = None,
        values: pd.Index | None = None,
        values_name: str | None = "feature",
    ) -> None:
        super().__init__(
            indices=indices, values=values, name=None, values_name=values_name
        )

    # --------------------- factory methods --------------------- #

    @classmethod
    def generate_default(
        cls,
        initial_index: int = 0,
        size: int = 10,
        prefix: str = "X",
        values_name: str = "feature",
    ) -> FeatureIndex:
        """Generate a default feature index with automatically named features.

        Creates a feature index with features named using a `prefix` and sequential
        indices. For single-feature indices, only the `prefix` is used. For larger
        indices, numbers are appended (e.g., "X0", "X1", ...).

        Parameters
        ----------
        initial_index : int, default=0
            Starting index for sequential numbering.
        size : int, default=10
            Number of features to generate. Must be positive.
        prefix : str, default="X"
            String prefix for feature names.
        values_name : str, default="feature"
            Name for the index of values.

        Returns
        -------
        feature_index : FeatureIndex
            A new `FeatureIndex` with automatically generated feature names.

        Raises
        ------
        ValueError
            If `size` is not a positive integer.
        TypeError
            If `initial_index` is not an integer, `prefix` is not a string,
            or `values_name` is not a string.

        Examples
        --------
        >>> from sigalg.core import FeatureIndex
        >>> features = FeatureIndex.generate_default(size=3, prefix="F")
        >>> list(features)
        ['F0', 'F1', 'F2']
        """
        if not isinstance(size, int) or size <= 0:
            raise ValueError("'size' must be a positive integer.")
        if not isinstance(initial_index, int):
            raise TypeError("'initial_index' must be an integer.")
        if values_name is not None and not isinstance(values_name, str):
            raise TypeError("If given, 'values_name' must be a string.")
        if not isinstance(prefix, str):
            raise TypeError("'prefix' must be a string.")

        if size == 1:
            indices = [prefix]
        else:
            indices = [
                f"{prefix}{i}" for i in range(initial_index, initial_index + size)
            ]
        return cls(indices=indices, values_name=values_name)

    # --------------------- data access methods --------------------- #

    def _getitem_hook(self, key: Any) -> FeatureIndex:
        """Internal hook for indexing operations to create events.

        This method is called by `__getitem__` from the parent `Index` class. In `FeatureIndex`, the purpose of this method is to ensure that `__getitem__` returns an instance of `FeatureIndex`. Items are retrieved by position.

        Parameters
        ----------
        key : int, slice, or list
            Indexing key for accessing features. An integer creates a single-element
            feature index, a slice creates a feature index with a slice of features,
            and a `list` creates a feature index with multiple features.

        Returns
        -------
        feature_index : FeatureIndex
            A `FeatureIndex` object containing the indexed features.

        Examples
        --------
        >>> from sigalg.core import FeatureIndex
        >>> features = FeatureIndex(indices=["F0", "F1", "F2", "F3"])
        >>> # Access via integer index
        >>> feature1 = features[0]
        >>> # Access via slice
        >>> feature2 = features[1:3]
        >>> # Access via list of positions
        >>> feature3 = features[[0, 2]]
        """  # noqa: D401
        if isinstance(key, int):
            # result = [self.values[key]]
            return self.values[key]
        else:
            FeatureIndex(
                indices=self.values[key].to_list(), values_name=self.values_name
            )

    # --------------------- equality --------------------- #

    def __eq__(self, other: FeatureIndex) -> bool:
        """Check equality with another feature index.

        Two feature indices are equal if they have the same features in the
        same order.

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the other object is a `FeatureIndex` with identical values,
            `False` otherwise.
        """
        return isinstance(other, FeatureIndex) and super().__eq__(other)
