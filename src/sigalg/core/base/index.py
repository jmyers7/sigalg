"""Base index class for ordered collections.

This module provides the `Index` class, which serves as the base class for
ordered collections of hashable items. It wraps a `pd.Index` and provides
validation, indexing, and iteration capabilities.

Classes
-------
Index
    Base class for ordered collections of hashable items.

Examples
--------
>>> import sigalg as sa
>>> idx = sa.Index(indices=["a", "b", "c"], name="MyIndex")
>>> len(idx)
3
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import pandas as pd


class Index:
    """Base class for ordered collections of hashable items.

    The `Index` class provides a foundation for representing ordered collections
    with validation, indexing, iteration, and equality operations. It wraps a
    `pd.Index` internally for efficient storage and manipulation.

    Parameters
    ----------
    indices : list of Hashable, optional
        List of hashable items to include in the index.
        Mutually exclusive with `values`.
    values : pd.Index, optional
        `pd.Index` object to use directly.
        Mutually exclusive with `indices`.
    name : str, optional
        Name identifier for the index.
    values_name : str, optional
        Name for the internal `pd.Index`.
    **kwargs
        Additional keyword arguments passed to subclasses.

    Raises
    ------
    ValueError
        If both `indices` and `values` are provided, or if neither is provided.
        If `indices` contains duplicate values.
    TypeError
        If `indices` is not a list, `values` is not a `pd.Index`,
        or any item in `indices` is not hashable.
    Examples
    --------
    >>> import sigalg as sa
    >>> idx = sa.Index(indices=["a", "b", "c"], name="MyIndex")
    >>> len(idx)
    3
    >>> list(idx)
    ['a', 'b', 'c']
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        indices: list[Hashable] | None = None,
        values: pd.Index | None = None,
        name: str | None = None,
        values_name: str | None = None,
        **kwargs,
    ) -> None:
        self._validate_parameters(
            indices=indices, values=values, name=name, values_name=values_name
        )

        if values is not None:
            self.values = values
            self.indices = values.to_list()
            self.values_name = values.name
        elif indices is not None:
            self.values = pd.Index(data=indices, name=values_name)
            self.indices = indices
            self.values_name = values_name
        self._name = name

    # --------------------- properties --------------------- #

    @property
    def name(self) -> str:
        """Get the name identifier for this index.

        Returns
        -------
        str
            The name of this index.
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """Set the name identifier for this index.

        Parameters
        ----------
        name : str
            New name for this index.

        Raises
        ------
        TypeError
            If name is not a string.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        self._name = name

    # --------------------- data access methods --------------------- #

    def __getitem__(self, key: Any) -> Any:
        """Access elements by (position) index or slice.

        Parameters
        ----------
        key : Any
            Index, slice, or other key for accessing elements.

        Returns
        -------
        Any
            The indexed element(s) from the index.
        """
        return self._getitem_hook(key=key)

    def _getitem_hook(self, key: Any) -> Any:
        """Hook for subclasses to customize indexing behavior.

        Parameters
        ----------
        key : Any
            Index, slice, or other key for accessing elements.

        Returns
        -------
        Any
            The indexed element(s) from the index.
        """
        return self.values[key]

    # --------------------- sequence methods --------------------- #

    def __len__(self) -> int:
        """Return the number of elements in the index.

        Returns
        -------
        int
            The number of elements in this index.
        """
        return len(self.values)

    def __iter__(self) -> iter:
        """Return an iterator over the elements.

        Yields
        ------
        Hashable
            Each element in the index in order.
        """
        return iter(self.values)

    # --------------------- equality --------------------- #

    def __eq__(self, other: Index) -> bool:
        """Check equality with another index.

        Two indices are equal if they have the same elements in the same order.

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        bool
            `True` if the other object is an `Index` with identical values,
            `False` otherwise.
        """
        return isinstance(other, Index) and self.values.equals(other.values)

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        indices: list[Hashable] | None = None,
        values: pd.Index | None = None,
        name: str | None = None,
        values_name: str | None = None,
    ) -> None:
        """Validate index construction parameters.

        Parameters
        ----------
        indices : list of Hashable, optional
            List of hashable items to validate.
        values : pd.Index, optional
            pandas Index to validate.
        name : str, optional
            Name to validate.
        values_name : str, optional
            Values name to validate.

        Raises
        ------
        ValueError
            If both `indices` and `values` are provided, or if neither is provided.
            If `indices` or `values` contain duplicate items.
        TypeError
            If `indices` is not a list, `values` is not a `pd.Index`,
            `name` is not a string, `values_name` is not a string, or any item
            in `indices` is not hashable.
        """
        if indices is not None and values is not None:
            raise ValueError("Cannot specify both 'indices' and 'values'.")
        if indices is None and values is None:
            raise ValueError("Must specify either 'indices' or 'values'.")
        if indices is not None:
            if not isinstance(indices, list):
                raise TypeError("indices must be a list of Hashable items.")
            for idx in indices:
                if not isinstance(idx, Hashable):
                    raise TypeError("All items in 'indices' must be Hashable.")
            if len(indices) != len(set(indices)):
                raise ValueError("All items in 'indices' must be unique.")
        if values is not None:
            if not isinstance(values, pd.Index):
                raise TypeError("values must be a pandas Index.")
            if len(values) != values.nunique():
                raise ValueError("All items in 'values' must be unique.")
        if name is not None and not isinstance(name, str):
            raise TypeError("name must be a string.")
        if values_name is not None and not isinstance(values_name, str):
            raise TypeError("values_name must be a string.")
