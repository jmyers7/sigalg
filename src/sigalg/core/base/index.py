"""Index module.

This module provides the `Index` class, which serves as the base class for
ordered collections of hashable items. It wraps a `pd.Index` and provides
validation, indexing, iteration capabilities, and other attributes.

Classes
-------
Index
    Base class for ordered collections of hashable items.

Examples
--------
>>> from sigalg.core import Index
>>> idx = Index().from_list(indices=["a", "b", "c"])
>>> idx # doctest: +NORMALIZE_WHITESPACE
Index:
['a', 'b', 'c']
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import pandas as pd


class Index:
    """A base class representing an ordered collection of hashable items.

    The `Index` class provides a foundation for representing ordered collections
    with validation, indexing, iteration, equality operations, and other attributes. It wraps a `pd.Index` internally.

    Parameters
    ----------
    name : Hashable | None, default=None
        Name identifier for the index.
    data_name : Hashable | None, default=None
        Name for the internal `pd.Index`.
    **kwargs
        Additional keyword arguments passed to subclasses.

    Raises
    ------
    TypeError
        If `name` or `data_name` is not `None` and is not hashable.

    Examples
    --------
    >>> from sigalg.core import Index
    >>> idx = Index(name="an_index").from_list(indices=["a", "b", "c"])
    >>> idx # doctest: +NORMALIZE_WHITESPACE
    Index 'an_index':
    ['a', 'b', 'c']
    """

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        name: Hashable | None = None,
        data_name: Hashable | None = None,
        **kwargs,
    ) -> None:
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("name must be hashable.")
        if data_name is not None and not isinstance(data_name, Hashable):
            raise TypeError("data_name must be hashable.")

        self._name = name
        self._data_name = data_name

        # cache for properties
        self._indices: list[Hashable] | None = None
        self._data: pd.Index | None = None

    def from_list(
        self,
        indices: list[Hashable],
    ) -> Index:
        """Create an `Index` from a list of hashable items.

        Parameters
        ----------
        indices : list[Hashable]
            List of hashable items to use for the index.

        Raises
        ------
        TypeError
            If `indices` is not a list of hashable items.
        ValueError
            If `indices` contains duplicate items.

        Returns
        -------
        self : Index
            The current `Index` instance with updated indices.
        """
        if not isinstance(indices, list):
            raise TypeError("indices must be a list of Hashable items.")
        for item in indices:
            if not isinstance(item, Hashable):
                raise TypeError("All items in 'indices' must be Hashable.")
        if len(indices) != len(set(indices)):
            raise ValueError("All items in 'indices' must be unique.")

        self._indices = indices
        return self

    def from_pandas(
        self,
        data: pd.Index,
    ) -> Index:
        """Create an `Index` from a `pd.Index`.

        Parameters
        ----------
        data : pd.Index
            `pd.Index` object to use for the index.

        Raises
        ------
        TypeError
            If `data` is not a `pd.Index`.

        Returns
        -------
        index : Index
            The current `Index` instance with updated data.

        Examples
        --------
        >>> from sigalg.core import Index
        >>> import pandas as pd
        >>> pd_index = pd.Index(['a', 'b', 'c'])
        >>> idx = Index(name="an_index").from_pandas(pd_index)
        >>> idx # doctest: +NORMALIZE_WHITESPACE
        Index 'an_index':
        ['a', 'b', 'c']
        """
        if not isinstance(data, pd.Index):
            raise TypeError("data must be a pd.Index.")

        if data.name is None:
            data.name = self._data_name

        self._data = data.copy()
        return self

    def from_sequence(
        self,
        size: int,
        initial_index: int = 0,
        prefix: Hashable | None = None,
    ) -> Index:
        """Create an `Index` with sequentially numbered items.

        Parameters
        ----------
        size : int
            Number of features to generate. Must be positive.
        initial_index : int, default=0
            Starting index for sequential numbering.
        prefix : Hashable | None, default=None
            Prefix for index names. If `None` or non-string hashable is given, then numerical indices are used.

        Returns
        -------
        index : Index
            A new `Index` with automatically generated indices.

        Raises
        ------
        ValueError
            If `size` is not a positive integer.
        TypeError
            If `initial_index` is not an integer, `prefix` is not hashable,
            `name` is not hashable, or `data_name` is not hashable (if given).

        Examples
        --------
        >>> from sigalg.core import Index
        >>> index1 = Index().from_sequence(size=3, prefix="F")
        >>> index1 # doctest: +NORMALIZE_WHITESPACE
        Index:
        ['F_0', 'F_1', 'F_2']
        >>> index2 = Index(name="an_index").from_sequence(size=2, initial_index=5)
        >>> index2 # doctest: +NORMALIZE_WHITESPACE
        Index 'an_index':
        [5, 6]
        """
        if not isinstance(size, int) or size <= 0:
            raise ValueError("'size' must be a positive integer.")
        if not isinstance(initial_index, int):
            raise TypeError("'initial_index' must be an integer.")
        if prefix is not None and not isinstance(prefix, Hashable):
            raise TypeError("If given, 'prefix' must be hashable.")

        if prefix is None or not isinstance(prefix, str):
            indices = list(range(initial_index, initial_index + size))
        else:
            if size == 1:
                indices = [prefix]
            else:
                indices = [
                    f"{prefix}_{i}" for i in range(initial_index, initial_index + size)
                ]
        return self.from_list(indices=indices)

    # --------------------- properties --------------------- #

    @property
    def indices(self) -> list[Hashable]:
        """Get the list of hashable items in the index.

        Returns
        -------
        indices : list[Hashable]
            The list of hashable items in this index.
        """
        if self._indices is None:
            self._indices = self.data.to_list()
        return self._indices

    @property
    def data(self) -> pd.Index:
        """Get the underlying `pd.Index`.

        Returns
        -------
        data : pd.Index
            The underlying `pd.Index` object.
        """
        if self._data is None:
            self._data = pd.Index(self._indices, name=self._data_name)
        return self._data

    @data.setter
    def data(self, data: pd.Index) -> None:
        """Set the underlying `pd.Index`.

        Parameters
        ----------
        data : pd.Index
            New `pd.Index` object to set.

        Raises
        ------
        TypeError
            If `data` is not a `pd.Index`.
        """
        if not isinstance(data, pd.Index):
            raise TypeError("data must be a pd.Index.")
        self._data = data

    @property
    def name(self) -> Hashable | None:
        """Get the name identifier for this index.

        Returns
        -------
        name : Hashable | None
            The name of this index.
        """
        return self._name

    @name.setter
    def name(self, name: Hashable | None) -> None:
        """Set the name identifier for this index.

        Parameters
        ----------
        name : Hashable | None
            New name for this index.

        Raises
        ------
        TypeError
            If `name` is not `None` and is not a hashable.
        """
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("name must be hashable.")
        self._name = name

    # --------------------- factory methods --------------------- #

    @classmethod
    def generate_sequence(
        cls,
        size: int,
        initial_index: int = 0,
        prefix: Hashable | None = None,
        name: Hashable | None = None,
        data_name: Hashable | None = None,
    ) -> Index:
        """Generate a sequential `Index`.

        Creates an `Index` with sequentially numbered items, optionally
        prefixed by a given string.

        Parameters
        ----------
        size : int
            Number of features to generate. Must be positive.
        initial_index : int, default=0
            Starting index for sequential numbering.
        prefix : Hashable | None, default=None
            Prefix for index names. If `None` or non-string hashable is given, then numerical indices are used.
        name : Hashable | None, default=None
            Name identifier for the index.
        data_name : Hashable | None, default=None
            Name for the index of values.

        Returns
        -------
        index : Index
            A new `Index` with automatically generated indices.

        Raises
        ------
        ValueError
            If `size` is not a positive integer.
        TypeError
            If `initial_index` is not an integer, `prefix` is not hashable,
            `name` is not hashable, or `data_name` is not hashable (if given).

        Examples
        --------
        >>> from sigalg.core import Index
        >>> index1 = Index.generate_sequence(size=3, prefix="F")
        >>> index1 # doctest: +NORMALIZE_WHITESPACE
        Index:
        ['F_0', 'F_1', 'F_2']
        >>> index2 = Index.generate_sequence(size=2, initial_index=5, name="an_index")
        >>> index2 # doctest: +NORMALIZE_WHITESPACE
        Index 'an_index':
        [5, 6]
        """
        return cls._generate_sequence(
            initial_index=initial_index,
            size=size,
            prefix=prefix,
            name=name,
            data_name=data_name,
        )

    @classmethod
    def _generate_sequence(
        cls,
        size: int,
        initial_index: int = 0,
        prefix: Hashable | None = None,
        name: Hashable | None = None,
        data_name: Hashable | None = None,
    ) -> Index:
        if not isinstance(size, int) or size <= 0:
            raise ValueError("'size' must be a positive integer.")
        if not isinstance(initial_index, int):
            raise TypeError("'initial_index' must be an integer.")
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("If given, 'name' must be hashable.")
        if data_name is not None and not isinstance(data_name, Hashable):
            raise TypeError("If given, 'data_name' must be hashable.")
        if prefix is not None and not isinstance(prefix, Hashable):
            raise TypeError("If given, 'prefix' must be hashable.")

        if prefix is None or not isinstance(prefix, str):
            indices = list(range(initial_index, initial_index + size))
        else:
            if size == 1:
                indices = [prefix]
            else:
                indices = [
                    f"{prefix}_{i}" for i in range(initial_index, initial_index + size)
                ]
        return cls(name=name, data_name=data_name).from_list(indices=indices)

    # --------------------- data access methods --------------------- #

    def __getitem__(self, pos: int | list[int] | slice) -> Any:
        """Access elements by (position) index or slice.

        Parameters
        ----------
        pos : int | list[int] | slice
            Index, slice, or other key for accessing elements positionally.

        Returns
        -------
        element : Any
            The indexed element(s) from the index.
        """
        return self._getitem_hook(pos=pos)

    def _getitem_hook(self, pos: int | list[int] | slice) -> Any:
        """Hook for subclasses to customize indexing behavior.

        Parameters
        ----------
        pos : int | list[int] | slice
            Index, slice, or other key for accessing elements positionally.

        Raise
        -----
        TypeError
            If `pos` is not an `int`, `list[int]`, or `slice`.

        Returns
        -------
        element : Any
            The indexed element(s) from the index.
        """  # noqa: D401
        if not isinstance(pos, (int, list, slice)):
            raise TypeError("pos must be int | list[int] | slice.")
        if isinstance(pos, list) and not all(isinstance(i, int) for i in pos):
            raise TypeError("pos list must contain only int.")

        data = self.data[pos]
        if isinstance(data, pd.Index):
            return Index(name=self.name, data_name=self.data.name).from_pandas(
                data=data
            )
        else:
            return data

    def __contains__(self, item: Hashable) -> bool:
        """Check if an item is in the index.

        Parameters
        ----------
        item : Hashable
            Item to check for membership in the index.

        Raises
        ------
        TypeError
            If `item` is not hashable.

        Returns
        -------
        contains : bool
            `True` if the item is in the index, `False` otherwise.
        """
        if not isinstance(item, Hashable):
            raise TypeError("item must be hashable.")
        return item in self.data

    # --------------------- sequence methods --------------------- #

    def __len__(self) -> int:
        """Return the number of elements in the index.

        Returns
        -------
        length : int
            The number of elements in this index.
        """
        return len(self.data)

    def __iter__(self) -> iter:
        """Return an iterator over the elements.

        Yields
        ------
        element : Hashable
            Each element in the index in order.
        """
        return iter(self.data)

    # --------------------- equality --------------------- #

    def __eq__(self, other: Index) -> bool:
        """Check equality with another index.

        Two indices are equal if they have the same elements in the same order. They may have different names and data names and still be considered equal.

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the other object is an `Index` with identical values,
            `False` otherwise.
        """
        return isinstance(other, Index) and self.data.equals(other.data)

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a string representation of the index.

        Returns
        -------
        repr_str : str
            String representation of the index.
        """
        if self._data is None and self._indices is None:
            return "Index with no data"
        if self.name is None:
            return f"Index:\n{self.data.to_list()}"
        else:
            return f"Index '{self.name}':\n{self.data.to_list()}"
