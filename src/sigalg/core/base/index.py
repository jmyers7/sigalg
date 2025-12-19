"""Base index class for ordered collections.

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
>>> idx = Index(indices=["a", "b", "c"], name="MyIndex")
>>> len(idx)
3
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import pandas as pd

from ...validation.index import IndexIn


class Index:
    """Base class for ordered collections of hashable items.

    The `Index` class provides a foundation for representing ordered collections
    with validation, indexing, iteration, equality operations, and other attributes. It wraps a `pd.Index` internally.

    Parameters
    ----------
    indices : list[Hashable]
        Ordered collection of unique hashable items. (Any iterable of hashable items is acceptable and will be coerced into a list internally.)
    name : Hashable, optional
        Name identifier for the index. Defaults to the class-level `index`.
    data_name : Hashable, optional
        Name for the internal `pd.Index`. Defaults to the class-level `data`.
    **kwargs
        Additional keyword arguments passed to subclasses.

    Raises
    ------
    pydantic.ValidationError
        If any of the parameters are invalid.

    Examples
    --------
    >>> from sigalg.core import Index
    >>> idx = Index(indices=["a", "b", "c"], name="MyIndex")
    >>> len(idx)
    3
    >>> list(idx)
    ['a', 'b', 'c']
    """

    DEFAULT_NAME = "index"
    DEFAULT_DATA_NAME = "data"
    DEFAULT_PREFIX = "index"

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        indices: list[Hashable],
        name: Hashable | None = None,
        data_name: Hashable | None = None,
        **kwargs,
    ) -> None:

        name = self.DEFAULT_NAME if name is None else name
        data_name = self.DEFAULT_DATA_NAME if data_name is None else data_name

        # input validation
        v = IndexIn(indices=indices, name=name, data_name=data_name)

        self.indices = v.indices
        self._name = v.name
        self._data_name = v.data_name

        # cache for properties
        self._data: pd.Index | None = None

    # --------------------- properties --------------------- #

    @property
    def data(self) -> pd.Index:
        """Get the underlying `pd.Index`.

        Returns
        -------
        data : pd.Index
            The underlying `pd.Index` object.
        """
        if self._data is None:
            self._data = pd.Index(self.indices, name=self._data_name)
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
    def name(self) -> Hashable:
        """Get the name identifier for this index.

        Returns
        -------
        name : Hashable
            The name of this index.
        """
        return self._name

    @name.setter
    def name(self, name: Hashable) -> None:
        """Set the name identifier for this index.

        Parameters
        ----------
        name : Hashable
            New name for this index.

        Raises
        ------
        TypeError
            If `name` is not a hashable.
        """
        if not isinstance(name, Hashable):
            raise TypeError("name must be hashable.")
        self._name = name

    # --------------------- factory methods --------------------- #

    @classmethod
    def from_pandas(
        cls,
        data: pd.Index,
        name: Hashable | None = None,
    ) -> Index:
        """Create an Index from a `pd.Index`.

        Parameters
        ----------
        data : pd.Index
            `pd.Index` object to use for the index.
        name : Hashable, optional
            Name identifier for the index.

        Raises
        ------
        TypeError
            If `data` is not a `pd.Index`.

        Returns
        -------
        index : Index
            A new `Index` instance created from the provided `pd.Index`.

        Examples
        --------
        >>> from sigalg.core import Index
        >>> import pandas as pd
        >>> pd_index = pd.Index(['a', 'b', 'c'])
        >>> idx = Index.from_pandas(pd_index, name='MyIndex')
        >>> list(idx)
        ['a', 'b', 'c']
        """
        if not isinstance(data, pd.Index):
            raise TypeError("data must be a pd.Index.")

        name = cls.DEFAULT_NAME if name is None else name

        indices = data.to_list()
        index = cls(indices=indices, name=name)
        index.data = data
        return index

    @classmethod
    def generate_default(
        cls,
        initial_index: int = 0,
        size: int = 10,
        prefix: str | None = None,
        name: Hashable | None = None,
        data_name: Hashable | None = None,
    ) -> Index:
        """Generate a default index with automatically named features.

        Creates an index with indices named using a `prefix` and sequential
        indices. For single indices, only the `prefix` is used. For larger
        indices, numbers are appended (e.g., "X0", "X1", ...).

        Parameters
        ----------
        initial_index : int, default=0
            Starting index for sequential numbering.
        size : int, default=10
            Number of features to generate. Must be positive.
        prefix : str, optional
            String prefix for index names.
        name : Hashable, optional
            Name identifier for the index.
        data_name : Hashable, optional
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
            If `initial_index` is not an integer, `prefix` is not a string,
            `name` is not hashable, or `data_name` is not hashable.

        Examples
        --------
        >>> from sigalg.core import Index
        >>> index = Index.generate_default(size=3, prefix="F")
        >>> list(index)
        ['F0', 'F1', 'F2']
        """
        if not isinstance(size, int) or size <= 0:
            raise ValueError("'size' must be a positive integer.")
        if not isinstance(initial_index, int):
            raise TypeError("'initial_index' must be an integer.")
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("If given, 'name' must be hashable.")
        if data_name is not None and not isinstance(data_name, Hashable):
            raise TypeError("If given, 'data_name' must be hashable.")
        if prefix is not None and not isinstance(prefix, str):
            raise TypeError("'prefix' must be a string.")

        name = cls.DEFAULT_NAME if name is None else name
        data_name = cls.DEFAULT_DATA_NAME if data_name is None else data_name
        prefix = cls.DEFAULT_PREFIX if prefix is None else prefix

        if size == 1:
            indices = [prefix]
        else:
            indices = [
                f"{prefix}{i}" for i in range(initial_index, initial_index + size)
            ]
        return cls(indices=indices, name=name, data_name=data_name)

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
            return Index.from_pandas(data=data, name=self.name)
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

        Two indices are equal if they have the same elements in the same order.

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
        return f"Index '{self.name}':\n{self.data.to_list()}"
