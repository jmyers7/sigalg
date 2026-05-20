"""A base class representing an ordered collection of hashable items."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import pandas as pd


class Index:
    """A base class representing an ordered collection of hashable items.

    Subclasses include `Domain`, `SampleSpace`, `Event`, and `Time`. Instances of the base class are used to index instances of `RandomVector` of dimension > 1. The underlying data structure is a `pd.Index` object stored in the `data` attribute.

    Parameters
    ----------
    name : Hashable, default="I"
        Name identifier for the index.
    **kwargs
        Additional keyword arguments passed to subclasses.

    Raises
    ------
    TypeError
        If `name` is not hashable.

    Examples
    --------
    >>> from sigalg.core import Index
    >>> I = Index().from_list(["a", "b", "c"])
    >>> print(I) # doctest: +NORMALIZE_WHITESPACE
    Index 'I':
    ['a', 'b', 'c']
    """

    # --------------------- constructors --------------------- #

    _properties = ["_data_name", "_indices", "_dimension", "_data"]

    def __init__(
        self,
        name: Hashable = "I",
        **kwargs,
    ) -> None:
        if not isinstance(name, Hashable):
            raise TypeError("name must be hashable.")

        self._name = name
        self._initialize_property_caches()

    def _initialize_property_caches(self) -> None:
        for property in self._properties:
            setattr(self, property, None)

    def from_list(
        self,
        indices: list,
        data_name: list | None = None,
    ) -> Index:
        """Create an index from a list.

        Parameters
        ----------
        indices : list
            A list of unique hashable items to use as the index. If the list contains tuples, all tuples must be the same length, and the underlying `pd.Index` will be a `pd.MultiIndex`.
        data_name : list | None, default=None
            A list of names for the underlying `pd.Index` object. If the list contains more than one name, the underlying `pd.Index` will be a `pd.MultiIndex`, and the number of names must match the length of the tuples in `indices`. If `None`, the underlying default will be set to `data_name=["index"]`.

        Raises
        ------
        TypeError
            If `indices` is not a list, or if `data_name` is not a list (when given).
        ValueError
            If `indices` contains duplicate items, or if `indices` is a list of tuples of different lengths, or if the length of `data_name` does not match the length of the tuples in `indices` (when `indices` is a list of tuples), or if the length of `data_name` is not 1 when `indices` is a list of non-tuples (when `data_name` is given).

        Returns
        -------
        self : Index
            The current `Index` instance with updated indices.

        Examples
        --------
        >>> from sigalg.core import Index
        >>> I1 = Index(name="I1").from_list(["a", "b", "c"])
        >>> print(I1) # doctest: +NORMALIZE_WHITESPACE
        Index 'I1':
        ['a', 'b', 'c']
        >>> print(I1.dimension)
        1
        >>> I2 = Index(name="I2").from_list([("a", 1), ("b", 2), ("c", 3)])
        >>> print(I2) # doctest: +NORMALIZE_WHITESPACE
        Index 'I2':
        [('a', 1), ('b', 2), ('c', 3)]
        >>> print(I2.dimension)
        2
        """
        if not isinstance(indices, list):
            raise TypeError("indices must be a list.")
        if len(indices) != len(set(indices)):
            raise ValueError("All items in 'indices' must be unique.")
        if len(indices) > 0:
            if isinstance(indices[0], tuple):
                tuple_length = len(indices[0])
                if not all(
                    isinstance(item, tuple) and len(item) == tuple_length
                    for item in indices
                ):
                    raise ValueError(
                        "All items in 'indices' must be tuples of the same length."
                    )
                is_multi_index = True
            else:
                is_multi_index = False

            if data_name is not None and not isinstance(data_name, list):
                raise TypeError("If given, data_name must be a list.")

            if is_multi_index:
                if data_name is not None and len(data_name) != tuple_length:
                    raise ValueError(
                        "If 'indices' is a list of tuples, 'data_name' must have the same length as the tuples."
                    )
                if data_name is None:
                    data_name = [f"index_{i}" for i in range(tuple_length)]
            else:
                if data_name is not None and len(data_name) != 1:
                    raise ValueError(
                        "If 'indices' is a list of non-tuples, 'data_name' must have length 1."
                    )
                if data_name is None:
                    data_name = ["index"]
        else:
            if data_name is None:
                data_name = ["index"]

        self._initialize_property_caches()
        self._indices = indices
        self._data_name = data_name
        self._dimension = len(data_name)
        return self

    def from_pandas(self, data: pd.Index) -> Index:
        """Create an index from a `pd.Index` object.

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
        >>> import pandas as pd
        >>> from sigalg.core import Index
        >>> data1 = pd.Index(["a", "b", "c"], name="index")
        >>> I1 = Index(name="I1").from_pandas(data1)
        >>> print(I1) # doctest: +NORMALIZE_WHITESPACE
        Index 'I1':
        ['a', 'b', 'c']
        >>> print(I1.dimension)
        1
        >>> data2 = pd.MultiIndex.from_tuples([("a", 1), ("b", 2), ("c", 3)], names=["letter", "number"])
        >>> I2 = Index(name="I2").from_pandas(data2)
        >>> print(I2) # doctest: +NORMALIZE_WHITESPACE
        Index 'I2':
        [('a', 1), ('b', 2), ('c', 3)]
        >>> print(I2.dimension)
        2
        """
        if not isinstance(data, pd.Index):
            raise TypeError("data must be a pd.Index.")

        self._initialize_property_caches()
        self._data = data.copy()
        self._data_name = data.names
        if isinstance(data, pd.MultiIndex):
            self._dimension = len(data.names)
        else:
            self._dimension = 1

        return self

    def from_sequence(
        self,
        size: int,
        initial_index: int = 0,
        prefix: Hashable | None = None,
        data_name: list | None = None,
    ) -> Index:
        """Create an index with sequentially numbered items.

        Parameters
        ----------
        size : int
            Number of features to generate. Must be positive.
        initial_index : int, default=0
            Starting index for sequential numbering.
        prefix : Hashable | None, default=None
            Prefix for index names. If `None` or non-string hashable is given, then numerical indices are used.
        data_name : list | None, default=None
            A list containing a single element for the name of the underlying `pd.Index` object. If `None`, the default will be set to `data_name=["index"]`.

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
            `name` is not hashable, or `data_name` is not a list with a single element (if given).

        Examples
        --------
        >>> from sigalg.core import Index
        >>> I_1 = Index(name="I_1").from_sequence(size=3, prefix="F")
        >>> print(I_1) # doctest: +NORMALIZE_WHITESPACE
        Index 'I_1':
        ['F_0', 'F_1', 'F_2']
        >>> I_2 = Index(name="I_2").from_sequence(size=2, initial_index=5)
        >>> print(I_2) # doctest: +NORMALIZE_WHITESPACE
        Index 'I_2':
        [5, 6]
        """
        if not isinstance(size, int) or size <= 0:
            raise ValueError("'size' must be a positive integer.")
        if not isinstance(initial_index, int):
            raise TypeError("'initial_index' must be an integer.")
        if prefix is not None and not isinstance(prefix, Hashable):
            raise TypeError("If given, 'prefix' must be hashable.")

        self._initialize_property_caches()

        if data_name is not None and not isinstance(data_name, list):
            raise TypeError(
                "If given, 'data_name' must be a list with a single element."
            )
        if data_name is not None and len(data_name) != 1:
            raise ValueError(
                "If given, 'data_name' must be a list with a single element."
            )
        if data_name is None:
            data_name = ["index"]

        if prefix is None or not isinstance(prefix, str):
            indices = list(range(initial_index, initial_index + size))
        else:
            if size == 1:
                indices = [prefix]
            else:
                indices = [
                    f"{prefix}_{i}" for i in range(initial_index, initial_index + size)
                ]
        return self.from_list(indices=indices, data_name=data_name)

    # --------------------- properties --------------------- #

    @property
    def indices(self) -> list | None:
        """Get the list of items in the index.

        Returns
        -------
        indices : list
            The list of items in this index.

        Examples
        --------
        >>> import pandas as pd
        >>> from sigalg.core import Index
        >>> pd_index = pd.Index(["a", "b", "c"], name="index")
        >>> I_1 = Index(name="I_1").from_pandas(pd_index)
        >>> print(I_1.indices)
        ['a', 'b', 'c']
        >>> I_2 = Index(name="I_2").from_list(["x", "y", "z"])
        >>> print(I_2.indices)
        ['x', 'y', 'z']
        """
        if self._indices is None and self._data is not None:
            self._indices = self.data.to_list()
        return self._indices

    @property
    def data(self) -> pd.Index | None:
        """Get the underlying `pd.Index` object.

        Returns
        -------
        data : pd.Index | None
            The underlying `pd.Index` object.

        Examples
        --------
        >>> import pandas as pd
        >>> from sigalg.core import Index
        >>> data = pd.Index(["a", "b", "c"], name="index")
        >>> I_1 = Index(name="I_1").from_pandas(data)
        >>> print(I_1.data)
        Index(['a', 'b', 'c'], dtype='str', name='index')
        >>> I_2 = Index(name="I_2").from_list(["x", "y", "z"], data_name=["letters"])
        >>> print(I_2.data)
        Index(['x', 'y', 'z'], dtype='str', name='letters')
        """
        if self._data is None and self._indices is not None:
            self._data = pd.Index(self._indices)
            self._data.names = self._data_name
        return self._data

    @property
    def data_name(self) -> list | None:
        """Get the name of the underlying `pd.Index` object.

        If the index is not a `pd.MultiIndex`, this will be a list containing a single element. If the index is a `pd.MultiIndex`, this will be a list of names corresponding to each level of the `MultiIndex`.

        Returns
        -------
        data_name : list | None
            The name of the underlying `pd.Index` object.

        Examples
        --------
        >>> import pandas as pd
        >>> from sigalg.core import Index
        >>> I_1 = Index(name="I_1").from_list(["x", "y", "z"], data_name=["letters"])
        >>> print(I_1.data_name)
        ['letters']
        >>> data = pd.MultiIndex.from_tuples([("a", 1), ("b", 2), ("c", 3)], names=["letter", "number"])
        >>> I_2 = Index(name="I_2").from_pandas(data)
        >>> print(I_2.data)
        MultiIndex([('a', 1),
                    ('b', 2),
                    ('c', 3)],
                   names=['letter', 'number'])
        >>> print(I_2.data_name)
        ['letter', 'number']
        >>> I_2.data_name = ["new_letter", "new_number"]
        >>> print(I_2.data)
        MultiIndex([('a', 1),
                    ('b', 2),
                    ('c', 3)],
                   names=['new_letter', 'new_number'])
        >>> print(I_2.data_name)
        ['new_letter', 'new_number']
        """
        return self._data_name

    @data_name.setter
    def data_name(self, data_name: list) -> None:
        """Set the name of the underlying `pd.Index` object.

        Parameters
        ----------
        data_name : list
            If the index is not a `pd.MultiIndex`, this should be a list containing a single element. If the index is a `pd.MultiIndex`, this should be a list of names corresponding to each level of the `MultiIndex`.

        Raises
        ------
        TypeError
            If `data_name` is not a list.
        """
        if not isinstance(data_name, list):
            raise TypeError("data_name must be a list.")
        if self._data is not None:
            self._data.names = data_name
            self._data_name = data_name

    @property
    def dimension(self) -> int | None:
        """Get the dimension of the index.

        The dimension is 1 if the underlying `pd.Index` is a regular `Index`, and is equal to the number of levels if the underlying `pd.Index` is a `MultiIndex`.

        Returns
        -------
        dimension : int | None
            The dimension of the index.
        """
        return self._dimension

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
    def name(self, name: Hashable) -> None:
        """Set the name identifier for this index.

        Parameters
        ----------
        name : Hashable
            New name for this index.

        Raises
        ------
        TypeError
            If `name` is not hashable.
        """
        if not isinstance(name, Hashable):
            raise TypeError("name must be hashable.")
        self._name = name

    def with_name(self, name: Hashable) -> Index:
        """Return a new index with the given name.

        Parameters
        ----------
        name : Hashable
            New name for the index.

        Returns
        -------
        index : Index
            A new `Index` with the specified name.
        """
        self.name = name
        return self

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
            return Index(name=self.name).from_pandas(data=data)
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
        if self.data is None:
            if self.name is None:
                return "Index: empty"
            else:
                return f"Index '{self.name}': empty"
        else:
            if self.name is None:
                return f"Index:\n{self.data.to_list()}"
            else:
                return f"Index '{self.name}':\n{self.data.to_list()}"
