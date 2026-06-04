"""A base class representing an ordered collection of hashable items."""

from __future__ import annotations

from collections.abc import Hashable
from itertools import product
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
    >>> import pandas as pd
    >>> from sigalg.core import Index
    >>> I = Index().from_list([1, 2, 4])
    >>> print(I)  # doctest: +NORMALIZE_WHITESPACE
    Index 'I':
    I
    1
    2
    4
    >>> J = Index(name="J").from_list([(1, 2), (3, 4)])
    >>> print(J)  # doctest: +NORMALIZE_WHITESPACE
    Index 'J':
    J_0  J_1
    1    2
    3    4
    >>> data = pd.Index(["a", "b", "c"], name="letter")
    >>> K = Index(name="K").from_pandas(data)
    >>> print(K)  # doctest: +NORMALIZE_WHITESPACE
    Index 'K':
    letter
         a
         b
         c
    """

    # --------------------- constructors --------------------- #

    _properties = ["_variable_names", "_indices", "_dimension", "_data"]

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
        indices: list[Hashable],
        variable_names: list[Hashable] | None = None,
    ) -> Index:
        """Create an index from a list.

        The name(s) of the underlying `pd.Index` object will be set to the names in the parameter `variable_names` according to the following rules:

        * If `indices` is a list of non-tuples, then `variable_names` must be `None` or a list with a single element. If `variable_names` is `None`, the name of the underlying `pd.Index` will be set to the name of this index.
        * If `indices` is a list of tuples, then `variable_names` must be `None`, a list with a single element, or a list with the same length as the tuples. If `variable_names` is `None`, the names of the underlying `pd.MultiIndex` will be set to the name of this index followed by an underscore and the level number (e.g. `["I_0", "I_1", ...]`). If `variable_names` is a list with a single element, the names of the underlying `pd.MultiIndex` will be set to that name followed by an underscore and the level number (e.g. `["name_0", "name_1", ...]`).

        Parameters
        ----------
        indices : list[Hashable]
            A list of unique hashable items to use as the index. If the list contains tuples, all tuples must be the same length, and the underlying `pd.Index` will be a `pd.MultiIndex`.
        variable_names : list[Hashable] | None, default=None
            A list of names for the underlying `pd.Index` object. See the description above for details.

        Raises
        ------
        TypeError
            If `indices` is not a list of hashable items, or if `variable_names` is not a list of hashable items (if given).
        ValueError
            If `indices` contains duplicate items, or if `variable_names` does not have the correct length according to the rules described above.

        Returns
        -------
        self : Index
            The current `Index` instance with updated indices.

        Examples
        --------
        >>> from sigalg.core import Index
        >>> I_1 = Index(name="I_1").from_list(["a", "b", "c"])
        >>> print(I_1) # doctest: +NORMALIZE_WHITESPACE
        Index 'I_1':
        I_1
          a
          b
          c
        >>> print(I_1.dimension)
        1
        >>> I_2 = Index(name="I_2").from_list([("a", 1), ("b", 2), ("c", 3)])
        >>> print(I_2) # doctest: +NORMALIZE_WHITESPACE
        Index 'I_2':
        I_2_0  I_2_1
            a      1
            b      2
            c      3
        >>> print(I_2.dimension)
        2
        """
        if not isinstance(indices, list) or not all(
            isinstance(item, Hashable) for item in indices
        ):
            raise TypeError("indices must be a list of hashable items.")
        if len(indices) != len(set(indices)):
            raise ValueError("All items in 'indices' must be unique.")
        if variable_names is not None and not isinstance(variable_names, list):
            raise TypeError("If given, variable_names must be a list.")
        if variable_names is not None and not all(
            isinstance(name, Hashable) for name in variable_names
        ):
            raise TypeError("All items in 'variable_names' must be hashable.")

        if len(indices) == 0:
            if variable_names is None:
                variable_names = [self.name]
            elif len(variable_names) != 1:
                raise ValueError(
                    "If 'indices' is empty, 'variable_names' must have length 1."
                )
            tuple_length = 0
        else:
            if isinstance(indices[0], tuple):
                tuple_length = len(indices[0])
                if not all(
                    isinstance(item, tuple) and len(item) == tuple_length
                    for item in indices
                ):
                    raise ValueError(
                        "All items in 'indices' must be tuples of the same length."
                    )
                if tuple_length == 1:
                    indices = [item[0] for item in indices]
            else:
                tuple_length = 1

            if tuple_length > 1:
                if variable_names is None:
                    variable_names = [f"{self.name}_{i}" for i in range(tuple_length)]
                elif len(variable_names) == 1:
                    variable_names = [
                        f"{variable_names[0]}_{i}" for i in range(tuple_length)
                    ]
                elif len(variable_names) != tuple_length:
                    raise ValueError(
                        "If 'indices' is a list of tuples, 'variable_names' must be None, have length 1, or must have length equal to the tuple length."
                    )
            else:
                if variable_names is None:
                    variable_names = [self.name]
                elif len(variable_names) != 1:
                    raise ValueError(
                        "If 'indices' is a list of non-tuples, 'variable_names' must be None or have length 1."
                    )

        self._initialize_property_caches()
        self._indices = indices
        self._variable_names = variable_names
        return self

    def from_product(
        self,
        indices1: list[Hashable],
        indices2: list[Hashable],
        variable_names: list[Hashable],
    ) -> Index:
        """Create an index from the Cartesian product of two lists.

        Parameters
        ----------
        indices1 : list[Hashable]
            First list of unique hashable items to use as the first component of the Cartesian product.
        indices2 : list[Hashable]
            Second list of unique hashable items to use as the second component of the Cartesian product.
        variable_names : list[Hashable]
            A list of variable names.

        Raises
        ------
        TypeError
            If `indices1` or `indices2` is not a list of hashable items, or if `variable_names` is not a list of hashable items.
        ValueError
            If `indices1` or `indices2` contains duplicate items.

        Examples
        --------
        >>> from sigalg.core import Index
        >>> list1 = [1, 2, 3]
        >>> list2 = ["a", "b"]
        >>> I = Index().from_product(list1, list1, variable_names=["x", "y"])
        >>> print(I)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I':
        x  y
        1  1
        1  2
        1  3
        2  1
        2  2
        2  3
        3  1
        3  2
        3  3
        >>> list3 = [("a", "red"), ("b", "blue")]
        >>> J = Index(name="J").from_product(list1, list3, variable_names=["x", "y", "z"])
        >>> print(J)  # doctest: +NORMALIZE_WHITESPACE
        Index 'J':
        x y    z
        1 a  red
        1 b blue
        2 a  red
        2 b blue
        3 a  red
        3 b blue
        """
        if not isinstance(indices1, list) or not all(
            isinstance(item, Hashable) for item in indices1
        ):
            raise TypeError("indices1 must be a list of hashable items.")
        if not isinstance(indices2, list) or not all(
            isinstance(item, Hashable) for item in indices2
        ):
            raise TypeError("indices2 must be a list of hashable items.")
        if len(indices1) != len(set(indices1)):
            raise ValueError("All items in 'indices1' must be unique.")
        if len(indices2) != len(set(indices2)):
            raise ValueError("All items in 'indices2' must be unique.")
        if not isinstance(variable_names, list) or not all(
            isinstance(name, Hashable) for name in variable_names
        ):
            raise TypeError("variable_names must be a list of hashable items.")

        product_indices = list(product(indices1, indices2))
        flattened_indices = [self._flatten(t) for t in product_indices]
        return self.from_list(flattened_indices, variable_names=variable_names)

    @classmethod
    def cartesian_product(
        cls,
        index1: Index,
        index2: Index,
        variable_names: list[Hashable] | None = None,
    ) -> Index:
        """Create an index from the Cartesian product of two `Index` instances.

        Parameters
        ----------
        index1 : Index
            The first `Index` instance.
        index2 : Index
            The second `Index` instance.
        variable_names : list[Hashable] | None, default=None
            A list of variable names for the resulting index. If `None`, the variable names will be set to the concatenation of the variable names of `index1` and `index2`.

        Raises
        ------
        TypeError
            If `index1` or `index2` is not an `Index` instance, or if `variable_names` is not a list of hashable items (if given).

        Examples
        --------
        >>> from sigalg.core import Index
        >>> I = Index().from_list([1, 2, 3], variable_names=["x"])
        >>> J = Index(name="J").from_list(["a", "b"], variable_names=["y"])
        >>> product_1 = Index.cartesian_product(I, J)
        >>> print(product_1)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I x J':
        x y
        1 a
        1 b
        2 a
        2 b
        3 a
        3 b
        >>> K = Index(name="K").from_list([("a", "red"), ("b", "blue")], variable_names=["u", "v"])
        >>> product_2 = Index.cartesian_product(I, K)
        >>> print(product_2)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I x K':
        x u    v
        1 a  red
        1 b blue
        2 a  red
        2 b blue
        3 a  red
        3 b blue
        """
        if not isinstance(index1, Index):
            raise TypeError("index1 must be an Index instance.")
        if not isinstance(index2, Index):
            raise TypeError("index2 must be an Index instance.")
        if variable_names is not None and not isinstance(variable_names, list):
            raise TypeError("If given, variable_names must be a list.")
        if variable_names is not None and not all(
            isinstance(name, Hashable) for name in variable_names
        ):
            raise TypeError("All items in variable_names must be hashable.")

        if variable_names is None:
            variable_names = index1.variable_names + index2.variable_names

        product_indices = list(product(index1.indices, index2.indices))
        flattened_indices = [cls._flatten(t) for t in product_indices]
        return cls(name=f"{index1.name} x {index2.name}").from_list(
            flattened_indices, variable_names=variable_names
        )

    @staticmethod
    def _flatten(t):
        if isinstance(t[0], tuple) and isinstance(t[1], tuple):
            return t[0] + t[1]
        if isinstance(t[0], tuple) and not isinstance(t[1], tuple):
            return t[0] + (t[1],)
        if not isinstance(t[0], tuple) and isinstance(t[1], tuple):
            return (t[0],) + t[1]
        if not isinstance(t[0], tuple) and not isinstance(t[1], tuple):
            return (t[0], t[1])

    def from_pandas(
        self, data: pd.Index, use_pandas_variable_names: bool = True
    ) -> Index:
        """Create an index from a `pd.Index` object.

        Parameters
        ----------
        data : pd.Index
            `pd.Index` object to use for the index.
        use_pandas_variable_names : bool, default=True
            Whether to use the `names` attribute of the `pd.Index` object as the `variable_names` of the current index.

        Raises
        ------
        TypeError
            If `data` is not a `pd.Index` or if `use_pandas_variable_names` is not a `bool`.

        Returns
        -------
        index : Index
            The current `Index` instance with updated data.

        Examples
        --------
        >>> import pandas as pd
        >>> from sigalg.core import Index
        >>> # Using the pandas name for the variable names
        >>> data1 = pd.Index(["a", "b", "c"], name="letter")
        >>> I1 = Index(name="I1").from_pandas(data1)
        >>> print(I1)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I1':
        letter
            a
            b
            c
        >>> # Using the default variable name (the name of the Index)
        >>> I2 = Index(name="I2").from_pandas(data1, use_pandas_variable_names=False)
        >>> print(I2)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I2':
        I2
        a
        b
        c
        >>> # Using the pandas names for the variable names in dimension 2
        >>> data3 = pd.MultiIndex.from_tuples(
        ...     [("a", 1), ("b", 2), ("c", 3)], names=["letter", "number"]
        ... )
        >>> I3 = Index(name="I3").from_pandas(data3)
        >>> print(I3)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I3':
        letter  number
            a       1
            b       2
            c       3
        >>> # Using the default variable names (the name of the Index with an index) in dimension 2
        >>> I4 = Index(name="I4").from_pandas(data3, use_pandas_variable_names=False)
        >>> print(I4)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I4':
        I4_0  I4_1
        a     1
        b     2
        c     3
        """
        if not isinstance(data, pd.Index):
            raise TypeError("data must be a pd.Index.")
        if not isinstance(use_pandas_variable_names, bool):
            raise TypeError("use_pandas_variable_names must be a bool.")

        if use_pandas_variable_names:
            variable_names = data.names
        elif isinstance(data, pd.MultiIndex):
            variable_names = [f"{self.name}_{i}" for i in range(data.nlevels)]
        else:
            variable_names = [self.name]

        self._initialize_property_caches()
        self._data = data.copy()
        self._variable_names = variable_names
        self._data.names = variable_names

        return self

    def from_sequence(
        self,
        size: int,
        initial_index: int = 0,
        prefix: Hashable | None = None,
        variable_name: Hashable | None = None,
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
        variable_name : Hashable | None, default=None
            An optional single element for the name of the underlying `pd.Index` object. If `None`, the default will be set to the name of the current index.

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
            `name` is not hashable, or `variable_name` is not a hashable (if given).

        Examples
        --------
        >>> from sigalg.core import Index
        >>> I1 = Index(name="I1").from_sequence(size=3, prefix="F")
        >>> print(I1)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I1':
         I1
        F_0
        F_1
        F_2
        >>> I2 = Index(name="I2").from_sequence(size=2, initial_index=5, variable_name="x")
        >>> print(I2)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I2':
         x
         5
         6
        """
        if not isinstance(size, int) or size <= 0:
            raise ValueError("'size' must be a positive integer.")
        if not isinstance(initial_index, int):
            raise TypeError("'initial_index' must be an integer.")
        if prefix is not None and not isinstance(prefix, Hashable):
            raise TypeError("If given, 'prefix' must be hashable.")
        if variable_name is not None and not isinstance(variable_name, Hashable):
            raise TypeError("If given, 'variable_name' must be hashable.")

        self._initialize_property_caches()

        if variable_name is None:
            variable_name = self.name

        if prefix is None:
            indices = list(range(initial_index, initial_index + size))
        else:
            if size == 1:
                indices = [prefix]
            else:
                indices = [
                    f"{prefix}_{i}" for i in range(initial_index, initial_index + size)
                ]
        return self.from_list(indices=indices, variable_names=[variable_name])

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
        >>> data = pd.Index(["a", "b", "c"], name="index")
        >>> I1 = Index(name="I1").from_pandas(data)
        >>> print(I1.indices)
        ['a', 'b', 'c']
        >>> I2 = Index(name="I2").from_list(["x", "y", "z"])
        >>> print(I2.indices)
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
        >>> I_2 = Index(name="I_2").from_list(["x", "y", "z"], variable_names=["letters"])
        >>> print(I_2.data)
        Index(['x', 'y', 'z'], dtype='str', name='letters')
        """
        if self._data is None and self._indices is not None:
            self._data = pd.Index(self._indices)
            self._data.names = self._variable_names
        return self._data

    @property
    def variable_names(self) -> list | None:
        """Get the variable names of the index.

        If the index is not a `pd.MultiIndex`, this will be a list containing a single element. If the index is a `pd.MultiIndex`, this will be a list of names corresponding to each level of the `MultiIndex`.

        Returns
        -------
        variable_names : list | None
            The variable names of the underlying `pd.Index` object.

        Examples
        --------
        >>> import pandas as pd
        >>> from sigalg.core import Index
        >>> I1 = Index(name="I1").from_list(["x", "y", "z"], variable_names=["letters"])
        >>> print(I1.variable_names)
        ['letters']
        >>> data = pd.MultiIndex.from_tuples([("a", 1), ("b", 2), ("c", 3)], names=["letter", "number"])
        >>> I2 = Index(name="I2").from_pandas(data)
        >>> print(I2.data)
        MultiIndex([('a', 1),
                    ('b', 2),
                    ('c', 3)],
                   names=['letter', 'number'])
        >>> print(I2.variable_names)
        ['letter', 'number']
        >>> I2.variable_names = ["new_letter", "new_number"]
        >>> print(I2.data)
        MultiIndex([('a', 1),
                    ('b', 2),
                    ('c', 3)],
                   names=['new_letter', 'new_number'])
        >>> print(I2.variable_names)
        ['new_letter', 'new_number']
        """
        return self._variable_names

    @variable_names.setter
    def variable_names(self, variable_names: list) -> None:
        """Set the variable names of index.

        Parameters
        ----------
        variable_names : list
            If the index is not a `pd.MultiIndex`, this should be a list containing a single element. If the index is a `pd.MultiIndex`, this should be a list of names corresponding to each level of the `MultiIndex`.

        Raises
        ------
        TypeError
            If `variable_names` is not a list.
        """
        if not isinstance(variable_names, list):
            raise TypeError("variable_names must be a list.")
        if self.data is not None:
            self._data.names = variable_names
            self._variable_names = variable_names

    def with_variable_names(self, variable_names: list, in_place: bool = True) -> Index:
        """Return a new index with the given variable names.

        Parameters
        ----------
        variable_names : list
            If the index is not a `pd.MultiIndex`, this should be a list containing a single element. If the index is a `pd.MultiIndex`, this should be a list of names corresponding to each level of the `MultiIndex`.
        in_place : bool, default=True
            If `True`, modify the current index in place. If `False`, return a new index with the specified variable names.

        Returns
        -------
        index : Index
            A new `Index` with the specified variable names.
        """
        if in_place:
            self.variable_names = variable_names
            return self
        else:
            return type(self)(name=self.name).from_list(
                self.indices, variable_names=variable_names
            )

    @property
    def dimension(self) -> int | None:
        """Get the dimension of the index.

        The dimension is 1 if the underlying `pd.Index` is a regular `Index`, and is equal to the number of levels if the underlying `pd.Index` is a `MultiIndex`.

        Returns
        -------
        dimension : int | None
            The dimension of the index.
        """
        if self._dimension is None and self.data is not None:
            if isinstance(self.data, pd.MultiIndex):
                self._dimension = self.data.nlevels
            else:
                self._dimension = 1
        return self._dimension

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
            return f"Index '{self.name}': empty"
        else:
            return (
                f"Index '{self.name}':\n{self.data.to_frame().to_string(index=False)}"
            )
