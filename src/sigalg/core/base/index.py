"""A base class representing an ordered collection of hashable items."""

from __future__ import annotations

from collections.abc import Hashable
from itertools import product
from typing import Any

import pandas as pd

from ...validation.index_validator import IndexLike, IndexValidator


class Index:
    """A base class representing an ordered collection of hashable items.

    Subclasses include `Domain`, `SampleSpace`, `Event`, and `Time`. Instances of `Index` are also used to index instances of `RandomVector` of dimension > 1.

    Parameters
    ----------
    indices : IndexLike | None, default=None
        A list of hashable items, a list of tuples, or a `pd.Index` object to use as the index. If `None`, an empty index will be created.
    name : Hashable, default="I"
        Name identifier for the index.
    variable_names : list[Hashable] | None, default=None
        A list of variable names.
    bypass_validation : bool, default=False
        If `True`, bypass validation of the input data. This is intended for use by subclasses.
    **kwargs
        Additional keyword arguments passed to subclasses.

    Examples
    --------
    Build an `Index` from a list of hashable items.

    >>> import pandas as pd
    >>> from sigalg.core import Index
    >>> lst = ["a", "b", "c"]
    >>> I1 = Index(indices=lst, name="I1")
    >>> print(I1)  # doctest: +NORMALIZE_WHITESPACE
    Index 'I1':
    index
        a
        b
        c

    Build an `Index` from a `pd.Index` object. Note that the name of the `pd.Index` becomes the variable name of the `Index`.

    >>> idx = pd.Index([1, 2, 3], name="x")
    >>> I2 = Index(indices=idx, name="I2")
    >>> print(I2)  # doctest: +NORMALIZE_WHITESPACE
    Index 'I2':
     x
     1
     2
     3

    Build an `Index` from a `pd.MultiIndex` object. Note that custom variable names are passed to the `Index` constructor.

    >>> multi_idx = pd.MultiIndex.from_tuples([(1, "a"), (2, "b")])
    >>> I3 = Index(indices=multi_idx, name="I3", variable_names=["num", "letter"])
    >>> print(I3)  # doctest: +NORMALIZE_WHITESPACE
    Index 'I3':
     num letter
     1      a
     2      b

    Build an `Index` from the same `pd.MultiIndex` object, but with default variable names based on a default name.

    >>> I = Index(indices=multi_idx)
    >>> print(I)  # doctest: +NORMALIZE_WHITESPACE
    Index 'I':
     index_0 index_1
           1       a
           2       b
    """

    _properties = [
        "_indices",
        "_dimension",
    ]

    _default_name = "I"
    _repr_name = "Index"
    _variable_names_prefix = "index"

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        indices: IndexLike | None = None,
        name: Hashable | None = None,
        variable_names: list[Hashable] | None = None,
        bypass_validation: bool = False,
        **kwargs,
    ) -> None:
        if name is None:
            name = type(self)._default_name

        if bypass_validation:
            self._data = indices
            self._variable_names = variable_names
        else:
            v = IndexValidator(
                indices=indices,
                name=name,
                variable_names=variable_names,
                variable_names_prefix=type(self)._variable_names_prefix,
            )
            self._data = v.indices
            self._variable_names = v.variable_names

        self._name = name
        self._initialize_property_caches()

    def _initialize_property_caches(self) -> None:
        for property in self._properties:
            setattr(self, property, None)

    @classmethod
    def from_sequence(
        cls,
        size: int,
        initial_index: int = 0,
        prefix: Hashable | None = None,
        name: Hashable | None = None,
        variable_name: Hashable | None = None,
    ) -> Index:
        """Create an index with sequentially numbered items.

        Parameters
        ----------
        size : int
            Number of indices to generate. Must be positive.
        initial_index : int, default=0
            Starting index for sequential numbering.
        prefix : Hashable | None, default=None
            Prefix for index names. If `None`, then numerical indices are used.
        name : Hashable, default="I"
            Name identifier for the index.
        variable_name : Hashable | None, default=None
            An optional single element for the variable name. If `None`, the default will be set to the name of the index.

        Returns
        -------
        index : Index
            A new `Index` with automatically generated indices.

        Raises
        ------
        ValueError
            If `size` is not a positive integer.
        TypeError
            If `initial_index` is not an integer, `prefix` is not hashable, or `variable_name` is not a hashable (if given).

        Examples
        --------
        Build an `Index` consisting of the numbers 0, 1, 2, with default name and variable name.

        >>> from sigalg.core import Index
        >>> I = Index.from_sequence(size=3)
        >>> print(I)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I':
        index
            0
            1
            2

        Build an `Index` consisting of the strings F_0, F_1, F_2.

        >>> I2 = Index.from_sequence(size=3, name="I2", prefix="F")
        >>> print(I2)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I2':
        index
          F_0
          F_1
          F_2

        Build an `Index` consisting of the numbers 5 and 6, with a custom variable name.

        >>> I3 = Index.from_sequence(size=2, name="I3", initial_index=5, variable_name="x")
        >>> print(I3)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I3':
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

        if name is None:
            name = cls._default_name

        if prefix is None:
            indices = list(range(initial_index, initial_index + size))
        else:
            if size == 1:
                indices = [prefix]
            else:
                indices = [
                    f"{prefix}_{i}" for i in range(initial_index, initial_index + size)
                ]

        v = IndexValidator(
            indices=indices,
            name=name,
            variable_names=[variable_name] if variable_name else None,
            variable_names_prefix=cls._variable_names_prefix,
        )

        return cls(indices=v.indices, name=v.name, variable_names=v.variable_names)

    @classmethod
    def from_product(
        cls,
        indices1: list[Hashable],
        indices2: list[Hashable],
        name: Hashable | None = None,
        variable_names: list[Hashable] | None = None,
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
        Build an `Index` from a pair of lists with custom variable names.

        >>> from sigalg.core import Index
        >>> list1 = [1, 2, 3]
        >>> list2 = ["a", "b"]
        >>> I = Index.from_product(list1, list2, variable_names=["x", "y"])
        >>> print(I)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I':
         x y
         1 a
         1 b
         2 a
         2 b
         3 a
         3 b

        Build an `Index` from a pair of lists with default variable names.

        >>> J = Index.from_product(list1, list2, name="J")
        >>> print(J)  # doctest: +NORMALIZE_WHITESPACE
        Index 'J':
         index_0 index_1
               1       a
               1       b
               2       a
               2       b
               3       a
               3       b

        Build an `Index` from a pair of lists, where the second list consists of tuples, along with custom variable names.

        >>> list3 = [("a", "red"), ("b", "blue")]
        >>> K = Index.from_product(list1, list3, name="K", variable_names=["x", "y", "z"])
        >>> print(K)  # doctest: +NORMALIZE_WHITESPACE
        Index 'K':
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
        if variable_names is not None:
            if not isinstance(variable_names, list) or not all(
                isinstance(name, Hashable) for name in variable_names
            ):
                raise TypeError("variable_names must be a list of hashable items.")

        if name is None:
            name = cls._default_name

        product_indices = list(product(indices1, indices2))
        flattened_indices = [cls._flatten(t) for t in product_indices]

        v = IndexValidator(
            indices=flattened_indices,
            name=name,
            variable_names=variable_names,
            variable_names_prefix=cls._variable_names_prefix,
        )

        return cls(indices=v.indices, name=v.name, variable_names=v.variable_names)

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
        Build an `Index` from the Cartesian product of two `Index` instances with default variable names.

        >>> from sigalg.core import Index
        >>> I = Index(indices=[1, 2, 3], variable_names=["x"])
        >>> J = Index(indices=["a", "b"], name="J", variable_names=["y"])
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

        Build an `Index` from the Cartesian product of two `Index` instances with custom variable names.

        >>> product_2 = Index.cartesian_product(I, J, variable_names=["u", "v"])
        >>> print(product_2)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I x J':
         u v
         1 a
         1 b
         2 a
         2 b
         3 a
         3 b
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
        v = IndexValidator(
            indices=flattened_indices,
            name=f"{index1.name} x {index2.name}",
            variable_names=variable_names,
            variable_names_prefix=cls._variable_names_prefix,
        )

        return cls(indices=v.indices, name=v.name, variable_names=v.variable_names)

    @classmethod
    def _promote(cls, instance):
        """Pass."""
        new = cls.__new__(cls)
        new.__dict__.update(instance.__dict__)
        return new

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
        >>> indices = pd.Index(["a", "b", "c"], name="index")
        >>> I1 = Index(indices=indices, name="I1")
        >>> print(I1.indices)
        ['a', 'b', 'c']
        >>> I2 = Index(indices=["x", "y", "z"], name="I2")
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
        >>> indices = pd.Index(["a", "b", "c"], name="index")
        >>> I = Index(indices=indices)
        >>> print(I.data)
        Index(['a', 'b', 'c'], dtype='str', name='index')
        >>> J = Index(indices=["x", "y", "z"], name="J", variable_names=["letters"])
        >>> print(J.data)
        Index(['x', 'y', 'z'], dtype='str', name='letters')
        """
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
        Get the variable names of an `Index` built from a list with custom variable names passed to the constructor.

        >>> import pandas as pd
        >>> from sigalg.core import Index
        >>> I1 = Index(["x", "y", "z"], name="I1", variable_names=["letters"])
        >>> print(I1.variable_names)
        ['letters']

        Get the variable names of an `Index` built from a `pd.MultiIndex` object with names.

        >>> indices = pd.MultiIndex.from_tuples(
        ...     [("a", 1), ("b", 2), ("c", 3)], names=["letter", "number"]
        ... )
        >>> I2 = Index(indices=indices, name="I2")
        >>> print(I2.variable_names)
        ['letter', 'number']

        Set new variable names. Notice that the `variable_names` property changes, and also the names of the underlying `pd.MultiIndex` are updated.

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
            return Index(indices=data, name=self.name)
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
            return f"{type(self)._repr_name} '{self.name}': empty"
        else:
            return f"{type(self)._repr_name} '{self.name}':\n{self.data.to_frame().to_string(index=False)}"
