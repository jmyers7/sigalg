"""A base class representing an ordered collection of hashable items."""

from __future__ import annotations

import re
from collections import Counter
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
    name : Hashable | None, default=None
        Name identifier for the index. If `None`, a default name `I` will be used.
    variable_names : list[Hashable] | None, default=None
        A list of variable names for the dimensions of the index. If `None`, a default variable name `index` will be used.
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

    Build an `Index` from the same `pd.MultiIndex` object, but with default variable names.

    >>> I = Index(indices=multi_idx)
    >>> print(I)  # doctest: +NORMALIZE_WHITESPACE
    Index 'I':
     index_0 index_1
           1       a
           2       b
    """

    _properties = ["_dimension"]
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
            Prefix for index values. If `None`, then numerical indices are used.
        name : Hashable | None, default=None
            Name identifier for the index. If `None`, a default name `I` will be used.
        variable_name : Hashable | None, default=None
            An optional single element for the variable name. If `None`, a default name `index` will be used.

        Returns
        -------
        index : Index
            A new `Index` with automatically generated indices.

        Raises
        ------
        ValueError
            If `size` is not a positive integer.
        TypeError
            If `initial_index` is not an integer, or `prefix`, `name, or `variable_name` is not hashable (if given).

        Examples
        --------
        Build an `Index` consisting of the numbers `0`, `1`, `2`, with default name and variable name.

        >>> from sigalg.core import Index
        >>> I = Index.from_sequence(size=3)
        >>> print(I)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I':
        index
            0
            1
            2

        Build an `Index` consisting of the strings `F_0`, `F_1`, `F_2`.

        >>> I2 = Index.from_sequence(size=3, name="I2", prefix="F")
        >>> print(I2)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I2':
        index
          F_0
          F_1
          F_2

        Build an `Index` consisting of the numbers `5` and `6`, with a custom variable name.

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
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("If given, 'name' must be hashable.")
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
    def cartesian_product(
        cls,
        indices: list,
        name: Hashable | None = None,
        variable_names: list[Hashable] | None = None,
    ) -> Index:
        """Create an index from the Cartesian product of a list of indices.

        Parameters
        ----------
        indices : list
            A list of either `IndexLike` or `Index` objects to serve as the factors of the Cartesian product.
        name: Hashable | None, default=None
            The name of the Cartesian product. If all items in `indices` are instances of `Index` and `name` is `None`, then a default will be generated from the names of the instances. Otherwise, if one or more is not an instance of `Index` and if `name` is `None`, then a default name of `I` will be used.
        variable_names : list[Hashable] | None, default=None
            A list of variable names for the resulting index. If `None`, the variable names will be set to the concatenation of the variable names of indices if they are all `Index` instances.

        Returns
        -------
        cartesian_product : Index
            The Cartesian product.

        Examples
        --------
        Build an `Index` from the Cartesian product of two `Index` instances.

        >>> from sigalg.core import Index
        >>> I = Index(indices=[1, 2, 3], variable_names=["x"])
        >>> J = Index(indices=["a", "b"], name="J", variable_names=["y"])
        >>> product_1 = Index.cartesian_product([I, J])
        >>> print(product_1)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I x J':
         x y
         1 a
         1 b
         2 a
         2 b
         3 a
         3 b

        Build an `Index` from the Cartesian product of two `Index` instances with custom variable names passed to the class method.

        >>> product_2 = Index.cartesian_product([I, J], variable_names=["u", "v"])
        >>> print(product_2)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I x J':
         u v
         1 a
         1 b
         2 a
         2 b
         3 a
         3 b

        Build an `Index` from two `Index` instances with identical variable names. Note that subscripts are automatically generated.

        >>> U = Index([(1, 2), (3, 4)], name="U", variable_names=["x", "y"])
        >>> V = Index([(5, 6), (7, 8)], name="V", variable_names=["x", "y"])
        >>> product_3 = Index.cartesian_product([U, V])
        >>> print(product_3)  # doctest: +NORMALIZE_WHITESPACE
        Index 'U x V':
        x_0  y_0  x_1  y_1
          1    2    5    6
          1    2    7    8
          3    4    5    6
          3    4    7    8

        Build an `Index` from a pair of lists with custom variable names passed to the class method.

        >>> from sigalg.core import Index
        >>> list1 = [1, 2, 3]
        >>> list2 = ["a", "b"]
        >>> I = Index.cartesian_product([list1, list2], variable_names=["x", "y"])
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

        >>> J = Index.cartesian_product([list1, list2], name="J")
        >>> print(J)  # doctest: +NORMALIZE_WHITESPACE
        Index 'J':
         index_0 index_1
               1       a
               1       b
               2       a
               2       b
               3       a
               3       b

        Build an `Index` from a pair of lists, where the second list consists of tuples.

        >>> list3 = [("a", "red"), ("b", "blue")]
        >>> K = Index.cartesian_product([list1, list3], name="K", variable_names=["x", "y", "z"])
        >>> print(K)  # doctest: +NORMALIZE_WHITESPACE
        Index 'K':
        x y    z
        1 a  red
        1 b blue
        2 a  red
        2 b blue
        3 a  red
        3 b blue

        Build an `Index` from a list of indices and and instance of `Index`.

        >>> L = Index.cartesian_product([list1, U], name="L")
        >>> print(L)  # doctest: +NORMALIZE_WHITESPACE
        Index 'L':
        index  x  y
            1  1  2
            1  3  4
            2  1  2
            2  3  4
            3  1  2
            3  3  4
        """
        if name is None:
            if all(isinstance(index, Index) for index in indices):
                name = " x ".join([index.name for index in indices])
            else:
                name = cls._default_name

        indices = [
            Index(index) if not isinstance(index, Index) else index for index in indices
        ]

        if variable_names is None:
            variable_names = cls._subscript_var_names(
                [index.variable_names for index in indices]
            )

        product_indices = list(product(*indices))
        flattened_indices = [cls._flatten(t) for t in product_indices]

        v = IndexValidator(
            indices=flattened_indices,
            name=name,
            variable_names=variable_names,
            variable_names_prefix=cls._variable_names_prefix,
        )

        return cls(indices=v.indices, name=v.name, variable_names=v.variable_names)

    def __matmul__(self, other: Index | IndexLike) -> Index:
        """Get the Cartesian product of this `Index` instance with another.

        Internally, calls the class method `Index.cartesian_product`.

        Parameters
        ----------
        other : Index | IndexLike
            The second factor in the Cartesian product.

        Returns
        -------
        cartesian_product : Index
            The Cartesian product.
        """
        return type(self).cartesian_product([self, other])

    @classmethod
    def cartesian_power(
        cls,
        index: IndexLike | Index,
        n: int,
        name: Hashable | None = None,
        variable_names: list[Hashable] | None = None,
    ) -> Index:
        """Form the Cartesian power of an index.

        Parameters
        ----------
        index : IndexLike | Index,
            The index used as the base of the Cartesian power.
        n : int
            The power of the Cartesian power.
        name : Hashable | None, default=None
            The name of the Cartesian power. If `None`, a default will be generated using the name of the current instance of `Index`.
        variable_names : list[Hashable] | None, default=None
            A list of variable names for the resulting index. If `None`, the variable names will be set to the variable names of `index` (if it is an instance of `Index`) with subscripts.

        Raises
        ------
        TypeError
            If `n` is not an integer.
        ValueError
            If `n` is not a positive integer.

        Returns
        -------
        cartesian_power : Index
            The Cartesian power.

        Examples
        --------
        Form the Cartesian power using the `cartesian_power` class method.

        >>> from sigalg.core import Index
        >>> I = Index([(1, "a"), (2, "b"), (3, "c")], variable_names=["x", "y"])
        >>> I_2 = Index.cartesian_power(I, 2)
        >>> print(I_2)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I ^ 2':
        x_0 y_0  x_1 y_1
          1   a    1   a
          1   a    2   b
          1   a    3   c
          2   b    1   a
          2   b    2   b
          2   b    3   c
          3   c    1   a
          3   c    2   b
          3   c    3   c

        Form the Cartesian power using the `^` operator.

        >>> print(I ^ 2)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I ^ 2':
        x_0 y_0  x_1 y_1
          1   a    1   a
          1   a    2   b
          1   a    3   c
          2   b    1   a
          2   b    2   b
          2   b    3   c
          3   c    1   a
          3   c    2   b
          3   c    3   c
        """
        if not isinstance(n, int):
            raise TypeError("n must be an integer.")
        if not isinstance(index, cls):
            index = cls(indices=index)
        if n <= 0:
            raise ValueError("n must be a positive integer.")

        if name is None:
            name = f"{index.name} ^ {n}"

        num_names = index.dimension

        if variable_names is None:
            variable_names = [f"{name}_0" for name in index.variable_names]
            variable_names_was_none = True
        else:
            input_variable_names = variable_names.copy()
            variable_names = input_variable_names[:num_names]
            variable_names_was_none = False

        power = index

        for k in range(n - 1):
            if variable_names_was_none:
                variable_names += [f"{name}_{k + 1}" for name in index.variable_names]
            else:
                variable_names = input_variable_names[: (k + 2) * num_names]

            power = type(index).cartesian_product(
                [power, index], variable_names=variable_names
            )

        return power.with_name(name)

    def __xor__(self, n: int) -> Index:
        """Form the Cartesian power of this instance of `Index`.

        Internally calls the `cartesian_power` method.

        Parameters
        ----------
        n : int
            The power of the Cartesian power.

        Returns
        -------
        cartesian_power : Index
            The Cartesian power.
        """
        return type(self).cartesian_power(index=self, n=n)

    @classmethod
    def _promote(cls, instance):
        """Pass."""
        new = cls.__new__(cls)
        new.__dict__.update(instance.__dict__)
        return new

    @staticmethod
    def _subscript_var_names(lists):
        names = [x for names in lists for x in names]
        if set(Counter(names).values()) == {1}:
            return names

        def base(s):
            m = re.fullmatch(r"(.+)_(\d+)", s)
            return (s, None) if not m else (m.group(1), int(m.group(2)))

        tuples = [base(s) for lst in lists for s in lst]
        bases = [t[0] for t in tuples]
        common_bases = {base for base, count in Counter(bases).items() if count >= 2}

        for base in common_bases:
            idx = 0
            for t in tuples:
                if t[0] == base:
                    tuples[tuples.index(t)] = (t[0], idx)
                    idx += 1

        return [f"{t[0]}_{t[1]}" if t[1] is not None else t[0] for t in tuples]

    @staticmethod
    def _flatten(t):
        return tuple(
            x for item in t for x in (item if isinstance(item, tuple) else (item,))
        )

    # --------------------- properties --------------------- #

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
        >>> indices = pd.Index(["a", "b", "c"], name="letter")
        >>> I = Index(indices=indices)
        >>> print(I.data)
        Index(['a', 'b', 'c'], dtype='str', name='letter')
        """
        return self._data

    @property
    def variable_names(self) -> list | None:
        """Get the variable names of the index.

        If the index is not a `pd.MultiIndex`, this will be a list containing a single element. If the index is a `pd.MultiIndex`, this will be a list of names corresponding to each level of the `MultiIndex`.

        Returns
        -------
        variable_names : list | None
            The variable names of the underlying `pd.Index` object, if set.

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
        """Set the name of the index and return `self` for chaining.

        Parameters
        ----------
        name : Hashable
            New name for the index.

        Returns
        -------
        self : Index
            The current index with a new name.
        """
        self.name = name
        return self

    # --------------------- data access methods --------------------- #

    def __getitem__(self, pos: int | list[int] | slice) -> Any:
        """Access elements by positions.

        Parameters
        ----------
        pos : int | list[int] | slice
            Index, slice, or other key for accessing elements positionally.

        Returns
        -------
        element : Any
            The indexed element(s) from the index.
        """
        if not isinstance(pos, (int, list, slice)):
            raise TypeError("pos must be int | list[int] | slice.")
        if isinstance(pos, list) and not all(isinstance(i, int) for i in pos):
            raise TypeError("pos list must contain only int.")

        data = self.data[pos]
        if isinstance(data, pd.Index):
            return type(self)(indices=data, name=self.name)
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
        return len(self.data) if self.data is not None else 0

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

        Two indices are equal if they have the same elements in the same order and with the same variable names. They may have different names and still be considered equal.

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
        return (
            isinstance(other, type(self))
            and self.data.equals(other.data)
            and self.variable_names == other.variable_names
        )

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

    # --------------------- set-theoretic operations --------------------- #

    def __and__(self, other: Index) -> Index:
        """Return the intersection of this index with another (`&` operator).

        Parameters
        ----------
        other : Index
            Another index from the same sample space.

        Raises
        ------
        TypeError
            If `other` is not an instance of `Index`.

        Returns
        -------
        intersection : Index
            An index containing elements present in both indices.
        """
        if not isinstance(other, Index):
            raise TypeError("other must be an instance of Index.")

        pts = set(self.data) & set(other.data)
        name = f"{self.name} intersect {other.name}"
        return type(self)(indices=list(pts), name=name)
