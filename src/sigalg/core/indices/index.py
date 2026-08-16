"""A base class representing an ordered collection of hashable items."""

from __future__ import annotations

from collections.abc import Hashable
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ...typing.index_like import IndexLike


class Index:
    """A base class representing an ordered collection of hashable items.

    Parameters
    ----------
    indices : IndexLike | None, default=None
        The object from which to construct the `Index`.
    variable_names : list[Hashable] | None, default=None
        A list of variable names for the dimensions of the index. If `None`, defaults will be generated.
    name : Hashable | None, default=None
        Name identifier for the index. If `None`, a default name will be generated.
    copy_data : bool, default=True
        If `indices` is a `pd.Index`, whether to internally make a copy of the index or not.
    **kwargs
        Additional keyword arguments passed to subclasses.

    Examples
    --------
    >>> import pandas as pd
    >>> from sigalg.core import Index

    Build an `Index` from a list of hashable items.

    >>> lst = ["a", "b", "c"]
    >>> I1 = Index(indices=lst, name="I1")
    >>> print(I1)  # doctest: +NORMALIZE_WHITESPACE
    Index 'I1':
     i
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

    Build an `Index` from the multi-index, but with default variable names.

    >>> multi_idx = pd.MultiIndex.from_tuples([(1, "a"), (2, "b")])
    >>> I4 = Index(indices=multi_idx, name="I4")
    >>> print(I4)  # doctest: +NORMALIZE_WHITESPACE
    Index 'I4':
     i_0  i_1
       1    a
       2    b
    """

    _properties = []
    _default_name = "I"
    _repr_name = "Index"
    _str_name = "Index"
    _variable_names_prefix = "i"

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        indices: IndexLike | None = None,
        variable_names: list[Hashable] | None = None,
        name: Hashable | None = None,
    ) -> None:
        from ...validation.index_validator import IndexValidator

        if name is None:
            name = type(self)._default_name

        v = IndexValidator(
            indices=indices,
            variable_names=variable_names,
            variable_names_prefix=type(self)._variable_names_prefix,
            name=name,
        )
        self.data = v.data
        self.name = name

    @classmethod
    def _from_validated(
        cls,
        *,
        data: pd.Index,
        name: Hashable,
    ) -> Index:
        idx = object.__new__(cls)
        idx.data = data
        idx.name = name
        return idx

    @classmethod
    def from_sequence(
        cls,
        size: int,
        initial_index: int = 0,
        prefix: Hashable | None = None,
        variable_name: Hashable | None = None,
        name: Hashable | None = None,
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
        variable_name : Hashable | None, default=None
            An optional single element for the variable name. If `None`, a default name will be generated.
        name : Hashable | None, default=None
            Name identifier for the index. If `None`, a default name will be generated.

        Raises
        ------
        ValueError
            If `size` is not a positive integer.
        TypeError
            If `initial_index` is not an integer, or `prefix`, `name, or `variable_name` is not hashable (if given).

        Returns
        -------
        index : Index
            A new `Index` with automatically generated indices.

        Examples
        --------
        Build an `Index` consisting of the numbers `0`, `1`, `2`, with default name and variable name.

        >>> from sigalg.core import Index
        >>> I = Index.from_sequence(size=3)
        >>> print(I)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I':
         i
         0
         1
         2

        Build an `Index` consisting of the strings `F_0`, `F_1`, `F_2`.

        >>> I2 = Index.from_sequence(size=3, name="I2", prefix="F")
        >>> print(I2)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I2':
           i
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
        if variable_name is None:
            variable_name = cls._variable_names_prefix

        if prefix is None:
            indices = list(range(initial_index, initial_index + size))
        else:
            if size == 1:
                indices = [prefix]
            else:
                indices = [
                    f"{prefix}_{i}" for i in range(initial_index, initial_index + size)
                ]

        data = pd.Index(data=indices, name=variable_name)
        return cls._from_validated(data=data, name=name)

    @classmethod
    def from_rand(
        cls,
        size: int,
        dim: int = 1,
        sample_range: tuple[int, int] | None = None,
        variable_names: list[Hashable] | None = None,
        name: Hashable | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> Index:
        """Generate a random `Index`.

        Parameters
        ----------
        size : int
            Number of indices to generate.
        dim : int, default=1
            The dimension of index.
        sample_range : tuple[int, int] | None, default=None
            A tuple specifying the range of values for the indices. If `None`, the range will be [0, size).
        variable_names : list[Hashable] | None, default=None
            A list of variable names for the dimensions of the index.
        name : Hashable | None, default=None
            Name identifier for the index.
        random_state : int | np.random.Generator | None, default=None
            A seed or random number generator for reproducibility.

        Returns
        -------
        index : Index
            A new `Index` with randomly generated indices.

        Examples
        --------
        Generate a random index with default parameters, except for the size and random state.

        >>> from sigalg.core import Index
        >>> I = Index.from_rand(size=4, random_state=42)
        >>> print(I)  # doctest: +NORMALIZE_WHITESPACE
        Index 'I':
         i
         0
         2
         3
         1

        Generate a 2-dimensional random index with a specified range and variable names.

        >>> J = Index.from_rand(
        ...     size=5,
        ...     dim=3,
        ...     sample_range=(0, 10),
        ...     random_state=42,
        ...     variable_names=["x", "y", "z"],
        ...     name="J",
        ... )
        >>> print(J)  # doctest: +NORMALIZE_WHITESPACE
        Index 'J':
         x  y  z
         5  0  5
         3  9  7
         7  7  7
         0  7  5
         4  1  8
        """
        from ._helpers import random_tuples

        if name is None:
            name = cls._default_name

        data = random_tuples(
            size=size, sample_range=sample_range, dim=dim, random_state=random_state
        )

        return cls(indices=data, name=name, variable_names=variable_names)

    @classmethod
    def cartesian_product(
        cls,
        factors: list[IndexLike | Index],
        variable_names: list[Hashable] | None = None,
        name: Hashable | None = None,
    ) -> Index:
        """Create an index from the Cartesian product of a list of indices.

        Parameters
        ----------
        factors : list[IndexLike | Index]
            The factors of the Cartesian product.
        variable_names : list[Hashable] | None, default=None
            A list of variable names for the resulting index. If `None`, the variable names will be set to the concatenation of the variable names of indices if they are all `Index` instances.
        name: Hashable | None, default=None
            The name of the Cartesian product. If all items in `indices` are instances of `Index` and `name` is `None`, then a default will be generated from the names of the instances. Otherwise, if one or more is not an instance of `Index` and if `name` is `None`, then a default name of `I` will be used.

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
         i_0 i_1
           1   a
           1   b
           2   a
           2   b
           3   a
           3   b

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
         i  x  y
         1  1  2
         1  3  4
         2  1  2
         2  3  4
         3  1  2
         3  3  4
        """
        from .._utils.utils import flatten, subscript_var_names

        if name is None:
            if all(isinstance(index, Index) for index in factors):
                name = " x ".join([index.name for index in factors])
            else:
                name = cls._default_name

        factors = [
            Index(index) if not isinstance(index, Index) else index for index in factors
        ]

        if variable_names is None:
            variable_names = subscript_var_names(
                [index.variable_names for index in factors]
            )

        product_indices = list(product(*factors))
        flattened_indices = [flatten(t) for t in product_indices]

        data = pd.MultiIndex.from_tuples(flattened_indices)

        return cls(indices=data, name=name, variable_names=variable_names)

    def __matmul__(self, other: IndexLike | Index) -> Index:
        """Get the Cartesian product of this `Index` instance with another.

        Internally, calls the class method `Index.cartesian_product`.

        Parameters
        ----------
        other : IndexLike | Index
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
        variable_names: list[Hashable] | None = None,
        name: Hashable | None = None,
    ) -> Index:
        """Form the Cartesian power of an index.

        Parameters
        ----------
        index : IndexLike | Index,
            The index used as the base of the Cartesian power.
        n : int
            The power of the Cartesian power.
        variable_names : list[Hashable] | None, default=None
            A list of variable names for the resulting index. If `None`, the variable names will be set to the variable names of `index` (if it is an instance of `Index`) with subscripts.
        name : Hashable | None, default=None
            The name of the Cartesian power. If `None`, a default will be generated using the name of the current instance of `Index`.

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
        new = cls.__new__(cls)
        new.__dict__.update(instance.__dict__)
        return new

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

    # --------------------- properties --------------------- #

    @property
    def variable_names(self) -> list[Hashable] | None:
        """Get the variable names of the index.

        If the index is not a `pd.MultiIndex`, this will be a list containing a single element. If the index is a `pd.MultiIndex`, this will be a list of names corresponding to each level of the `MultiIndex`.

        Returns
        -------
        variable_names : list[Hashable] | None
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

        Print the default variable names of an `Index` built from a list of integers.

        >>> I3 = Index(indices=[0, 1], name="I3")
        >>> print(I3.variable_names)
        ['i']

        Print the default variable names of an `Index` built from a list of tuples.

        >>> I4 = Index(indices=[(1, 2), (3, 4)], name="I4")
        >>> print(I4.variable_names)
        ['i_0', 'i_1']
        """
        return list(self.data.names) if self.data is not None else None

    @property
    def dimension(self) -> int | None:
        """Get the dimension of the index.

        The dimension is 1 if the underlying `pd.Index` is a regular `Index`, and is equal to the number of levels if the underlying `pd.Index` is a `MultiIndex`.

        Returns
        -------
        dimension : int | None
            The dimension of the index.
        """
        return self.data.nlevels if self.data is not None else None

    # --------------------- data methods --------------------- #

    def __getitem__(self, pos: int | list[int] | slice) -> Hashable | Index:
        """Access elements by positions.

        Parameters
        ----------
        pos : int | list[int] | slice
            Index, slice, or other key for accessing elements positionally.

        Returns
        -------
        element : Hashable | Index
            The indexed element(s) from the index.
        """
        if not isinstance(pos, (int, list, slice)):
            raise TypeError("pos must be int | list[int] | slice.")
        if isinstance(pos, list) and not all(isinstance(i, int) for i in pos):
            raise TypeError("pos list must contain only int.")

        data = self.data[pos]
        if isinstance(data, pd.Index):
            return type(self)._from_validated(data=data, name=self.name)
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
        return bool(item in self.data)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Return the index's data as a NumPy array.

        Parameters
        ----------
        dtype : data-type | None, default=None
            The desired data-type for the array. If `None`, the data-type of the underlying data is used.
        copy : bool | None, default=None
            Whether to return a copy of the data. If `None`, the default behavior is used.

        Returns
        -------
        np.ndarray
            The index's data as a NumPy array.
        """
        arr = self.data.values
        if dtype is not None:
            arr = np.asarray(arr, dtype=dtype)
        if copy:
            arr = arr.copy()

        return arr

    def to_numpy(self, dtype=None, copy=None) -> np.ndarray:
        """Return the index's data as a NumPy array.

        Parameters
        ----------
        dtype : data-type | None, default=None
            The desired data-type for the array. If `None`, the data-type of the underlying data is used.
        copy : bool | None, default=None
            Whether to return a copy of the data. If `None`, the default behavior is used.

        Returns
        -------
        np.ndarray
            The index's data as a NumPy array.
        """
        return self.__array__(dtype=dtype, copy=copy)

    def sort(self, ascending: bool = True) -> Index:
        """Return a sorted copy of the index.

        Parameters
        ----------
        ascending : bool, default=True
            Whether to sort in ascending order. If `False`, sort in descending order.

        Returns
        -------
        sorted_index : Index
            A new index with elements sorted.
        """
        sorted_data = self.data.copy().sort_values(ascending=ascending)
        return type(self)._from_validated(data=sorted_data, name=self.name)

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

        Two indices are equal if they have the same variable names (as sets) and are equal (as sets).

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the indices are considered equal according to the above criteria, `False` otherwise.
        """
        from .._utils.index_helpers import align_index

        if not isinstance(other, Index):
            return False
        try:
            _ = align_index(self.data, by=other.data)
        except ValueError:
            return False

        return True

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a concise string representation of the index.

        Returns
        -------
        repr_str : str
            String representation of the index.
        """
        if self.data is None:
            return f"{type(self)._repr_name}(empty)"
        else:
            return f"{type(self)._repr_name}(num_indices={len(self.data)}, name={self.name})"

    def __str__(self) -> str:
        """Return a detailed string representation of the index.

        Returns
        -------
        repr_str : str
            String representation of the index.
        """
        if self.data is None:
            return f"{type(self)._str_name} '{self.name}': empty"
        else:
            return f"{type(self)._str_name} '{self.name}':\n{self.data.to_frame().to_string(index=False)}"

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
        if self.dimension > 1 or other.dimension > 1:
            raise NotImplementedError(
                "Intersection is not yet implemented for indices of dimension > 1."
            )

        if self.variable_names != other.variable_names:
            raise ValueError(
                "Cannot intersect two indices whose variable names are not equal."
            )

        pts = set(self.data) & set(other.data)
        name = f"{self.name} intersect {other.name}"
        data = pd.Index(pts, name=self.variable_names[0])
        return type(self)._from_validated(data=data, name=name)
