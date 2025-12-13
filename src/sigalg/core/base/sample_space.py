"""
Sample spaces for probability theory.

This module provides the `SampleSpace` class, which represents the set of all
possible outcomes in a probability experiment. Sample spaces serve as the
foundation for defining events, sigma-algebras, and probability measures.

Classes
-------
SampleSpace
    Represents a sample space as a collection of outcomes.
SampleSpaceMethods
    Mixin providing sample space methods to other classes.

Examples
--------
>>> import sigalg as sa
>>> sample_space = sa.SampleSpace(indices=["H", "T"], name="CoinFlip")
>>> sample_space
Sample space 'CoinFlip':
['H', 'T']
>>> event = sample_space.get_event(["H"], name="Heads")
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import pandas as pd

from .index import Index

if TYPE_CHECKING:
    from ..probability_measures import ProbabilityMeasure
    from ..sigma_algebras import SigmaAlgebra
    from . import ProbabilitySpace
    from .event import Event
    from .event_space import EventSpace


class SampleSpace(Index):
    """A sample space representing all possible outcomes of a probability experiment.

    A sample space is a fundamental object in probability theory that contains all
    possible outcomes (sample points) that can occur in a random experiment. It serves
    as the domain for events, random variables, and probability measures.

    The sample space can be constructed either from a list of hashable indices or
    from an existing `pd.Index` object. Sample spaces support operations like
    creating events, converting to probability spaces, and iterating over outcomes.

    Parameters
    ----------
    indices : list[Hashable], optional
        List of hashable items representing sample points. Mutually exclusive with `values`.
    values : pd.Index, optional
        `pd.Index` object containing sample points. Mutually exclusive with `indices`.
    name : str, default="Omega"
        Name identifier for the sample space.
    values_name : str, default="sample"
        Name for the index of values.

    Raises
    ------
    ValueError
        If both `indices` and `values` are provided, or if neither is provided.
        If `indices` contains duplicate values.
    TypeError
        If `indices` is not a list or `values` is not a `pd.Index`.

    Examples
    --------
    >>> import sigalg as sa
    >>> import pandas as pd
    >>> # Construction with list
    >>> space1 = sa.SampleSpace(indices=["omega0", "omega1", "omega2"], name="Omega")
    >>> # Construction with pandas Index
    >>> idx = pd.Index(["a", "b", "c"], name="sample")
    >>> space2 = sa.SampleSpace(values=idx, name="S")
    >>> # Get an event from the sample space
    >>> event = space1.get_event(["omega0", "omega1"], name="A")
    """

    # --------------------- constructor --------------------- #
    def __init__(
        self,
        indices: list[Hashable] | None = None,
        values: pd.Index | None = None,
        name: str = "Omega",
        values_name: str = "sample",
    ) -> None:
        super().__init__(
            indices=indices, values=values, name=name, values_name=values_name
        )

    # --------------------- factory methods --------------------- #

    @classmethod
    def generate_default(
        cls,
        initial_index: int = 0,
        size: int = 10,
        prefix: str = "omega",
        name: str = "Omega",
        values_name: str = "sample",
    ) -> SampleSpace:
        """Generate a default sample space with automatically named sample points.

        Creates a sample space with sample points named using a `prefix` and sequential
        indices. For single-element spaces, only the `prefix` is used. For larger spaces, indices are appended to the `prefix` (e.g., "`omega0`", "`omega1`", ...).

        Parameters
        ----------
        initial_index : int, default=0
            Starting index for sequential numbering.
        size : int, default=10
            Number of sample points to generate. Must be positive.
        prefix : str, default="omega"
            String prefix for sample point names.
        name : str, default="Omega"
            Name identifier for the sample space.
        values_name : str, default="sample"
            Name for the index of values.

        Returns
        -------
        sample_space : SampleSpace
            A new `SampleSpace` with automatically generated sample points.

        Raises
        ------
        ValueError
            If `size` is not a positive integer.
        TypeError
            If `initial_index` is not an integer or `prefix` is not a string.

        Examples
        --------
        >>> import sigalg as sa
        >>> space = sa.SampleSpace.generate_default(size=5, prefix="s", initial_index=1)
        >>> list(space)
        ['s1', 's2', 's3', 's4', 's5']
        """
        if not isinstance(size, int) or size <= 0:
            raise ValueError("'size' must be a positive integer.")
        if not isinstance(initial_index, int):
            raise TypeError("'initial_index' must be an integer.")
        if not isinstance(prefix, str):
            raise TypeError("'prefix' must be a string.")

        if size == 1:
            indices = [prefix]
        else:
            indices = [
                f"{prefix}{i}" for i in range(initial_index, initial_index + size)
            ]
        return cls(indices=indices, name=name, values_name=values_name)

    # --------------------- conversion methods --------------------- #

    def make_probability_space(
        self,
        sigma_algebra: SigmaAlgebra | None = None,
        probability_measure: ProbabilityMeasure | None = None,
    ) -> ProbabilitySpace:
        """Convert this sample space to a probability space.

        Creates a `ProbabilitySpace` object with this sample space as the underlying
        space. Optionally specify a sigma-algebra and probability measure. If not
        provided, defaults will be used.

        Parameters
        ----------
        sigma_algebra : SigmaAlgebra, optional
            Sigma-algebra to use. If `None`, a power set sigma-algebra will be created.
        probability_measure : ProbabilityMeasure, optional
            Probability measure to use. If `None`, a uniform probability measure will be created.

        Returns
        -------
        probability_space : ProbabilitySpace
            A `ProbabilitySpace` object with this sample space.

        Examples
        --------
        >>> import sigalg as sa
        >>> space = sa.SampleSpace(indices=["s0", "s1", "s2"])
        >>> # Create with default uniform measure
        >>> prob_space = space.make_probability_space()
        >>> # Create with custom probability measure
        >>> probs = {"s0": 0.5, "s1": 0.3, "s2": 0.2}
        >>> measure = sa.ProbabilityMeasure(probabilities=probs, sample_space=space)
        >>> prob_space = space.make_probability_space(probability_measure=measure)
        """
        from . import ProbabilitySpace

        return ProbabilitySpace(
            sample_space=self,
            sigma_algebra=sigma_algebra,
            probability_measure=probability_measure,
        )

    def make_event_space(self, sigma_algebra: SigmaAlgebra | None = None) -> EventSpace:
        """Convert this sample space to an event space.

        Creates an `EventSpace` object with this sample space as the underlying space.
        Optionally specify a sigma-algebra to define which events are measurable.

        Parameters
        ----------
        sigma_algebra : SigmaAlgebra, optional
            Sigma-algebra to use. If `None`, a power set sigma-algebra will be created,
            making all subsets measurable.

        Returns
        -------
        event_space : EventSpace
            An `EventSpace` object with this sample space.

        Examples
        --------
        >>> import sigalg as sa
        >>> space = sa.SampleSpace(indices=["s0", "s1", "s2", "s3"])
        >>> # Create with default power set sigma-algebra
        >>> event_space = space.make_event_space()
        >>> # Create with custom sigma-algebra
        >>> sigma = sa.SigmaAlgebra(
        ...     sample_space=space,
        ...     sample_id_to_atom_id={"s0": 0, "s1": 0, "s2": 1, "s3": 1}
        ... )
        >>> event_space = space.make_event_space(sigma_algebra=sigma)
        """
        from .event_space import EventSpace

        return EventSpace(sample_space=self, sigma_algebra=sigma_algebra)

    # --------------------- data access methods --------------------- #

    def get_event(self, event_indices: list[Hashable], name: str = "A") -> Event:
        """Create an event from a list of sample point indices.

        Constructs an `Event` object representing a subset of this sample space.
        All provided indices must exist in the sample space.

        Parameters
        ----------
        event_indices : list of Hashable
            List of sample point indices to include in the event.
            Must be hashable items that exist in this sample space.
        name : str, default="A"
            Name identifier for the event.

        Returns
        -------
        event : Event
            An `Event` object containing the specified sample points.

        Raises
        ------
        TypeError
            If `event_indices` is not a list.
        ValueError
            If any index in `event_indices` is not found in the sample space.

        Examples
        --------
        >>> import sigalg as sa
        >>> space = sa.SampleSpace(indices=["omega0", "omega1", "omega2", "omega3"])
        >>> # Create event with specific sample points
        >>> event = space.get_event(["omega0", "omega1"], name="A")
        >>> # Create event with empty list
        >>> empty_event = space.get_event([])
        """
        from .event import Event

        if not isinstance(event_indices, list):
            raise TypeError("event_indices must be a list of Hashable items.")
        for idx in event_indices:
            if idx not in self.values:
                raise ValueError(f"Index '{idx}' not found in sample space.")
        return Event(sample_space=self, event_indices=event_indices, name=name)

    def _getitem_hook(self, key):
        """Internal hook for indexing operations to create events.

        This method is called by `__getitem__` from the parent `Index` class. In `SampleSpace`, the purpose of this method is to ensure that `__getitem__` returns an instance of `Event`. Items are retrieved by position.

        Parameters
        ----------
        key : int, slice, tuple, or list
            Indexing key. Can be:
            - An integer: Creates single-element event
            - A slice: Creates event with slice of sample points
            - A tuple (index, name): Creates event with custom name
            - A list: Creates event with multiple sample points

        Returns
        -------
        event : Event
            An `Event` object containing the indexed sample points.

        Examples
        --------
        >>> import sigalg as sa
        >>> space = sa.SampleSpace(indices=["omega0", "omega1", "omega2", "omega3"])
        >>> # Access via integer index
        >>> event1 = space[0, "E"]
        >>> # Access via slice
        >>> event2 = space[1:3, "D"]
        >>> # Access via list of positions
        >>> event3 = space[[0, 2], "C"]
        """
        from .event import Event

        if isinstance(key, tuple) and len(key) == 2:
            item_idx, name = key
        else:
            item_idx = key
            name = "A"
        event_series = self.values[item_idx]
        if isinstance(item_idx, int):
            event_indices = [event_series]
        else:
            event_indices = event_series.to_list()
        return Event(sample_space=self, event_indices=event_indices, name=name)

    # --------------------- sequence methods --------------------- #

    def __len__(self) -> int:
        """Return the number of sample points in the sample space.

        Returns
        -------
        size : int
            The cardinality (size) of the sample space.

        Examples
        --------
        >>> import sigalg as sa
        >>> space = sa.SampleSpace(indices=["omega0", "omega1", "omega2"])
        >>> len(space)
        3
        """
        return len(self.values)

    def __iter__(self) -> iter:
        """Return an iterator over the sample points.

        Yields
        ------
        Hashable
            Each sample point (index) in the sample space in order.

        Examples
        --------
        >>> import sigalg as sa
        >>> space = sa.SampleSpace(indices=["omega0", "omega1", "omega2"])
        >>> for outcome in space:
        ...     print(outcome)
        omega0
        omega1
        omega2
        """
        return iter(self.values)

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a string representation of the sample space.

        Returns
        -------
        repr_str : str
            A formatted string showing the sample space name and its sample points.

        Examples
        --------
        >>> import sigalg as sa
        >>> space = sa.SampleSpace(indices=["H", "T"], name="CoinFlip")
        >>> repr(space)
        "Sample space 'CoinFlip':\n['H', 'T']"
        """
        return f"Sample space '{self.name}':\n{self.values.to_list()}"

    # --------------------- equality --------------------- #

    def __eq__(self, other: SampleSpace) -> bool:
        """Check equality with another sample space.

        Two sample spaces are equal if they have the same sample points in the
        same order.

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        equal : bool
            `True` if the other object is a `SampleSpace` with identical values,
            `False` otherwise.

        Examples
        --------
        >>> import sigalg as sa
        >>> space1 = sa.SampleSpace(indices=["omega0", "omega1"], name="S")
        >>> space2 = sa.SampleSpace(indices=["omega0", "omega1"], name="S")
        >>> space1 == space2
        True
        >>> space3 = sa.SampleSpace(indices=["omega1", "omega0"], name="S")
        >>> space1 == space3  # Different order
        False
        """
        return isinstance(other, SampleSpace) and self.values.equals(other.values)


class SampleSpaceMethods:
    """Mixin class providing sample space methods to other classes.

    This mixin provides convenience methods for classes that have a `sample_space`
    attribute, allowing them to delegate sample space operations to that attribute.

    The class assumes the implementing class has a `sample_space` attribute that
    is a `SampleSpace` instance.

    Examples
    --------
    >>> class MyClass(SampleSpaceMethods):
    ...     def __init__(self, sample_space):
    ...         self.sample_space = sample_space
    >>> import sigalg as sa
    >>> space = sa.SampleSpace(indices=["a", "b", "c"])
    >>> obj = MyClass(space)
    >>> event = obj.get_event(["a", "b"], name="E")
    """

    def get_event(self, event_indices: list[Hashable], name: str = "A") -> Event:
        """Create an event from a list of sample point indices.

        Delegates to the `sample_space.get_event` method.

        Parameters
        ----------
        event_indices : list of Hashable
            List of sample point indices to include in the event.
        name : str, default="A"
            Name identifier for the event.

        Returns
        -------
        event : Event
            An `Event` object containing the specified sample points.

        Raises
        ------
        TypeError
            If `event_indices` is not a list.
        ValueError
            If any index in `event_indices` is not found in the sample space.
        """
        return self.sample_space.get_event(event_indices, name)

    def _getitem_hook(self, key):
        """Internal hook for indexing operations.

        Delegates to the `sample_space._getitem_hook` method.

        Parameters
        ----------
        key : int, slice, tuple, or list
            Indexing key for creating events.

        Returns
        -------
        event : Event
            An `Event` object based on the indexing operation.
        """
        return self.sample_space._getitem_hook(key)
