"""Event module.

This module provides the `Event` class, which represents a subset of a sample space. Events support set-theoretic operations (union, intersection, complement, difference) and subset/superset relationships.

Classes
-------
Event
    Represents an event as a subset of a sample space.

Examples
--------
>>> from sigalg.core import Event, SampleSpace
>>> Omega = SampleSpace.generate_sequence(size=4)
>>> A = Event(name="A", sample_space=Omega).from_list(["omega_0", "omega_1"])
>>> B = Event(name="B", sample_space=Omega).from_list(["omega_1", "omega_2"])
>>> union = A | B
>>> union # doctest: +NORMALIZE_WHITESPACE
Event 'A union B':
['omega_0', 'omega_1', 'omega_2']
>>> intersection = A & B
>>> intersection # doctest: +NORMALIZE_WHITESPACE
Event 'A intersect B':
['omega_1']
>>> complement = ~A
>>> complement # doctest: +NORMALIZE_WHITESPACE
Event 'A complement':
['omega_2', 'omega_3']
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

from .index import Index
from .sample_space import SampleSpaceMethods

if TYPE_CHECKING:
    from .sample_space import SampleSpace


class Event(SampleSpaceMethods, Index):
    """A class representing an event in a sample space.

    In the mathematical theory, an event is supposed to be a measurable subset of a sample space with respect to a given sigma-algebra. However, in SigAlg, we do *not* enforce this requirement.

    Parameters
    ----------
    sample_space : SampleSpace
            The sample space to which this event belongs.
    name : Hashable | None, default="A"
        Name identifier for the event.
    data_name : Hashable | None, default="sample"
        Name for the index of values.

    Raises
    ------
    TypeError
        If `sample_space` is not a `SampleSpace` instance.

    Examples
    --------
    >>> from sigalg.core import Event, SampleSpace
    >>> Omega = SampleSpace.generate_sequence(size=4)
    >>> A = Event(name="A", sample_space=Omega).from_list(["omega_0", "omega_1"])
    >>> B = Event(name="B", sample_space=Omega).from_list(["omega_1", "omega_2"])
    >>> union = A | B
    >>> union # doctest: +NORMALIZE_WHITESPACE
    Event 'A union B':
    ['omega_0', 'omega_1', 'omega_2']
    >>> intersection = A & B
    >>> intersection # doctest: +NORMALIZE_WHITESPACE
    Event 'A intersect B':
    ['omega_1']
    >>> complement = ~A
    >>> complement # doctest: +NORMALIZE_WHITESPACE
    Event 'A complement':
    ['omega_2', 'omega_3']
    """

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        sample_space: SampleSpace,
        name: Hashable | None = "A",
        data_name: Hashable | None = "sample",
    ) -> None:
        from .sample_space import SampleSpace

        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")
        self.sample_space = sample_space
        super().__init__(name=name, data_name=data_name)

    def from_list(
        self,
        indices: list[Hashable],
    ) -> Event:
        """Create an Event from a list of sample point indices.

        Parameters
        ----------
        indices : list[Hashable]
            List of sample point indices to include in the event.

        Returns
        -------
        self : Event
            The event instance with the specified sample points.

        Examples
        --------
        >>> from sigalg.core import Event, SampleSpace
        >>> Omega = SampleSpace.generate_sequence(size=4)
        >>> A = Event(name="A", sample_space=Omega).from_list(indices=["omega_0", "omega_2"])
        >>> A # doctest: +NORMALIZE_WHITESPACE
        Event 'A':
        ['omega_0', 'omega_2']
        """
        self._validate_parameters(indices=indices, sample_space=self.sample_space)
        pts = set(indices)
        ordered_indices = [idx for idx in self.sample_space.data if idx in pts]
        self._indices = ordered_indices
        return self

    # --------------------- data access methods --------------------- #

    def _getitem_hook(self, pos: int | list[int] | slice) -> Event | Hashable:
        """Internal hook for indexing operations to create events.

        This method is called by `__getitem__` from the parent `Index` class. In `Event`, the purpose of this method is to ensure that `__getitem__` returns an instance of `Event`. Items are retrieved by position.

        Parameters
        ----------
        pos : int, slice, tuple, or list
            Indexing key for accessing sample points. An integer creates a single-element event, a slice creates an event with a slice of sample points, a tuple `(index, name)` creates an event with a custom name, and a `list` creates an event with multiple sample points.

        Returns
        -------
        event : Event | Hashable
            An `Event` object containing the indexed sample points, or a single hashable if `pos` is an `int`.

        Examples
        --------
        >>> from sigalg.core import SampleSpace
        >>> Omega = SampleSpace.generate_sequence(size=5)
        >>> A = Omega.get_event(["omega_0", "omega_2", "omega_4"], name="A")
        >>> # Access via integer index
        >>> E = A[0, "E"]
        >>> # Access via slice
        >>> D = A[1:3, "D"]
        >>> # Access via list of positions
        >>> C = A[[0, 2], "C"]
        """  # noqa: D401
        from .event import Event

        if isinstance(pos, tuple):
            if len(pos) != 2:
                raise TypeError("Use `Event[idx]` or `Event[idx, name]`.")
            item_idx, name = pos
            if not isinstance(name, Hashable):
                raise TypeError("Event name must be hashable.")
        else:
            item_idx, name = pos, "A"

        if not isinstance(item_idx, (int, slice, list)):
            raise TypeError("Index must be an int, slice, or list[int].")

        item = self.data[item_idx]

        if isinstance(item_idx, int):
            return item
        else:
            return Event(name=name, sample_space=self.sample_space).from_list(
                indices=item.to_list()
            )

    # --------------------- set-theoretic operations --------------------- #

    def complement(self) -> Event:
        """Return the complement of this event.

        Returns
        -------
        event : Event
            An event containing all sample points not in this event.

        Examples
        --------
        >>> from sigalg.core import Event, SampleSpace
        >>> Omega = SampleSpace.generate_sequence(size=3)
        >>> A = Event(name="A", sample_space=Omega).from_list(indices=["omega_0"])
        >>> A.complement() # doctest: +NORMALIZE_WHITESPACE
        Event 'A complement':
        ['omega_1', 'omega_2']
        """
        return ~self

    def intersection(self, other: Event) -> Event:
        """Return the intersection of this event with another event.

        Parameters
        ----------
        other : Event
            Another event from the same sample space.

        Returns
        -------
        event : Event
            An event containing sample points in both events.

        Raises
        ------
        ValueError
            If events are from different sample spaces.

        Examples
        --------
        >>> from sigalg.core import Event, SampleSpace
        >>> Omega = SampleSpace.generate_sequence(size=3)
        >>> A = Event(name="A", sample_space=Omega).from_list(indices=["omega_0", "omega_1"])
        >>> B = Event(name="B", sample_space=Omega).from_list(indices=["omega_1", "omega_2"])
        >>> A.intersection(B) # doctest: +NORMALIZE_WHITESPACE
        Event 'A intersect B':
        ['omega_1']
        """
        return self & other

    def union(self, other: Event) -> Event:
        """Return the union of this event with another event.

        Parameters
        ----------
        other : Event
            Another event from the same sample space.

        Returns
        -------
        event : Event
            An event containing sample points in either event.

        Raises
        ------
        ValueError
            If events are from different sample spaces.

        Examples
        --------
        >>> from sigalg.core import Event, SampleSpace
        >>> Omega = SampleSpace.generate_sequence(size=3)
        >>> A = Event(name="A", sample_space=Omega).from_list(indices=["omega_0"])
        >>> B = Event(name="B", sample_space=Omega).from_list(indices=["omega_1"])
        >>> A.union(B) # doctest: +NORMALIZE_WHITESPACE
        Event 'A union B':
        ['omega_0', 'omega_1']
        """
        return self | other

    def difference(self, other: Event) -> Event:
        """Return the set difference of this event and another event.

        Parameters
        ----------
        other : Event
            Another event from the same sample space.

        Returns
        -------
        event : Event
            An event containing sample points in this event but not in `other`.

        Raises
        ------
        ValueError
            If events are from different sample spaces.

        Examples
        --------
        >>> from sigalg.core import Event, SampleSpace
        >>> Omega = SampleSpace.generate_sequence(size=3)
        >>> A = Event(name="A", sample_space=Omega).from_list(indices=["omega_0", "omega_1"])
        >>> B = Event(name="B", sample_space=Omega).from_list(indices=["omega_1", "omega_2"])
        >>> A.difference(B) # doctest: +NORMALIZE_WHITESPACE
        Event 'A difference B':
        ['omega_0']
        """
        return self - other

    # --------------------- set-theoretic operators --------------------- #

    def __invert__(self) -> Event:
        """Return the complement of this event (`~` operator).

        Returns
        -------
        event : Event
            An event containing all sample points not in this event.
        """
        space = self.sample_space.data
        pts = set(self.data)
        comp = [idx for idx in space if idx not in pts]
        return Event(
            name=f"{self.name} complement", sample_space=self.sample_space
        ).from_list(indices=comp)

    def __or__(self, other: Event) -> Event:
        """Return the union of this event with another event (`|` operator).

        Parameters
        ----------
        other : Event
            Another event from the same sample space.

        Returns
        -------
        event : Event
            An event containing sample points in either event.

        Raises
        ------
        ValueError
            If events are from different sample spaces.
        """
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        pts = set(self.data) | set(other.data)
        return Event(
            name=f"{self.name} union {other.name}", sample_space=self.sample_space
        ).from_list(
            indices=list(pts),
        )

    def __and__(self, other: Event) -> Event:
        """Return the intersection of this event with another event (`&` operator).

        Parameters
        ----------
        other : Event
            Another event from the same sample space.

        Returns
        -------
        event : Event
            An event containing sample points in both events.

        Raises
        ------
        ValueError
            If events are from different sample spaces.
        """
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        pts = set(self.data) & set(other.data)
        return Event(
            name=f"{self.name} intersect {other.name}", sample_space=self.sample_space
        ).from_list(
            indices=list(pts),
        )

    def __sub__(self, other: Event) -> Event:
        """Return the set difference of this event and another event (`-` operator).

        Parameters
        ----------
        other : Event
            Another event from the same sample space.

        Returns
        -------
        event : Event
            An event containing sample points in this event but not in `other`.

        Raises
        ------
        ValueError
            If events are from different sample spaces.
        """
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        pts = set(self.data) - set(other.data)
        return Event(
            name=f"{self.name} difference {other.name}", sample_space=self.sample_space
        ).from_list(
            indices=list(pts),
        )

    # --------------------- sub/superset methods --------------------- #

    def __le__(self, other: Event) -> bool:
        """Check if this event is a subset of another event (`<=` operator).

        Parameters
        ----------
        other : Event
            Another event from the same sample space.

        Returns
        -------
        is_le : bool
            True if this event is a subset of the other event.

        Raises
        ------
        ValueError
            If events are from different sample spaces.
        """
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        return set(self.data).issubset(set(other.data))

    def __lt__(self, other: Event) -> bool:
        """Check if this event is a proper subset of another event (`<` operator).

        Parameters
        ----------
        other : Event
            Another event from the same sample space.

        Returns
        -------
        is_lt : bool
            True if this event is a proper subset of the other event.

        Raises
        ------
        ValueError
            If events are from different sample spaces.
        """
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        return set(self.data) < set(other.data)

    def __ge__(self, other: Event) -> bool:
        """Check if this event is a superset of another event (`>=` operator).

        Parameters
        ----------
        other : Event
            Another event from the same sample space.

        Returns
        -------
        is_ge : bool
            True if this event is a superset of the other event.

        Raises
        ------
        ValueError
            If events are from different sample spaces.
        """
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        return set(self.data).issuperset(set(other.data))

    def __gt__(self, other: Event) -> bool:
        """Check if this event is a proper superset of another event (`>` operator).

        Parameters
        ----------
        other : Event
            Another event from the same sample space.

        Returns
        -------
        is_gt : bool
            True if this event is a proper superset of the other event.

        Raises
        ------
        ValueError
            If events are from different sample spaces.
        """
        if self.sample_space != other.sample_space:
            raise ValueError("Events must come from the same sample space.")
        return set(self.data) > set(other.data)

    # --------------------- equality --------------------- #

    def __eq__(self, other) -> bool:
        """Check equality with another event.

        Two events are equal if they belong to the same sample space and
        contain the same sample points in the same order.

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the other object is an `Event` with identical sample space
            and values, `False` otherwise.
        """
        return (
            isinstance(other, Event)
            and self.sample_space == other.sample_space
            and self.data.equals(other.data)
        )

    # --------------------- conversion methods --------------------- #

    def to_sample_space(self) -> SampleSpace:
        """Convert this event to a sample space.

        Creates a new `SampleSpace` containing only the sample points in this event.

        Returns
        -------
        sample_space : SampleSpace
            A sample space containing this event's outcomes.

        Examples
        --------
        >>> from sigalg.core import Event, SampleSpace
        >>> Omega = SampleSpace.generate_sequence(size=3)
        >>> A = Event(name="A", sample_space=Omega).from_list(indices=["omega_0", "omega_1"])
        >>> A.to_sample_space() # doctest: +NORMALIZE_WHITESPACE
        Sample space 'A':
        ['omega_0', 'omega_1']
        """
        from ..base import SampleSpace

        return SampleSpace(name=self.name, data_name=self.data.name).from_list(
            self.data.to_list()
        )

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a string representation of the event.

        Returns
        -------
        repr_str : str
            A formatted string showing the event name and its sample points.
        """
        return f"Event '{self.name}':\n{self.data.to_list()}"

    # --------------------- validation methods --------------------- #

    @staticmethod
    def _validate_parameters(
        indices: list[Hashable],
        sample_space: SampleSpace,
    ):
        """Validate parameters for the Event constructor.

        Parameters
        ----------
        indices : list[Hashable]
            List of sample point indices to include in the event.
        sample_space : SampleSpace
            The sample space to which this event belongs.

        Raises
        ------
        TypeError
            If `sample_space` is not a `SampleSpace` instance or `indices`
            is not a `list`.
        ValueError
            If any index in `indices` is not found in the sample space.
        """
        from .sample_space import SampleSpace

        if not isinstance(indices, list):
            raise TypeError("indices must be a list.")
        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")
        if any(idx not in sample_space.data for idx in indices):
            raise ValueError("All indices must be in the sample space.")
