"""Events for probability theory.

This module provides the `Event` class, which represents a subset of a sample space. Events support set-theoretic operations (union, intersection, complement, difference) and subset/superset relationships.

Classes
-------
Event
    Represents an event as a subset of a sample space.

Examples
--------
>>> from sigalg.core import Event, SampleSpace
>>> Omega = SampleSpace(indices=["omega0", "omega1", "omega2", "omega3"])
>>> A = Event(sample_space=Omega, event_indices=["omega0", "omega1"], name="A")
>>> B = Event(sample_space=Omega, event_indices=["omega1", "omega2"], name="B")
>>> union = A | B
>>> intersection = A & B
>>> complement = ~A
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

from .index import Index
from .sample_space import SampleSpaceMethods

if TYPE_CHECKING:
    from .sample_space import SampleSpace


class Event(SampleSpaceMethods, Index):
    """An event representing a subset of a sample space.

    Events are fundamental objects in probability theory representing collections
    of outcomes from a sample space. They support set-theoretic operations and
    maintain order according to the underlying sample space.

    In the mathematical theory, an event is supposed to be a measurable subset of a sample space with respect to a given sigma-algebra. However, in SigAlg, we do *not* enforce this requirement.

    Parameters
    ----------
    sample_space : SampleSpace
        The sample space to which this event belongs.
    event_indices : list[Hashable]
        `list[Hashable]` of sample point indices to include in the event.
        All indices must exist in the sample space.
    name : str, default="A"
        Name identifier for the event.
    values_name : str, default="sample"
        Name for the index of values.

    Raises
    ------
    TypeError
        If `sample_space` is not a `SampleSpace` instance or `event_indices`
        is not a `list`.
    ValueError
        If any index in `event_indices` is not found in the sample space.

    Examples
    --------
    >>> from sigalg.core import Event, SampleSpace
    >>> Omega = SampleSpace(indices=["omega0", "omega1", "omega2", "omega3"])
    >>> A = Event(sample_space=Omega, event_indices=["omega0", "omega1"], name="A")
    >>> len(A)
    2
    >>> # Set operations
    >>> B = Event(sample_space=Omega, event_indices=["omega1", "omega2"], name="B")
    >>> union = A | B
    >>> intersection = A & B
    >>> complement = ~A
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        sample_space: SampleSpace,
        event_indices: list[Hashable],
        name: str = "A",
        values_name: str = "sample",
    ) -> None:
        self._validate_event_parameters(
            sample_space=sample_space,
            event_indices=event_indices,
        )
        pts = set(event_indices)
        ordered = [idx for idx in sample_space.data if idx in pts]
        super().__init__(indices=ordered, name=name, data_name=values_name)
        self.sample_space = sample_space

    def _getitem_hook(self, key) -> Event | Hashable:
        from .event import Event

        if isinstance(key, tuple) and len(key) == 2:
            item_idx, name = key
        else:
            item_idx = key
            name = "A"
        if isinstance(item_idx, int):
            return self.data[item_idx]
        else:
            return Event(
                sample_space=self.sample_space,
                event_indices=self.data[item_idx].to_list(),
                name=name,
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
        >>> Omega = SampleSpace(indices=["omega0", "omega1", "omega2"])
        >>> A = Event(sample_space=Omega, event_indices=["omega0"], name="A")
        >>> comp = A.complement()
        >>> len(comp)
        2
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
        >>> Omega = SampleSpace(indices=["omega0", "omega1", "omega2"])
        >>> A = Event(sample_space=Omega, event_indices=["omega0", "omega1"], name="A")
        >>> B = Event(sample_space=Omega, event_indices=["omega1", "omega2"], name="B")
        >>> intersect = A.intersection(B)
        >>> len(intersect)
        1
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
        >>> Omega = SampleSpace(indices=["omega0", "omega1", "omega2"])
        >>> A = Event(sample_space=Omega, event_indices=["omega0"], name="A")
        >>> B = Event(sample_space=Omega, event_indices=["omega1"], name="B")
        >>> union = A.union(B)
        >>> len(union)
        2
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
        >>> Omega = SampleSpace(indices=["omega0", "omega1", "omega2"])
        >>> A = Event(sample_space=Omega, event_indices=["omega0", "omega1"], name="A")
        >>> B = Event(sample_space=Omega, event_indices=["omega1", "omega2"], name="B")
        >>> diff = A.difference(B)
        >>> len(diff)
        1
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
        return Event(self.sample_space, comp, name=f"{self.name} complement")

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
            self.sample_space, list(pts), name=f"{self.name} union {other.name}"
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
            self.sample_space, list(pts), name=f"{self.name} intersect {other.name}"
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
            self.sample_space, list(pts), name=f"{self.name} difference {other.name}"
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
        >>> Omega = SampleSpace(indices=["omega0", "omega1", "omega2"])
        >>> A = Event(sample_space=Omega, event_indices=["omega0", "omega1"], name="A")
        >>> new_space = A.to_sample_space()
        >>> len(new_space)
        2
        """
        from ..base import SampleSpace

        return SampleSpace(self.data.to_list())

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
    def _validate_event_parameters(
        sample_space: SampleSpace,
        event_indices: list[Hashable],
    ) -> None:
        """Validate event construction parameters.

        Parameters
        ----------
        sample_space : SampleSpace
            The sample space to validate.
        event_indices : list[Hashable]
            The `list[Hashable]` of event indices to validate.

        Raises
        ------
        TypeError
            If `sample_space` is not a `SampleSpace` instance or `event_indices`
            is not a `list`.
        ValueError
            If any index in `event_indices` is not found in the sample space.
        """
        from .sample_space import SampleSpace

        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")
        if not isinstance(event_indices, list):
            raise TypeError("event_indices must be a list.")
        for idx in event_indices:
            if idx not in sample_space.data:
                raise ValueError(f"Index '{idx}' not found in sample_space.")
