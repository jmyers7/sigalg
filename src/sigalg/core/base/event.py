"""A class representing an event."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

from .index import Index
from .sample_space import SampleSpaceMethods

if TYPE_CHECKING:
    from .sample_space import SampleSpace


class Event(SampleSpaceMethods, Index):
    r"""A class representing an event space.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    sample_space : SampleSpace
        The sample space to which this event belongs.
    name : Hashable | None, default="A"
        Name identifier for the event.
    data_name : Hashable | None, default="sample"
        Name for the underlying `pd.Index`.

    Raises
    ------
    TypeError
        If `sample_space` is not a `SampleSpace` instance.

    Examples
    --------
    >>> from sigalg.core import Event, SampleSpace
    >>> Omega = SampleSpace().from_sequence(size=4)
    >>> # Extract an event by calling the `Event` constructor
    >>> A = Event(sample_space=Omega, name="A").from_list([0, 2])
    >>> print(A) # doctest: +NORMALIZE_WHITESPACE
    Event 'A':
    [0, 2]
    >>> # Extract an event directly from the sample space
    >>> B = Omega.get_event([1, 3], name="B")
    >>> print(B) # doctest: +NORMALIZE_WHITESPACE
    Event 'B':
    [1, 3]

    Notes
    -----
    Let $\mathcal{F}$ be a $\sigma$-algebra on a sample space $\Omega$. An *event* (relative to $\mathcal{F}$) is a subset $A$ of $\Omega$ in $\mathcal{F}$. In general measure theory, an event is called an $\mathcal{F}$-measurable set.

    However, in SigAlg, we do *not* enforce the requirement that an event is measurable with respect to a pre-existing $\sigma$-algebra. Any subset of a sample space can be represented as an instance of `Event`.
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
        """Create an Event from a list of sample points.

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
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> A = Event(sample_space=Omega, name="A").from_list([0, 2])
        >>> print(A) # doctest: +NORMALIZE_WHITESPACE
        Event 'A':
        [0, 2]
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
        >>> Omega = SampleSpace().from_sequence(size=5, prefix="omega")
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
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> A = Omega.get_event([0])
        >>> A.complement() # doctest: +NORMALIZE_WHITESPACE
        Event 'A complement':
        [1, 2]
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

        Examples
        --------
        >>> from sigalg.core import Event, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> A = Omega.get_event([0, 1])
        >>> B = Omega.get_event([1, 2], name="B")
        >>> A.intersection(B) # doctest: +NORMALIZE_WHITESPACE
        Event 'A intersect B':
        [1]
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

        Examples
        --------
        >>> from sigalg.core import Event, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> A = Omega.get_event([0])
        >>> B = Omega.get_event([1], name="B")
        >>> A.union(B) # doctest: +NORMALIZE_WHITESPACE
        Event 'A union B':
        [0, 1]
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

        Examples
        --------
        >>> from sigalg.core import Event, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> A = Omega.get_event([0, 1])
        >>> B = Omega.get_event([1, 2], name="B")
        >>> A.difference(B) # doctest: +NORMALIZE_WHITESPACE
        Event 'A difference B':
        [0]
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

        Raises
        ------
        ValueError
            If events are from different sample spaces.

        Returns
        -------
        event : Event
            An event containing sample points in either event.
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

        Raises
        ------
        ValueError
            If events are from different sample spaces.

        Returns
        -------
        event : Event
            An event containing sample points in both events.
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

        Raises
        ------
        ValueError
            If events are from different sample spaces.

        Returns
        -------
        event : Event
            An event containing sample points in this event but not in `other`.
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

        Raises
        ------
        ValueError
            If events are from different sample spaces.

        Returns
        -------
        is_le : bool
            True if this event is a subset of the other event.
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

        Raises
        ------
        ValueError
            If events are from different sample spaces.

        Returns
        -------
        is_lt : bool
            True if this event is a proper subset of the other event.
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

        Raises
        ------
        ValueError
            If events are from different sample spaces.

        Returns
        -------
        is_ge : bool
            True if this event is a superset of the other event.
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

        Raises
        ------
        ValueError
            If events are from different sample spaces.

        Returns
        -------
        is_gt : bool
            True if this event is a proper superset of the other event.
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
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> A = Omega.get_event([0, 1])
        >>> A.to_sample_space() # doctest: +NORMALIZE_WHITESPACE
        Sample space 'A':
        [0, 1]
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
