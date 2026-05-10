"""A class representing an event."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

from .index import Index

if TYPE_CHECKING:
    from ..base.sample_space import SampleSpace
    from ..random_objects.random_variable import RandomVariable
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra


class Event(Index):
    r"""A class representing an event.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    sig_alg : SigmaAlgebra | None, default=None
        The sigma-algebra with respect to which this event is measurable.
    name : Hashable | None, default="A"
        Name identifier for the event.

    Raises
    ------
    TypeError
        If `sig_alg` is not a `SigmaAlgebra` instance.

    Examples
    --------
    >>> from sigalg.core import SampleSpace, SigmaAlgebra
    >>> Omega = SampleSpace().from_sequence(size=4)
    >>> F = SigmaAlgebra.power_set(Omega)
    >>> # Extract an event by calling the `Event` constructor
    >>> A = Event(sig_alg=F, name="A").from_list([0, 2])
    >>> print(A) # doctest: +NORMALIZE_WHITESPACE
    Event 'A':
    [0, 2]
    >>> # Extract an event directly from the sigma-algebra
    >>> B = F.get_event([1, 3], name="B")
    >>> print(B) # doctest: +NORMALIZE_WHITESPACE
    Event 'B':
    [1, 3]

    Notes
    -----
    Let $\mathcal{F}$ be a $\sigma$-algebra on a sample space $\Omega$. An *event* (relative to $\mathcal{F}$) is a subset $A$ of $\Omega$ in $\mathcal{F}$. In general measure theory, an event is called an $\mathcal{F}$-measurable set.
    """

    # --------------------- constructors --------------------- #

    def __init__(
        self, sig_alg: SigmaAlgebra | None = None, name: Hashable | None = "A"
    ) -> None:
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be a SigmaAlgebra instance.")
        self._sig_alg = sig_alg
        data_name = sig_alg.sample_space.data_name if sig_alg is not None else None
        super().__init__(name=name, data_name=data_name)

        # caches
        self._indicator: RandomVariable | None = None

    def from_list(
        self,
        indices: list[Hashable],
    ) -> Event:
        """Create an Event from a list of sample points.

        Parameters
        ----------
        indices : list[Hashable]
            List of sample point indices to include in the event.

        Raises
        ------
        TypeError
            If `indices` is not a list of hashable objects.
        ValueError
            If the event defined by `indices` is not measurable with respect to the sigma-algebra, or if it is not a subset of the sample space.

        Returns
        -------
        self : Event
            The event instance with the specified sample points.

        Examples
        --------
        >>> from sigalg.core import Event, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=5)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 1,
        ...         1: 1,
        ...         2: 0,
        ...         3: 0,
        ...         4: 2,
        ...     }
        ... )
        >>> A = Event(sig_alg=F, name="A").from_list([0, 1])
        >>> print(A)  # doctest: +NORMALIZE_WHITESPACE
        Event 'A':
        [0, 1]
        >>> # Try to get a non-measurable event
        >>> B = Event(sig_alg=F, name="B").from_list([0, 2])
        Traceback (most recent call last):
            ...
        ValueError: The provided indices do not form a measurable event.
        """
        if not isinstance(indices, list):
            raise TypeError("The indices must form a list of Hashables.")
        if self.sig_alg is None:
            raise ValueError("Cannot create an event without a sigma-algebra.")
        event_set = set(indices)
        sample_space_set = set(self.sample_space)
        if not event_set.issubset(sample_space_set):
            raise ValueError(
                "The event is not a subset of the sample space of the sigma-algebra."
            )

        result = super().from_list(
            [omega for omega in self.sample_space if omega in event_set]
        )

        _ = self.indicator  # this checks for measurability

        return result

    def from_pandas(self, data, overwrite_data_name=False):  # noqa: D102
        raise NotImplementedError(
            "Events cannot be created from pandas data. Use `from_list` instead."
        )

    def from_sequence(self, size, initial_index=0, prefix=None):  # noqa: D102
        raise NotImplementedError(
            "Events cannot be created from sequences. Use `from_list` instead."
        )

    # --------------------- properties --------------------- #

    @property
    def sig_alg(self) -> SigmaAlgebra | None:
        """Get the sigma-algebra containing this event.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra containing this event.
        """
        return self._sig_alg

    @sig_alg.setter
    def sig_alg(self, sig_alg: SigmaAlgebra) -> None:
        """Set the sigma-algebra containing this event.

        Setting the sigma-algebra will clear any existing data.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The new sigma-algebra for this event.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be a SigmaAlgebra instance.")

        self._sig_alg = sig_alg
        self._indices = None
        self._data = None
        self._indicator = None

    @property
    def sample_space(self) -> SampleSpace | None:
        """Get the ambient sample space of the event.

        Returns
        -------
        sample_space : SampleSpace | None
            The ambient sample space of the event.
        """
        return self.sig_alg.sample_space if self.sig_alg is not None else None

    @property
    def indicator(self) -> RandomVariable | None:
        """Get the indicator random variable of the event.

        Returns
        -------
        indicator : RandomVariable | None
            The indicator random variable of the event, or `None` if it has not been created yet.


        Examples
        --------
        >>> from sigalg.core import Event, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 1,
        ...         1: 0,
        ...         2: 1,
        ...     }
        ... )
        >>> A = Event(sig_alg=F, name="A").from_list([0, 2])
        >>> A.indicator # doctest: +NORMALIZE_WHITESPACE
        Random variable 'I_A':
                I_A
        sample
        0         1
        1         0
        2         1
        """
        from ..random_objects.random_variable import RandomVariable

        if self._indicator is None and self.indices is not None:
            outputs = {
                omega: 1 if omega in self.indices else 0 for omega in self.sample_space
            }
            name = f"I_{self.name}" if self.name is not None else "indicator"

            try:
                indicator = RandomVariable(
                    domain=self.sample_space, sig_alg=self.sig_alg, name=name
                ).from_dict(outputs)
            except ValueError as e:
                raise ValueError(
                    "The provided indices do not form a measurable event."
                ) from e

            self._indicator = indicator

        return self._indicator

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
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=5, prefix="omega")
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_event(["omega_0", "omega_2", "omega_4"], name="A")
        >>> # Access via integer index
        >>> E = A[0, "E"]
        >>> # Access via slice
        >>> D = A[1:3, "D"]
        >>> # Access via list of positions
        >>> C = A[[0, 2], "C"]
        """  # noqa: D401
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
            return self.sig_alg.get_event(item.to_list(), name)

    # --------------------- set-theoretic operations --------------------- #

    def complement(self) -> Event:
        """Return the complement of this event.

        Returns
        -------
        event : Event
            An event containing all sample points not in this event.

        Examples
        --------
        >>> from sigalg.core import Event, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_event([0])
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
        >>> from sigalg.core import Event, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_event([0, 1])
        >>> B = F.get_event([1, 2], name="B")
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
        >>> from sigalg.core import Event, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_event([0])
        >>> B = F.get_event([1], name="B")
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
        >>> from sigalg.core import Event, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_event([0, 1])
        >>> B = F.get_event([1, 2], name="B")
        >>> A.difference(B) # doctest: +NORMALIZE_WHITESPACE
        Event 'A difference B':
        [0]
        """
        return self - other

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
        return self.sig_alg.get_event(comp, name=f"{self.name} complement")

    def __and__(self, other: Event) -> Event:
        """Return the intersection of this event with another event (`&` operator).

        Parameters
        ----------
        other : Event
            Another event from the same sigma-algebra.

        Raises
        ------
        ValueError
            If events are from different sigma-algebras.

        Returns
        -------
        event : Event
            An event containing sample points in both events.
        """
        if self.sig_alg != other.sig_alg:
            raise ValueError("Events must belong to the same sigma-algebra.")
        pts = set(self.data) & set(other.data)
        return self.sig_alg.get_event(
            list(pts), name=f"{self.name} intersect {other.name}"
        )

    def __or__(self, other: Event) -> Event:
        """Return the union of this event with another event (`|` operator).

        Parameters
        ----------
        other : Event
            Another event from the same sigma-algebra.

        Raises
        ------
        ValueError
            If events are from different sigma-algebras.

        Returns
        -------
        event : Event
            An event containing sample points in either event.
        """
        if self.sig_alg != other.sig_alg:
            raise ValueError("Events must belong to the same sigma-algebra.")
        pts = set(self.data) | set(other.data)
        return self.sig_alg.get_event(list(pts), name=f"{self.name} union {other.name}")

    def __sub__(self, other: Event) -> Event:
        """Return the set difference of this event and another event (`-` operator).

        Parameters
        ----------
        other : Event
            Another event from the same sigma-algebra.

        Raises
        ------
        ValueError
            If events are from different sigma-algebras.

        Returns
        -------
        event : Event
            An event containing sample points in this event but not in `other`.
        """
        if self.sig_alg != other.sig_alg:
            raise ValueError("Events must belong to the same sigma-algebra.")
        pts = set(self.data) - set(other.data)
        return self.sig_alg.get_event(
            list(pts), name=f"{self.name} difference {other.name}"
        )

    # --------------------- sub/superset methods --------------------- #

    def __le__(self, other: Event) -> bool:
        """Check if this event is a subset of another event (`<=` operator).

        Parameters
        ----------
        other : Event
            Another event from the same sigma-algebra.

        Raises
        ------
        ValueError
            If events are from different sigma-algebras.

        Returns
        -------
        is_le : bool
            True if this event is a subset of the other event.
        """
        if self.sig_alg != other.sig_alg:
            raise ValueError("Events must belong to the same sigma-algebra.")
        return set(self.data).issubset(set(other.data))

    def __lt__(self, other: Event) -> bool:
        """Check if this event is a proper subset of another event (`<` operator).

        Parameters
        ----------
        other : Event
            Another event from the same sigma-algebra.

        Raises
        ------
        ValueError
            If events are from different sigma-algebras.

        Returns
        -------
        is_lt : bool
            True if this event is a proper subset of the other event.
        """
        if self.sig_alg != other.sig_alg:
            raise ValueError("Events must belong to the same sigma-algebra.")
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
            Another event from the same sigma-algebra.

        Raises
        ------
        ValueError
            If events are from different sigma-algebras.

        Returns
        -------
        is_gt : bool
            True if this event is a proper superset of the other event.
        """
        if self.sig_alg != other.sig_alg:
            raise ValueError("Events must belong to the same sigma-algebra.")
        return set(self.data) > set(other.data)

    # --------------------- equality --------------------- #

    def __eq__(self, other) -> bool:
        """Check equality with another event.

        Two events are equal if they belong to the same sigma-algebra and
        contain the same sample points in the same order.

        Parameters
        ----------
        other : object
            Another object to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the other object is an `Event` with identical sigma-algebra
            and values, `False` otherwise.
        """
        return (
            isinstance(other, Event)
            and self.sig_alg == other.sig_alg
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
        >>> from sigalg.core import Event, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_event([0, 1])
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
        if self.data is None:
            if self.name is None:
                return "Event: empty"
            else:
                return f"Event '{self.name}': empty"
        else:
            if self.name is None:
                return f"Event:\n{self.data.to_list()}"
            else:
                return f"Event '{self.name}':\n{self.data.to_list()}"
