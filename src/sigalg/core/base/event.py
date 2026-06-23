"""A class representing an event."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import pandas as pd

from .index import Index

if TYPE_CHECKING:
    from ...validation.index_validator import IndexLike
    from ..probability_measures.probability_measure import ProbabilityMeasure
    from ..random_objects.random_variable import RandomVariable
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from .probability_space import ProbabilitySpace
    from .sample_space import SampleSpace


class Event(Index):
    r"""A class representing an event.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    sample_space : SampleSpace | None, default=None
        The sample space containing this event.
    sig_alg : SigmaAlgebra | None, default=None
        The sigma-algebra containing this event.
    prob_measure : ProbabilityMeasure | None, default=None
        The probability measure associated with this event.
    name : Hashable, default="A"
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

    _properties = Index._properties + ["_indicator", "_is_atom", "_atom_id"]

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        indices: IndexLike | None = None,
        name: Hashable = "A",
        variable_names: list[Hashable] | None = None,
        bypass_validation: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            indices=indices,
            name=name,
            variable_names=variable_names,
            bypass_validation=bypass_validation,
        )

    @classmethod
    def from_list(
        cls,
        indices: list,
        sig_alg: SigmaAlgebra,
        prob_measure: ProbabilityMeasure | None = None,
        name: Hashable = "A",
    ) -> Event:
        """Create an Event from a list of sample points.

        Parameters
        ----------
        indices : list
            List of sample point indices to include in the event.

        Raises
        ------
        TypeError
            If `indices` is not a list.
        ValueError
            If the event defined by `indices` is not measurable with respect to the sigma-algebra, or if it is not a subset of the sample space.

        Returns
        -------
        self : Event
            The event instance with the specified sample points.

        Examples
        --------
        >>> from sigalg.core import Event, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=5)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 1,
        ...         1: 1,
        ...         2: 0,
        ...         3: 0,
        ...         4: 2,
        ...     },
        ... )
        >>> A = Event.from_list(indices=[0, 1], sig_alg=F)
        >>> print(A)  # doctest: +NORMALIZE_WHITESPACE
        Event 'A':
        [0, 1]
        >>> # Try to get a non-measurable event
        >>> B = Event.from_list(indices=[0, 2], sig_alg=F, name="B")
        Traceback (most recent call last):
            ...
        ValueError: The event is not measurable.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .probability_space import ProbabilitySpace

        if not isinstance(indices, list):
            raise TypeError("The indices must be a list.")
        if not isinstance(sig_alg, SigmaAlgebra):
            raise ValueError("sig_alg must be a SigmaAlgebra")

        prob_space = ProbabilitySpace(
            sig_alg=sig_alg,
            prob_measure=prob_measure,
        )

        event_set = set(indices)
        sample_space_set = set(prob_space.sample_space)

        if not event_set.issubset(sample_space_set):
            raise ValueError(
                "The event is not a subset of the sample space of the sigma-algebra."
            )

        ordered_event = [
            omega for omega in prob_space.sample_space if omega in event_set
        ]
        event = cls(
            indices=ordered_event,
            name=name,
            variable_names=prob_space.sample_space.variable_names,
        )

        event._prob_space = prob_space

        is_measurable, event._atom_id = event._test_measurability_and_atom(
            ordered_event
        )

        if not is_measurable:
            raise ValueError("The event is not measurable.")
        event._is_atom = event._atom_id is not None

        return event

    def _test_measurability_and_atom(self, ordered_event: list) -> bool:
        """Test whether the event defined by `ordered_event` is measurable with respect to the sigma-algebra, and if so, whether it is an atom.

        Parameters
        ----------
        ordered_event : list
            A list of sample points defining the event, ordered according to the sample space.

        Returns
        -------
        is_measurable : bool
            `True` if the event is measurable with respect to the sigma-algebra, `False` otherwise.
        atom_id : Hashable | None
            If the event is an atom, returns its atom ID; otherwise returns `None`.
        """
        indicator_data = pd.Series(
            {
                sample_point: 1 if sample_point in ordered_event else 0
                for sample_point in self.sample_space
            },
            name="indicator",
        )

        combined_data = pd.concat(
            [indicator_data, self.sig_alg.data], axis=1
        ).drop_duplicates()

        if len(combined_data) != self.sig_alg.num_atoms:
            is_measurable = False
            atom_id = None
        else:
            is_measurable = True
            if combined_data["indicator"].sum() == 1:
                if isinstance(self.sig_alg.data, pd.DataFrame):
                    sig_alg_cols = self.sig_alg.data.columns.to_list()
                    atom_id = combined_data[combined_data["indicator"] == 1][
                        sig_alg_cols
                    ]
                else:
                    sig_alg_cols = self.sig_alg.data.name
                    atom_id = combined_data[combined_data["indicator"] == 1][
                        sig_alg_cols
                    ].item()

            else:
                atom_id = None

        return is_measurable, atom_id

    # --------------------- properties --------------------- #

    @property
    def prob_space(self) -> ProbabilitySpace | None:
        """Get the probability space associated with this event.

        Returns
        -------
        prob_space : ProbabilitySpace | None
            The probability space associated with this event.
        """
        return self._prob_space

    @property
    def sample_space(self) -> SampleSpace | None:
        """Get the ambient sample space of the event.

        Returns
        -------
        sample_space : SampleSpace | None
            The ambient sample space of the event.
        """
        return self.prob_space.sample_space

    @property
    def sig_alg(self) -> SigmaAlgebra | None:
        """Get the sigma-algebra containing this event.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra containing this event.
        """
        return self.prob_space.sig_alg

    @property
    def prob_measure(self) -> ProbabilityMeasure | None:
        """Get the probability measure associated with this event.

        Returns
        -------
        prob_measure : ProbabilityMeasure | None
            The probability measure associated with this event.
        """
        return self.prob_space.prob_measure

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
        Omega
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
                    domain=self.sample_space,
                    sig_alg=self.sig_alg,
                    prob_measure=self.prob_measure,
                    name=name,
                ).from_dict(outputs)
            except ValueError as e:
                raise ValueError(
                    "The provided indices do not form a measurable event."
                ) from e

            self._indicator = indicator

        return self._indicator

    # TODO: write unit tests
    @property
    def is_atom(self) -> bool | None:
        """Return whether this event is an atom in the sigma-algebra."""
        return self._is_atom

    # TODO: write unit tests
    @property
    def atom_id(self) -> Hashable | None:
        """Return the atom ID if this event is an atom, or `None` otherwise."""
        return self._atom_id

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
        Define the power-set `SigmaAlgebra` on the sample space Omega = {0,1,2,3,4} and the event A = {0,2,4}.
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=5)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_event(indices=[0, 2, 4], name="A")

        Access the element in position 1 of the event A, namely 2.

        >>> E = A[1, "E"]
        >>> print(E)
        2

        Access elements of A by passing a slice of positions.

        >>> D = A[1:3, "D"]
        >>> print(D)  # doctest: +NORMALIZE_WHITESPACE
        Event 'D':
        [2, 4]

        Access elements of A by passing a list of positions.

        >>> C = A[[0, 2], "C"]
        >>> print(C)  # doctest: +NORMALIZE_WHITESPACE
        Event 'C':
        [0, 4]
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
        >>> Omega = SampleSpace.from_sequence(size=3)
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
        >>> Omega = SampleSpace.from_sequence(size=3)
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
        >>> Omega = SampleSpace.from_sequence(size=3)
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
        >>> Omega = SampleSpace.from_sequence(size=3)
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
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_event([0, 1])
        >>> A.to_sample_space() # doctest: +NORMALIZE_WHITESPACE
        Sample space 'A':
         A
         0
         1
        """
        from ..base import SampleSpace

        return SampleSpace(indices=self.data.to_list(), name=self.name)

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
