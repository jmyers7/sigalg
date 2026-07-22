"""A class representing an event."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

import pandas as pd

from .index import Index

if TYPE_CHECKING:
    from ..measures.probability_measure import ProbabilityMeasure
    from ..random_objects.random_variable import RandomVariable
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra
    from .probability_space import ProbabilitySpace
    from .sample_space import SampleSpace


class Event(Index):
    r"""A class representing an event.

    The constructor exists only becaus `Event` is a subclass of `Index`, but the user should construct events primarily using the `from_list` class method and the `get_event` method on a `SigmaAlgebra` instance or `ProbabilitySpace` instance.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    indices : IndexLike | None, default=None
        An `IndexLike` object containing the points in the event. If `None`, an empty event will be created.
    name : Hashable | None, default=None
        Name identifier for the index. If `None`, a default name `A` will be used.
    variable_names : list[Hashable] | None, default=None
        A list of variable names for the dimensions of the index. If `None`, a default variable name `sample` will be used.

    Raises
    ------
    TypeError
        If `sig_alg` is not a `SigmaAlgebra` instance.

    Examples
    --------
    Extract an event by calling the `from_list` class method.

    >>> from sigalg.core import Event, SampleSpace, SigmaAlgebra
    >>> Omega = SampleSpace.from_sequence(size=4)
    >>> F = SigmaAlgebra.power_set(Omega)
    >>> A = Event.from_list([0, 2], sig_alg=F)
    >>> print(A) # doctest: +NORMALIZE_WHITESPACE
    Event 'A':
     sample
          0
          2

    Extract an event directly from the sigma-algebra

    >>> B = F.get_event([1, 3], name="B")
    >>> print(B) # doctest: +NORMALIZE_WHITESPACE
    Event 'B':
     sample
          1
          3

    Notes
    -----
    Let $\mathcal{F}$ be a $\sigma$-algebra on a sample space $\Omega$. An *event* (relative to $\mathcal{F}$) is a subset $A$ of $\Omega$ in $\mathcal{F}$. In general measure theory, an event is called an $\mathcal{F}$-measurable set.
    """

    _properties = Index._properties + [
        "_sig_alg",
        "_prob_measure",
        "_indicator",
        "_is_atom",
        "_atom_id",
        "_prob_space",
    ]

    _default_name = "A"
    _repr_name = "Event"
    _variable_names_prefix = "sample"

    # --------------------- constructors --------------------- #

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
        sig_alg : SigmaAlgebra
            The sigma-algebra to which the event belongs.
        prob_measure : ProbabilityMeasure | None, default=None
            An optional probability measure from the probability space from which the event is drawn. If `None`, the uniform probability measure on the sigma-algebra will be used.

        Raises
        ------
        TypeError
            If `indices` is not a list, if `sig_alg` is not an instance of `SigmaAlgebra`, or if `prob_measure` is not an instance of `ProbabilityMeasure` (if given).
        ValueError
            If `indices` is not a subset of the sample space of the sigma-algebra, or if the event defined by `indices` is not measurable with respect to the sigma-algebra.

        Returns
        -------
        event : Event
            The event instance with the specified sample points.

        Examples
        --------
        Define a sigma-algebra with three atoms on the sample space Omega = {0,1,2,3,4}.

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

        Get the event A = {0,1} from the sample space.

        >>> A = Event.from_list(indices=[0, 1], sig_alg=F)
        >>> print(A)  # doctest: +NORMALIZE_WHITESPACE
        Event 'A':
         sample
              0
              1

        Try to build a non-measurable event.

        >>> B = Event.from_list(indices=[0, 2], sig_alg=F, name="B")
        Traceback (most recent call last):
            ...
        ValueError: The event is not measurable.
        """
        from ..measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .probability_space import ProbabilitySpace

        if not isinstance(indices, list):
            raise TypeError("The indices must be a list.")
        if not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be a SigmaAlgebra")
        if prob_measure is not None and not isinstance(
            prob_measure, ProbabilityMeasure
        ):
            raise TypeError("prob_measure must be a ProbabilityMeasure")

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
        """Get the sample space associated with this event.

        Returns
        -------
        sample_space : SampleSpace | None
            The ambient sample space of the event.
        """
        return self.prob_space.sample_space if self.prob_space is not None else None

    @property
    def sig_alg(self) -> SigmaAlgebra | None:
        """Get the sigma-algebra containing this event.

        The `sig_alg` property is settable. The new sigma-algebra must be a sub-sigma-algebra of the existing one, and the event must be measurable with respect to the new sigma-algebra. The probability measure will be updated to the restriction of the existing probability measure to the new sigma-algebra.

        Returns
        -------
        sig_alg : SigmaAlgebra | None
            The sigma-algebra containing this event.

        Examples
        --------
        Define a probability space.

        >>> from sigalg.core import ProbabilityMeasure, ProbabilitySpace, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=5)
        >>> F = SigmaAlgebra(
        ...    sample_space=Omega,
        ...    mapping={
        ...        0: 1,
        ...        1: 1,
        ...        2: 0,
        ...        3: 0,
        ...        4: 2,
        ...    },
        ... )
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.25,
        ...         1: 0.1,
        ...         2: 0.65,
        ...     },
        ... )
        >>> prob_space = ProbabilitySpace(Omega, F, P)

        Extract an event from the probability space and print its `sig_alg` property.

        >>> A = prob_space.get_event([0, 1])
        >>> print(A.sig_alg)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                atom_ID
        sample
        0             1
        1             1
        2             0
        3             0
        4             2

        Define a new sigma-algebra, a sub-sigma-algebra of the first.

        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 1,
        ...     },
        ...     name="G",
        ... )

        Set the sigma-algebra and print the updated probability space.

        >>> A.sig_alg = G
        >>> print(A.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, G, P|G)
        =================================
        <BLANKLINE>
        * Sample space 'Omega':
         sample
              0
              1
              2
              3
              4
        <BLANKLINE>
        * Sigma algebra 'G':
                atom_ID
        sample
        0             0
        1             0
        2             1
        3             1
        4             1
        <BLANKLINE>
        * Probability measure 'P|G':
                probability
        atom_ID
        0                0.1
        1                0.9
        """
        return self.prob_space.sig_alg if self.prob_space is not None else None

    @sig_alg.setter
    def sig_alg(self, sig_alg: SigmaAlgebra) -> None:
        """Set the sigma-algebra associated with this event.

        The new sigma-algebra must be a sub-sigma-algebra of the existing one, and the event must be measurable with respect to the new sigma-algebra. The probability measure will be updated to the restriction of the existing probability measure to the new sigma-algebra.

        Parameters
        ----------
        sig_alg : SigmaAlgebra
            The new sigma-algebra.

        Raises
        ------
        TypeError
            If `sig_alg` is not an instance of `SigmaAlgebra`.
        ValueError
            If the current instance of `Event` has a `prob_space` attribute equal to `None`, or if the current instance is not in the new sigma-algebra.
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        if not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("sig_alg must be an instance of SigmaAlgebra.")
        if self.prob_space is None:
            raise ValueError(
                "Cannot set a new sigma-algebra for an event whose prob_space attribute is `None`."
            )
        if self not in sig_alg:
            raise ValueError("The event must be in the new sigma-algebra.")

        self.prob_space.sig_alg = sig_alg
        self._indicator = None

    @property
    def prob_measure(self) -> ProbabilityMeasure | None:
        """Get the probability measure associated with this event.

        The `prob_measure` property is settable. The new probability measure must be defined on a sub-sigma-algebra of the current sigma-algebra and the event must be measureable with respect to the new sigma-algebra.

        Returns
        -------
        prob_measure : ProbabilityMeasure | None
            The probability measure associated with this event.

        Examples
        --------
        Define a probability space.

        >>> from sigalg.core import ProbabilityMeasure, ProbabilitySpace, SampleSpace, SigmaAlgebra
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
        >>> P = ProbabilityMeasure(
        ...     sig_alg=F,
        ...     mapping={
        ...         0: 0.25,
        ...         1: 0.1,
        ...         2: 0.65,
        ...     },
        ... )
        >>> prob_space = ProbabilitySpace(Omega, F, P)

        Extract an event from a probability space and print its `prob_space` property.

        >>> A = prob_space.get_event([0, 1])
        >>> print(A.prob_measure)  # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        atom_ID
        1               0.10
        0               0.25
        2               0.65

        Define a new probability measure on a sub-sigma-algebra of the first.

        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...         4: 1,
        ...     },
        ...     name="G",
        ... )
        >>> Q = ProbabilityMeasure(
        ...     sig_alg=G,
        ...     mapping={
        ...         0: 0.4,
        ...         1: 0.6,
        ...     },
        ...     name="Q"
        ... )

        Set a new sigma-algebra and print the update probability space of the event.

        >>> A.prob_measure = Q
        >>> print(A.prob_space)  # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, G, Q)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
         sample
              0
              1
              2
              3
              4
        <BLANKLINE>
        * Sigma algebra 'G':
                atom_ID
        sample
        0             0
        1             0
        2             1
        3             1
        4             1
        <BLANKLINE>
        * Probability measure 'Q':
                probability
        atom_ID
        0                0.4
        1                0.6
        """
        return self.prob_space.prob_measure if self.prob_space is not None else None

    @prob_measure.setter
    def prob_measure(self, prob_measure: ProbabilityMeasure) -> None:
        """Set the probability measure associated with this event.

        The new probability measure must be defined on a sub-sigma-algebra of the current sigma-algebra and the event must be measureable with respect to the new sigma-algebra.

        Parameters
        ----------
        prob_measure : ProbabilityMeasure
            The new probability measure.

        Raises
        ------
        TypeError
            If `prob_measure` is not an instance of `ProbabilityMeasure`.
        ValueError
            If the current instance of `Event` has a `prob_space` attribute equal to `None`, or if the current instance is not in the sigma-algebra of the new probability measure.
        """
        from ..measures.probability_measure import ProbabilityMeasure

        if not isinstance(prob_measure, ProbabilityMeasure):
            raise TypeError("prob_measure must be an instance of ProbabilityMeasure.")
        if self.prob_space is None:
            raise ValueError(
                "Cannot set a new probability measure for an event whose prob_space attribute is `None`."
            )
        if self not in prob_measure.sig_alg:
            raise ValueError(
                "The event must be in the sigma-algebra of the new probability measure."
            )

        self.prob_space.prob_measure = prob_measure
        self._indicator = None

    @property
    def indicator(self) -> RandomVariable | None:
        """Get the indicator random variable of the event.

        Returns
        -------
        indicator : RandomVariable | None
            The indicator random variable of the event.

        Examples
        --------
        Define a sample space and a sigma-algebra.

        >>> from sigalg.core import Event, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 1,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> A = F.get_event([0, 2])

        Print the indicator random variable of the event.

        >>> print(A.indicator)  # doctest: +NORMALIZE_WHITESPACE
        Random variable 'I_A':
                I_A
        sample
        0         1
        1         0
        2         1
        """
        from ..random_objects.random_variable import RandomVariable

        if self._indicator is None and self.data is not None:
            name = f"I_{self.name}"
            mapping = (
                pd.Series([1] * len(self), index=self.data, name=name)
                .reindex(index=self.sample_space.data)
                .fillna(value=0)
                .astype(int)
            )

            try:
                indicator = RandomVariable(
                    sample_space=self.sample_space,
                    sig_alg=self.sig_alg,
                    prob_measure=self.prob_measure,
                    mapping=mapping,
                    name=name,
                )
            except ValueError as e:
                raise ValueError(
                    "The provided indices do not form a measurable event."
                ) from e

            self._indicator = indicator

        return self._indicator

    # TODO: write unit tests
    @property
    def is_atom(self) -> bool | None:
        """Return whether this event is an atom in the sigma-algebra.

        Returns
        -------
        is_atom : bool | None
            Whether the current event is an atom or not.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
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
        >>> A = F.get_event([0, 1])
        >>> print(A.is_atom)
        True
        >>> B = F.get_event([0, 1, 2, 3], name="B")
        >>> print(B.is_atom)
        False
        """
        return self._is_atom

    # TODO: write unit tests
    @property
    def atom_id(self) -> Hashable | None:
        """Return the atom ID if this event is an atom, or `None` otherwise.

        Returns
        -------
        atom_id : Hashable | None
            The atom ID of the current event if it is an atom, and `None` otherwise.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
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
        >>> A = F.get_event([0, 1])
        >>> print(A.atom_id)
        1
        >>> B = F.get_event([0, 1, 2, 3], name="B")
        >>> print(B.atom_id)
        None
        """
        return self._atom_id

    # --------------------- data access methods --------------------- #

    def __getitem__(self, key: any) -> Event | Hashable:
        """Internal hook for indexing operations to create events.

        If `key` is an integer, an event is created from a single point retrieved by position given by `key; a slice creates an event with a slice of sample points, a tuple `(index, name)` creates an event with a custom name, and a `list` creates an event with multiple sample points.

        Parameters
        ----------
        key : any
            Indexing key for accessing sample points by position.

        Returns
        -------
        event : Event | Hashable
            An `Event` object containing the indexed sample points, or a single hashable if `key` is an `int`.

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
         sample
              2
              4

        Access elements of A by passing a list of positions.

        >>> C = A[[0, 2], "C"]
        >>> print(C)  # doctest: +NORMALIZE_WHITESPACE
        Event 'C':
         sample
              0
              4
        """  # noqa: D401
        if isinstance(key, tuple):
            if len(key) != 2:
                raise TypeError("Use `Event[idx]` or `Event[idx, name]`.")
            item_idx, name = key
            if not isinstance(name, Hashable):
                raise TypeError("Event name must be hashable.")
        else:
            item_idx, name = key, "A"

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
         sample
              1
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

        Examples
        --------
        >>> from sigalg.core import Event, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_event([0, 1])
        >>> B = F.get_event([1, 2], name="B")
        >>> A.intersection(B) # doctest: +NORMALIZE_WHITESPACE
        Event 'A intersect B':
         sample
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

        Examples
        --------
        >>> from sigalg.core import Event, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_event([0])
        >>> B = F.get_event([1], name="B")
        >>> A.union(B) # doctest: +NORMALIZE_WHITESPACE
        Event 'A union B':
         sample
              0
              1
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
         sample
              0
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
        if self.sig_alg <= other.sig_alg:
            super_sig_alg = other.sig_alg
        elif other.sig_alg <= self.sig_alg:
            super_sig_alg = self.sig_alg
        else:
            raise ValueError("Events must belong to the same sigma-algebra.")

        pts = set(self.data) & set(other.data)
        return super_sig_alg.get_event(
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
        if self.sig_alg <= other.sig_alg:
            super_sig_alg = other.sig_alg
        elif other.sig_alg <= self.sig_alg:
            super_sig_alg = self.sig_alg
        else:
            raise ValueError("Events must belong to the same sigma-algebra.")

        pts = set(self.data) | set(other.data)
        return super_sig_alg.get_event(
            list(pts), name=f"{self.name} union {other.name}"
        )

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
        if self.sig_alg <= other.sig_alg:
            super_sig_alg = other.sig_alg
        elif other.sig_alg <= self.sig_alg:
            super_sig_alg = self.sig_alg
        else:
            raise ValueError("Events must belong to the same sigma-algebra.")

        pts = set(self.data) - set(other.data)
        return super_sig_alg.get_event(
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
         sample
              0
              1
        """
        from ..base import SampleSpace

        return SampleSpace(indices=self.data.to_list(), name=self.name)
