"""Sample spaces for probability theory.

This module provides the `SampleSpace` class, which models the indices or labels of all possible outcomes in a random experiment.

Classes
-------
SampleSpace
    Represents a sample space as a collection of outcomes.
SampleSpaceMethods
    Mixin providing sample space methods to other classes.

Examples
--------
>>> from sigalg.core import SampleSpace
>>> Omega = SampleSpace(indices=["H", "T"], name="CoinFlip")
>>> Omega
Sample space 'CoinFlip':
['H', 'T']
>>> event = Omega.get_event(["H"], name="Heads")
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

from .index import Index

if TYPE_CHECKING:
    from ..probability_measures import ProbabilityMeasure
    from ..sigma_algebras import SigmaAlgebra
    from . import ProbabilitySpace
    from .event import Event
    from .event_space import EventSpace


class SampleSpace(Index):
    """A sample space modeling all possible outcomes of a random experiment.

    An instance of `SampleSpace` is not intended to contain data; rather, it is used to model only the labels or indices of possible outcomes of a random experiment. Data is encoded in instances of `RandomVariable` and `RandomVector`.

    Sample spaces support operations like creating events, converting to probability spaces, and iterating over outcomes.

    Parameters
    ----------
    indices : list[Hashable]
        Ordered collection of unique hashable items. (Any iterable of hashable items is acceptable and will be coerced into a list internally.)
    name : Hashable, optional
        Name identifier for the sample space. Defaults to the class-level `Omega`.
    data_name : Hashable, optional
        Name for the internal `pd.Index`. Defaults to the class-level `sample`.

    Raises
    ------
    ValueError
        If parameters are invalid.

    Examples
    --------
    >>> from sigalg.core import SampleSpace
    >>> import pandas as pd
    >>> # Construction with list
    >>> Omega1 = SampleSpace(indices=["omega0", "omega1", "omega2"], name="Omega1")
    >>> # Construction with pd.Index
    >>> idx = pd.Index(["a", "b", "c"], name="sample")
    >>> Omega2 = SampleSpace.from_pandas(data=idx, name="Omega2")
    >>> # Get an event from the sample space
    >>> A = Omega1.get_event(["omega0", "omega1"], name="A")
    """

    DEFAULT_NAME = "Omega"
    DEFAULT_DATA_NAME = "sample"
    DEFAULT_PREFIX = "omega"

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
        >>> from sigalg.core import SampleSpace, ProbabilityMeasure
        >>> Omega = SampleSpace(indices=["s0", "s1", "s2"])
        >>> # Create with default uniform measure
        >>> prob_space = Omega.make_probability_space()
        >>> # Create with custom probability measure
        >>> probs = {"s0": 0.5, "s1": 0.3, "s2": 0.2}
        >>> P = ProbabilityMeasure(probabilities=probs, sample_space=Omega, name="P")
        >>> prob_space = Omega.make_probability_space(probability_measure=P)
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
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace(indices=["s0", "s1", "s2", "s3"])
        >>> # Create with default power set sigma-algebra
        >>> event_space = Omega.make_event_space()
        >>> # Create with custom sigma-algebra
        >>> F = SigmaAlgebra(
        ...     sample_id_to_atom_id={"s0": 0, "s1": 0, "s2": 1, "s3": 1},
        ...     sample_space=Omega,
        ... )
        >>> event_space = Omega.make_event_space(sigma_algebra=F)
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
        >>> from sigalg.core import SampleSpace
        >>> Omega = SampleSpace(indices=["omega0", "omega1", "omega2", "omega3"])
        >>> # Create event with specific sample points
        >>> A = Omega.get_event(["omega0", "omega1"], name="A")
        >>> # Create event with empty list
        >>> empty_event = Omega.get_event([])
        """
        from .event import Event

        if not isinstance(event_indices, list):
            raise TypeError("event_indices must be a list of Hashable items.")
        for idx in event_indices:
            if idx not in self.data:
                raise ValueError(f"Index '{idx}' not found in sample space.")
        return Event(sample_space=self, event_indices=event_indices, name=name)

    def _getitem_hook(self, pos: int | list[int] | slice) -> Event | Hashable:
        """Internal hook for indexing operations to create events.

        This method is called by `__getitem__` from the parent `Index` class. In `SampleSpace`, the purpose of this method is to ensure that `__getitem__` returns an instance of `Event`. Items are retrieved by position.

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
        >>> Omega = SampleSpace(indices=["omega0", "omega1", "omega2", "omega3"])
        >>> # Access via integer index
        >>> E = Omega[0, "E"]
        >>> # Access via slice
        >>> D = Omega[1:3, "D"]
        >>> # Access via list of positions
        >>> C = Omega[[0, 2], "C"]
        """  # noqa: D401
        from .event import Event

        if isinstance(pos, tuple):
            if len(pos) != 2:
                raise TypeError("Use `Omega[idx]` or `Omega[idx, name]`.")
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
            return Event(sample_space=self, event_indices=item.to_list(), name=name)

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        r"""Return a string representation of the sample space.

        Returns
        -------
        repr_str : str
            A formatted string showing the sample space name and its sample points.

        Examples
        --------
        >>> from sigalg.core import SampleSpace
        >>> Omega = SampleSpace(indices=["H", "T"], name="CoinFlip")
        >>> repr(Omega)
        "Sample space 'CoinFlip':\n['H', 'T']"
        """
        return f"Sample space '{self.name}':\n{self.data.to_list()}"

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
        """
        return isinstance(other, SampleSpace) and super().__eq__(other)


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
    >>> from sigalg.core import SampleSpace
    >>> Omega = SampleSpace(indices=["a", "b", "c"])
    >>> obj = MyClass(Omega)
    >>> E = obj.get_event(["a", "b"], name="E")
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
