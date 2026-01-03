"""Sample space module.

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
>>> Omega = SampleSpace(name="CoinFlip").from_list(["H", "T"])
>>> Omega # doctest: +NORMALIZE_WHITESPACE
Sample space 'CoinFlip':
['H', 'T']
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
    """A class representing a sample space.

    An instance of `SampleSpace` is not intended to contain data; rather, it is used to model only the labels or indices of possible outcomes of a random experiment. Data is encoded in instances of `RandomVariable` and `RandomVector`.

    Sample spaces support operations like creating events, converting to probability spaces, and iterating over outcomes.

    Parameters
    ----------
    name : Hashable | None, default="Omega"
        Name identifier for the sample space.
    data_name : Hashable | None, default="sample"
        Name for the internal `pd.Index`.

    Examples
    --------
    >>> from sigalg.core import SampleSpace
    >>> import pandas as pd
    >>> # Construction with list
    >>> Omega_1 = SampleSpace(name="Omega_1").from_list(["omega_0", "omega_1", "omega_2"])
    >>> Omega_1 # doctest: +NORMALIZE_WHITESPACE
    Sample space 'Omega_1':
    ['omega_0', 'omega_1', 'omega_2']
    >>> # Construction with pd.Index
    >>> idx = pd.Index(["a", "b", "c"], name="sample")
    >>> Omega_2 = SampleSpace(name="Omega_2").from_pandas(data=idx)
    >>> Omega_2 # doctest: +NORMALIZE_WHITESPACE
    Sample space 'Omega_2':
    ['a', 'b', 'c']
    """

    def __init__(
        self,
        name: Hashable | None = "Omega",
        data_name: Hashable | None = "sample",
    ) -> None:
        super().__init__(name=name, data_name=data_name)

    # --------------------- factory methods --------------------- #

    @classmethod
    def generate_sequence(
        cls,
        size: int,
        initial_index: int = 0,
        prefix: Hashable | None = "omega",
        name: Hashable | None = "Omega",
        data_name: Hashable | None = "sample",
    ) -> SampleSpace:
        """Generate a default `SampleSpace` with sequential indices.

        Creates a `SampleSpace` with sequentially numbered sample points, optionally
        prefixed by a given string.

        Parameters
        ----------
        size : int
            Number of sample points to generate.
        initial_index : int, default=0
            Starting integer for generating sample point names.
        prefix : Hashable | None, default="omega"
            Prefix for naming sample points. If the prefix is a non-string hashable or
            `None`, numerical indices are used instead.
        name : Hashable | None, default="Omega"
            Name identifier for the sample space.
        data_name : Hashable | None, default="sample"
            Name for the internal `pd.Index`.

        Examples
        --------
        >>> from sigalg.core import SampleSpace
        >>> # Generate sample space with string prefix
        >>> Omega1 = SampleSpace.generate_sequence(size=3, prefix="s")
        >>> Omega1 # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega':
        ['s_0', 's_1', 's_2']
        >>> # Generate sample space with numerical indices
        >>> Omega2 = SampleSpace.generate_sequence(size=2, initial_index=5, prefix=None, name="Numbers")
        >>> Omega2 # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Numbers':
        [5, 6]
        """
        return cls._generate_sequence(
            initial_index=initial_index,
            size=size,
            prefix=prefix,
            name=name,
            data_name=data_name,
        )

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
        >>> Omega = SampleSpace().from_list(["s0", "s1", "s2"])
        >>> # Create with default uniform measure
        >>> prob_space = Omega.make_probability_space()
        >>> # Create with custom probability measure
        >>> probs = {"s0": 0.5, "s1": 0.3, "s2": 0.2}
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict(probs)
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
            Sigma-algebra to use. If `None`, a power set sigma-algebra will be created.

        Returns
        -------
        event_space : EventSpace
            An `EventSpace` object with this sample space.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_list(["s0", "s1", "s2", "s3"])
        >>> # Create with default power set sigma-algebra
        >>> event_space = Omega.make_event_space()
        >>> # Create with custom sigma-algebra
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     sample_id_to_atom_id={"s0": 0, "s1": 0, "s2": 1, "s3": 1},
        ... )
        >>> event_space = Omega.make_event_space(sigma_algebra=F)
        """
        from .event_space import EventSpace

        return EventSpace(sample_space=self, sigma_algebra=sigma_algebra)

    # --------------------- data access methods --------------------- #

    def get_event(self, event_indices: list[Hashable], name: Hashable = "A") -> Event:
        """Create an event from a list of sample point indices.

        Constructs an `Event` object representing a subset of this sample space.
        All provided indices must exist in the sample space.

        Parameters
        ----------
        event_indices : list of Hashable
            List of sample point indices to include in the event.
            Must be hashable items that exist in this sample space.
        name : Hashable, default="A"
            Name identifier for the event.

        Returns
        -------
        event : Event
            An `Event` object containing the specified sample points.

        Examples
        --------
        >>> from sigalg.core import SampleSpace
        >>> Omega = SampleSpace().from_list(["omega0", "omega1", "omega2", "omega3"])
        >>> # Create event with specific sample points
        >>> A = Omega.get_event(["omega0", "omega1"], name="A")
        >>> # Create event with empty list
        >>> empty_event = Omega.get_event([])
        """
        from .event import Event

        return Event(sample_space=self, name=name).from_list(indices=event_indices)

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
        >>> Omega = SampleSpace().from_list(["omega0", "omega1", "omega2", "omega3"])
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
            return Event(name=name, sample_space=self).from_list(item.to_list())

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
        >>> Omega = SampleSpace(name="CoinFlip").from_list(["H", "T"])
        >>> repr(Omega)
        "Sample space 'CoinFlip':\n['H', 'T']"
        """
        if self._data is None and self._indices is None:
            return "Sample with no data"
        if self.name is None:
            return f"Sample space:\n{self.data.to_list()}"
        else:
            return f"Sample space '{self.name}':\n{self.data.to_list()}"

    # --------------------- equality --------------------- #

    def __eq__(self, other: SampleSpace) -> bool:
        """Check equality with another sample space.

        Two sample spaces are equal if they have the same sample points in the
        same order. They can have different names and different data names and still
        be considered equal.

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
    >>> Omega = SampleSpace().from_list(["a", "b", "c"])
    >>> obj = MyClass(Omega)
    >>> E = obj.get_event(["a", "b"], name="E")
    """

    def get_event(self, event_indices: list[Hashable], name: Hashable = "A") -> Event:
        """Create an event from a list of sample point indices.

        Delegates to the `sample_space.get_event` method.

        Parameters
        ----------
        event_indices : list of Hashable
            List of sample point indices to include in the event.
        name : Hashable, default="A"
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
