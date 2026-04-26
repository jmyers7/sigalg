"""A class representing a sample space."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

from .index import Index

if TYPE_CHECKING:
    from ..probability_measures import ProbabilityMeasure
    from ..sigma_algebras import SigmaAlgebra
    from .event import Event
    from .event_space import EventSpace
    from .probability_space import ProbabilitySpace


class SampleSpace(Index):
    r"""A class representing a sample space.

    See the Notes section below for the mathematical details.

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
    >>> Omega_1 = SampleSpace(name="Omega_1").from_list(["red", "green", "blue"])
    >>> Omega_1 # doctest: +NORMALIZE_WHITESPACE
    Sample space 'Omega_1':
    ['red', 'green', 'blue']
    >>> # Construction with pd.Index
    >>> data = pd.Index(["a", "b", "c"], name="sample")
    >>> Omega_2 = SampleSpace(name="Omega_2").from_pandas(data=data)
    >>> Omega_2 # doctest: +NORMALIZE_WHITESPACE
    Sample space 'Omega_2':
    ['a', 'b', 'c']

    Notes
    -----
    In the abstract, a *sample space* is just a set $\Omega$. However, in the context of probability theory, sample spaces are often conceptualized as the set of all possible outcomes of a random experiment. Each element $\omega \in \Omega$ is called a *sample point* or *outcome*. The sample space serves as the foundational building block for defining events (subsets of $\Omega$) and probability measures (functions that assign probabilities to events).

    Sample spaces are not meant to contain data. Instead, data is meant to be encoded in random variables and vectors defined on the sample space, which are functions on sample spaces.

    See also the [notebook](https://johnmyers-phd.com/sigalg/dictionary/){target="_blank"} on the docs website.
    """

    def __init__(
        self,
        name: Hashable | None = "Omega",
        data_name: Hashable | None = "sample",
    ) -> None:
        """The only purpose of this __init__ is to call the superclass's __init__ with new default values for the parameters `name` and `data_name`."""  # noqa: D401
        super().__init__(name=name, data_name=data_name)

    # --------------------- conversion methods --------------------- #

    def make_probability_space(
        self,
        sig_alg: SigmaAlgebra | None = None,
        prob_measure: ProbabilityMeasure | None = None,
    ) -> ProbabilitySpace:
        """Convert this sample space to a probability space by adding a sigma-algebra and probability measure.

        Parameters
        ----------
        sig_alg : SigmaAlgebra | None, default=None
            Sigma-algebra to use. If `None`, a power set sigma-algebra will be created.
        prob_measure : ProbabilityMeasure | None, default=None
            Probability measure to use. If `None`, a uniform probability measure will be created.

        Raises
        ------
        TypeError
            If `sig_alg` is not a `SigmaAlgebra` or `None`, or if `prob_measure` is not a `ProbabilityMeasure` or `None`.

        Returns
        -------
        probability_space : ProbabilitySpace
            A `ProbabilitySpace` object with this sample space.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_list(["a", "b", "c"])
        >>> # Create with default uniform measure and power-set sigma-algebra
        >>> prob_space = Omega.make_probability_space()
        >>> print(prob_space) # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, power_set, P)
        =======================================
        <BLANKLINE>
        * Sample space 'Omega':
        ['a', 'b', 'c']
        <BLANKLINE>
        * Sigma algebra 'power_set':
                atom ID
        sample
        a             0
        b             1
        c             2
        <BLANKLINE>
        * Probability measure 'P':
                probability
        sample
        a          0.333333
        b          0.333333
        c          0.333333
        >>> # Create with custom probability measure and sigma-algebra
        >>> probs = {"a": 0.5, "b": 0.3, "c": 0.2}
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict(probs)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict({"a": 0, "b": 1, "c": 1})
        >>> prob_space = Omega.make_probability_space(sig_alg=F, prob_measure=P)
        >>> print(prob_space) # doctest: +NORMALIZE_WHITESPACE
        Probability space (Omega, F, P)
        ===============================
        <BLANKLINE>
        * Sample space 'Omega':
        ['a', 'b', 'c']
        <BLANKLINE>
        * Sigma algebra 'F':
                atom ID
        sample
        a             0
        b             1
        c             1
        <BLANKLINE>
        * Probability measure 'P':
                probability
        sample
        a               0.5
        b               0.3
        c               0.2
        """
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .probability_space import ProbabilitySpace

        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("`sig_alg` must be a `SigmaAlgebra` or `None`.")
        if prob_measure is not None and not isinstance(
            prob_measure, ProbabilityMeasure
        ):
            raise TypeError(
                "`prob_measure` must be a `ProbabilityMeasure` or `None`."
            )

        return ProbabilitySpace(
            sample_space=self,
            sig_alg=sig_alg,
            prob_measure=prob_measure,
        )

    def make_event_space(self, sig_alg: SigmaAlgebra | None = None) -> EventSpace:
        """Convert this sample space to an event space by adding a sigma-algebra.

        Parameters
        ----------
        sig_alg : SigmaAlgebra | None, default=None
            Sigma-algebra to use. If `None`, a power set sigma-algebra will be created.

        Raises
        ------
        TypeError
            If `sig_alg` is not a `SigmaAlgebra` or `None`.

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
        >>> print(event_space) # doctest: +NORMALIZE_WHITESPACE
        Event space (Omega, power_set)
        ==============================
        <BLANKLINE>
        * Sample space 'Omega':
        ['s0', 's1', 's2', 's3']
        <BLANKLINE>
        * Sigma algebra 'power_set':
                atom ID
        sample
        s0            0
        s1            1
        s2            2
        s3            3
        >>> # Create with custom sigma-algebra
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     sample_id_to_atom_id={"s0": 0, "s1": 0, "s2": 1, "s3": 1},
        ... )
        >>> event_space = Omega.make_event_space(sig_alg=F)
        >>> print(event_space) # doctest: +NORMALIZE_WHITESPACE
        Event space (Omega, F)
        ======================
        <BLANKLINE>
        * Sample space 'Omega':
        ['s0', 's1', 's2', 's3']
        <BLANKLINE>
        * Sigma algebra 'F':
                atom ID
        sample
        s0            0
        s1            0
        s2            1
        s3            1
        """
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .event_space import EventSpace

        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("`sig_alg` must be a `SigmaAlgebra` or `None`.")

        return EventSpace(sample_space=self, sig_alg=sig_alg)

    # --------------------- data access methods --------------------- #

    def get_event(self, event_indices: list[Hashable], name: Hashable = "A") -> Event:
        """Create an event from a list of sample points.

        Parameters
        ----------
        event_indices : list[Hashable]
            List of sample points to include in the event. Must be hashable items that exist in this sample space.
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
        >>> A = Omega.get_event(["omega0", "omega1"], name="A")
        >>> print(A) # doctest: +NORMALIZE_WHITESPACE
        Event 'A':
        ['omega0', 'omega1']
        """
        from .event import Event

        return Event(sample_space=self, name=name).from_list(indices=event_indices)

    def _getitem_hook(
        self,
        pos: (
            list[int]
            | slice
            | tuple[list[int], Hashable]
            | tuple[slice, Hashable]
            | int
        ),
    ) -> Event | Hashable:
        """Internal hook for indexing operations to create events.

        This method is called by `__getitem__` from the parent `Index` class. In `SampleSpace`, the purpose of this method is to ensure that `__getitem__` returns an instance of `Event`. Items are retrieved by position.

        Parameters
        ----------
        pos : list[int] | slice | tuple[list[int], Hashable] | tuple[slice, Hashable] | int
            Indexing key for accessing sample points. A list of integers returns the event with the sample points at those positions, a slice returns the event with the sample points in that slice, and an integer returns the single sample point at that position. Optionally, a custom name can be provided by using a tuple of the form `(index, name)`, where `index` is either a list of integers or a slice, and `name` is a hashable identifier for the event.

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
        >>> print(E) # doctest: +NORMALIZE_WHITESPACE
        omega0
        >>> # Access via slice
        >>> D = Omega[1:3, "D"]
        >>> print(D) # doctest: +NORMALIZE_WHITESPACE
        Event 'D':
        ['omega1', 'omega2']
        >>> # Access via list of positions
        >>> C = Omega[[0, 2], "C"]
        >>> print(C) # doctest: +NORMALIZE_WHITESPACE
        Event 'C':
        ['omega0', 'omega2']
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
    """Mixin class providing sample space methods to other classes."""

    def get_event(self, event_indices: list[Hashable], name: Hashable = "A") -> Event:
        """Create an event from a list of sample points.

        Calls `SampleSpace.get_event`. See the docstring of `SampleSpace.get_event` for details.

        Parameters
        ----------
        event_indices : list[Hashable]
            List of sample points to include in the event.
        name : Hashable, default="A"
            Name identifier for the event.

        Returns
        -------
        event : Event
            An `Event` object containing the specified sample points.
        """
        return self.sample_space.get_event(event_indices, name)
