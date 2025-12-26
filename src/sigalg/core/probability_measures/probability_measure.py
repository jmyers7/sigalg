"""Probability measure module.

This module defines the `ProbabilityMeasure` class, which represents a probability measure on a sample space. It includes methods for computing probabilities of events, conditional probabilities, and checking for independence between events. The module also provides factory methods for creating probability measures from `pd.Series` and for creating uniform probability measures.

Classes
-------
ProbabilityMeasure
    Represents a probability measure on a sample space.
ProbabilityMeasureMethods
    Mixin class providing probability measure methods to other classes.

Examples
--------
>>> from sigalg.core import ProbabilityMeasure, SampleSpace
>>> sample_space = SampleSpace(indices=["omega0", "omega1", "omega2"])
>>> probabilities = {"omega0": 0.2, "omega1": 0.5, "omega2": 0.3}
>>> prob_measure = ProbabilityMeasure(probabilities=probabilities, sample_space=sample_space, name="P")
>>> float(prob_measure("omega1"))
0.5
>>> A = sample_space.get_event(["omega0", "omega1"], name="A")
>>> float(prob_measure(A))
0.7
>>> uniform_measure = ProbabilityMeasure.uniform(sample_space, name="Q")
>>> float(uniform_measure(["omega0", "omega1"]))
0.6666666666666666
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd

from ...validation.sample_space_mapping_in import SampleSpaceMappingIn

if TYPE_CHECKING:
    from ..base.event import Event
    from ..base.sample_space import SampleSpace


class ProbabilityMeasure:
    """Class representing a probability measure on a sample space.

    A probability measure is a mapping from sample space indices to probabilities with the following properties: All probabilities are non-negative real numbers and they sum to 1. The class provides methods to compute probabilities of events, conditional probabilities, and to check for independence between events.

    Parameters
    ----------
    probabilities : Mapping[Hashable, Real]
        A mapping from sample space indices to their associated probabilities.
    sample_space : SampleSpace
        The sample space on which the probability measure is defined.
    name : Hashable, default="P"
        A name for the probability measure.

    Raises
    ------
    TypeError
        If `probabilities` is not a mapping from Hashable to Real, or if `sample_space` is not a SampleSpace instance, or if `name` is not Hashable.
    ValueError
        If the probabilities do not sum to 1, or if any probability is negative, or if the keys of `probabilities` do not match the indices of `sample_space`.

    Examples
    --------
    >>> from sigalg.core import ProbabilityMeasure, SampleSpace
    >>> sample_space = SampleSpace(indices=["omega0", "omega1", "omega2"])
    >>> probabilities = {"omega0": 0.2, "omega1": 0.5, "omega2": 0.3}
    >>> prob_measure = ProbabilityMeasure(probabilities=probabilities, sample_space=sample_space, name="P")
    >>> float(prob_measure("omega1"))
    0.5
    >>> A = sample_space.get_event(["omega0", "omega1"], name="A")
    >>> float(prob_measure(A))
    0.7
    """

    # --------------------- constructor --------------------- #

    def __init__(
        self,
        probabilities: Mapping[Hashable, Real],
        sample_space: SampleSpace,
        name: Hashable = "P",
    ) -> None:

        v = SampleSpaceMappingIn(
            mapping=probabilities,
            sample_space=sample_space,
            name=name,
            kind="probabilities",
        )

        self.probabilities = v.mapping
        self.sample_space = v.sample_space
        self._name = v.name

        # caches for properties
        self._data: pd.Series | None = None

    # --------------------- properties --------------------- #

    @property
    def data(self) -> pd.Series:
        """Get the probability values as a `pd.Series`.

        Returns
        -------
        data: pd.Series
            A `pd.Series` with sample space indices as the index and their associated probabilities as values.
        """
        if self._data is None:
            self._data = pd.Series(
                data=[self.probabilities[idx] for idx in self.sample_space.data],
                index=self.sample_space.data,
                name=self.name,
            )
        return self._data

    @data.setter
    def data(self, data: pd.Series) -> None:
        """Set the probability values from a `pd.Series`.

        The `data` property is not meant to be set directly by the user. This setter is provided so that the `from_pandas` factory method can set the property.

        Parameters
        ----------
        data: pd.Series
            A `pd.Series` with sample space indices as the index and their associated probabilities as values.

        Raises
        ------
        TypeError
            If `data` is not a `pd.Series`, or if `data.to_dict()` is not a mapping from Hashable to Real.
        ValueError
            If the probabilities do not sum to 1, or if any probability is negative, or if the keys of `data.to_dict()` do not match the indices of `sample_space`.
        """
        if not isinstance(data, pd.Series):
            raise TypeError("data must be a pandas Series instance.")
        v = SampleSpaceMappingIn(
            mapping=data.to_dict(),
            sample_space=self.sample_space,
            name=self.name,
            kind="probabilities",
        )
        self.probabilities = v.mapping
        self._data = data

    @property
    def name(self) -> Hashable:
        """Get the name of the probability measure.

        Returns
        -------
        name: Hashable
            The name of the probability measure.
        """
        return self._name

    @name.setter
    def name(self, name: Hashable) -> None:
        """Set the name of the probability measure.

        Parameters
        ----------
        name: Hashable
            The new name of the probability measure.

        Raises
        ------
        TypeError
            If `name` is not Hashable.
        """
        if not isinstance(name, Hashable):
            raise TypeError("name must be hashable.")
        self._name = name
        self.values.name = name

    # --------------------- methods --------------------- #

    def P(self, key: Hashable | list[Hashable] | Event) -> Real:
        """Get the probability of a sample point or event.

        This method is an alias for the `__call__` method.
        """
        return self(key)

    def conditional_probability(self, event_A: Event, event_B: Event) -> Real:
        """Compute the conditional probability P(A|B).

        Parameters
        ----------
        event_A : Event
            The event A.
        event_B : Event
            The event B.

        Raises
        ------
        ValueError
            If `event_A` or `event_B` are from a different sample space than this probability measure's sample space, or if P(B) = 0.
        """
        if event_A.sample_space != self.sample_space:
            raise ValueError(
                "event_A must be from this probability space's sample space."
            )
        if event_B.sample_space != self.sample_space:
            raise ValueError(
                "event_B must be from this probability space's sample space."
            )
        prob_B = self.P(event_B)
        if prob_B < 1e-10:
            raise ValueError("Cannot compute conditional probability: P(B) = 0")
        intersection_indices = [idx for idx in event_A.data if idx in event_B.data]
        if not intersection_indices:
            return 0.0
        intersection_event = self.sample_space.get_event(intersection_indices)
        prob_intersection = self.P(intersection_event)
        return prob_intersection / prob_B

    def are_independent(
        self, event_A: Event, event_B: Event, tolerance: Real = 1e-10
    ) -> bool:
        """Check if two events are independent.

        Parameters
        ----------
        event_A : Event
            The event A.
        event_B : Event
            The event B.
        tolerance : Real, default=1e-10
            The numerical tolerance for checking independence.

        Raises
        ------
        ValueError
            If `event_A` or `event_B` are from a different sample space than this probability measure's sample space.

        Returns
        -------
        is_independent : bool
            `True` if the events are independent, `False` otherwise.
        """
        if event_A.sample_space != self.sample_space:
            raise ValueError(
                "event_A must be from this probability space's sample space."
            )
        if event_B.sample_space != self.sample_space:
            raise ValueError(
                "event_B must be from this probability space's sample space."
            )
        prob_A = self.P(event_A)
        prob_B = self.P(event_B)
        prob_intersection = self.P(event_A & event_B)
        return bool(abs(prob_intersection - prob_A * prob_B) < tolerance)

    # --------------------- factory methods --------------------- #

    @classmethod
    def from_pandas(
        cls,
        data: pd.Series,
        name: Hashable = "P",
    ) -> ProbabilityMeasure:
        """Create a `ProbabilityMeasure` from a `pd.Series`.

        Parameters
        ----------
        data : pd.Series
            A `pd.Series` with sample space indices as the index and their associated probabilities as values.
        name : Hashable, default="P"
            A name for the probability measure.

        Raises
        ------
        TypeError
            If `data` is not a `pd.Series`.

        Returns
        -------
        prob_measure: ProbabilityMeasure
            A ProbabilityMeasure instance created from the provided pandas Series.

        Examples
        --------
        >>> import pandas as pd
        >>> from sigalg.core import ProbabilityMeasure
        >>> # Create a probability measure from a series with custom index
        >>> data = pd.Series(data=[0.2, 0.5, 0.3], index=["omega0", "omega1", "omega2"])
        >>> prob_measure = ProbabilityMeasure.from_pandas(data, name="P")
        >>> prob_measure("omega1")
        0.5
        >>> # Check the automatically generated sample space
        >>> prob_measure.sample_space
        Sample space 'Omega':
        ['omega0', 'omega1', 'omega2']
        >>> # Change the name of the sample space
        >>> prob_measure.sample_space.name = "S"
        >>> prob_measure.sample_space
        Sample space 'S':
        ['omega0', 'omega1', 'omega2']
        >>> # Create a probability measure from a series with default index
        >>> new_data = pd.Series(data=[0.6, 0.4])
        >>> new_prob_measure = ProbabilityMeasure.from_pandas(new_data)
        >>> new_prob_measure(0)
        0.6
        >>> new_prob_measure.sample_space
        Sample space 'Omega':
        [0, 1]
        """
        from ..base.sample_space import SampleSpace

        if not isinstance(data, pd.Series):
            raise TypeError("data must be a pandas Series.")
        sample_space = SampleSpace.from_pandas(data.index, name="Omega")
        probabilities = data.to_dict()
        prob_measure = cls(
            probabilities=probabilities,
            sample_space=sample_space,
            name=name,
        )
        data.name = None  # erase name
        prob_measure.data = data
        return prob_measure

    @classmethod
    def uniform(
        cls, sample_space: SampleSpace, name: Hashable = "P"
    ) -> ProbabilityMeasure:
        """Create a uniform `ProbabilityMeasure` on the given sample space.

        Parameters
        ----------
        sample_space : SampleSpace
            The sample space on which to define the uniform probability measure.
        name : Hashable, default="P"
            A name for the probability measure.

        Raises
        ------
        ValueError
            If the sample space is empty.

        Returns
        -------
        prob_measure: ProbabilityMeasure
            A uniform ProbabilityMeasure instance on the provided sample space.
        """
        n = len(sample_space)
        if n == 0:
            raise ValueError(
                "Cannot create uniform distribution on empty sample space."
            )
        probabilities = dict.fromkeys(sample_space.data, 1.0 / n)
        return cls(probabilities=probabilities, sample_space=sample_space, name=name)

    # --------------------- access methods --------------------- #

    def __call__(self, key: Hashable | list[Hashable] | Event) -> Real:
        """Get the probability of a sample point or event.

        Parameters
        ----------
        key : Hashable | list[Hashable] | Event
            A sample space index, a list of sample space indices, or an Event.

        Raises
        ------
        TypeError
            If `key` is not a Hashable, list of Hashables, or Event.
        ValueError
            If `key` is an Event from a different sample space.
        KeyError
            If any index in `key` is not found in the sample space.

        Returns
        -------
        probability : Real
            The probability associated with the given sample point(s) or event.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, SampleSpace
        >>> sample_space = SampleSpace(indices=["omega0", "omega1", "omega2"])
        >>> probabilities = {"omega0": 0.2, "omega1": 0.5, "omega2": 0.3}
        >>> prob_measure = ProbabilityMeasure(probabilities=probabilities, sample_space=sample_space)
        >>> # Probability of a single sample point
        >>> float(prob_measure("omega1"))
        0.5
        >>> # Probability of multiple sample points
        >>> float(prob_measure(["omega0", "omega2"]))
        0.5
        >>> # Probability of an event
        >>> A = sample_space.get_event(["omega0", "omega1"], name="A")
        >>> float(prob_measure(A))
        0.7
        """
        from ..base import Event

        if not isinstance(key, (Hashable, list, Event)):
            raise TypeError("Key must be a Hashable, list of Hashables, or Event.")

        if isinstance(key, Event):
            if key.sample_space != self.sample_space:
                raise ValueError("Event must be from the same sample space.")
            return self.data.loc[list(key)].sum()
        elif isinstance(key, list):
            for idx in key:
                if idx not in self.probabilities:
                    raise KeyError(f"Index '{idx}' not found in sample space.")
            return sum(self.probabilities[idx] for idx in key)
        else:
            if key not in self.probabilities:
                raise KeyError(f"Index '{key}' not found in sample space.")
            return self.probabilities[key]

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Get the string representation of the probability measure.

        Returns
        -------
        repr_str : str
            A string representation of the probability measure.
        """
        return f"Probability measure '{self.name}':\n{self.data.to_frame()}"

    # --------------------- equality --------------------- #

    def __eq__(self, other: ProbabilityMeasure) -> bool:
        """Check equality with another probability measure.

        Two probability measures are considered equal if they have the same sample space and identical probability values for each index. They may have different names and still be considered equal.

        Parameters
        ----------
        other : ProbabilityMeasure
            The other probability measure to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the two probability measures are equal, `False` otherwise.
        """
        if not isinstance(other, ProbabilityMeasure):
            return False
        if self.sample_space != other.sample_space:
            return False
        return self.data.equals(other.data)


class ProbabilityMeasureMethods:
    """Mixin class providing probability measure methods to other classes.

    This mixin provides convenience methods for classes that have a `probability_measure` attribute, allowing them to delegate probability measure operations to that attribute.

    The class assumes the implementing class has a `probability_measure` attribute that
    is a `ProbabilityMeasure` instance.

    Examples
    --------
    >>> class MyClass(ProbabilityMeasureMethods):
    ...     def __init__(self, probability_measure):
    ...         self.probability_measure = probability_measure
    >>> from sigalg.core import SampleSpace, ProbabilityMeasure
    >>> Omega = SampleSpace(indices=["a", "b", "c"])
    >>> probabilities = {"a": 0.2, "b": 0.5, "c": 0.3}
    >>> prob_measure = ProbabilityMeasure(probabilities=probabilities, sample_space=Omega)
    >>> obj = MyClass(prob_measure)
    >>> float(obj.P(["a", "b"]))
    0.7
    """

    def P(self, key: Hashable | list[Hashable] | Event) -> Real:
        """Get the probability of a sample point or event.

        Delegates to the `P` method of the `probability_measure` attribute.

        Parameters
        ----------
        key : Hashable | list[Hashable] | Event
            A sample space index, a list of sample space indices, or an Event.

        Returns
        -------
        probability : Real
            The probability associated with the given sample point(s) or event.
        """
        return self.probability_measure.P(key)

    def conditional_probability(self, event_A: Event, event_B: Event) -> Real:
        """Compute the conditional probability P(A|B).

        Delegates to the `conditional_probability` method of the `probability_measure` attribute.

        Parameters
        ----------
        event_A : Event
            The event A.
        event_B : Event
            The event B.
        """
        return self.probability_measure.conditional_probability(event_A, event_B)

    def are_independent(
        self, event_A: Event, event_B: Event, tolerance: Real = 1e-10
    ) -> bool:
        """Check if two events are independent.

        Delegates to the `are_independent` method of the `probability_measure` attribute.

        Parameters
        ----------
        event_A : Event
            The event A.
        event_B : Event
            The event B.
        tolerance : Real, default=1e-10
            The numerical tolerance for checking independence.
        """
        return self.probability_measure.are_independent(event_A, event_B, tolerance)
