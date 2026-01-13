"""Probability measure module.

This module defines the `ProbabilityMeasure` class, which represents a probability measure on a sample space. It includes methods for computing probabilities of events, conditional probabilities, and checking for independence between events.

Classes
-------
ProbabilityMeasure
    Represents a probability measure on a sample space.
ProbabilityMeasureMethods
    Mixin class providing probability measure methods to other classes.

Examples
--------
>>> from sigalg.core import ProbabilityMeasure, SampleSpace
>>> sample_space = SampleSpace.generate_sequence(size=3)
>>> probabilities = {"omega_0": 0.2, "omega_1": 0.5, "omega_2": 0.3}
>>> P = ProbabilityMeasure(sample_space=sample_space).from_dict(probabilities)
>>> float(P("omega_1"))
0.5
>>> A = sample_space.get_event(["omega_0", "omega_1"], name="A")
>>> float(P(A))
0.7
>>> uniform_measure = ProbabilityMeasure.uniform(sample_space, name="Q")
>>> float(uniform_measure(["omega_0", "omega_1"]))
0.6666666666666666
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from numbers import Real
from typing import TYPE_CHECKING

import pandas as pd

from ...validation.sample_space_mapping_in import SampleSpaceMappingIn

if TYPE_CHECKING:
    from ..base.event import Event
    from ..base.sample_space import SampleSpace
    from ..featurized_spaces.feature_vector import FeatureVector
    from ..random_objects.random_vector import RandomVector
    from ..sigma_algebras.sigma_algebra import SigmaAlgebra


class ProbabilityMeasure:
    """A class representing a probability measure on a sample space.

    A probability measure is a mapping from sample space indices to probabilities with the following properties: All probabilities are non-negative real numbers and they sum to 1. The class provides methods to compute probabilities of events, conditional probabilities, and to check for independence between events.

    Parameters
    ----------
    sample_space : SampleSpace
        The sample space on which the probability measure is defined.
    name : Hashable, default="P"
        A name for the probability measure.

    Raises
    ------
    TypeError
        If `sample_space` is not a `SampleSpace` instance (if given), or if `name` is not hashable (if given).

    Examples
    --------
    >>> from sigalg.core import ProbabilityMeasure, SampleSpace
    >>> sample_space = SampleSpace.generate_sequence(size=3)
    >>> probabilities = {"omega_0": 0.2, "omega_1": 0.5, "omega_2": 0.3}
    >>> P = ProbabilityMeasure(sample_space=sample_space).from_dict(probabilities)
    >>> float(P("omega_1"))
    0.5
    >>> A = sample_space.get_event(["omega_0", "omega_1"], name="A")
    >>> float(P(A))
    0.7
    """

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        sample_space: SampleSpace | None = None,
        name: Hashable | None = "P",
    ) -> None:
        from ..base.sample_space import SampleSpace

        if sample_space is not None and not isinstance(sample_space, SampleSpace):
            raise TypeError("If given, sample_space must be a SampleSpace instance.")
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("If given, name must be hashable.")
        self.sample_space = sample_space
        self._name = name

        # caches for properties
        self._data: pd.Series | None = None
        self._probabilities: Mapping[Hashable, Real] | None = None

    def from_dict(self, probabilities: Mapping[Hashable, Real]) -> ProbabilityMeasure:
        """Create a `ProbabilityMeasure` from a dictionary.

        If a `sample_space` was not provided during initialization, it will be created from the keys of the provided dictionary. If it was provided, the keys of the dictionary must match the sample space.

        Parameters
        ----------
        probabilities : Mapping[Hashable, Real]
            A mapping from sample space indices to their probabilities.
        """
        from ..base.sample_space import SampleSpace

        v = SampleSpaceMappingIn(
            mapping=probabilities, sample_space=self.sample_space, kind="probabilities"
        )

        if self.sample_space is None:
            self.sample_space = SampleSpace().from_list(list(v.mapping.keys()))

        self._probabilities = v.mapping
        return self

    def from_pandas(self, data: pd.Series) -> ProbabilityMeasure:
        """Create a `ProbabilityMeasure` from a `pd.Series`.

        If a `sample_space` was not provided during initialization, it will be created from the index of the provided `pd.Series`. If it was provided, the index of the `pd.Series` must match the sample space.

        Parameters
        ----------
        data: pd.Series
            A `pd.Series` with sample space indices as the index and their associated probabilities as values

        Raises
        ------
        TypeError
            If `data` is not a `pd.Series`.
        """
        from ..base.sample_space import SampleSpace

        if not isinstance(data, pd.Series):
            raise TypeError("data must be a pandas Series.")
        v = SampleSpaceMappingIn(
            mapping=data.to_dict(), sample_space=self.sample_space, kind="probabilities"
        )

        if self.sample_space is None:
            self.sample_space = SampleSpace().from_pandas(data.index)

        self._data = pd.Series(v.mapping, name="probability")
        self._data.index.name = self.sample_space.data.name
        return self

    # --------------------- properties --------------------- #

    @property
    def probabilities(self) -> Mapping[Hashable, Real]:
        """Get the mapping from sample IDs to their probabilities.

        Returns
        -------
        probabilities : Mapping[Hashable, Real]
            A mapping from sample IDs to their probabilities.
        """
        if self._probabilities is None:
            self._probabilities = self.data.to_dict()
        return self._probabilities

    @property
    def data(self) -> pd.Series:
        """Get the probability values as a `pd.Series`.

        Returns
        -------
        data: pd.Series
            A `pd.Series` with sample space indices as the index and their associated probabilities as values.
        """
        if self._data is None:
            self._data = pd.Series(data=self._probabilities, name="probability")
            self._data.index.name = self.sample_space.data.name
        return self._data

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

    def with_name(self, name: Hashable) -> ProbabilityMeasure:
        """Set the name of the probability measure and return self for chaining.

        Parameters
        ----------
        name : Hashable
            The new name for the random vector.

        Returns
        -------
        self : ProbabilityMeasure
            The current instance with the updated name.
        """
        self.name = name
        return self

    # --------------------- methods --------------------- #

    def P(self, key: Hashable | list[Hashable] | Event) -> Real:
        """Get the probability of a sample point or event.

        This method is an alias for the `__call__` method.
        """
        return self(key)

    def integrate(self, rv: RandomVector) -> pd.Series | Real:
        """Compute the Lebesgue integral of a random variable/vector with respect to the probability measure.

        Parameters
        ----------
        rv : RandomVector
            The integrand.

        Raises
        ------
        TypeError
            If `rv` is not a `RandomVector` whose domain is the sample space of the probability measure.

        Returns
        -------
        integral : pd.Series | Real
            Either a real number representing the integral if `rv` is an instance of `RandomVariable`, or a `pd.Series` whose values are the integrals of the components of `rv` is a `RandomVector` of dimension greater than 1.

        Examples
        --------
        >>> from sigalg.core import ProbabilityMeasure, RandomVariable, RandomVector, SampleSpace
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> P = ProbabilityMeasure(sample_space=Omega).from_dict({0: 0.2, 1: 0.5, 2: 0.3})
        >>> # Integral of random variable (i.e., 1-dimensional random vector)
        >>> X = RandomVariable(domain=Omega, name="X").from_dict({0: 1, 1: 2, 2: 3})
        >>> float(P.integrate(X))
        2.0999999999999996
        >>> # Integral of random vector of dimension > 1
        >>> Y = RandomVector(domain=Omega, name="Y").from_dict({0: (1, 4), 1: (2, 5), 2: (3, 6)})
        >>> P.integrate(Y) # doctest: +NORMALIZE_WHITESPACE
        feature
        Y_0    2.1
        Y_1    5.1
        dtype: float64
        """
        from ...core.random_objects.random_vector import RandomVector

        if not isinstance(rv, RandomVector) or rv.domain != self.sample_space:
            raise TypeError(
                "rv must be a RandomVector whose domain is the sample space of the probability measure."
            )

        probabilities = self.data
        return rv.data.mul(probabilities, axis=0).sum()

    def expectation(
        self,
        rv: RandomVector,
        sigma_algebra: SigmaAlgebra | None = None,
    ) -> RandomVector:
        """Compute the expectation of a `RandomVector` with respect to the probability measure, optionally conditioned on a `SigmaAlgebra`.

        The conditional expectation of a random variable is another random variable that is constant on each atom of the sigma algebra, its value on an atom being the mean value of the original random variable on that atom. This mean value is computed with respect to the conditional probabilities of the atom. If an atom has probability 0, the expected value is defined to be 0 on this atom.

        The unconditional expectation is the same as the conditional expectation with respect to the trivial sigma algebra (with a single atom equal to the entire sample space), so this description applies to the unconditional expectation too.

        Parameters
        ----------
        rv : RandomVector
            The random vector for which to compute the expectation.
        sigma_algebra : SigmaAlgebra | None, default=None
            The sigma algebra to condition on. If `None`, computes the unconditional expectation.

        Returns
        -------
        exp : RandomVector
            The expected value of the random variable.

        Examples
        --------
        >>> from sigalg.core import expectation, ProbabilityMeasure, RandomVector, SampleSpace, SigmaAlgebra
        >>> domain = SampleSpace().from_sequence(size=3, prefix="omega")
        >>> outputs = {"omega_0": (1, 2), "omega_1": (3, 4), "omega_2": (5, 6)}
        >>> P = ProbabilityMeasure(sample_space=domain).from_dict({"omega_0": 0.2, "omega_1": 0.5, "omega_2": 0.3})
        >>> X = RandomVector(domain).from_dict(outputs)
        >>> # Compute unconditional expectation
        >>> P.expectation(X) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'E(X)':
        expectation   E(X)_0  E(X)_1
        sample
        omega_0          3.2     4.2
        omega_1          3.2     4.2
        omega_2          3.2     4.2
        >>> # Compute conditional expectation given a sigma algebra
        >>> F = SigmaAlgebra(domain).from_dict({"omega_0": 0, "omega_1": 0, "omega_2": 1})
        >>> P.expectation(X, F) # doctest: +NORMALIZE_WHITESPACE
        Random vector 'E(X|F)':
        expectation  E(X_0|F)  E(X_1|F)
        sample
        omega_0       2.428571  3.428571
        omega_1       2.428571  3.428571
        omega_2       5.000000  6.000000
        """
        from ..random_objects.operators import expectation

        return expectation(
            rv=rv,
            sigma_algebra=sigma_algebra,
            probability_measure=self,
        )

    def conditional_probability(self, event: Event, given: Event) -> Real:
        """Compute the conditional probability P(A|B).

        Parameters
        ----------
        event : Event
            The event A.
        given : Event
            The event B.

        Raises
        ------
        ValueError
            If `event` or `given` are from a different sample space than this probability measure's sample space, or if P(B) = 0.
        """
        if event.sample_space != self.sample_space:
            raise ValueError(
                "event must be from this probability space's sample space."
            )
        if given.sample_space != self.sample_space:
            raise ValueError(
                "given must be from this probability space's sample space."
            )
        prob_given = self.P(given)
        if prob_given < 1e-10:
            raise ValueError("Cannot compute conditional probability: P(given) = 0")
        return self.P(event & given) / prob_given

    def are_independent(
        self,
        event1: Event | None = None,
        event2: Event | None = None,
        algebra1: SigmaAlgebra | None = None,
        algebra2: SigmaAlgebra | None = None,
        tolerance: Real = 1e-10,
    ) -> bool:
        """Check if two events or sigma algebras are independent.

        Parameters
        ----------
        event1 : Event | None, default=None
            The first event.
        event2 : Event | None, default=None
            The second event.
        algebra1 : SigmaAlgebra | None, default=None
            The first sigma algebra.
        algebra2 : SigmaAlgebra | None, default=None
            The second sigma algebra.
        tolerance : Real, default=1e-10
            The numerical tolerance for checking independence.

        Raises
        ------
        ValueError
            If neither events nor sigma algebras are provided, or if both are provided,
            or if the provided objects are from a different sample space.
        TypeError
            If the provided objects are not of the correct type.

        Returns
        -------
        is_independent : bool
            `True` if the events or sigma algebras are independent, `False` otherwise.
        """
        from ..base.event import Event
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra

        events_provided = event1 is not None and event2 is not None
        algebras_provided = algebra1 is not None and algebra2 is not None

        P = self.P

        if not events_provided and not algebras_provided:
            raise ValueError("Must provide either two events or two sigma algebras.")
        if events_provided and algebras_provided:
            raise ValueError("Cannot provide both events and sigma algebras.")

        if events_provided:
            if not isinstance(event1, Event) or not isinstance(event2, Event):
                raise TypeError("event1 and event2 must be Event instances.")

            for event in (event1, event2):
                if event.sample_space != self.sample_space:
                    raise ValueError(
                        "Event must be from this probability measure's sample space."
                    )

            return bool(abs(P(event1 & event2) - P(event1) * P(event2)) < tolerance)

        if not isinstance(algebra1, SigmaAlgebra) or not isinstance(
            algebra2, SigmaAlgebra
        ):
            raise TypeError("algebra1 and algebra2 must be SigmaAlgebra instances.")

        for algebra in (algebra1, algebra2):
            if algebra.sample_space != self.sample_space:
                raise ValueError(
                    "Sigma algebra must be from this probability measure's sample space."
                )

        atoms1 = algebra1.to_atoms()
        atoms2 = algebra2.to_atoms()
        for atom1 in atoms1:
            for atom2 in atoms2:
                if not self.are_independent(
                    event1=atom1, event2=atom2, tolerance=tolerance
                ):
                    return False
        return True

    # --------------------- factory methods --------------------- #

    @classmethod
    def from_features(
        cls,
        rv: RandomVector,
        pmf: Callable[[FeatureVector | Hashable], Real],
        name: Hashable | None = "P",
    ) -> ProbabilityMeasure:
        """Add a probability measure on the domain of a random vector using a function of the features.

        Parameters
        ----------
        rv : RandomVector
            The random vector whose domain will receive the probability measure.
        pmf : Callable[[FeatureVector | Hashable], Real]
            Function mapping feature vectors (in dimension > 1) or hashable values (in dimension 1) to probability values. Must return non-negative values that sum to 1.
        name: Hashable | None, default="P",
            The name of the probability measure.

        Returns
        -------
        prob_measure : ProbabilityMeasure
            The resulting probability measure.

        Examples
        --------
        >>> from sigalg.core import (
        ...     FeatureVector, ProbabilityMeasure, RandomVector, SampleSpace
        ... )
        >>> domain = SampleSpace.generate_sequence(size=4)
        >>> outputs = {
        ...     "omega_0": (0, 0),
        ...     "omega_1": (0, 1),
        ...     "omega_2": (1, 0),
        ...     "omega_3": (1, 1),
        ... }
        >>> X = RandomVector(domain=domain).from_dict(outputs)
        >>> def pmf(v: FeatureVector) -> Real:
        ...     v0, v1 = v
        ...     return 0.75**v0 * 0.25 ** (1 - v0) * 0.6**v1 * 0.4 ** (1 - v1)
        >>> P = ProbabilityMeasure.from_features(rv=X, pmf=pmf)
        >>> P # doctest: +NORMALIZE_WHITESPACE
        Probability measure 'P':
                probability
        sample
        omega_0         0.10
        omega_1         0.15
        omega_2         0.30
        omega_3         0.45
        """
        probabilities = {
            sample_index: pmf(sample_features)
            for sample_index, sample_features in rv.iter_features()
        }
        return cls(sample_space=rv.domain).from_dict(probabilities)

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
        return cls(sample_space=sample_space, name=name).from_dict(probabilities)

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
        >>> sample_space = SampleSpace.generate_sequence(size=3)
        >>> probabilities = {"omega_0": 0.2, "omega_1": 0.5, "omega_2": 0.3}
        >>> P = ProbabilityMeasure(sample_space=sample_space).from_dict(probabilities)
        >>> # Probability of a single sample point
        >>> float(P("omega_1"))
        0.5
        >>> # Probability of multiple sample points
        >>> float(P(["omega_0", "omega_2"]))
        0.5
        >>> # Probability of an event
        >>> A = sample_space.get_event(["omega_0", "omega_1"], name="A")
        >>> float(P(A))
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
    >>> Omega = SampleSpace().from_list(["a", "b", "c"])
    >>> probabilities = {"a": 0.2, "b": 0.5, "c": 0.3}
    >>> P = ProbabilityMeasure(sample_space=Omega).from_dict(probabilities)
    >>> obj = MyClass(P)
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

    def conditional_probability(self, event: Event, given: Event) -> Real:
        """Compute the conditional probability P(A|B).

        Parameters
        ----------
        event : Event
            The event A.
        given : Event
            The event B.

        Raises
        ------
        ValueError
            If `event` or `given` are from a different sample space than this probability measure's sample space, or if P(B) = 0.
        """
        return self.probability_measure.conditional_probability(event, given)

    def are_independent(
        self,
        event1: Event | None = None,
        event2: Event | None = None,
        algebra1: SigmaAlgebra | None = None,
        algebra2: SigmaAlgebra | None = None,
        tolerance: Real = 1e-10,
    ) -> bool:
        """Check if two events or sigma algebras are independent.

        Parameters
        ----------
        event1 : Event | None, default=None
            The first event.
        event2 : Event | None, default=None
            The second event.
        algebra1 : SigmaAlgebra | None, default=None
            The first sigma algebra.
        algebra2 : SigmaAlgebra | None, default=None
            The second sigma algebra.
        tolerance : Real, default=1e-10
            The numerical tolerance for checking independence.

        Raises
        ------
        ValueError
            If neither events nor sigma algebras are provided, or if both are provided,
            or if the provided objects are from a different sample space.
        TypeError
            If the provided objects are not of the correct type.

        Returns
        -------
        is_independent : bool
            `True` if the events or sigma algebras are independent, `False` otherwise.
        """
        return self.probability_measure.are_independent(
            event1=event1,
            event2=event2,
            algebra1=algebra1,
            algebra2=algebra2,
            tolerance=tolerance,
        )
