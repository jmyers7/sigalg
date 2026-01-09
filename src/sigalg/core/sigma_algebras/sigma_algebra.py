"""Sigma algebra module.

This module defines the SigmaAlgebra class, which represents a sigma algebra
over a given sample space. It includes methods for checking measurability of
events, retrieving atoms, and factory methods for creating common types of
sigma algebras such as power sets and trivial sigma algebras.

Classes
-------
SigmaAlgebra
    Represents a sigma algebra over a sample space.
SigmaAlgebraMethods
    Mixin class providing additional methods for sigma algebras.

Examples
--------
>>> from sigalg.core import SampleSpace, SigmaAlgebra
>>> sample_space = SampleSpace.generate_sequence(size=3, prefix="s", initial_index=1)
>>> F = SigmaAlgebra.trivial(sample_space, name="F")
>>> # Show the atom ids for each sample point
>>> F # doctest: +NORMALIZE_WHITESPACE
Sigma algebra 'F':
     atom ID
sample
s_1        0
s_2        0
s_3        0
>>> G = SigmaAlgebra.power_set(sample_space, name="G")
>>> G # doctest: +NORMALIZE_WHITESPACE
Sigma algebra 'G':
     atom ID
sample
s_1        0
s_2        1
s_3        2
>>> sample_id_to_atom_id = {"s_1": "A", "s_2": "A", "s_3": "B"}
>>> H = SigmaAlgebra(name="H").from_dict(
...     sample_id_to_atom_id=sample_id_to_atom_id,
... )
>>> H # doctest: +NORMALIZE_WHITESPACE
Sigma algebra 'H':
     atom ID
sample
s_1        A
s_2        A
s_3        B
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING

import pandas as pd

from ...validation.sample_space_mapping_in import SampleSpaceMappingIn

if TYPE_CHECKING:
    from ..base.event import Event
    from ..base.sample_space import SampleSpace
    from ..random_objects.random_vector import RandomVector


class SigmaAlgebra:
    """A class representing a sigma algebra over a sample space.

    This class represents a sigma algebra defined by a mapping from sample IDs
    to atom IDs within a given sample space.

    Parameters
    ----------
    sample_space : SampleSpace | None, default=None
            The sample space over which the sigma algebra is defined. If `None`, it will be inferred either the `from_dict` or `from_pandas` methods.
    name : Hashable | None, default="F"
        The name of the sigma algebra.

    Raises
    ------
    TypeError
        If `name` is provided and is not a hashable type, or if `sample_space` is provided and is not a `SampleSpace` instance.

    Examples
    --------
    >>> from sigalg.core import SampleSpace, SigmaAlgebra
    >>> sample_id_to_atom_id = {"s_1": "A", "s_2": "A", "s_3": "B"}
    >>> F = SigmaAlgebra(name="F").from_dict(
    ...     sample_id_to_atom_id=sample_id_to_atom_id,
    ... )
    >>> F # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'F':
        atom ID
    sample
    s_1        A
    s_2        A
    s_3        B
    """

    # --------------------- constructors --------------------- #

    def __init__(
        self,
        sample_space: SampleSpace | None = None,
        name: Hashable | None = "F",
    ) -> None:
        from ..base.sample_space import SampleSpace

        if sample_space is not None and not isinstance(sample_space, SampleSpace):
            raise TypeError("If given, sample_space must be a SampleSpace instance.")
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("If given, name must be a hashable type.")
        self.sample_space = sample_space
        self._name = name

        # caches for properties
        self._data: pd.Series | None = None
        self._sample_id_to_atom_id: Mapping[Hashable, Hashable] | None = None
        self._num_atoms: int | None = None
        self._atom_ids: list[Hashable] | None = None
        self._atom_id_to_sample_ids: dict[Hashable, list[Hashable]] | None = None
        self._atom_id_to_event: dict[Hashable, Event] | None = None
        self._atom_id_to_cardinality: dict[Hashable, int] | None = None

    def from_dict(
        self, sample_id_to_atom_id: Mapping[Hashable, Hashable]
    ) -> SigmaAlgebra:
        """Initialize the sigma algebra from a dictionary mapping sample IDs to atom IDs.

        If a `sample_space` was not provided during initialization, it will be created from the keys of the provided mapping. If it was provided, the keys of the mapping must match the sample space.

        Parameters
        ----------
        sample_id_to_atom_id : Mapping[Hashable, Hashable]
            A mapping from sample IDs to atom IDs.

        Returns
        -------
        self : SigmaAlgebra
            The current `SigmaAlgebra` instance with updated mapping.
        """
        from ..base.sample_space import SampleSpace

        v = SampleSpaceMappingIn(
            mapping=sample_id_to_atom_id, sample_space=self.sample_space
        )

        if self.sample_space is None:
            self.sample_space = SampleSpace().from_list(list(v.mapping.keys()))

        self._sample_id_to_atom_id = v.mapping
        return self

    def from_pandas(self, data: pd.Series) -> SigmaAlgebra:
        """Create a `SigmaAlgebra` from a `pd.Series`.

        If a `sample_space` was not provided during initialization, it will be created from the index of the provided `pd.Series`. If it was provided, the index of the `pd.Series` must match the sample space.

        Parameters
        ----------
        data : pd.Series
            `pd.Series` object to use for the sigma algebra.

        Raises
        ------
        TypeError
            If `data` is not a `pd.Series`.

        Returns
        -------
        self : SigmaAlgebra
            The current `SigmaAlgebra` instance with updated data.

        Examples
        --------
        >>> from sigalg.core import SigmaAlgebra
        >>> import pandas as pd
        >>> # Create a sigma algebra from a series with custom index
        >>> data = pd.Series(['A', 'A', 'B'], index=['s_0', 's_1', 's_2'])
        >>> F = SigmaAlgebra().from_pandas(data)
        >>> F # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
            atom ID
        sample
        s_0          A
        s_1          A
        s_2          B
        >>> # Check the automatically generated sample space
        >>> F.sample_space # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega':
        ['s_0', 's_1', 's_2']
        >>> # Change the name of the sample space
        >>> F.sample_space.name = 'S'
        >>> F.sample_space # doctest: +NORMALIZE_WHITESPACE
        Sample space 'S':
        ['s_0', 's_1', 's_2']
        >>> # Create another sigma algebra from series with default index
        >>> new_data = pd.Series([0, 0, 1])
        >>> G = SigmaAlgebra(name="G").from_pandas(new_data)
        >>> G # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
                atom ID
        sample
        0             0
        1             0
        2             1
        >>> G.sample_space # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega':
        [0, 1, 2]
        """
        from ..base.sample_space import SampleSpace

        if not isinstance(data, pd.Series):
            raise TypeError("data must be a pandas Series.")
        _ = SampleSpaceMappingIn(mapping=data.to_dict(), sample_space=self.sample_space)

        if self.sample_space is None:
            self.sample_space = SampleSpace().from_pandas(data.index)

        self._data = data.copy()
        self._data.name = "atom ID"
        return self

    # --------------------- properties --------------------- #

    @property
    def sample_id_to_atom_id(self) -> Mapping[Hashable, Hashable]:
        """Get the mapping from sample IDs to atom IDs.

        Returns
        -------
        sample_id_to_atom_id : Mapping[Hashable, Hashable]
            A mapping from sample IDs to atom IDs.
        """
        if self._sample_id_to_atom_id is None:
            self._sample_id_to_atom_id = self.data.to_dict()
        return self._sample_id_to_atom_id

    @property
    def data(self) -> pd.Series:
        """Get the underlying `pd.Series`.

        Returns
        -------
        data: pd.Series
            A `pd.Series` mapping sample IDs to atom IDs.
        """
        if self._data is None:
            self._data = pd.Series(data=self._sample_id_to_atom_id, name="atom ID")
            self._data.index.name = self.sample_space.data.name
        return self._data

    # @data.setter
    # def data(self, data: pd.Series) -> None:
    #     """Set the underlying `pd.Series`.

    #     The `data` property is not meant to be set directly by the user. This setter is provided so that the `from_pandas` factory method can set the property.

    #     Parameters
    #     ----------
    #     data : pd.Series
    #         New `pd.Series` object to set.

    #     Raises
    #     ------
    #     TypeError
    #         If `data` is not a `pd.Series`.
    #     """
    #     if not isinstance(data, pd.Series):
    #         raise TypeError("data must be a pandas Series.")
    #     self._data = data

    @property
    def name(self) -> Hashable:
        """Get the name identifier for this sigma algebra.

        Returns
        -------
        name : Hashable
            The name of this sigma algebra.
        """
        return self._name

    @name.setter
    def name(self, name: Hashable) -> None:
        """Set the name identifier for this sigma algebra.

        Parameters
        ----------
        name : Hashable
            New name for this sigma algebra.

        Raises
        ------
        TypeError
            If `name` is not a hashable.
        """
        if not isinstance(name, Hashable):
            raise TypeError("name must be a hashable type.")
        self._name = name
        if self._data is not None:
            self._data.name = name

    @property
    def num_atoms(self) -> int:
        """Get the number of atoms in this sigma algebra.

        Returns
        -------
        num_atoms : int
            The number of atoms in this sigma algebra.
        """
        if self._num_atoms is None:
            self._num_atoms = self.data.nunique()
        return self._num_atoms

    @property
    def atom_ids(self) -> list[Hashable]:
        """Get a list of atom IDs in this sigma algebra.

        Returns
        -------
        atom_ids : list[Hashable]
            A list of atom IDs in this sigma algebra.
        """
        if self._atom_ids is None:
            self._atom_ids = list(self.data.unique())
        return self._atom_ids

    @property
    def atom_id_to_sample_ids(self) -> dict[Hashable, list[Hashable]]:
        """Get a mapping from atom IDs to lists of sample IDs in this sigma algebra.

        Returns
        -------
        atom_id_to_sample_ids : dict[Hashable, list[Hashable]]
            A dictionary mapping each atom ID to a list of sample IDs contained in that atom.
        """
        if self._atom_id_to_sample_ids is None:
            atom_id_to_sample_ids = {}
            for sample_id, atom_id in self.sample_id_to_atom_id.items():
                if atom_id not in atom_id_to_sample_ids:
                    atom_id_to_sample_ids[atom_id] = []
                atom_id_to_sample_ids[atom_id].append(sample_id)
            self._atom_id_to_sample_ids = atom_id_to_sample_ids
        return self._atom_id_to_sample_ids

    @property
    def atom_id_to_event(self) -> dict[Hashable, Event]:
        """Get a mapping from atom IDs to `Event` objects in this sigma algebra.

        Returns
        -------
        atom_id_to_event : dict[Hashable, Event]
            A dictionary mapping each atom ID to its corresponding `Event` object.
        """
        if self._atom_id_to_event is None:
            atom_id_to_event = {
                atom_id: self.sample_space.get_event(sample_ids, name=atom_id)
                for atom_id, sample_ids in self.atom_id_to_sample_ids.items()
            }
            self._atom_id_to_event = atom_id_to_event
        return self._atom_id_to_event

    @property
    def atom_id_to_cardinality(self) -> dict[Hashable, int]:
        """Get a mapping from atom IDs to their cardinalities in this sigma algebra.

        Returns
        -------
        atom_id_to_cardinality : dict[Hashable, int]
            A dictionary mapping each atom ID to the number of sample IDs it contains.
        """
        if self._atom_id_to_cardinality is None:
            self._atom_id_to_cardinality = {
                atom_id: len(event) for atom_id, event in self.atom_id_to_event.items()
            }
        return self._atom_id_to_cardinality

    # --------------------- factory methods --------------------- #

    @classmethod
    def power_set(
        cls,
        sample_space: SampleSpace,
        name: Hashable = "power_set",
    ) -> SigmaAlgebra:
        """Create the power-set sigma algebra over a given sample space.

        The power-set sigma algebra contains all possible subsets of the sample space,
        meaning each sample point is its own atom. It is the finest sigma algebra possible over the given sample space.

        Parameters
        ----------
        sample_space : SampleSpace
            The sample space over which to create the power-set sigma algebra.
        name : Hashable, optional
            Name identifier for the sigma algebra.

        Returns
        -------
        sigma_algebra : SigmaAlgebra
            A new `SigmaAlgebra` instance representing the power-set sigma algebra.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> sample_space = SampleSpace.generate_sequence(size=3, initial_index=1, prefix="s")
        >>> G = SigmaAlgebra.power_set(sample_space, name="G")
        >>> # Each sample point is its own atom in the power-set sigma algebra
        >>> G # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
            atom ID
        sample
        s_1        0
        s_2        1
        s_3        2
        """
        sample_id_to_atom_id = {
            index: idx for idx, index in enumerate(sample_space.data)
        }
        return cls(name=name).from_dict(sample_id_to_atom_id=sample_id_to_atom_id)

    @classmethod
    def trivial(
        cls,
        sample_space: SampleSpace,
        name: Hashable = "trivial",
    ) -> SigmaAlgebra:
        """Create the trivial sigma algebra over a given sample space.

        The trivial sigma algebra contains only the empty set and the entire sample space, meaning all sample points belong to the same atom. It is the coarsest sigma algebra possible over the given sample space.

        Parameters
        ----------
        sample_space : SampleSpace
            The sample space over which to create the trivial sigma algebra.
        name : Hashable, optional
            Name identifier for the sigma algebra.

        Returns
        -------
        sigma_algebra : SigmaAlgebra
            A new `SigmaAlgebra` instance representing the trivial sigma algebra.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> sample_space = SampleSpace.generate_sequence(size=3, initial_index=1, prefix="s")
        >>> F = SigmaAlgebra.trivial(sample_space, name="F")
        >>> # All sample points belong to the same atom in the trivial sigma algebra
        >>> F # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                atom ID
        sample
        s_1        0
        s_2        0
        s_3        0
        """
        sample_id_to_atom_id = dict.fromkeys(sample_space.data, 0)
        return cls(name=name).from_dict(sample_id_to_atom_id=sample_id_to_atom_id)

    @classmethod
    def from_random_vector(
        cls,
        rv: RandomVector,
        discretize: bool = False,
        n_bins: int = 10,
        use_pca: bool = False,
        n_components: int | None = None,
    ) -> SigmaAlgebra:
        """Create a sigma algebra induced by a random vector.

        Parameters
        ----------
        rv : RandomVector
            The random vector to induce the sigma algebra from.
        discretize : bool, default=False
            Whether to discretize continuous data using binning.
        n_bins : int, default=10
            Number of bins per dimension (only used if discretize=True).
        use_pca : bool, default=False
            Whether to apply PCA before discretization (only used if discretize=True).
        n_components : int | None, default=None
            Number of principal components (only used if discretize=True and use_pca=True).

        Returns
        -------
        sigma_algebra : SigmaAlgebra
            A new `SigmaAlgebra` instance induced by the given random vector.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import KBinsDiscretizer

        from ..random_objects import RandomVector

        if not isinstance(rv, RandomVector):
            raise TypeError("rv must be a RandomVector instance.")

        name = f"sigma({rv.name})" if rv.name is not None else None

        if not discretize:
            return cls(sample_space=rv.domain, name=name).from_dict(rv.outputs)

        data = rv.data.values.reshape(-1, 1) if rv.dimension == 1 else rv.data.values

        if use_pca:
            pca = PCA(n_components=n_components)
            data = pca.fit_transform(data)

        discretizer = KBinsDiscretizer(
            n_bins=n_bins,
            encode="ordinal",
            strategy="quantile",
            quantile_method="averaged_inverted_cdf",
            subsample=None,
        )
        discretized = discretizer.fit_transform(data)

        sample_id_to_atom_id = {
            sample_id: tuple(discretized[idx].astype(int))
            for idx, sample_id in enumerate(rv.domain.data)
        }

        return cls(
            sample_space=rv.domain, name=f"{name}_discrete" if name else None
        ).from_dict(sample_id_to_atom_id)

    @classmethod
    def from_event(cls, event: Event) -> SigmaAlgebra:
        """Create the sigma algebra generated by a single event.

        Parameters
        ----------
        event : Event
            The event to generate the sigma algebra from.

        Returns
        -------
        sigma_algebra : SigmaAlgebra
            A new `SigmaAlgebra` instance generated by the given event.
        """
        from ..base import Event

        if not isinstance(event, Event):
            raise TypeError("event must be an Event instance.")

        sample_space = event.sample_space
        sample_id_to_atom_id = {}
        for sample_id in sample_space.data:
            if sample_id in event.data:
                sample_id_to_atom_id[sample_id] = 1
            else:
                sample_id_to_atom_id[sample_id] = 0

        name = f"sigma({event.name})" if event.name is not None else None
        return cls(
            sample_id_to_atom_id=sample_id_to_atom_id,
            sample_space=sample_space,
            name=name,
        )

    # --------------------- methods --------------------- #

    def to_atoms(self) -> list[Event]:
        """
        Get a list of atoms as `Event` objects in this sigma algebra.

        Returns
        -------
        atoms : list[Event]
            A list of `Event` objects representing the atoms in this sigma algebra.
        """
        return list(self.atom_id_to_event.values())

    def is_measurable(self, event: Event) -> bool:
        """Check if an event is measurable with respect to this sigma algebra.

        Parameters
        ----------
        event : Event
            The event to check for measurability.

        Raises
        ------
        TypeError
            If `event` is not an `Event` instance.
        ValueError
            If `event` does not have the same sample space as this sigma algebra.

        Returns
        -------
        is_measurable : bool
            `True` if the event is measurable with respect to this sigma algebra, `False` otherwise.

        Examples
        --------
        >>> from sigalg.core import Event, SampleSpace, SigmaAlgebra
        >>> sample_space = SampleSpace.generate_sequence(size=3, initial_index=1, prefix="s")
        >>> sample_id_to_atom_id = {"s_1": "A", "s_2": "A", "s_3": "B"}
        >>> sigma_algebra = SigmaAlgebra(sample_space=sample_space).from_dict(
        ...     sample_id_to_atom_id=sample_id_to_atom_id,
        ... )
        >>> A = Event(sample_space=sample_space, name="A").from_list(["s_1", "s_2"])
        >>> B = Event(sample_space=sample_space, name="B").from_list(["s_3"])
        >>> C = Event(sample_space=sample_space, name="C").from_list(["s_1"])
        >>> sigma_algebra.is_measurable(A)
        True
        >>> sigma_algebra.is_measurable(B)
        True
        >>> sigma_algebra.is_measurable(C)
        False
        """
        from ..base import Event

        if not isinstance(event, Event):
            raise TypeError("event must be an Event instance.")
        if event.sample_space != self.sample_space:
            raise ValueError(
                "event must have the same sample_space as the sigma_algebra."
            )

        event_sample_ids = set(event.data)
        for event_sample_id in event_sample_ids:
            atom_id = self.sample_id_to_atom_id[event_sample_id]
            atom_sample_ids = set(self.atom_id_to_sample_ids[atom_id])
            if not event_sample_ids.issuperset(atom_sample_ids):
                return False
        return True

    def get_atom_containing(self, sample_id: Hashable) -> Event:
        """Get the atom containing a given sample ID.

        Parameters
        ----------
        sample_id : Hashable
            The sample ID for which to retrieve the containing atom.

        Raises
        ------
        ValueError
            If `sample_id` is not in the sample space of this sigma algebra.

        Returns
        -------
        atom : Event
            The `Event` object representing the atom that contains the given sample ID.
        """
        from ..base import Event

        if sample_id not in self.sample_id_to_atom_id:
            raise ValueError(f"Sample ID '{sample_id}' not in sample space.")
        atom_id = self.sample_id_to_atom_id[sample_id]
        sample_ids = self.atom_id_to_sample_ids[atom_id]
        return Event(sample_space=self.sample_space).from_list(sample_ids)

    def __contains__(self, event: Event) -> bool:
        """Check if an event is measurable with respect to this sigma algebra.

        Parameters
        ----------
        event : Event
            The event to check for measurability.

        Returns
        -------
        contains : bool
            `True` if the event is measurable with respect to this sigma algebra, `False` otherwise.
        """
        return self.is_measurable(event)

    def __or__(self, other: SigmaAlgebra) -> SigmaAlgebra:
        """Get the join (least upper bound) of this sigma algebra with another.

        Parameters
        ----------
        other : SigmaAlgebra
            The other sigma algebra to join with.

        Returns
        -------
        join_sigma_algebra : SigmaAlgebra
            A new `SigmaAlgebra` instance representing the join of the two sigma algebras.
        """
        from .lattice_operations import join

        return join([self, other])

    # --------------------- iter method --------------------- #

    def __iter__(self) -> iter:
        """Iterate over the atom IDs and atoms (as `Events`) in this sigma algebra.

        Returns
        -------
        iterator : iter
            An iterator over tuples of (atom_id, Event) for each atom in the sigma algebra.
        """
        return iter(self.atom_id_to_event.items())

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a string representation of the sigma algebra.

        Returns
        -------
        repr_str : str
            A string representation of the sigma algebra.
        """
        return f"Sigma algebra '{self.name}':\n{self.data.to_frame()}"

    # --------------------- equality --------------------- #

    def __eq__(self, other: SigmaAlgebra) -> bool:
        """Check equality with another sigma algebra.

        Two sigma algebras are equal if they have the same sample space and contain the same atoms. They may have different names and still be considered equal.

        Parameters
        ----------
        other : SigmaAlgebra
            The other sigma algebra to compare with.

        Returns
        -------
        is_equal : bool
            `True` if the other object is a `SigmaAlgebra` with the same sample space and atoms, `False` otherwise.
        """
        if not isinstance(other, SigmaAlgebra):
            return False
        if self.sample_space != other.sample_space:
            return False
        return self <= other and other <= self

    # --------------------- order relations --------------------- #

    def __le__(self, other: SigmaAlgebra) -> bool:
        """Check if this sigma algebra is a sub-algebra of another.

        Parameters
        ----------
        other : SigmaAlgebra
            The other sigma algebra to compare with.

        Raises
        ------
        ValueError
            If the sample spaces of the two sigma algebras are not the same.

        Returns
        -------
        is_subalgebra : bool
            `True` if this sigma algebra is a sub-algebra of the other, `False` otherwise.
        """
        if not isinstance(other, SigmaAlgebra):
            return NotImplemented
        if self.sample_space != other.sample_space:
            raise ValueError(
                "Sigma algebras must have the same sample space for comparison."
            )
        from .comparison import is_subalgebra

        return is_subalgebra(sub_algebra=self, super_algebra=other)

    def __lt__(self, other: SigmaAlgebra) -> bool:
        """
        Check if this sigma algebra is a proper sub-algebra of another.

        Parameters
        ----------
        other : SigmaAlgebra
            The other sigma algebra to compare with.

        Returns
        -------
        is_proper_subalgebra : bool
            `True` if this sigma algebra is a proper sub-algebra of the other, `False` otherwise.
        """
        if not isinstance(other, SigmaAlgebra):
            return NotImplemented
        return self <= other and self != other

    def __ge__(self, other: SigmaAlgebra) -> bool:
        """Check if this sigma algebra is a super-algebra of another.

        Parameters
        ----------
        other : SigmaAlgebra
            The other sigma algebra to compare with.

        Raises
        ------
        ValueError
            If the sample spaces of the two sigma algebras are not the same.

        Returns
        -------
        is_superalgebra : bool
            `True` if this sigma algebra is a super-algebra of the other, `False` otherwise.
        """
        if not isinstance(other, SigmaAlgebra):
            return NotImplemented
        if self.sample_space != other.sample_space:
            raise ValueError(
                "Sigma algebras must have the same sample space for comparison."
            )
        from .comparison import is_subalgebra

        return is_subalgebra(sub_algebra=other, super_algebra=self)

    def __gt__(self, other: SigmaAlgebra) -> bool:
        """Check if this sigma algebra is a proper super-algebra of another.

        Parameters
        ----------
        other : SigmaAlgebra
            The other sigma algebra to compare with.

        Returns
        -------
        is_proper_superalgebra : bool
            `True` if this sigma algebra is a proper super-algebra of the other, `False` otherwise.
        """
        if not isinstance(other, SigmaAlgebra):
            return NotImplemented
        return self >= other and self != other


class SigmaAlgebraMethods:
    """Mixin class providing sigma algebra methods to other classes.

    This mixin provides convenience methods for classes that have a `sigma_algebra`
    attribute, allowing them to delegate sigma algebra operations to that attribute.

    The class assumes the implementing class has a `sigma_algebra` attribute that
    is a `SigmaAlgebra` instance.

    Examples
    --------
    >>> class MyClass(SigmaAlgebraMethods):
    ...     def __init__(self, sigma_algebra):
    ...         self.sigma_algebra = sigma_algebra
    >>> from sigalg.core import SigmaAlgebra
    >>> F = SigmaAlgebra().from_dict({"a": 0, "b": 0, "c": 1})
    >>> obj = MyClass(F)
    >>> A = F.sample_space.get_event(["a", "b"])
    >>> obj.is_measurable(A)
    True
    """

    def is_measurable(self, event: Event) -> bool:
        """Check if an event is measurable with respect to the sigma algebra.

        Delegates to the `is_measurable` method of the `sigma_algebra` attribute.

        Parameters
        ----------
        event : Event
            The event to check for measurability.

        Returns
        -------
        is_measurable : bool
            `True` if the event is measurable with respect to the sigma algebra, `False` otherwise.
        """
        return self.sigma_algebra.is_measurable(event)

    def get_atom_containing(self, sample_id: Hashable) -> Event:
        """Get the atom containing a given sample ID.

        Delegates to the `get_atom_containing` method of the `sigma_algebra` attribute.

        Parameters
        ----------
        sample_id : Hashable
            The sample ID for which to retrieve the containing atom.

        Returns
        -------
        atom : Event
            The `Event` object representing the atom that contains the given sample ID.
        """
        return self.sigma_algebra.get_atom_containing(sample_id)
