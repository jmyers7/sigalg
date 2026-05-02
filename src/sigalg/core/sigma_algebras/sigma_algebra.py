"""A class representing a sigma-algebra."""

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
    r"""A class representing a sigma-algebra on a sample space.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    sample_space : SampleSpace | None, default=None
        The sample space over which the sigma-algebra is defined.
    name : Hashable | None, default="F"
        The name of the sigma-algebra.

    Raises
    ------
    TypeError
        If `name` is provided and is not a hashable type, or if `sample_space` is provided and is not a `SampleSpace` instance.

    Examples
    --------
    >>> from sigalg.core import SampleSpace, SigmaAlgebra
    >>> Omega = SampleSpace().from_sequence(size=3)
    >>> # Define a sigma-algebra with atoms A_0 = {0, 0} and A_1 = {2}
    >>> atom_ids = {0: 0, 1: 0, 2: 1}
    >>> F = SigmaAlgebra(name="F").from_dict(atom_ids)
    >>> print(F) # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'F':
        atom ID
    sample
    0         0
    1         0
    2         1

    Notes
    -----
    A *$\sigma$-algebra* $\mathcal{F}$ on a set $\Omega$ is a collection of subsets of $\Omega$ that contains $\Omega$, and is closed under complementation and countable unions. In the case that $\Omega$ is finite (as it always is, in SigAlg), then $\mathcal{F}$ obviously needs only to be closed under finite unions.

    A set in $\mathcal{F}$ is called an *event*. In general measure theory, events are called *$\mathcal{F}$-measurable* sets, or just *measurable* sets if the identity of the $\sigma$-algebra is clear.

    A $\sigma$-algebra $\mathcal{F}$ on a finite set $\Omega$ determines its *atoms*, which are the nonempty events $A\in \mathcal{F}$ that are *minimal* with respect to subset inclusion, in the sense that if $B\in \mathcal{F}$ is nonempty and $B\subset A$, then necessarily $A=B$. And conversely, $\mathcal{F}$ is completely recoverable from its atoms, in the sense that every $B\in \mathcal{F}$ is a union of atoms. The atoms partition the set $\Omega$, which means that the atoms are pairwise disjoint and their union is all of $\Omega$.

    If $\{A_i\}_{i\in I}$ is the set of atoms, indexed by a finite set $I$, then there is a mapping $\Omega \to I$ given by $\omega \mapsto i$, where $A_i$ is the unique atom that contains $\omega$. This mapping is what SigAlg uses to represent $\sigma$-algebras. The indices in $I$ are called *atom identifiers*. See the Example above.

    See also the [notebook](https://johnmyers-phd.com/sigalg/dictionary/){target="_blank"} on the docs website.
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
        self._atom_space: SampleSpace | None = None
        self._num_atoms: int | None = None
        self._atom_ids: list[Hashable] | None = None
        self._atom_id_to_sample_ids: dict[Hashable, list[Hashable]] | None = None
        self._atom_id_to_event: dict[Hashable, Event] | None = None
        self._atom_id_to_cardinality: dict[Hashable, int] | None = None
        self._is_power_set: bool | None = None

    def from_dict(
        self, sample_id_to_atom_id: Mapping[Hashable, Hashable]
    ) -> SigmaAlgebra:
        """Generate the sigma-algebra from a dictionary mapping sample points to atom IDs.

        If a `sample_space` was not provided during initialization, it will be created from the keys of the provided mapping. If it was provided, the keys of the mapping must match the sample space.

        Parameters
        ----------
        sample_id_to_atom_id : Mapping[Hashable, Hashable]
            A mapping from sample points to atom IDs.

        Returns
        -------
        self : SigmaAlgebra
            The current `SigmaAlgebra` instance.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> # Define a sigma-algebra with atoms A_0 = {0, 0} and A_1 = {2}
        >>> atom_ids = {0: 0, 1: 0, 2: 1}
        >>> F = SigmaAlgebra(name="F").from_dict(atom_ids)
        >>> print(F) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
            atom ID
        sample
        0         0
        1         0
        2         1
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
        """Generate the sigma-algebra from a `pd.Series` mapping sample points to atom IDs.

        If a `sample_space` was not provided during initialization, it will be created from the index of the provided `pd.Series`. If it was provided, the index of the `pd.Series` must match the sample space.

        Parameters
        ----------
        data : pd.Series
            `pd.Series` object to use for the sigma-algebra.

        Raises
        ------
        TypeError
            If `data` is not a `pd.Series`.

        Returns
        -------
        self : SigmaAlgebra
            The current `SigmaAlgebra` instance.

        Examples
        --------
        >>> from sigalg.core import SigmaAlgebra
        >>> import pandas as pd
        >>> # Create a sigma algebra from a series with custom index
        >>> data = pd.Series([0, 0, 1], index=['s_0', 's_1', 's_2'])
        >>> F = SigmaAlgebra().from_pandas(data)
        >>> print(F) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
            atom ID
        sample
        s_0          0
        s_1          0
        s_2          1
        >>> # Check the automatically generated sample space
        >>> print(F.sample_space) # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega':
        ['s_0', 's_1', 's_2']
        >>> # Change the name of the sample space
        >>> F.sample_space.name = 'S'
        >>> print(F.sample_space) # doctest: +NORMALIZE_WHITESPACE
        Sample space 'S':
        ['s_0', 's_1', 's_2']
        >>> # Create another sigma algebra from series with default index
        >>> new_data = pd.Series([0, 0, 1])
        >>> G = SigmaAlgebra(name="G").from_pandas(new_data)
        >>> print(G) # doctest: +NORMALIZE_WHITESPACE
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

    @classmethod
    def from_random_vector(
        cls,
        rv: RandomVector,
    ) -> SigmaAlgebra:
        r"""Create a sigma-algebra induced by a random vector.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        rv : RandomVector
            The random vector from which to generate the sigma-algebra.

        Returns
        -------
        sig_alg : SigmaAlgebra
            A new `SigmaAlgebra` instance induced by the given random vector.

        Examples
        --------
        >>> from sigalg.core import RandomVector, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> X = RandomVector(domain=Omega).from_dict(
        ...     {
        ...             0: (1, 2),
        ...             1: (1, 2),
        ...             2: (2, 4),
        ...     }
        ... )
        >>> sigma_X = SigmaAlgebra.from_random_vector(rv=X)
        >>> print(sigma_X) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'sigma(X)':
            atom ID
        sample
        0       (1, 2)
        1       (1, 2)
        2       (2, 4)

        Notes
        -----
        Let $X: \Omega \to \mathbb{R}^d$ be a function defined on a sample space $\Omega$. The *$\sigma$-algebra induced by $X$*, denoted $\sigma(X)$, is the $\sigma$-algebra generated by the preimages of Borel sets in $\mathbb{R}^d$ under $X$. In SigAlg, in which $\Omega$ is finite and $\sigma$-algebras are determined by their atoms, we may take the atom identifiers to be the unique values of $X$ on $\Omega$.
        """
        from ..random_objects import RandomVector

        if not isinstance(rv, RandomVector):
            raise TypeError("rv must be a RandomVector instance.")

        name = f"sigma({rv.name})" if rv.name is not None else None

        return cls(sample_space=rv.domain, name=name).from_dict(rv.outputs)

    @classmethod
    def from_event(cls, event: Event) -> SigmaAlgebra:
        r"""Create the sigma-algebra generated by a single event.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        event : Event
            The event to generate the sigma-algebra from.

        Raises
        ------
        TypeError
            If `event` is not an `Event` instance.
        ValueError
            If `event` is empty.

        Returns
        -------
        sig_alg : SigmaAlgebra
            A new `SigmaAlgebra` instance generated by the given event.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_event([0, 2])
        >>> print(SigmaAlgebra.from_event(A)) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'sigma(A)':
                atom ID
        sample
        0             1
        1             0
        2             1

        Notes
        -----
        Let $A$ be a nonempty subset of a finite set $\Omega$. The *$\sigma$-algebra generated by $A$*, denoted $\sigma(A)$, has two atoms given by $A$ and its complement $A^c$.
        """
        from ..base import Event

        if not isinstance(event, Event):
            raise TypeError("event must be an Event instance.")
        if len(event) == 0:
            raise ValueError("event must be nonempty.")

        sample_space = event.sample_space
        sample_id_to_atom_id = {}
        for sample_id in sample_space.data:
            if sample_id in event.data:
                sample_id_to_atom_id[sample_id] = 1
            else:
                sample_id_to_atom_id[sample_id] = 0

        name = f"sigma({event.name})" if event.name is not None else None
        return cls(
            sample_space=sample_space,
            name=name,
        ).from_dict(sample_id_to_atom_id)

    @classmethod
    def power_set(
        cls,
        sample_space: SampleSpace,
        name: Hashable = "power_set",
    ) -> SigmaAlgebra:
        r"""Create the power-set sigma-algebra over a given sample space.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        sample_space : SampleSpace
            The sample space over which to create the power-set sigma-algebra.
        name : Hashable, optional
            Name identifier for the sigma algebra.

        Returns
        -------
        sig_alg : SigmaAlgebra
            A new `SigmaAlgebra` instance representing the power-set sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> sample_space = SampleSpace().from_sequence(size=3)
        >>> G = SigmaAlgebra.power_set(sample_space, name="G")
        >>> print(G) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
            atom ID
        sample
        0        0
        1        1
        2        2

        Notes
        -----
        The *power-set $\sigma$-algebra* on a set $\Omega$ consists of all subsets of $\Omega$. Its atoms are all singleton subsets. It is the finest $\sigma$-algebra on $\Omega$.
        """
        sample_id_to_atom_id = {
            sample_point: num for num, sample_point in enumerate(sample_space.data)
        }
        return cls(sample_space=sample_space, name=name).from_dict(
            sample_id_to_atom_id=sample_id_to_atom_id
        )

    @classmethod
    def trivial(
        cls,
        sample_space: SampleSpace,
        name: Hashable = "trivial",
    ) -> SigmaAlgebra:
        r"""Create the trivial sigma-algebra over a given sample space.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        sample_space : SampleSpace
            The sample space over which to create the trivial sigma-algebra.
        name : Hashable, optional
            Name identifier for the sigma-algebra.

        Returns
        -------
        sig_alg : SigmaAlgebra
            A new `SigmaAlgebra` instance representing the trivial sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> sample_space = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra.trivial(sample_space, name="F")
        >>> print(F) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                atom ID
        sample
        0        0
        1        0
        2        0

        Notes
        -----
        The *trivial $\sigma$-algebra* on a set $\Omega$ consists of only the sets $\Omega$ and $\emptyset$. Its single atom is $\Omega$ itself. It is the coarsest $\sigma$-algebra on $\Omega$.
        """
        sample_id_to_atom_id = dict.fromkeys(sample_space.data, 0)
        return cls(name=name).from_dict(sample_id_to_atom_id=sample_id_to_atom_id)

    # --------------------- properties --------------------- #

    @property
    def sample_id_to_atom_id(self) -> Mapping[Hashable, Hashable]:
        """Get the mapping from sample points to atom IDs.

        Returns
        -------
        sample_id_to_atom_id : Mapping[Hashable, Hashable]
            A mapping from sample IDs to atom IDs.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> # Define a sigma-algebra with atoms A_0 = {0, 0} and A_1 = {2}
        >>> atom_ids = {0: 0, 1: 0, 2: 1}
        >>> F = SigmaAlgebra(name="F").from_dict(atom_ids)
        >>> print(F) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
            atom ID
        sample
        0         0
        1         0
        2         1
        >>> print(F.sample_id_to_atom_id)
        {0: 0, 1: 0, 2: 1}
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
            A `pd.Series` mapping sample points to atom IDs.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> # Define a sigma-algebra with atoms A_0 = {0, 0} and A_1 = {2}
        >>> atom_ids = {0: 0, 1: 0, 2: 1}
        >>> F = SigmaAlgebra(name="F").from_dict(atom_ids)
        >>> print(F) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
            atom ID
        sample
        0         0
        1         0
        2         1
        >>> print(F.data) # doctest: +NORMALIZE_WHITESPACE
                sample
        0    0
        1    0
        2    1
        Name: atom ID, dtype: int64
        """
        if self._data is None:
            self._data = pd.Series(data=self._sample_id_to_atom_id, name="atom ID")
            self._data.index.name = self.sample_space.data.name
        return self._data

    @property
    def atom_space(self) -> SampleSpace:
        """Get the sample space of atom identifiers.

        The order that the atom identifiers appear in the sample space is the same as the order they appear in the underlying `pd.Series` of the sigma-algebra.

        Returns
        -------
        atom_space: SampleSpace
            The sample space whose points are the atom identifiers of the sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         0: 1,
        ...         1: 0,
        ...         2: 1,
        ...     }
        ... )
        >>> print(F.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'atom_space':
        [1, 0]
        """
        from ..base.sample_space import SampleSpace

        if self._atom_space is None:
            self._atom_space = SampleSpace(
                name="atom_space", data_name="atom ID"
            ).from_list(self.atom_ids)
        return self._atom_space

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

    def with_name(self, name: Hashable) -> SigmaAlgebra:
        """Set the name of the sigma-algebra and return self for chaining.

        Parameters
        ----------
        name : Hashable
            The new name for the sigma algebra.

        Returns
        -------
        self : SigmaAlgebra
            The current instance with the updated name.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> # Define a sigma-algebra with atoms A_0 = {0, 0} and A_1 = {2}
        >>> atom_ids = {0: 0, 1: 0, 2: 1}
        >>> F = SigmaAlgebra(name="F").from_dict(atom_ids)
        >>> print(F) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
            atom ID
        sample
        0         0
        1         0
        2         1
        >>> print(F.with_name("sig_alg")) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'sig_alg':
            atom ID
        sample
        0         0
        1         0
        2         1
        """
        self.name = name
        return self

    @property
    def num_atoms(self) -> int:
        """Get the number of atoms in this sigma-algebra.

        Returns
        -------
        num_atoms : int
            The number of atoms in this sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> # Define a sigma-algebra with atoms A_0 = {0, 0} and A_1 = {2}
        >>> atom_ids = {0: 0, 1: 0, 2: 1}
        >>> F = SigmaAlgebra(name="F").from_dict(atom_ids)
        >>> print(F) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
            atom ID
        sample
        0         0
        1         0
        2         1
        >>> print(F.num_atoms)
        2
        """
        if self._num_atoms is None:
            self._num_atoms = self.data.nunique()
        return self._num_atoms

    @property
    def atom_ids(self) -> list[Hashable]:
        """Get a list of atom IDs in this sigma-algebra.

        Returns
        -------
        atom_ids : list[Hashable]
            A list of atom IDs in this sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> # Define a sigma-algebra with atoms A_0 = {0, 0} and A_1 = {2}
        >>> atom_ids = {0: 0, 1: 0, 2: 1}
        >>> F = SigmaAlgebra(name="F").from_dict(atom_ids)
        >>> print(F) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
            atom ID
        sample
        0         0
        1         0
        2         1
        >>> print(F.atom_ids)
        [0, 1]
        """
        if self._atom_ids is None:
            self._atom_ids = list(self.data.drop_duplicates())
        return self._atom_ids

    @property
    def atom_id_to_sample_ids(self) -> dict[Hashable, list[Hashable]]:
        """Get a mapping from atom IDs to lists of sample points.

        Returns
        -------
        atom_id_to_sample_ids : dict[Hashable, list[Hashable]]
            A dictionary mapping each atom ID to a list of sample points contained in that atom.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> # Define a sigma-algebra with atoms A_0 = {0, 0} and A_1 = {2}
        >>> atom_ids = {0: 0, 1: 0, 2: 1}
        >>> F = SigmaAlgebra(name="F").from_dict(atom_ids)
        >>> print(F) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
            atom ID
        sample
        0         0
        1         0
        2         1
        >>> print(F.atom_id_to_sample_ids)
        {0: [0, 1], 1: [2]}
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
        r"""Get a mapping from atom IDs to `Event` objects in this sigma-algebra.

        Returns
        -------
        atom_id_to_event : dict[Hashable, Event]
            A dictionary mapping each atom ID to its corresponding `Event` object.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> # Define a sigma-algebra with atoms A_0 = {0, 0} and A_1 = {2}
        >>> atom_ids = {0: 0, 1: 0, 2: 1}
        >>> F = SigmaAlgebra(name="F").from_dict(atom_ids)
        >>> print(F) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
            atom ID
        sample
        0         0
        1         0
        2         1
        >>> for atom_id, event in F.atom_id_to_event.items():
        ...     print(f"Atom ID: {atom_id}\n{event}\n") # doctest: +NORMALIZE_WHITESPACE
        Atom ID: 0
        Event '0':
        [0, 1]
        <BLANKLINE>
        Atom ID: 1
        Event '1':
        [2]
        <BLANKLINE>
        """
        if self._atom_id_to_event is None:
            atom_id_to_event = {
                atom_id: self.get_event(sample_ids, name=atom_id)
                for atom_id, sample_ids in self.atom_id_to_sample_ids.items()
            }
            self._atom_id_to_event = atom_id_to_event
        return self._atom_id_to_event

    @property
    def atom_id_to_cardinality(self) -> dict[Hashable, int]:
        """Get a mapping from atom IDs to their cardinalities in this sigma-algebra.

        Returns
        -------
        atom_id_to_cardinality : dict[Hashable, int]
            A dictionary mapping each atom ID to the number of sample points it contains.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> # Define a sigma-algebra with atoms A_0 = {0, 0} and A_1 = {2}
        >>> atom_ids = {0: 0, 1: 0, 2: 1}
        >>> F = SigmaAlgebra(name="F").from_dict(atom_ids)
        >>> print(F) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
            atom ID
        sample
        0         0
        1         0
        2         1
        >>> print(F.atom_id_to_cardinality)
        {0: 2, 1: 1}
        """
        if self._atom_id_to_cardinality is None:
            self._atom_id_to_cardinality = {
                atom_id: len(event) for atom_id, event in self.atom_id_to_event.items()
            }
        return self._atom_id_to_cardinality

    @property
    def is_power_set(self) -> bool:
        """Boolean flag signaling a power-set sigma-algebra.

        Returns
        -------
        is_power_set: bool
            A boolean signaling whether the sigma-algebra is the power-set sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> atom_ids = dict(zip(Omega, [0, 0, 1]))
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(atom_ids)
        >>> print(F.is_power_set)
        False
        >>> atom_ids = dict(zip(Omega, [1, 6, 7]))
        >>> G = SigmaAlgebra(sample_space=Omega).from_dict(atom_ids)
        >>> print(G.is_power_set)
        True
        """
        if self._is_power_set is None:
            self._is_power_set = self.num_atoms == len(self.sample_space)
        return self._is_power_set

    # --------------------- atom and event methods --------------------- #

    def to_atoms(self) -> list[Event]:
        r"""Get a list of atoms as `Event` objects in this sigma-algebra.

        Returns
        -------
        atoms : list[Event]
            A list of `Event` objects representing the atoms in this sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> # Define a sigma-algebra with atoms A_0 = {0, 0} and A_1 = {2}
        >>> atom_ids = {0: 0, 1: 0, 2: 1}
        >>> F = SigmaAlgebra(name="F").from_dict(atom_ids)
        >>> print(F) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
            atom ID
        sample
        0         0
        1         0
        2         1
        >>> for atom in F.to_atoms():
        ...     print(atom, "\n") # doctest: +NORMALIZE_WHITESPACE
        Event '0':
        [0, 1]
        <BLANKLINE>
        Event '1':
        [2]
        <BLANKLINE>
        """
        return list(self.atom_id_to_event.values())

    def get_event(self, event_indices: list[Hashable], name: Hashable = "A") -> Event:
        """Create a measurable event from a list of sample points.

        Parameters
        ----------
        event_indices : list[Hashable]
            List of sample points to include in the event. Must form a measurable set
            (i.e., a union of atoms).
        name : Hashable, default="A"
            Name identifier for the event.

        Raises
        ------
        ValueError
            If the provided indices do not form a measurable event with respect to
            this sigma-algebra.

        Returns
        -------
        event : Event
            An `Event` object containing the specified sample points.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> atom_ids = {0: 0, 1: 0, 2: 1, 3: 1}
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(atom_ids)
        >>> # Create a measurable event (union of atoms)
        >>> A = F.get_event([0, 1], name="A")
        >>> print(A) # doctest: +NORMALIZE_WHITESPACE
        Event 'A':
        [0, 1]
        >>> # Trying to create a non-measurable event raises an error
        >>> try:
        ...     B = F.get_event([0, 2], name="B")  # Not a union of atoms
        ... except ValueError as e:
        ...     print(e)
        The provided indices do not form a measurable event.
        """
        from ..base.event import Event

        return Event(sig_alg=self, name=name).from_list(event_indices)

    def get_atom_containing(self, sample_id: Hashable) -> Event:
        """Get the atom containing a given sample point.

        Parameters
        ----------
        sample_id : Hashable
            The sample point for which to retrieve the containing atom.

        Raises
        ------
        ValueError
            If `sample_id` is not in the sample space of this sigma-algebra.

        Returns
        -------
        atom : Event
            The `Event` object representing the atom that contains the given sample point.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> # Define a sigma-algebra with atoms A_0 = {0, 0} and A_1 = {2}
        >>> atom_ids = {0: 0, 1: 0, 2: 1}
        >>> F = SigmaAlgebra(name="F").from_dict(atom_ids)
        >>> print(F) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
            atom ID
        sample
        0         0
        1         0
        2         1
        >>> print(F.get_atom_containing(0)) # doctest: +NORMALIZE_WHITESPACE
        Event 'A':
        [0, 1]
        """
        from ..base import Event

        if sample_id not in self.sample_id_to_atom_id:
            raise ValueError(f"Sample ID '{sample_id}' not in sample space.")
        atom_id = self.sample_id_to_atom_id[sample_id]
        sample_ids = self.atom_id_to_sample_ids[atom_id]
        return Event(sig_alg=self).from_list(sample_ids)

    # --------------------- measurability methods --------------------- #

    def is_measurable(self, event: Event) -> bool:
        r"""Check if an event is measurable with respect to this sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        event : Event
            The event to check for measurability.

        Raises
        ------
        TypeError
            If `event` is not an `Event` instance.
        ValueError
            If `event` does not have the same sample space as this sigma-algebra.

        Returns
        -------
        is_measurable : bool
            `True` if the event is measurable with respect to this sigma-algebra, `False` otherwise.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> # Define a sigma-algebra with atoms A_0 = {0, 1} and A_1 = {2}
        >>> atom_ids = {0: 0, 1: 0, 2: 1}
        >>> F = SigmaAlgebra(name="F").from_dict(atom_ids)
        >>> print(F) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
            atom ID
        sample
        0         0
        1         0
        2         1
        >>> # Create measurable events
        >>> A = F.get_event([0, 1], name="A")
        >>> B = F.get_event([2], name="B")
        >>> # Create a non-measurable event using the power set
        >>> power_set = SigmaAlgebra.power_set(Omega)
        >>> C = power_set.get_event([0], name="C")
        >>> print(F.is_measurable(A))
        True
        >>> print(F.is_measurable(B))
        True
        >>> print(F.is_measurable(C))
        False

        Notes
        -----
        Let $\mathcal{F}$ be a $\sigma$-algebra on a set $\Omega$. Then a subset $A\subset \Omega$ is said to be *$\mathcal{F}$-measurable* if $A\in \mathcal{F}$.
        """
        from ..base import Event

        if not isinstance(event, Event):
            raise TypeError("event must be an Event instance.")
        if event.sample_space != self.sample_space:
            raise ValueError("event must have the same sample_space as the sig_alg.")

        return event.indicator.is_measurable(sig_alg=self)

    def __contains__(self, event: Event) -> bool:
        """Check if an event is measurable with respect to this sigma-algebra.

        Parameters
        ----------
        event : Event
            The event to check for measurability.

        Returns
        -------
        contains : bool
            `True` if the event is measurable with respect to this sigma-algebra, `False` otherwise.
        """
        return self.is_measurable(event)

    # --------------------- sequences methods --------------------- #

    def __iter__(self) -> iter:
        """Iterate over the atom IDs and atoms (as `Events`) in this sigma-algebra.

        Returns
        -------
        iterator : iter
            An iterator over tuples of (atom_id, Event) for each atom in the sigma-algebra.
        """
        return iter(self.atom_id_to_event.items())

    def __len__(self) -> int:
        """Get the number of atoms in the sigma-algebra."""
        return self.num_atoms

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        """Return a string representation of the sigma-algebra.

        Returns
        -------
        repr_str : str
            A string representation of the sigma-algebra.
        """
        return f"Sigma algebra '{self.name}':\n{self.data.to_frame()}"

    # --------------------- equality --------------------- #

    def __eq__(self, other: SigmaAlgebra) -> bool:
        """Check equality with another sigma-algebra.

        Two sigma-algebras are equal if they have the same sample space and contain the same atoms. They may have different names and still be considered equal.

        Parameters
        ----------
        other : SigmaAlgebra
            The other sigma-algebra to compare with.

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

    # --------------------- lattice methods --------------------- #

    def __or__(self, other: SigmaAlgebra) -> SigmaAlgebra:
        r"""Get the join (least upper bound) of this sigma-algebra with another.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        other : SigmaAlgebra
            The other sigma-algebra to join with.

        Returns
        -------
        join_sigma_algebra : SigmaAlgebra
            A new `SigmaAlgebra` instance representing the join of the two sigma-algebras.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=6)
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...             0: 0,
        ...             1: 0,
        ...             2: 0,
        ...             3: 1,
        ...             4: 1,
        ...             5: 1,
        ...     }
        ... )
        >>> G = SigmaAlgebra(sample_space=Omega, name="G").from_dict(
        ...     {
        ...             0: 0,
        ...             1: 1,
        ...             2: 1,
        ...             3: 1,
        ...             4: 0,
        ...             5: 0,
        ...     }
        ... )
        >>> print(F | G) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'join':
            atom ID
        sample
        0       (0, 0)
        1       (0, 1)
        2       (0, 1)
        3       (1, 1)
        4       (1, 0)
        5       (1, 0)

        Notes
        -----
        Let $\{\mathcal{F}_i\}_{k\in K}$ be a finite collection of $\sigma$-algebras on a finite set $\Omega$. The *join* (or *least upper bound*) of the collection, denoted $\bigvee_{k\in K} \mathcal{F}_k$, is the coarsest $\sigma$-algebra that contains all of the $\mathcal{F}_k$. Its atoms are given by the nonempty intersections of atoms from each $\mathcal{F}_k$. In particular, the atom identifiers for the join can be represented as tuples of the atom identifiers from each $\mathcal{F}_k$.
        """
        from .lattice import Lattice

        return Lattice.join([self, other])

    def __le__(self, other: SigmaAlgebra) -> bool:
        """Check if this sigma-algebra is a sub-algebra of another.

        Parameters
        ----------
        other : SigmaAlgebra
            The other sigma-algebra to compare with.

        Raises
        ------
        ValueError
            If the sample spaces of the two sigma-algebras are not the same.

        Returns
        -------
        is_subalgebra : bool
            `True` if this sigma-algebra is a sub-algebra of the other, `False` otherwise.
        """
        from .lattice import Lattice

        if not isinstance(other, SigmaAlgebra):
            return NotImplemented
        if self.sample_space != other.sample_space:
            raise ValueError(
                "Sigma-algebras must have the same sample space for comparison."
            )

        return Lattice.is_subalgebra(sub_algebra=self, super_algebra=other)

    def __lt__(self, other: SigmaAlgebra) -> bool:
        """
        Check if this sigma-algebra is a proper sub-algebra of another.

        Parameters
        ----------
        other : SigmaAlgebra
            The other sigma-algebra to compare with.

        Returns
        -------
        is_proper_subalgebra : bool
            `True` if this sigma-algebra is a proper sub-algebra of the other, `False` otherwise.
        """
        if not isinstance(other, SigmaAlgebra):
            return NotImplemented
        return self <= other and self != other

    def __ge__(self, other: SigmaAlgebra) -> bool:
        """Check if this sigma-algebra is a super-algebra of another.

        Parameters
        ----------
        other : SigmaAlgebra
            The other sigma-algebra to compare with.

        Raises
        ------
        ValueError
            If the sample spaces of the two sigma-algebras are not the same.

        Returns
        -------
        is_superalgebra : bool
            `True` if this sigma-algebra is a super-algebra of the other, `False` otherwise.
        """
        from .lattice import Lattice

        if not isinstance(other, SigmaAlgebra):
            return NotImplemented
        if self.sample_space != other.sample_space:
            raise ValueError(
                "Sigma-algebras must have the same sample space for comparison."
            )

        return Lattice.is_subalgebra(sub_algebra=other, super_algebra=self)

    def __gt__(self, other: SigmaAlgebra) -> bool:
        """Check if this sigma-algebra is a proper super-algebra of another.

        Parameters
        ----------
        other : SigmaAlgebra
            The other sigma-algebra to compare with.

        Returns
        -------
        is_proper_superalgebra : bool
            `True` if this sigma-algebra is a proper super-algebra of the other, `False` otherwise.
        """
        if not isinstance(other, SigmaAlgebra):
            return NotImplemented
        return self >= other and self != other


class SigmaAlgebraMethods:
    """Mixin class providing sigma-algebra methods to other classes."""

    def get_event(self, event_indices: list[Hashable], name: Hashable = "A") -> Event:
        """Create a measurable event from a list of sample points.

        Calls `SigmaAlgebra.get_event`. See the docstring of `SigmaAlgebra.get_event` for details.

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
        return self.sig_alg.get_event(event_indices, name)

    def is_measurable(self, event: Event) -> bool:
        """Check if an event is measurable with respect to the sigma-algebra.

        Calls `SigmaAlgebra.is_measurable`. See the docstring of `SigmaAlgebra.is_measurable` for details.

        Parameters
        ----------
        event : Event
            The event to check for measurability.

        Returns
        -------
        is_measurable : bool
            `True` if the event is measurable with respect to the sigma-algebra, `False` otherwise.
        """
        return self.sig_alg.is_measurable(event)

    def get_atom_containing(self, sample_id: Hashable) -> Event:
        """Get the atom containing a given sample point.

        Calls `SigmaAlgebra.get_atom_containing`. See the docstring of `SigmaAlgebra.get_atom_containing` for details.

        Parameters
        ----------
        sample_id : Hashable
            The sample point for which to retrieve the containing atom.

        Returns
        -------
        atom : Event
            The `Event` object representing the atom that contains the given sample point.
        """
        return self.sig_alg.get_atom_containing(sample_id)
