"""A class representing a sigma-algebra."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ...validation.mapping_validator import MappingLike
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
    mapping: MappingLike | None, default=None
        The mapping object assigning sample points to atom IDs.
    index : Index | None, default=None
        The index for the atom IDs. See the Examples section below for usage.
    name : Hashable, default="F"
        The name of the sigma-algebra.

    Raises
    ------
    TypeError
        If `name` is not a hashable type, or if `sample_space` is provided and is not a `SampleSpace` instance.

    Examples
    --------
    Construct a `SigmaAlgebra` on the sample space Omega = {0, 1, 2} with atoms A_0 = {1, 2} and A_1 = {1}.

    >>> from sigalg.core import Index, SampleSpace, SigmaAlgebra
    >>> Omega = SampleSpace.from_sequence(size=3)
    >>> mapping = {
    ...     0: 1,
    ...     1: 0,
    ...     2: 0,
    ... }
    >>> F = SigmaAlgebra(sample_space=Omega, mapping=mapping)
    >>> print(F)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'F':
           atom_ID
    sample
    0            1
    1            0
    2            0

    Construct a `SigmaAlgebra` on the same sample space with atoms A_{(1,2)} = {0}, A_{(0,2)} = {1} and A_{(0,1)} = {2}. Note the "2-dimensional" atom IDs.

    >>> mapping = {
    ...     0: (1, 2),
    ...     1: (0, 2),
    ...     2: (0, 1),
    ... }
    >>> G = SigmaAlgebra(sample_space=Omega, mapping=mapping, name="G")
    >>> print(G)  # doctest: +NORMALIZE_WHITESPACE
    Sigma algebra 'G':
           atom_ID
    sample
    0       (1, 2)
    1       (0, 2)
    2       (0, 1)

    Notes
    -----
    A *$\sigma$-algebra* $\mathcal{F}$ on a set $\Omega$ is a collection of subsets of $\Omega$ that contains $\Omega$, and is closed under complementation and countable unions. In the case that $\Omega$ is finite (as it always is, in SigAlg), then $\mathcal{F}$ obviously needs only to be closed under finite unions.

    A $\sigma$-algebra $\mathcal{F}$ determines its *atoms*, which are the nonempty sets $A\in \mathcal{F}$ that are *minimal* with respect to subset inclusion, in the following sense: if $B\in \mathcal{F}$ is nonempty and $B\subset A$, then necessarily $A=B$. And conversely, provided that $\Omega$ is finite, the $\sigma$-algebra $\mathcal{F}$ is completely recoverable from its atoms, in the sense that every event $A\in \mathcal{F}$ is a disjoint union of atoms.

    If $\{A_i\}_{i\in I}$ is the set of atoms, indexed by a finite set $I$, then there is a mapping $\Omega \to I$ given by $\omega \mapsto i$, where $A_i$ is the unique atom that contains $\omega$. This mapping is what SigAlg uses to represent $\sigma$-algebras. The indices in $I$ are called *atom identifiers*. The atom identifiers may consist of tuples, in which case the $\sigma$-algebra is said to have *multi-dimensional* atom identifiers, and the *dimension* of the $\sigma$-algebra is the common length of the tuples.
    """

    # --------------------- constructors --------------------- #

    _properties = [
        "_sample_id_to_atom_id",
        "_dimension",
        "_atom_space",
        "_atom_indicator_df",
        "_num_atoms",
        "_atom_ids",
        "_atom_id_to_sample_ids",
        "_atom_id_to_event",
        "_atom_id_to_cardinality",
        "_is_power_set",
        "_is_trivial",
        "_to_atoms",
    ]

    def __init__(
        self,
        sample_space: SampleSpace | None = None,
        mapping: MappingLike | None = None,
        name: Hashable = "F",
        variable_names: list[Hashable] | None = None,
    ) -> None:
        from ...validation.mapping_validator import MappingValidator

        v = MappingValidator(
            mapping=mapping,
            domain=sample_space,
            name=name,
            output_name="atom_ID",
        )

        if isinstance(v.mapping, pd.DataFrame):
            self._data = v.mapping.apply(tuple, axis=1)
            self._data.name = "atom_ID"
        else:
            self._data = v.mapping

        if variable_names is not None and (
            not isinstance(variable_names, list)
            or any(not isinstance(name, Hashable) for name in variable_names)
        ):
            raise ValueError(
                "If given, variable_names must be a list of hashable items."
            )

        self._initialize_property_caches()

        self._sample_space = v.domain
        self._name = v.name
        self._index = v.index

        if variable_names is None:
            if self.dimension is not None:
                self._variable_names = (
                    [f"atom_{i}" for i in range(self.dimension)]
                    if self.dimension > 1
                    else ["atom"]
                )
            else:
                self._variable_names = None
        else:
            self._variable_names = variable_names

    def _initialize_property_caches(self) -> None:
        for property in self._properties:
            setattr(self, property, None)

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
        Create the power-set sigma-algebra on the sample space Omega1 = {0, 1, 2}.

        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega1 = SampleSpace.from_sequence(size=3, name="Omega1")
        >>> G = SigmaAlgebra.power_set(Omega1, name="G")
        >>> print(G)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
                atom_ID
        sample
        0             0
        1             1
        2             2
        >>> print(G.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega1':
         sample
               0
               1
               2

        Create the power-set sigma-algebra on the sample space Omega2 = {(1,a), (1,b), (2,a), (3,a)}.

        >>> Omega2 = SampleSpace.from_product(
        ...     [1, 2], ["a", "b"], name="Omega2", variable_names=["number", "letter"]
        ... )
        >>> F = SigmaAlgebra.power_set(Omega2, name="F")
        >>> print(F)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                      atom_ID
        number letter
        1      a       (1, a)
               b       (1, b)
        2      a       (2, a)
               b       (2, b)
        >>> print(F.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega2':
         number letter
             1      a
             1      b
             2      a
             2      b

        Notes
        -----
        The *power-set $\sigma$-algebra* on a set $\Omega$ consists of all subsets of $\Omega$. Its atoms are all singleton subsets. It is the finest $\sigma$-algebra on $\Omega$.
        """
        mapping = dict(zip(sample_space, sample_space))
        result = cls(sample_space=sample_space, mapping=mapping, name=name)
        result._atom_space = sample_space
        return result

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
        >>> Omega1 = SampleSpace.from_sequence(size=3, name="Omega1")
        >>> G = SigmaAlgebra.trivial(Omega1, name="G")
        >>> print(G)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'G':
               atom_ID
        sample
        0            0
        1            0
        2            0
        >>> print(G.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'G_atom':
         atom
            0
        >>> Omega2 = SampleSpace.from_product(
        ...     [1, 2], ["a", "b"], name="Omega2", variable_names=["number", "letter"]
        ... )
        >>> F = SigmaAlgebra.trivial(Omega2, name="F")
        >>> print(F)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
                      atom_ID
        number letter
        1      a            0
               b            0
        2      a            0
               b            0
        >>> print(F.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'F_atom':
         atom
            0

        Notes
        -----
        The *trivial $\sigma$-algebra* on a set $\Omega$ consists of only the sets $\Omega$ and $\emptyset$. Its single atom is $\Omega$ itself. It is the coarsest $\sigma$-algebra on $\Omega$.
        """
        mapping = dict.fromkeys(sample_space.data, 0)
        return cls(sample_space=sample_space, mapping=mapping, name=name)

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
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra.power_set(Omega)
        >>> A = F.get_event([0, 2])
        >>> sigma_A = SigmaAlgebra.from_event(A)
        >>> print(sigma_A) # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'sigma(A)':
               atom_ID
        sample
        0            1
        1            0
        2            1

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
            mapping=sample_id_to_atom_id,
            name=name,
        )

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
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> X = RandomVector(
        ...     domain=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (1, 2),
        ...         2: (2, 4),
        ...     },
        ... )
        >>> sigma_X = SigmaAlgebra.from_random_vector(rv=X)
        >>> print(sigma_X)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'sigma_X':
               atom_ID
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

        name = f"sigma_{rv.name}"

        return cls(sample_space=rv.domain, mapping=rv.point_outputs, name=name)

    # --------------------- properties --------------------- #

    @property
    def sample_space(self) -> SampleSpace | None:
        """Get the sample space over which this sigma-algebra is defined.

        The `sample_space` property is settable. If the `SigmaAlgebra` instance already has a sample space, the new sample space must contain the same number of sample points.

        Returns
        -------
        sample_space : SampleSpace | None
            The sample space of this sigma-algebra.

        Examples
        --------
        Define a `SigmaAlgebra` on the sample space Omega = {0, 1, 2} with atoms A_0 = {1} and A_1 = {0, 2}. Print the underlying sample space.

        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 1,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ...     name="F",
        ... )
        >>> print(F.sample_space)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega':
         sample
              0
              1
              2

        Set a new sample space on the sigma-algebra to Omega_new = {a, b, c}. Notice that Omega_new is in bijective correspondence to the first sample space Omega.

        >>> Omega_new = SampleSpace(["a", "b", "c"], name="Omega_new")
        >>> F.sample_space = Omega_new
        >>> print(F.sample_space)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega_new':
         sample
              a
              b
              c
        >>> print(F)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'F':
               atom_ID
        sample
        a            1
        b            0
        c            1
        """
        return self._sample_space

    @sample_space.setter
    def sample_space(self, sample_space: SampleSpace) -> None:
        """Set the sample space of this sigma-algebra.

        If the `SigmaAlgebra` instance already has a sample space, the new sample space must contain the same number of sample points.

        Parameters
        ----------
        sample_space : SampleSpace
            The new sample space for this sigma-algebra.

        Raises
        ------
        TypeError
            If `sample_space` is not a `SampleSpace` instance.
        ValueError
            If the new sample space does not have the same number of points as the existing sample space.
        """
        from ..base.sample_space import SampleSpace

        if not isinstance(sample_space, SampleSpace):
            raise TypeError("sample_space must be a SampleSpace instance.")

        if self.sample_space is not None:
            if len(sample_space) != len(self.sample_space):
                raise ValueError(
                    "New sample space must have the same number of points as the existing sample space."
                )

            if self.data is not None:
                self.data.index = sample_space.data

            self._sample_id_to_atom_id = None
            self._atom_id_to_sample_ids = None
            self._atom_id_to_event = None
            self._to_atoms = None

        self._sample_space = sample_space

    @property
    def sample_id_to_atom_id(self) -> Mapping[Hashable, Hashable] | None:
        """Get the mapping from sample points to atom IDs.

        Returns
        -------
        sample_id_to_atom_id : Mapping[Hashable, Hashable] | None
            A mapping from sample IDs to atom IDs.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ...     name="F",
        ... )
        >>> print(F.sample_id_to_atom_id)
        {0: 0, 1: 0, 2: 1}
        """
        if self._sample_id_to_atom_id is None and self.data is not None:
            if isinstance(self.data, pd.Series):
                self._sample_id_to_atom_id = self.data.to_dict()
            else:
                self._sample_id_to_atom_id = self.data.apply(tuple, axis=1).to_dict()
        return self._sample_id_to_atom_id

    @property
    def data(self) -> pd.Series | pd.DataFrame | None:
        """Get the underlying `pd.Series` or `pd.DataFrame`.

        Returns
        -------
        data: pd.Series | pd.DataFrame | None
            A `pd.Series` or `pd.DataFrame` mapping sample points to atom IDs.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ...     name="F",
        ... )
        >>> print(F.data) # doctest: +NORMALIZE_WHITESPACE
        sample
        0    0
        1    0
        2    1
        Name: atom_ID, dtype: int64
        """
        return self._data

    @property
    def atom_space(self) -> SampleSpace | None:
        """Get the sample space of atom identifiers.

        Returns
        -------
        atom_space: SampleSpace | None
            The sample space whose points are the atom identifiers of the sigma-algebra.

        Examples
        --------
        Define a `SigmaAlgebra` on the sample space Omega = {0,1,2} with atoms A_0 = {1,2} and A_1 = {0}.
        >>> from sigalg.core import Index, SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 1,
        ...         1: 0,
        ...         2: 0,
        ...     },
        ... )

        The atom space is an instance of `SampleSpace` consisting of the atom IDs, 0 and 1.

        >>> print(F.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'F_atom':
         atom
            1
            0

        Create a second sigma-algebra with 2-dimensional atom IDs.

        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (0, 2),
        ...         2: (0, 1),
        ...     },
        ...     name="G",
        ... )
        >>> print(G.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'G_atom':
         atom_0  atom_1
              1       2
              0       2
              0       1

        Define a third sigma-algebra with custom variable names for its atom space.

        >>> H = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: (1, 2),
        ...         1: (0, 2),
        ...         2: (0, 1),
        ...     },
        ...     name="H",
        ...     variable_names=["x", "y"],
        ... )
        >>> print(H.atom_space)  # doctest: +NORMALIZE_WHITESPACE
        Sample space 'H_atom':
         x  y
         1  2
         0  2
         0  1
        """
        from ..base.sample_space import SampleSpace

        if self._atom_space is None and self.data is not None:
            self._atom_space = SampleSpace(
                self.atom_ids,
                name=f"{self.name}_atom",
                variable_names=self.variable_names,
            )

        return self._atom_space

    @property
    def variable_names(self) -> list[Hashable] | None:
        """Pass."""
        return self._variable_names

    @property
    def dimension(self) -> int | None:
        """Pass."""
        if self._dimension is None and self.data is not None:
            first_ID = next(iter(self.data))
            if isinstance(first_ID, tuple):
                self._dimension = len(first_ID)
            else:
                self._dimension = 1
        return self._dimension

    @property
    def atom_indicator_df(self) -> pd.DataFrame | None:
        """Get a `pd.DataFrame` whose columns are indicators for membership of each sample point in the atoms of the sigma-algebra.

        Returns
        -------
        atom_indicator_df : pd.DataFrame | None
            A DataFrame where each column corresponds to an atom of the sigma-algebra and each row corresponds to a sample point. The entries are 1 if the sample point belongs to the atom and 0 otherwise.

        Examples
        --------
        Define a `SigmaAlgebra` on the sample space Omega = {0,1,2,3,4,5} with atoms A_a = {2,3}, A_b = {0,1}, and A_c = {4,5}.

        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: "b",
        ...         1: "b",
        ...         2: "a",
        ...         3: "a",
        ...         4: "c",
        ...         5: "c",
        ...     },
        ... )
        >>> print(F.atom_indicator_df)  # doctest: +NORMALIZE_WHITESPACE
                a  b  c
        sample
        0       0  1  0
        1       0  1  0
        2       1  0  0
        3       1  0  0
        4       0  0  1
        5       0  0  1

        Define a sigma-algebra with 2-dimensional atom IDs.

        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: ("a", "b"),
        ...         1: ("a", "b"),
        ...         2: ("a", "b"),
        ...         3: ("c", "d"),
        ...         4: ("c", "d"),
        ...         5: ("a", "b"),
        ...     },
        ...     name="G",
        ... )
        >>> print(G.atom_indicator_df)  # doctest: +NORMALIZE_WHITESPACE
                (a, b)  (c, d)
        sample
        0            1       0
        1            1       0
        2            1       0
        3            0       1
        4            0       1
        5            1       0
        """
        if self._atom_indicator_df is None and self.data is not None:
            self._atom_indicator_df = pd.get_dummies(self.data).astype(int)

        return self._atom_indicator_df

    @property
    def name(self) -> Hashable | None:
        """Get the name identifier for this sigma algebra.

        Returns
        -------
        name : Hashable | None
            The name of this sigma algebra.
        """
        return self._name

    @name.setter
    def name(self, name: Hashable | None) -> None:
        """Set the name identifier for this sigma algebra.

        Parameters
        ----------
        name : Hashable | None
            New name for this sigma algebra.

        Raises
        ------
        TypeError
            If `name` is not a hashable.
        """
        if name is not None and not isinstance(name, Hashable):
            raise TypeError("name must be a hashable type.")
        self._name = name

    def with_name(self, name: Hashable | None) -> SigmaAlgebra:
        """Set the name of the sigma-algebra and return self for chaining.

        Parameters
        ----------
        name : Hashable | None
            The new name for the sigma algebra.

        Returns
        -------
        self : SigmaAlgebra
            The current instance with the updated name.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> print(F.with_name("sig_alg"))  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'sig_alg':
               atom_ID
        sample
        0            0
        1            0
        2            1
        """
        self.name = name
        return self

    @property
    def num_atoms(self) -> int | None:
        """Get the number of atoms in this sigma-algebra.

        Returns
        -------
        num_atoms : int | None
            The number of atoms in this sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> print(F.num_atoms)
        2
        """
        if self._num_atoms is None and self.data is not None:
            if isinstance(self.data, pd.DataFrame):
                self._num_atoms = len(self.data.drop_duplicates())
            else:
                self._num_atoms = self.data.nunique()
        return self._num_atoms

    @property
    def atom_ids(self) -> list[Hashable] | None:
        """Get a list of atom IDs in this sigma-algebra.

        Returns
        -------
        atom_ids : list[Hashable] | None
            A list of atom IDs in this sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> print(F.atom_ids)
        [0, 1]
        """
        if self._atom_ids is None and self.data is not None:
            if isinstance(self.data, pd.DataFrame):
                self._atom_ids = list(
                    self.data.drop_duplicates().itertuples(index=False, name=None)
                )
            else:
                self._atom_ids = list(self.data.drop_duplicates())
        return self._atom_ids

    @property
    def atom_id_to_sample_ids(self) -> dict[Hashable, list[Hashable]] | None:
        """Get a mapping from atom IDs to lists of sample points.

        Returns
        -------
        atom_id_to_sample_ids : dict[Hashable, list[Hashable]] | None
            A dictionary mapping each atom ID to a list of sample points contained in that atom.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> print(F.atom_id_to_sample_ids)
        {0: [0, 1], 1: [2]}
        """
        if (
            self._atom_id_to_sample_ids is None
            and self.sample_id_to_atom_id is not None
        ):
            atom_id_to_sample_ids = {}
            for sample_id, atom_id in self.sample_id_to_atom_id.items():
                if atom_id not in atom_id_to_sample_ids:
                    atom_id_to_sample_ids[atom_id] = []
                atom_id_to_sample_ids[atom_id].append(sample_id)
            self._atom_id_to_sample_ids = atom_id_to_sample_ids
        return self._atom_id_to_sample_ids

    @property
    def atom_id_to_event(self) -> dict[Hashable, Event] | None:
        r"""Get a mapping from atom IDs to `Event` objects in this sigma-algebra.

        Returns
        -------
        atom_id_to_event : dict[Hashable, Event] | None
            A dictionary mapping each atom ID to its corresponding `Event` object.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> for atom_id, event in F.atom_id_to_event.items():
        ...     print(f"Atom ID: {atom_id}\n{event}\n") # doctest: +NORMALIZE_WHITESPACE
        Atom ID: 0
        Event '0':
         sample
              0
              1
        <BLANKLINE>
        Atom ID: 1
        Event '1':
         sample
              2
        <BLANKLINE>
        """
        if self._atom_id_to_event is None and self.atom_id_to_sample_ids is not None:
            atom_id_to_event = {
                atom_id: self.get_event(sample_ids, name=atom_id)
                for atom_id, sample_ids in self.atom_id_to_sample_ids.items()
            }
            self._atom_id_to_event = atom_id_to_event
        return self._atom_id_to_event

    @property
    def atom_id_to_cardinality(self) -> dict[Hashable, int] | None:
        """Get a mapping from atom IDs to their cardinalities in this sigma-algebra.

        Returns
        -------
        atom_id_to_cardinality : dict[Hashable, int] | None
            A dictionary mapping each atom ID to the number of sample points it contains.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace.from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> print(F.atom_id_to_cardinality)
        {0: 2, 1: 1}
        """
        if (
            self._atom_id_to_cardinality is None
            and self.atom_id_to_sample_ids is not None
        ):
            self._atom_id_to_cardinality = {
                atom_id: len(lst) for atom_id, lst in self.atom_id_to_sample_ids.items()
            }
        return self._atom_id_to_cardinality

    @property
    def is_power_set(self) -> bool | None:
        """Boolean flag signaling a power-set sigma-algebra.

        Returns
        -------
        is_power_set: bool | None
            A boolean signaling whether the sigma-algebra is the power-set sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> atom_ids = dict(zip(Omega, [0, 0, 1]))
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> print(F.is_power_set)
        False
        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 1,
        ...         1: 6,
        ...         2: 7,
        ...     },
        ...     name="G",
        ... )
        >>> print(G.is_power_set)
        True
        """
        if self._is_power_set is None and self.data is not None:
            self._is_power_set = self.num_atoms == len(self._sample_space)
        return self._is_power_set

    @property
    def is_trivial(self) -> bool | None:
        """Boolean flag signaling a trivial sigma-algebra.

        Returns
        -------
        is_trivial: bool | None
            A boolean signaling whether the sigma-algebra is the trivial sigma-algebra.

        Examples
        --------
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> atom_ids = dict(zip(Omega, [0, 0, 1]))
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> print(F.is_trivial)
        False
        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 1,
        ...         1: 1,
        ...         2: 1,
        ...     },
        ...     name="G",
        ... )
        >>> print(G.is_trivial)
        True
        """
        if self._is_trivial is None and self.data is not None:
            self._is_trivial = self.num_atoms == 1
        return self._is_trivial

    # TODO: possibly rename?
    @property
    def to_atoms(self) -> list[Event] | None:
        r"""Get a list of atoms as `Event` objects in this sigma-algebra.

        An alias for the `to_atoms` method.

        Returns
        -------
        atoms : list[Event] | None
            A list of `Event` objects representing the atoms in this sigma-algebra.

        Examples
        --------
        Define a `SigmaAlgebra` on the sample space Omega = {0,1,2} with atoms A_0 = {0,1} and A_1 = {2}.

        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> for atom in F.to_atoms:
        ...     print(atom, "\n") # doctest: +NORMALIZE_WHITESPACE
        Event '0':
         sample
              0
              1
        <BLANKLINE>
        Event '1':
         sample
              2
        <BLANKLINE>
        """
        if self._to_atoms is None and self.atom_id_to_event is not None:
            self._to_atoms = list(self.atom_id_to_event.values())
        return self._to_atoms

    # --------------------- atom and event methods --------------------- #

    def get_event(self, indices: list[Hashable], name: Hashable = "A") -> Event:
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
        Create a `SigmaAlgebra` on the sample space Omega = {0,1,2,3} with atoms A_0 = {0,1} and A_1 = {2,3}.
        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=4)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...         3: 1,
        ...     },
        ... )

        Get the event A = {0,1}, which is just the atom A_0 and is thus measurable with respect to the sigma-algebra.

        >>> A = F.get_event([0, 1], name="A")
        >>> print(A)  # doctest: +NORMALIZE_WHITESPACE
        Event 'A':
         sample
              0
              1

        Try to create the "event" B = {0,2}. Note that this set is not measurable with respect to the sigma-algebra because it is not a union of atoms.

        >>> try:
        ...     B = F.get_event([0, 2], name="B")
        ... except ValueError as e:
        ...     print(e)
        The event is not measurable.
        """
        from ..base.event import Event

        return Event.from_list(indices=indices, sig_alg=self, name=name)

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
        Define a `SigmaAlgebra` on the sample space Omega = {0,1,2} with atoms A_0 = {0,1} and A_1 = {2}.

        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
        >>> print(F.get_atom_containing(0)) # doctest: +NORMALIZE_WHITESPACE
        Event 'A':
         sample
              0
              1
        """
        from ..base import Event

        if sample_id not in self.sample_id_to_atom_id:
            raise ValueError(f"Sample ID '{sample_id}' not in sample space.")
        atom_id = self.sample_id_to_atom_id[sample_id]
        sample_ids = self.atom_id_to_sample_ids[atom_id]
        return Event.from_list(indices=sample_ids, sig_alg=self)

    # --------------------- measurability methods --------------------- #

    def is_measurable(
        self,
        event: Event | None = None,
        event_list: list[Hashable] | None = None,
    ) -> bool:
        r"""Check if an event is measurable with respect to this sigma-algebra.

        See the Notes section below for the mathematical details.

        Parameters
        ----------
        event : Event | None, default=None
            The event to check for measurability.
        event_list : list[Hashable] | None, default=None
            A list of sample points to check for measurability.

        Raises
        ------
        TypeError
            If `event` is not an `Event` instance or if `event_list` is not a list of hashable sample points.
        ValueError
            If both `event` and `event_list` are provided or if neither is provided, or if `event` does not have the same sample space as this sigma-algebra.

        Returns
        -------
        is_measurable : bool
            `True` if the event is measurable with respect to this sigma-algebra, `False` otherwise.

        Examples
        --------
        Define a `SigmaAlgebra` on the sample space Omega = {0,1,2} with atoms A_0 = {0,1} and A_1 = {2}.

        >>> from sigalg.core import SampleSpace, SigmaAlgebra
        >>> Omega = SampleSpace().from_sequence(size=3)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 1,
        ...     },
        ... )
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

        if event is not None and not isinstance(event, Event):
            raise TypeError("event must be an Event instance.")
        if event_list is not None and not isinstance(event_list, list):
            raise TypeError("event_list must be a list of sample points.")
        if event is not None and event_list is not None:
            raise ValueError("Only one of event or event_list should be provided.")
        if event is None and event_list is None:
            raise ValueError("Either event or event_list must be provided.")
        if event is not None and event.sample_space != self._sample_space:
            raise ValueError("event must have the same sample_space as the sig_alg.")

        if event is not None:
            return event.indicator.is_measurable(sig_alg=self)
        else:
            try:
                self.get_event(event_list)
                return True
            except ValueError:
                return False

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

    # --------------------- sequence methods --------------------- #

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
        if self.data is None:
            return f"Sigma algebra '{self.name}': empty"
        else:
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
        if self._sample_space != other._sample_space:
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
        >>> Omega = SampleSpace.from_sequence(size=6)
        >>> F = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 0,
        ...         2: 0,
        ...         3: 1,
        ...         4: 1,
        ...         5: 1,
        ...     },
        ... )
        >>> G = SigmaAlgebra(
        ...     sample_space=Omega,
        ...     mapping={
        ...         0: 0,
        ...         1: 1,
        ...         2: 1,
        ...         3: 1,
        ...         4: 0,
        ...         5: 0,
        ...     },
        ...     name="G",
        ... )
        >>> join = F | G
        >>> print(join)  # doctest: +NORMALIZE_WHITESPACE
        Sigma algebra 'join':
               atom_ID
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
        if self._sample_space != other._sample_space:
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
        if self._sample_space != other._sample_space:
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

    def get_event(self, indices: list[Hashable], name: Hashable = "A") -> Event:
        """Create a measurable event from a list of sample points.

        Calls `SigmaAlgebra.get_event`. See the docstring of `SigmaAlgebra.get_event` for details.

        Parameters
        ----------
        event_indices : list[Hashable]
            List of sample points to include in the event. Must form a measurable set
            (i.e., a union of atoms).
        name : Hashable, default="A"
            Name identifier for the event.

        Returns
        -------
        event : Event
            An `Event` object containing the specified sample points.
        """
        return self.sig_alg.get_event(indices, name)

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
