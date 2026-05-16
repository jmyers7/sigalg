"""A class representing a sample space."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

from .domain import Domain

if TYPE_CHECKING:
    from ..probability_measures import ProbabilityMeasure
    from ..sigma_algebras import SigmaAlgebra
    from .event_space import EventSpace
    from .probability_space import ProbabilitySpace


class SampleSpace(Domain):
    r"""A class representing a sample space.

    See the Notes section below for the mathematical details.

    Parameters
    ----------
    name : Hashable | None, default="Omega"
        Name identifier for the sample space.

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
    In the abstract, a *sample space* is just a set $\Omega$. However, in the context of probability theory, sample spaces are often conceptualized as the set of all possible outcomes of a random experiment. Each element $\omega \in \Omega$ is called a *sample point* or *outcome*. The sample space serves as the foundational building block for defining events (subsets of sample spaces contained in $\sigma$-algebras) and probability measures (functions that assign probabilities to events).

    Sample spaces are not meant to contain data. Instead, data is meant to be encoded in random variables and vectors defined on the sample space, which are functions on sample spaces.
    """

    def __init__(self, name: Hashable | None = "Omega") -> None:
        """The only purpose of this __init__ is to call the superclass's __init__ with a new default name parameter."""  # noqa: D401
        super().__init__(name=name)

    def from_list(
        self,
        indices: list[Hashable],
        data_name: Hashable | None = "sample",
    ) -> SampleSpace:
        """Create a sample space from a list of hashable items.

        This method calls the superclass's `from_list` method with a new default `data_name` parameter.

        Parameters
        ----------
        indices : list[Hashable]
            List of hashable items to use for the index.
        data_name : Hashable | None, default="sample"
            Name for the underlying `pd.Index` object. If `None`, the `pd.Index` will be unnamed.

        Raises
        ------
        TypeError
            If `indices` is not a list of hashable items, or if `data_name` is not hashable (if given).
        ValueError
            If `indices` contains duplicate items.

        Returns
        -------
        self : SampleSpace
            The current `SampleSpace` instance with updated indices.

        Examples
        --------
        >>> from sigalg.core import SampleSpace
        >>> Omega = SampleSpace().from_list(["a", "b", "c"])
        >>> print(Omega) # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega':
        ['a', 'b', 'c']
        """
        return super().from_list(indices=indices, data_name=data_name)

    def from_sequence(
        self,
        size: int,
        initial_index: int = 0,
        prefix: Hashable | None = None,
        data_name: Hashable | None = "sample",
    ) -> SampleSpace:
        """Create a sample space with sequentially numbered items.

        This method calls the superclass's `from_sequence` method with a new default `data_name` parameter.

        Parameters
        ----------
        size : int
            Number of features to generate. Must be positive.
        initial_index : int, default=0
            Starting index for sequential numbering.
        prefix : Hashable | None, default=None
            Prefix for index names. If `None` or non-string hashable is given, then numerical indices are used.

        Returns
        -------
        sample_space : SampleSpace
            A new `SampleSpace` with automatically generated indices.

        Raises
        ------
        ValueError
            If `size` is not a positive integer.
        TypeError
            If `initial_index` is not an integer, `prefix` is not hashable,
            `name` is not hashable, or `data_name` is not hashable (if given).

        Examples
        --------
        >>> from sigalg.core import SampleSpace
        >>> Omega = SampleSpace(name="Omega").from_sequence(size=3, prefix="F")
        >>> print(Omega) # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega':
        ['F_0', 'F_1', 'F_2']
        >>> Omega_2 = SampleSpace(name="Omega_2").from_sequence(size=2, initial_index=5)
        >>> print(Omega_2) # doctest: +NORMALIZE_WHITESPACE
        Sample space 'Omega_2':
        [5, 6]
        """
        return super().from_sequence(
            size=size,
            initial_index=initial_index,
            prefix=prefix,
            data_name=data_name,
        )

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
        Probability space (Omega, power_set, uniform)
        =============================================
        <BLANKLINE>
        * Sample space 'Omega':
        ['a', 'b', 'c']
        <BLANKLINE>
        * Sigma algebra 'power_set':
            atom ID
        sample
        a            a
        b            b
        c            c
        <BLANKLINE>
        * Probability measure 'uniform':
                probability
        sample
        a          0.333333
        b          0.333333
        c          0.333333
        >>> # Create with custom probability measure and sigma-algebra
        >>> F = SigmaAlgebra(sample_space=Omega).from_dict(
        ...     {
        ...         "a": 0,
        ...         "b": 1,
        ...         "c": 1,
        ...     }
        ... )
        >>> P = ProbabilityMeasure(sig_alg=F).from_dict(
        ...     {
        ...         0: 0.2,
        ...         1: 0.8,
        ...     }
        ... )
        >>> prob_space = Omega.make_probability_space(sig_alg=F, prob_measure=P)
        >>> print(prob_space)  # doctest: +NORMALIZE_WHITESPACE
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
        atom ID
        0                0.2
        1                0.8
        """
        from ..probability_measures.probability_measure import ProbabilityMeasure
        from ..sigma_algebras.sigma_algebra import SigmaAlgebra
        from .probability_space import ProbabilitySpace

        if sig_alg is not None and not isinstance(sig_alg, SigmaAlgebra):
            raise TypeError("`sig_alg` must be a `SigmaAlgebra` or `None`.")
        if prob_measure is not None and not isinstance(
            prob_measure, ProbabilityMeasure
        ):
            raise TypeError("`prob_measure` must be a `ProbabilityMeasure` or `None`.")

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
        s0          s0
        s1          s1
        s2          s2
        s3          s3
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
        if sig_alg is None:
            sig_alg = SigmaAlgebra.power_set(sample_space=self)

        return EventSpace(sample_space=self, sig_alg=sig_alg)

    # --------------------- representation --------------------- #

    def __repr__(self) -> str:
        r"""Return a string representation of the sample space.

        Returns
        -------
        repr_str : str
            A formatted string showing the sample space name and its sample points.
        """
        if self.data is None:
            if self.name is None:
                return "Sample space: empty"
            else:
                return f"Sample space '{self.name}': empty"
        else:
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
